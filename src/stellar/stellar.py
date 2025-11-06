import torch
import numpy as np

class STELLAR:
    def __init__(self, model, max_length):
        self.model = model
        self.scheduler = model.noise_scheduler
        self.device = model.device
        self.unet = model.unet
        self.vae = model.vae
        self.control_model = model.control_model
        self.text_encoder = model.text_encoder
        self.NORMALIZER = model.NORMALIZER

        self.max_length = max_length + 1

    def next_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
    ):
        """
        predict the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            assert image.dim() == 4, print("input dims should be 4 !")
            latents = self.vae.encode(image.to(self.device)).latent_dist.sample()
            latents = latents * self.NORMALIZER
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.NORMALIZER * latents.detach()
        image = self.model.vae.decode(latents).sample
        if return_type == 'np':
            image = image.clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    def latent2image_grad(self, latents):
        latents = 1 / self.NORMALIZER * latents
        image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def inversion(
            self,
            image: torch.Tensor,
            hint: torch.Tensor,
            cond,
            num_inference_steps=50,
            guidance_scale=7.5,
            return_intermediates=False,
            **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """

        assert image.shape[0] == len(cond), print("Unequal batch size for image and cond.")
        assert image.shape[0] == hint.shape[0], print("Unequal batch size for image and hint.")

        cond_embeddings = self.model.get_text_conditioning(cond)
        if guidance_scale > 1.:
            uncond = [""] * len(cond)
            uncond_embeddings = self.model.get_text_conditioning(uncond)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        else:
            text_embeddings = cond_embeddings

        latents = self.image2latent(image)
        start_latents = latents


        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(reversed(self.scheduler.timesteps)):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
                hint_input = torch.cat([hint] * 2)
            else:
                model_inputs = latents
                hint_input = hint

            control_input = self.control_model(hint_input, model_inputs, t, text_embeddings)

            noise_pred = self.unet(
                x=model_inputs,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                control=control_input
            ).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
        if return_intermediates:
            return latents, latents_list

        return latents, start_latents

    @torch.no_grad()
    def __call__(
            self,
            hint,
            cond,
            batch_size=1,
            height=256,
            width=256,
            num_inference_steps=50,
            guidance_scale=2,
            latents=None,
            unconditioning=None,
            ref_intermediate_latents=None,
            return_intermediates=False,
            **kwds):

        cond_embeddings = self.model.get_text_conditioning(cond)
        if guidance_scale > 1.:
            uncond = [""] * len(cond)
            uncond_embeddings = self.model.get_text_conditioning(uncond)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        else:
            text_embeddings = cond_embeddings

        batch_size = len(cond)
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=self.device)
        else:
            pass

        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]

        for i, t in enumerate(self.scheduler.timesteps):
            if ref_intermediate_latents is not None:
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
                hint_input = torch.cat([hint] * 2)
            else:
                model_inputs = latents
                hint_input = hint

            control_input = self.control_model(hint_input, model_inputs, t, text_embeddings)

            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
            noise_pred = self.unet(
                x=model_inputs,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                control=control_input,
            ).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

            latents, pred_x0 = self.step(noise_pred, t, latents)
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates: # False
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, latents_list, pred_x0_list
        return image