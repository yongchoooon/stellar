import torch

def text_editing(model, source_image, style_image, style_text, target_text, ddim_steps=50, scale=2):

    with torch.no_grad():
        prompts = [style_text, target_text]
        inversion_prompt = [style_text]
        start_code, latents_list = model.inversion(source_image,
                                                style_image,
                                                inversion_prompt,
                                                guidance_scale=scale,
                                                num_inference_steps=ddim_steps,
                                                return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1)
        image_stellar = model(
                style_image.expand(len(prompts), -1, -1, -1),
                prompts,
                num_inference_steps=ddim_steps,
                latents=start_code,
                guidance_scale=scale,
        )
        image_stellar = image_stellar.clamp(0, 1)
        image_stellar = image_stellar.cpu().permute(0, 2, 3, 1).numpy()

    return [
        image_stellar[0],
        image_stellar[1],
    ] 
