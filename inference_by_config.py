import os
import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from pytorch_lightning import seed_everything

from utils import create_model, load_state_dict
from src.stellar.stellar import STELLAR
from src.stellar.stellar_app import text_editing

def load_image(image_path, image_height=256, image_width=256):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = Image.open(image_path)
    image = T.ToTensor()(T.Resize((image_height, image_width))(img.convert("RGB")))
    image = image.to(device)
    return image.unsqueeze(0)

def main(cfg):
    cfg_path = f'configs_inference/{cfg.config}.yaml'
    cfg_inf = OmegaConf.load(cfg_path).inference

    ckpt_path = cfg_inf.ckpt_path
    dataset_dir = cfg_inf.dataset_dir
    dataset_style_dir = cfg_inf.dataset_style_dir
    dataset_style_txt = cfg_inf.dataset_style_txt
    dataset_target_txt = cfg_inf.dataset_target_txt
    output_dir = cfg_inf.output_dir
    seed = cfg_inf.seed
    guidance_scale = cfg_inf.guidance_scale
    num_inference_steps = cfg_inf.num_inference_steps
    max_length = cfg_inf.max_length

    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    model = create_model(cfg_path).cuda()

    model.load_state_dict(load_state_dict(ckpt_path), strict=False)
    model.eval()

    style_dir = os.path.join(dataset_dir, dataset_style_dir)
    style_images_path = {image_name: os.path.join(style_dir, image_name) for image_name in os.listdir(style_dir)}

    style_txt = os.path.join(dataset_dir, dataset_style_txt)
    style_dict = {}
    with open(style_txt, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                image_name, text = line.strip().split(' ')[:]
                style_dict[image_name] = text

    target_txt = os.path.join(dataset_dir, dataset_target_txt)
    target_dict = {}
    with open(target_txt, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                image_name, text = line.strip().split(' ')[:]
                target_dict[image_name] = text

    pipeline = STELLAR(model, max_length)

    style_image_keys = sorted(list(style_images_path.keys()))

    for image_name in tqdm(style_image_keys):

        image_path = style_images_path[image_name]
        style_text = style_dict[image_name]
        target_text = target_dict[image_name]
        w,h = Image.open(image_path).size
        source_image = load_image(image_path)
        style_image = load_image(image_path)

        result = text_editing(
            pipeline, 
            source_image, 
            style_image, 
            style_text, 
            target_text,
            ddim_steps=num_inference_steps,
            scale=guidance_scale,
        )

        _, image_stellar = result[:]
        image_stellar = Image.fromarray((image_stellar * 255).astype(np.uint8)).resize((w, h))
        image_stellar.save(os.path.join(output_dir, image_name))

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    cfg = parser.parse_args()

    main(cfg)