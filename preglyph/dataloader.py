import os
import random
import string
import torchvision.transforms as transforms
import torch.utils.data as data
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from random import choice, randint
from utils import *

import cv2
import numpy as np
import math


class LanguageAdaptiveLabelDataset(data.Dataset):

    def __init__(self, lang, length, font_path, character_path, min_len, max_len, rec_image_shape) -> None:
        super().__init__()

        self.lang = lang
        self.character = self.get_multilingual_character(lang, character_path)

        self.length = length
        self.font_dir = font_path

        self.min_len = min_len
        self.max_len = max_len
        
        self.rec_image_shape = rec_image_shape
        self.imgC, self.imgH, self.imgW = self.rec_image_shape
        self.fixed_ratio = self.imgW / self.imgH

        self.grayscale = transforms.Grayscale()

        self.words = []
        self.word_path = self.get_word_path(lang)
        with open(self.word_path, 'r') as f:
            lines = f.readlines()
            for word in lines:
                self.words.append(word.strip())

    def __len__(self):
        
        return self.length
        
    def resize_norm_img(self, img, max_wh_ratio):
        assert self.imgC == img.shape[2]
        imgW = int((self.imgH * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(self.imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(self.imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, self.imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((self.imgC, self.imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def get_multilingual_character(self, lang, character_path):        
        with open(character_path, 'r') as f:
            lines = f.readlines()
            characters = lines[0]
            
        if lang == "ar":
            return characters
        else:
            return string.digits + string.punctuation + characters
        
    def get_word_path(self, lang):
        if lang == "ko":
            return "words/words_ko_2k.txt"
        elif lang == "jp":
            return "words/words_jp_2k.txt"
        elif lang == "ar":
            return "words/words_ar_2k.txt"
        else:
            raise ValueError("Unsupported language. Supported languages are: ko, jp, ar.")

    def __getitem__(self, index):

        while True:

            if random.random() < 0.5:
                text_len = randint(self.min_len, self.max_len)
                text = "".join([choice(self.character) for i in range(text_len)])
            else:
                text = random.choice(self.words)

            try:
                font = ImageFont.truetype(os.path.join(self.font_dir, choice(os.listdir(self.font_dir))), 128)
                
                std_l, std_t, std_r, std_b = font.getbbox(text)
                std_h, std_w = std_b - std_t, std_r - std_l
                if std_h == 0 or std_w == 0:
                    continue
            except:
                continue
            
            try:
                image = Image.new('RGB', (std_w, std_h), color = (0,0,0))
                draw = ImageDraw.Draw(image)
                draw.text((0, 0), text, fill = (255,255,255), font=font, anchor="lt")
            except:
                continue
            image = transforms.ToTensor()(image)
            image = image.permute(1, 2, 0).numpy()

            image = self.resize_norm_img(image, self.fixed_ratio)

            batch = {
                "image": image,
                "text": text
            }

            return batch


def get_dataloader(cfgs, datype="train"):

    dataset_cfgs = OmegaConf.load(cfgs.dataset_cfg_path)
    print(f"Extracting data from {dataset_cfgs.target}")
    Dataset = eval(dataset_cfgs.target)
    dataset = Dataset(dataset_cfgs.params, datype = datype)

    return data.DataLoader(dataset=dataset, batch_size=cfgs.batch_size, shuffle=cfgs.shuffle, num_workers=cfgs.num_workers, drop_last=True)

