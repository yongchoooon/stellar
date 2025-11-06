
import torch
import torch.nn as nn
import math
import string
import numpy as np
import pytorch_lightning as pl

from typing import Union
from paddleocr import PaddleOCR
from transformers import get_cosine_schedule_with_warmup

from utils import instantiate_from_config

def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast
class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None
        self._emb_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def emb_key(self) -> str:
        return self._emb_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @emb_key.setter
    def emb_key(self, value: str):
        self._emb_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @emb_key.deleter
    def emb_key(self):
        del self._emb_key


def get_lang_for_ppocr(lang):
    if lang == "ko":
        return "korean"
    elif lang == "jp":
        return "japan"
    elif lang == "ar":
        return "arabic"
    else:
        raise ValueError(f"Unsupported language: {lang}")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.tile(self.pe[None, ...].to(x.device), (x.shape[0], 1, 1))
        return self.dropout(x)


class PPOCREncoder(nn.Module):
    def __init__(self, freeze=True, *args, **kwargs):
        super().__init__()
        self.lang = get_lang_for_ppocr(kwargs.pop("lang", ""))
        self.ocr = PaddleOCR(lang = self.lang, **kwargs)
        self.tr = self.ocr.text_recognizer
        _, self.imgH, self.imgW = self.tr.rec_image_shape
        self.fixed_ratio = self.imgW / self.imgH

    def forward(self, x):
        device = x.device
        x = x.cpu().numpy()  # Convert torch tensor to numpy array

        self.tr.input_tensor.copy_from_cpu(x)
        self.tr.predictor.run()

        x = [t.copy_to_cpu() for t in self.tr.output_tensors]
        x = x[0]
        x = torch.tensor(x, device=device)  # Convert back to torch tensor and retain device

        if self.lang == "arabic":
            x = torch.flip(x, dims=[1])

        return x
    
    def encode(self, x):
        return self(x)


class LanguageAdaptiveLabelEncoder(AbstractEmbModel, pl.LightningModule):

    def __init__(self,
                 lang,
                 character_path,
                 max_len,
                 emb_dim,
                 n_heads=8,
                 n_trans_layers=12,
                 trainable=False,
                 lr=1e-4,
                 clip_dim=1024,
                 visual_len=197,
                 visual_dim=768,
                 visual_config=None,
                 cosine_annealing = False,
                 warmup = False,
                 eta_min = None,
                 T_max = None,
                 multi_step_lr = False,
                 milestone = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lang = lang
        self.character_path = character_path

        self.max_len = max_len
        self.emd_dim = emb_dim
        self.n_heads = n_heads
        self.n_trans_layers = n_trans_layers

        self.cosine_annealing = cosine_annealing
        self.eta_min = eta_min
        self.T_max = T_max

        self.multi_step_lr = multi_step_lr
        self.milestone = milestone

        self.warmup = warmup
        
        self.character = self.get_multilingual_character(lang, character_path)
        self.num_cls = len(self.character) + 1

        self.label_embedding = nn.Embedding(self.num_cls, self.emd_dim)
        self.pos_embedding = PositionalEncoding(d_model=self.emd_dim, max_len=self.max_len)
        transformer_block = nn.TransformerEncoderLayer(d_model=self.emd_dim, nhead=self.n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_block, num_layers=self.n_trans_layers)

        if trainable:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.visual_encoder = instantiate_from_config(visual_config)

            self.learning_rate = lr
            self.clip_dim = clip_dim
            self.visual_len = visual_len
            self.visual_dim = visual_dim

            self.text_head = nn.Sequential(*[
                nn.InstanceNorm1d(self.max_len),
                nn.Linear(self.emd_dim, self.clip_dim, bias=False),
                nn.Conv1d(in_channels=self.max_len, out_channels=1, kernel_size=1)
            ])

            self.visual_head = nn.Sequential(*[
                nn.InstanceNorm1d(self.visual_len),
                nn.Linear(self.visual_dim, self.clip_dim, bias=False),
                nn.Conv1d(in_channels=self.visual_len, out_channels=1, kernel_size=1)
            ])


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_index(self, labels):

        indexes = []
        for label in labels:
            if len(label) > self.max_len:
                label = label[:self.max_len]
            index = [self.character.find(c) + 1 for c in label]
            index = index + [0] * (self.max_len - len(index))
            indexes.append(index)

        return torch.tensor(indexes, device=next(self.parameters()).device)

    def get_multilingual_character(self, lang, character_path):        
        with open(character_path, 'r') as f:
            lines = f.readlines()
            characters = lines[0]
        if lang == "ar":
            return characters
        else:
            return string.digits + string.punctuation + characters

    def get_embeddings(self, x):

        emb = self.label_embedding(x)
        emb = self.pos_embedding(emb)
        out = self.encoder(emb)

        return out

    def forward(self, labels):

        idx = self.get_index(labels)
        out = self.get_embeddings(idx)

        return out

    def encode(self, text):
        return self(text)

    def get_loss(self, text_out, visual_out, clip_target):

        text_out = text_out / text_out.norm(dim=1, keepdim=True)
        visual_out = visual_out / visual_out.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * visual_out @ text_out.T
        logits_per_text = logits_per_image.T
        clip_loss_image = nn.functional.cross_entropy(logits_per_image, clip_target)
        clip_loss_text = nn.functional.cross_entropy(logits_per_text, clip_target)
        clip_loss = (clip_loss_image + clip_loss_text) / 2
        return clip_loss,  logits_per_text

    def training_step(self, batch, batch_idx):

        text = batch["text"]
        image = batch["image"]
        idx = self.get_index(text)
        text_emb = self.get_embeddings(idx)
        visual_emb = self.visual_encoder(image)
        text_out = self.text_head(text_emb).squeeze(1)
        visual_out = self.visual_head(visual_emb).squeeze(1)

        cls_target = idx
        clip_target = torch.arange(0, idx.shape[0], 1).to(cls_target)

        clip_loss, logits_per_text = self.get_loss(text_out, visual_out, clip_target)
        loss_dict = {}
        loss_dict["loss/clip_loss"] = clip_loss
        clip_idx = torch.max(logits_per_text, dim=-1).indices
        clip_acc = (clip_idx == clip_target).to(dtype=torch.float32).mean()
        loss_dict["acc/clip_acc"] = clip_acc
        self.log_dict(loss_dict, prog_bar=True, batch_size=len(text),
                      logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return clip_loss

    def configure_optimizers(self):

        lr = self.learning_rate
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )

        return_dict = dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval="step"
            ),
        )
        return return_dict
