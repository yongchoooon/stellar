# reference to https://github.com/ZYM-PKU/UDiffText, many thanks.
from typing import Union
import torch
import torch.nn as nn
import math
import string

from paddleocr import PaddleOCR

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
        x = x.cpu().numpy()

        self.tr.input_tensor.copy_from_cpu(x)
        self.tr.predictor.run()

        x = [t.copy_to_cpu() for t in self.tr.output_tensors]
        x = x[0]
        x = torch.tensor(x, device=device)

        if self.lang == "arabic":
            x = torch.flip(x, dims=[1])

        return x
    
    def encode(self, x):
        return self(x)

class LanguageAdaptiveLabelEncoder(AbstractEmbModel):

    def __init__(self,
                 lang,
                 character_path,
                 max_len,
                 emb_dim,
                 n_heads=8,
                 n_trans_layers=12,
                 ckpt_path=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lang = lang
        self.character_path = character_path

        self.max_len = max_len
        self.emd_dim = emb_dim
        self.n_heads = n_heads
        self.n_trans_layers = n_trans_layers

        self.character = self.get_multilingual_character(lang, character_path)
        self.num_cls = len(self.character) + 1

        self.label_embedding = nn.Embedding(self.num_cls, self.emd_dim)
        self.pos_embedding = PositionalEncoding(d_model=self.emd_dim, max_len=self.max_len)
        transformer_block = nn.TransformerEncoderLayer(d_model=self.emd_dim, nhead=self.n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_block, num_layers=self.n_trans_layers)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

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
