import torch
import torch.nn as nn
import torch.nn.functional as F
from paddleocr import PaddleOCR

def get_lang_for_ppocr(lang):
    if lang == "ko":
        return "korean"
    elif lang == "jp":
        return "japan"
    elif lang == "ar":
        return "arabic"
    else:
        raise ValueError(f"Unsupported language: {lang}")

class PPOCRv4Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lang           = config.lang
        self.use_gpu        = torch.cuda.is_available() and config.use_gpu
        if self.use_gpu == False:
            print("====================================================")
            print("Warning: GPU not available. Using CPU for inference.")
            print("====================================================")
        self.iter_size      = config.iter_size
        self.max_length     = config.max_length + 1

        self.ocr = PaddleOCR(
            det=False,
            rec=True,
            use_angle_cls=False,
            lang=get_lang_for_ppocr(self.lang),
            use_gpu=self.use_gpu,
            show_log=False
        )

    def forward(self, images, mode="train"):
        B = images.size(0)

        imgs = (images.detach().cpu().permute(0,2,3,1).numpy() * 255).astype("uint8")
        imgs = [img for img in imgs]

        rec_results = self.ocr.ocr(imgs, det=False, rec=True, cls=False)

        sequences = []
        confidences = []
        loss_weights = []
        
        for rec_res in rec_results[0]:
            text, score = rec_res
            if text:
                seq_ids = [self.char2id.get(ch, 0) for ch in text]
                char_scores = [score] * len(seq_ids)
                w = float(score)
            else:
                seq_ids, char_scores, w = [0], [0.1], 1e-4

            sequences.append(seq_ids)
            confidences.append(char_scores)
            loss_weights.append(w)

        B, L, C = len(sequences), self.max_length, self.num_classes
        logits_batch = torch.zeros((B, L, C), device=images.device)
        
        confidence_margin = 10.0
    
        for i, (seq, conf_list) in enumerate(zip(sequences, confidences)):
            for t, (cid, conf) in enumerate(zip(seq[:L], conf_list[:L])):

                logits_batch[i, t, cid] = confidence_margin * conf
                
                noise_scale = 0.1
                noise = torch.randn(C, device=images.device) * noise_scale
                
                mask = torch.ones(C, device=images.device)
                mask[cid] = 0
                
                logits_batch[i, t] += noise * mask
        
        pad_positions = torch.zeros((B, L), dtype=torch.bool, device=images.device)
        for i, seq in enumerate(sequences):
            if len(seq) < L:
                pad_positions[i, len(seq):] = True
        
        padding_logits = torch.ones((B, L, C), device=images.device) * (-confidence_margin)
        logits_batch = torch.where(pad_positions.unsqueeze(-1), padding_logits, logits_batch)

        avg_weight = sum(loss_weights) / len(loss_weights)

        out = {
            "logits": logits_batch,
            "loss_weight": avg_weight,
            "name": "ppocr"
        }
        
        outputs = [ out for _ in range(self.iter_size) ]
        
        return outputs
