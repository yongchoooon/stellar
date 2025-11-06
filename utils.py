import os
import glob
import torch
import importlib
from omegaconf import OmegaConf
from src.trainer.utils import instantiate_from_config

def save_config(cfgs, config_path, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(cfgs, os.path.join(log_dir, os.path.basename(config_path)))

def get_last_checkpoint(cfgs, resume_num):
    ckpt_dir = os.path.join(cfgs.lightning.default_root_dir, "lightning_logs", f"version_{resume_num}", "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if ckpts:
        ckpt_nums = [int(os.path.basename(ckpt).split("-")[0].split("=")[-1]) for ckpt in ckpts]
        max_index = ckpt_nums.index(max(ckpt_nums))
        last_ckpt = ckpts[max_index] if max_index >= 0 else None
    else:
        last_ckpt = None
    return last_ckpt

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def create_data(config):
    data_cls = get_obj_from_str(config.target)
    data = data_cls(data_config=config)
    return data