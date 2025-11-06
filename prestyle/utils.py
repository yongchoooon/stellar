import os
import glob
import torch
import importlib
from omegaconf import OmegaConf

def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def create_data(config):
    data_cls = get_obj_from_str(config.target)
    data = data_cls(data_config=config)
    return data

def save_config(cfgs, config_path, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(cfgs, os.path.join(log_dir, os.path.basename(config_path)))

def get_model(cfgs):
    model = instantiate_from_config(cfgs.model)
    if "load_ckpt_path" in cfgs:
        model.load_state_dict(torch.load(cfgs.load_ckpt_path, map_location="cpu")["state_dict"], strict=False)
    return model

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