import argparse
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from utils import get_dataloader, get_model, save_config, get_last_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.enabled = False


def train(cfgs, config_path, resume_num=None):
    """Main training function."""
    dataloader = get_dataloader(cfgs)
    model = get_model(cfgs)

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=cfgs.check_freq, save_top_k=-1
    )
    tb_logger = TensorBoardLogger(
        save_dir=cfgs.lightning.default_root_dir,
        name="lightning_logs",
        version=resume_num if resume_num is not None else None
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        **cfgs.lightning
    )

    save_config(cfgs, config_path, trainer.logger.log_dir)

    ckpt_path = get_last_checkpoint(cfgs, resume_num) if resume_num else None
    trainer.fit(model=model, train_dataloaders=dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Pretraining Text Glyph Structure Encoder in MultiLingual")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_num", type=int)
    args = parser.parse_args()

    config_path = f'configs/{args.config}.yaml'
    cfgs = OmegaConf.load(config_path)

    train(cfgs, config_path, args.resume_num)