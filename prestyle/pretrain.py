import argparse
import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import get_model, create_data, instantiate_from_config, save_config, get_last_checkpoint


def train(cfgs, config_path, resume_num=None):
    model = get_model(cfgs)

    all_cfgs = cfgs.copy()

    data_config = cfgs.pop("data", OmegaConf.create())
    data_opt = data_config
    data = create_data(data_opt)
    data.prepare_data()
    data.setup(stage='fit')

    checkpoint_callback = ModelCheckpoint(every_n_epochs=cfgs.check_freq, save_top_k=-1)
    imagelogger_callback = instantiate_from_config(cfgs.imagelogger_callback)

    tb_logger = TensorBoardLogger(
        save_dir=cfgs.lightning.default_root_dir,
        name="lightning_logs",
        version=resume_num if resume_num is not None else None
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, imagelogger_callback],
        **cfgs.lightning
    )

    save_config(all_cfgs, config_path, trainer.logger.log_dir)

    ckpt_path = get_last_checkpoint(cfgs, resume_num) if resume_num else None

    trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Pretraining Text Glyph Structure Encoder in MultiLingual")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_num", type=int)
    args = parser.parse_args()

    config_path = f'configs/{args.config}.yaml'
    cfgs = OmegaConf.load(config_path)

    train(cfgs, config_path, args.resume_num)