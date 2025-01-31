import shutil
import logging
import os
import sys

import torch
from omegaconf import DictConfig, OmegaConf
from datamodule.data_module import DataModule
from lightning import ModelModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import check_availability

# Set environment variables and logger level
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "disabled"
logging.basicConfig(level=logging.WARNING)


def main(cfg):
    checkpoint = ModelCheckpoint(
        monitor="decoder_acc_val", mode="max", save_weights_only=True, filename="epoch={epoch}.ckpt"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Set modules and trainer
    modelmodule = ModelModule(cfg)
    if cfg.trainer.resume_from_checkpoint and cfg.train:
        modelmodule = modelmodule.load_from_checkpoint(cfg.trainer.resume_from_checkpoint, cfg=cfg)
        print("Loaded checkpoint from", cfg.trainer.resume_from_checkpoint)
    datamodule = DataModule(cfg)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,  # 분산 학습 비활성화
        precision="bf16",
        amp_backend="native",
        strategy="ddp" if cfg.trainer.use_ddp else None,  # 필요에 따라 DDP 전략 사용
        log_every_n_steps=500,
        max_epochs=-1,
        max_steps=cfg.scheduler.get("num_training_steps", -1),
        logger=WandbLogger(
            name=cfg.train_name,
            project="cross-modal-sync",
            entity="quoqa-nlp"
        ),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=callbacks,
    )

    # Training and testing
    if cfg.train:
        trainer.fit(model=modelmodule, datamodule=datamodule)
        trainer.test(ckpt_path='/home/work/SyncVSR/LRS/video/Vox+LRS2+LRS3.ckpt', datamodule=datamodule)
        shutil.copy('/home/work/SyncVSR/LRS/video/Vox+LRS2+LRS3.ckpt', f"./{cfg.train_name}.ckpt")
    else:
        modelmodule = modelmodule.load_from_checkpoint(cfg.trainer.resume_from_checkpoint, cfg=cfg, strict= False)#check_availability("fairseq")) # strict=False can be done if you were 
        trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli()))
