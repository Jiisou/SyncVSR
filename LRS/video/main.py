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
        accelerator="cpu",
        strategy=None,  # DDP 전략 비활성화
        # accelerator="gpu", #MPS(Metal Performance Shaders, macOS GPU 가속) 환경에서는 DDP 전략이 지원되지 않습니다.
        devices="auto", # "auto"
        precision="bf16", # 16 as float16 and 32 as full precision and bf16 as brain float: https://lightning.ai/docs/pytorch/1.5.9/advanced/mixed_precision.html
        amp_backend="native",
        # strategy="ddp", -> cuda 환경에서만 사용 가능
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
        trainer.test(ckpt_path="/Users/jisu/Desktop/dev/prometheus/SyncVSR/LRS/video/Vox+LRS2+LRS3.ckpt", datamodule=datamodule) # checkpoint.best_model_path
        shutil.copy("/Users/jisu/Desktop/dev/prometheus/SyncVSR/LRS/video/Vox+LRS2+LRS3.ckpt", f"./{cfg.train_name}.ckpt") #checkpoint.best_model_path
    else:
        modelmodule = modelmodule.load_from_checkpoint(cfg.trainer.resume_from_checkpoint, cfg=cfg, strict=check_availability("fairseq")) # strict=False can be done if you were 
        trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli()))
