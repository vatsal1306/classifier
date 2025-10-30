import logging
import os
import sys
from typing import Literal

import wandb
from dotenv import load_dotenv

import src.config as config

load_dotenv(dotenv_path='../../.env', verbose=True)


def setup_logger(log_file_path):
    """
    Sets up a logger that writes to both the console and a specified file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Logger setup complete.")


def init_wandb(sync_mode: Literal["online", "offline", "disabled"] = 'online'):
    """
    Initialize WandB.
    """
    wandb.login(key=os.environ.get('WANDB_API_KEY', ''))
    wandb.init(
        project="nsfw",
        name=config.RUN_NAME,
        dir=os.path.join(config.RUNS_DIR, config.RUN_NAME),
        notes=config.DESCRIPTION,
        mode=sync_mode,
        config={
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.TRAIN_BATCH_SIZE,
            "weight_decay": config.WEIGHT_DECAY,
            "optimizer": config.OPTIMIZER,
            "scheduler": config.SCHEDULER,
            "model": config.MODEL_NAME,
            "classes": getattr(config, "CLASS_NAMES", ["safe", "not_safe", "kiss"]),
        })

    # Define metrics consistent with train.py logging keys
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("train_accuracy", step_metric="epoch")
    wandb.define_metric("train_f1_macro", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("val_accuracy", step_metric="epoch")
    wandb.define_metric("val_f1_macro", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")

    return wandb
