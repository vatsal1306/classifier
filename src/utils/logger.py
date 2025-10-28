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

    Args:
        log_file_path (str): The full path to the log file.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')

    # Create a handler to write to the console (stdout)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Create a handler to write to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Logger setup complete.")


def init_wandb(sync_mode: Literal["online", "offline", "disabled"] = 'online'):
    """
    Initialize WandB and TensorBoard loggers.
    Returns:
        wb: WandB logger instance.
        writer: TensorBoard SummaryWriter instance.
    """
    # Initialize WandB
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
        })

    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("train_mae", step_metric="epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("test_mae", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")

    return wandb
