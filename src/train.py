import logging
import os
import shutil
import sys
from time import time

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.config import RUN_NAME
from src.data.dataloader import build_dataloader
from src.models import get_model
from src.utils.logger import setup_logger, init_wandb
from src.utils.utils import cleanup_models


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    """
    Runs a single training epoch.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch}", unit="batch")
    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Step the scheduler only if it's the per-batch type
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch - 1 + i / len(dataloader))

        # Statistics
        total_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)


        # Use optimizer.param_groups for the most up-to-date LR
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=f"{total_loss / (i + 1):.4f}", acc=f"{correct_predictions / total_samples:.4f}",
                                 lr=f"{current_lr:.6f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Runs a single validation epoch.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(loss=f"{total_loss / (len(progress_bar))}",
                                     acc=f"{correct_predictions / total_samples}")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def main():
    # --- Setup Run Directory ---
    run_dir = os.path.join(config.RUNS_DIR, RUN_NAME)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Copy config file for reproducibility
    shutil.copy("src/config.py", os.path.join(run_dir, "config.py"))

    # --- Setup Logger ---
    setup_logger(os.path.join(run_dir, "train.log"))
    wb = init_wandb()

    logging.info(f"Starting new run: {RUN_NAME}")
    logging.info(f"Device: {config.DEVICE}")

    # --- Dataloaders ---
    logging.info("Building dataloaders...")
    train_loader = build_dataloader('train', config)
    test_loader = build_dataloader('test', config)
    logging.info("Dataloaders built successfully.")

    # --- Model, Optimizer, Loss , Scheduler ---
    logging.info(f"Loading model: {config.MODEL_NAME}")
    model = get_model(config.MODEL_NAME, config.PRETRAINED, config.OUTPUT_FEATURES).to(config.DEVICE)

    if config.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9,
                                    weight_decay=config.WEIGHT_DECAY)
    else:
        logging.error(f"Unsupported optimizer: {config.OPTIMIZER}")
        return

    if config.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0,
                                                                         T_mult=config.T_MULT, eta_min=config.ETA_MIN)
    elif config.SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.ETA_MIN)
    else:
        logging.error(f"Unsupported scheduler: {config.SCHEDULER}")
        return

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoints_dir, "model_best.pth")

    logging.info("Starting training...")
    tr_start = time()
    for epoch in range(1, config.EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{config.EPOCHS} ---")
        current_lr = optimizer.param_groups[0]['lr'] # Get LR before training epoch
        logging.info(f"Current Learning Rate: {current_lr}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, config.DEVICE,
                                                epoch)
        logging.info(f"Epoch {epoch} Training -> Loss: {train_loss}, Accuracy: {train_acc}")

        val_loss, val_acc = validate_one_epoch(model, test_loader, criterion, config.DEVICE)
        logging.info(f"Epoch {epoch} Validation -> Loss: {val_loss}, Accuracy: {val_acc}")
        
        
        # Step the scheduler only if it's the per-epoch type
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # --- Log to WandB ---
        if wb is not None:
            wb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)

        # --- Save Best Checkpoint (based on val_loss) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved to {best_model_path} (Val Loss: {val_loss})")


        # --- Save Checkpoint ---
        if epoch % config.SAVE_CHECKPOINT_EPOCHS == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"model_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    # log training time in wandb
    if wb is not None:
        wb.summary["training_time"] = time() - tr_start
    logging.info("--- Training Complete ---")
    # --- Cleanup old models ---
    cleanup_models(dir_path=checkpoints_dir)
    logging.info("Redundant checkpoints cleaned up.")


if __name__ == "__main__":
    main()
