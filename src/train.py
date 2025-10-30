# train.py
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
from sklearn.metrics import f1_score

import config
from src.config import RUN_NAME
from src.data.dataloader import build_dataloader
from src.models import get_model
from src.utils.logger import setup_logger, init_wandb
from src.utils.utils import cleanup_models


def _unpack_batch(batch):
    """
    Support (images, labels) or (images, labels, *extras).
    """
    if isinstance(batch, (list, tuple)):
        images, labels = batch[0], batch[1]
        return images, labels
    raise ValueError("Unexpected batch structure.")


# (Point 4) Centralized criterion factory honoring config
def _create_criterion():
    loss_name = str(getattr(config, "LOSS_FUNCTION", "CrossEntropy")).lower()
    label_smoothing = float(getattr(config, "LABEL_SMOOTHING", 0.0))

    if loss_name in {"crossentropy", "cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    raise ValueError(f"Unsupported LOSS_FUNCTION: {loss_name}")


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    """
    Runs a single training epoch for multi-class classification (3 classes).
    - outputs: logits [B, C]
    - labels:  long [B]
    - loss:    CrossEntropyLoss
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # for F1 macro
    all_preds = []
    all_targets = []

    progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch}", unit="batch")
    for i, batch in enumerate(progress_bar):
        images, labels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()  # class indices

        # Forward
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)  # logits [B, 3]
        loss = criterion(outputs, labels)  # CE on logits

        # Backward + opt
        loss.backward()
        optimizer.step()

        # Per-batch scheduler stepping for CosineAnnealingWarmRestarts
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch - 1 + i / max(1, len(dataloader)))

        # (Point 3) Sample-weighted loss accumulation
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(outputs, dim=1)  # [B]
        correct_predictions += (preds == labels).sum().item()

        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())

        # Postfix with sample-weighted running avg
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(
            loss=f"{total_loss / max(1, total_samples):.4f}",
            acc=f"{correct_predictions / max(1, total_samples):.4f}",
            lr=f"{current_lr:.6f}"
        )

    avg_loss = total_loss / max(1, total_samples)  # (Point 3)
    accuracy = correct_predictions / max(1, total_samples)

    # Macro F1 across 3 classes
    all_preds = torch.cat(all_preds).numpy() if all_preds else []
    all_targets = torch.cat(all_targets).numpy() if all_targets else []
    f1_macro = f1_score(all_targets, all_preds, average="macro") if len(all_targets) else 0.0

    return avg_loss, accuracy, f1_macro


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Runs a single validation epoch for multi-class classification (3 classes).
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # for F1 macro
    all_preds = []
    all_targets = []

    progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            images, labels = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            outputs = model(images)  # logits [B, 3]
            loss = criterion(outputs, labels)

            # (Point 3) Sample-weighted accumulation
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

            progress_bar.set_postfix(
                loss=f"{total_loss / max(1, total_samples):.4f}",
                acc=f"{correct_predictions / max(1, total_samples):.4f}"
            )

    avg_loss = total_loss / max(1, total_samples)  # (Point 3)
    accuracy = correct_predictions / max(1, total_samples)

    all_preds = torch.cat(all_preds).numpy() if all_preds else []
    all_targets = torch.cat(all_targets).numpy() if all_targets else []
    f1_macro = f1_score(all_targets, all_preds, average="macro") if len(all_targets) else 0.0

    return avg_loss, accuracy, f1_macro


def main():
    # --- Setup Run Directory ---
    run_dir = os.path.join(config.RUNS_DIR, RUN_NAME)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Copy config file for reproducibility
    shutil.copy("src/config.py", os.path.join(run_dir, "config.py"))

    # --- Setup Logger / WandB ---
    setup_logger(os.path.join(run_dir, "train.log"))
    wb = init_wandb()

    logging.info(f"Starting new run: {RUN_NAME}")
    logging.info(f"Device: {config.DEVICE}")
    # (Point 5) Log classes
    logging.info(f"Classes: {getattr(config, 'CLASS_NAMES', ['0', '1', '2'])}")

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.T_0, T_mult=config.T_MULT, eta_min=config.ETA_MIN
        )
    elif config.SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.ETA_MIN
        )
    else:
        logging.error(f"Unsupported scheduler: {config.SCHEDULER}")
        return

    # (Point 4) Create criterion via factory
    criterion = _create_criterion()

    logging.info("Starting training...")
    tr_start = time()
    for epoch in range(1, config.EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{config.EPOCHS} ---")
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current Learning Rate: {current_lr}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, config.DEVICE, epoch
        )
        logging.info(f"Epoch {epoch} Training -> Loss: {train_loss}, "
                     f"Accuracy: {train_acc}, F1(macro): {train_f1}")

        val_loss, val_acc, val_f1 = validate_one_epoch(model, test_loader, criterion, config.DEVICE)
        logging.info(f"Epoch {epoch} Validation -> Loss: {val_loss}, "
                     f"Accuracy: {val_acc}, F1(macro): {val_f1}")

        # Step the scheduler only if it's the per-epoch type
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # --- Log to WandB ---
        if wb is not None:
            wb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_f1_macro": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1,
                "lr": optimizer.param_groups[0]['lr'],
            }, step=epoch)

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
