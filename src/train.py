import logging
import os
import shutil

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.config import RUN_NAME
from src.data.dataloader import build_dataloader
from src.models import get_model
from src.utils.logger import setup_logger


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs a single training epoch.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix(loss=total_loss / total_samples, acc=correct_predictions / total_samples)

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

    progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
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

            progress_bar.set_postfix(loss=total_loss / total_samples, acc=correct_predictions / total_samples)

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

    logging.info(f"Starting new run: {RUN_NAME}")
    logging.info(f"Device: {config.DEVICE}")

    # --- Dataloaders ---
    logging.info("Building dataloaders...")
    train_loader = build_dataloader('train', config)
    test_loader = build_dataloader('test', config)
    logging.info("Dataloaders built successfully.")

    # --- Model, Optimizer, Loss ---
    logging.info(f"Loading model: {config.MODEL_NAME}")
    model = get_model(config.MODEL_NAME, config.PRETRAINED, config.OUTPUT_FEATURES).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    logging.info("Starting training...")
    for epoch in range(1, config.EPOCHS + 1):
        logging.info(f"--- Epoch {epoch}/{config.EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        logging.info(f"Epoch {epoch} Training -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc = validate_one_epoch(model, test_loader, criterion, config.DEVICE)
        logging.info(f"Epoch {epoch} Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # --- Save Checkpoint ---
        if epoch % config.SAVE_CHECKPOINT_EPOCHS == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    logging.info("--- Training Complete ---")


if __name__ == "__main__":
    main()
