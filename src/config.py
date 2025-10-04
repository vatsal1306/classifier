import torch

# --- DATASET & DATALOADER ---
DATA_DIR = "dataset/filtered_data"  # Directory where the processed shards are stored
TRAIN_BATCH_SIZE = 128  # Batch size for training (32 human, 32 non-human)
TEST_BATCH_SIZE = 128  # Batch size for validation/testing

# --- MODEL ---
MODEL_NAME = "resnet50"  # Model architecture to use (e.g., "resnet18", "resnet34")
PRETRAINED = True  # Whether to use a model pre-trained on ImageNet
OUTPUT_FEATURES = 1  # Number of output features (1 for binary classification with BCEWithLogitsLoss)

# --- TRAINING ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100  # Total number of training epochs
LEARNING_RATE = 1e-4  # Initial learning rate for the optimizer
OPTIMIZER = "AdamW"  # Optimizer to use (e.g., "AdamW", "SGD")
LOSS_FUNCTION = "BCEWithLogitsLoss"  # Loss function for training

# --- CHECKPOINTS & LOGGING ---
RUNS_DIR = "runs"  # Main directory to store all training runs
RUN_NAME = "test_run"  # Name for the current run (used in the run directory)
SAVE_CHECKPOINT_EPOCHS = 1  # Save a model checkpoint every N epochs
