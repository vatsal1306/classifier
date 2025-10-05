import torch

# --- DATASET & DATALOADER ---
DATA_DIR = "dataset/filtered_data"  # Directory where the processed shards are stored
TRAIN_BATCH_SIZE = 256  # Batch size for training (32 human, 32 non-human)
TEST_BATCH_SIZE = 256  # Batch size for validation/testing

# --- MODEL ---
MODEL_NAME = "resnet50"  # Model architecture to use (e.g., "resnet18", "resnet34")
PRETRAINED = True  # Whether to use a model pre-trained on ImageNet
OUTPUT_FEATURES = 1  # Number of output features (1 for binary classification with BCEWithLogitsLoss)

# --- TRAINING ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100  # Total number of training epochs
LEARNING_RATE = 0.0001  # Initial learning rate for the optimizer
OPTIMIZER = "AdamW"  # Optimizer to use (e.g., "AdamW", "SGD")
SCHEDULER = "CosineAnnealingWarmRestarts"
T_0 = 10         # Number of epochs for the first restart.
T_MULT = 2       # A factor to increase T_i after a restart. T_i = T_i * T_mult
ETA_MIN = 1e-12   # Minimum learning rate.
WEIGHT_DECAY = 0.0001  # Weight decay factor for regularization
LOSS_FUNCTION = "BCEWithLogitsLoss"  # Loss function for training

# --- CHECKPOINTS & LOGGING ---
RUNS_DIR = "runs"  # Main directory to store all training runs
DESCRIPTION = "First run with resnet50 on filtered dataset"  # Description for the current run
RUN_NAME = "first_run"  # Name for the current run (used in the run directory)
SAVE_CHECKPOINT_EPOCHS = 20  # Save a model checkpoint every N epochs
