import torch

# --- DATASET & DATALOADER ---
DATA_DIR = "dataset/multiclass"
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256

# --- CLASSES ---
# 0 -> safe, 1 -> not_safe, 2 -> kiss
CLASS_NAMES = ["safe", "not_safe", "kiss"]
NUM_CLASSES = 3

# --- MODEL ---
MODEL_NAME = "efficientnet_v2_s"
PRETRAINED = True
OUTPUT_FEATURES = NUM_CLASSES  # IMPORTANT: 3 logits for multi-class

# --- TRAINING ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 400
LEARNING_RATE = 1e-3
OPTIMIZER = "SGD"  # ["AdamW", "SGD"]
SCHEDULER = "CosineAnnealingWarmRestarts"  # ["CosineAnnealingLR", "CosineAnnealingWarmRestarts"]
T_0 = 10
T_MULT = 2
ETA_MIN = 1e-10
WEIGHT_DECAY = 0.05

# Loss (multi-class)
LOSS_FUNCTION = "CrossEntropy"
LABEL_SMOOTHING = 0.1

# --- SAMPLER (anchored oversampling epoch rule) ---
# Epoch ends when the majority (anchor) class is exhausted (no replacement for anchor, with replacement for others)
ANCHOR_CLASS = "auto"            # or set to an int 0/1/2
DROP_LAST = False                 # drop last partial batch
BALANCED_PER_BATCH = True        # aim for as-even-as-possible within each batch
SEED = 50

# --- CHECKPOINTS & LOGGING ---
RUNS_DIR = "runs"
DESCRIPTION = "efficientnet s multiclass run increase lr, reduce eta_min, increase wd"
RUN_NAME = "eff_s_lr1e3_wd0.05"
SAVE_CHECKPOINT_EPOCHS = 20
