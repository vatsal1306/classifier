import argparse
import logging
import os
import pickle
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.data.dataloader import ImageClassDataset
from src.data.transformations import get_test_transforms
from src.models import get_model
from src.utils.utils import import_vars_from_path

logger = logging.getLogger(__name__)


def run_predictions(checkpoint_path, cfg):
    """
    Runs the model on the test set and saves all predictions to a pickle file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(cfg.RUNS_DIR, cfg.RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    output_pkl_path = os.path.join(run_dir, "predictions.pkl")

    # --- Load Model ---
    logger.info(f"Loading model from {checkpoint_path}")
    model = get_model(cfg.MODEL_NAME, pretrained=False, num_classes=cfg.OUTPUT_FEATURES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Dataloader (with paths) ---
    logger.info("Building test dataloader for prediction...")

    def collate_fn_predict(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        paths = [item[2] for item in batch]
        return images, labels, paths

    transform = get_test_transforms()
    test_root = os.path.join(cfg.DATA_DIR, "test")
    ds_safe = ImageClassDataset(os.path.join(test_root, "safe"), label=0, transform=transform, return_path=True)
    ds_not_safe = ImageClassDataset(os.path.join(test_root, "not_safe"), label=1, transform=transform, return_path=True)
    ds_kiss = ImageClassDataset(os.path.join(test_root, "kiss"), label=2, transform=transform, return_path=True)
    full_dataset = ConcatDataset([ds_safe, ds_not_safe, ds_kiss])

    dataloader = DataLoader(
        full_dataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_predict
    )

    # --- Prediction Loop ---
    results = []
    logger.info("Running predictions...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Predicting"):
            images = images.to(device, non_blocking=True)
            logits = model(images)  # [B, 3]
            probs = F.softmax(logits, dim=1).cpu()  # [B, 3]
            preds = torch.argmax(probs, dim=1)

            for i in range(len(paths)):
                results.append({
                    "image_path": paths[i],
                    "true_label": int(labels[i].item()),
                    "predicted_label": int(preds[i].item()),
                    "probs": probs[i].tolist(),  # list of length 3 [p0,p1,p2]
                    "pred_name": cfg.CLASS_NAMES[int(preds[i].item())],
                })

    # --- Save to Pickle File ---
    logger.info(f"Saving {len(results)} predictions to {output_pkl_path}")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info("--- Prediction Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model predictions and save results to predictions.pkl.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.py file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"config.py not found at {args.config}")

    cfg = import_vars_from_path(args.config)
    run_predictions(args.checkpoint, cfg)
