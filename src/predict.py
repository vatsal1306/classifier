import argparse
import logging
import os
import pickle
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

# We will need the updated dataloader and other modules
from src.data.dataloader import ImageClassDataset
from src.data.transformations import get_test_transforms
from src.models import get_model
from src.utils.utils import import_vars_from_path

logger = logging.getLogger(__name__)


def run_predictions(checkpoint_path, config):
    """
    Runs the model on the test set and saves all predictions to a pickle file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    run_dir = os.path.join(config.RUNS_DIR, config.RUN_NAME)
    output_pkl_path = os.path.join(run_dir, "predictions.pkl")

    # --- Load Model ---
    logger.info(f"Loading model from {checkpoint_path}")
    model = get_model(config.MODEL_NAME, pretrained=False, num_classes=config.OUTPUT_FEATURES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Dataloader (modified to get image paths) ---
    logger.info("Building test dataloader for prediction...")

    # We need a custom collate function to handle the image paths (strings)
    def collate_fn_predict(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return images, labels, paths

    # Build the dataset with return_path=True
    not_safe_dataset = ImageClassDataset(os.path.join(config.DATA_DIR, "test", "not_safe"), label=1,
                                         transform=get_test_transforms(), return_path=True)
    safe_dataset = ImageClassDataset(os.path.join(config.DATA_DIR, "test", "safe"), label=0,
                                     transform=get_test_transforms(), return_path=True)
    full_dataset = ConcatDataset([not_safe_dataset, safe_dataset])

    dataloader = DataLoader(
        full_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        collate_fn=collate_fn_predict
    )

    # --- Prediction Loop ---
    results = []
    logger.info("Running predictions...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            preds = (probs > 0.5).int()

            for i in range(len(paths)):
                results.append({
                    "image_path": paths[i],
                    "true_label": labels[i].item(),
                    "predicted_label": preds[i].item(),
                    "probability_not_safe": probs[i].item()
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

    config = import_vars_from_path(args.config)
    run_predictions(args.checkpoint, config)
