import argparse
import logging
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.data.dataloader import ImageClassDataset
from src.data.transformations import get_test_transforms
from src.models import get_model
from src.utils.utils import import_vars_from_path

logger = logging.getLogger(__name__)


def collate_fn_for_inference(batch):
    """
    Custom collate function for the inference dataloader.
    It handles batches where original images (as numpy arrays) have different sizes.

    Args:
        batch (list): A list of tuples, where each tuple is
                      (transformed_tensor, label, original_numpy_image).

    Returns:
        tuple: A tuple containing:
               - A stacked tensor of transformed images.
               - A tensor of labels.
               - A list of original numpy images.
    """
    # 1. Stack the transformed tensors and labels, which are uniform in size.
    transformed_images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch])

    # 2. Keep the original numpy images as a list, since they have variable sizes.
    original_images = [item[2] for item in batch]

    return transformed_images, labels, original_images


def visualize_mistakes(mistakes, output_path, grid_size=5):
    """
    Creates and saves a collage of the most confident misclassified images using OpenCV.
    """
    if not mistakes:
        logger.info("No mistakes to visualize.")
        return

    mistakes.sort(key=lambda x: x[2], reverse=True)
    num_images = min(len(mistakes), grid_size * grid_size)

    # Use the first image to determine a base size for the grid
    sample_img = mistakes[0][0]
    h, w, _ = sample_img.shape

    # Create a grid to hold the images
    grid_image = np.full((grid_size * h, grid_size * w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 255)  # Red in BGR
    line_type = 2

    for i in range(num_images):
        image_rgb, true_label, confidence = mistakes[i]
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Resize image to fit the grid cell, just in case they vary slightly
        resized_img = cv2.resize(image_bgr, (w, h))

        row = i // grid_size
        col = i % grid_size

        # Calculate paste location
        y_offset, x_offset = row * h, col * w
        grid_image[y_offset:y_offset + h, x_offset:x_offset + w] = resized_img

        # Add text
        true_text = "True: Human" if true_label == 1 else "True: Non-Human"
        pred_text = "Pred: Non-Human" if true_label == 1 else "Pred: Human"
        conf_text = f"Conf: {confidence:.2f}"

        cv2.putText(grid_image, true_text, (x_offset + 5, y_offset + 15), font, font_scale, font_color, line_type)
        cv2.putText(grid_image, pred_text, (x_offset + 5, y_offset + 30), font, font_scale, font_color, line_type)
        cv2.putText(grid_image, conf_text, (x_offset + 5, y_offset + 45), font, font_scale, font_color, line_type)

    cv2.imwrite(output_path, grid_image)
    logger.info(f"Mistakes collage saved to {output_path}")


def run_inference(checkpoint_path, config):
    """
    Runs inference on the test set, calculates accuracy, and visualizes mistakes.
    """
    data_dir = config.DATA_DIR
    output_dir = os.path.join(config.RUNS_DIR, config.RUN_NAME, "inference_results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {checkpoint_path}")
    model = get_model(config.MODEL_NAME, pretrained=False, num_classes=config.OUTPUT_FEATURES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print("Building test dataloader...")
    # Use return_numpy=True to get original images for visualization
    human_dataset = ImageClassDataset(
        os.path.join(data_dir, "test", "human"), label=1, transform=get_test_transforms(), return_path=True
    )
    non_human_dataset = ImageClassDataset(
        os.path.join(data_dir, "test", "non_human"), label=0, transform=get_test_transforms(), return_path=True
    )
    test_dataset = ConcatDataset([human_dataset, non_human_dataset])
    dataloader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False, num_workers=4,
                            pin_memory=True,
                            collate_fn=collate_fn_for_inference)

    correct_predictions = 0
    total_samples = 0
    mistakes = []

    print("Running inference...")
    with torch.no_grad():
        for transformed_images, labels, original_images in tqdm(dataloader, desc="Inference"):
            transformed_images = transformed_images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(transformed_images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Find and store mistakes for visualization
            misclassified_mask = (preds != labels).view(-1)
            if misclassified_mask.any():
                # Get the indices of the misclassified samples
                misclassified_indices = misclassified_mask.nonzero(as_tuple=True)[0]

                for idx in misclassified_indices:
                    original_img = original_images[idx]
                    true_label = int(labels[idx].item())
                    pred_prob = probs[idx].item()

                    # Confidence is the probability of the predicted class
                    confidence = pred_prob if preds[idx].item() == 1 else 1 - pred_prob

                    mistakes.append((original_img, true_label, confidence))

    # --- Results ---
    if total_samples > 0:
        accuracy = correct_predictions / total_samples
        logger.info(f"\n--- Inference Complete ---")
        logger.info(f"Accuracy on Test Set: {accuracy:.4f}")
    else:
        logger.warning("No samples were processed during inference.")

    # --- Visualize ---
    visualize_mistakes(mistakes, os.path.join(output_dir, "worst_mistakes.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and visualize model mistakes.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--config_pth", type=str, required=True, help="Path to the config.py file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = import_vars_from_path(args.config_pth)
    run_inference(args.checkpoint, config)
