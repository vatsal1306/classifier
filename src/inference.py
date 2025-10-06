import argparse
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


def visualize_mistakes(mistakes, output_path, grid_size=5):
    """
    Creates and saves a collage of misclassified images using OpenCV.
    """
    if not mistakes:
        print("No mistakes to visualize.")
        return

    mistakes.sort(key=lambda x: x[2], reverse=True)
    num_images = min(len(mistakes), grid_size * grid_size)
    if num_images == 0:
        return

    # Assuming all images are the same size, get size from the first one
    img_h, img_w, _ = mistakes[0][0].shape

    # Create a blank white grid to hold the images
    grid_img_h = grid_size * img_h
    grid_img_w = grid_size * img_w
    grid_image = np.full((grid_img_h, grid_img_w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 255)  # Red in BGR
    font_thickness = 1

    for i in range(num_images):
        image_rgb, true_label, confidence, pred_label = mistakes[i]

        # Resize image to fit the grid cell
        resized_image_rgb = cv2.resize(image_rgb, (img_w, img_h))
        # Convert to BGR for OpenCV text and drawing functions
        image_bgr = cv2.cvtColor(resized_image_rgb, cv2.COLOR_RGB2BGR)

        row = i // grid_size
        col = i % grid_size
        y_offset, x_offset = row * img_h, col * img_w

        # Paste image onto the grid
        grid_image[y_offset:y_offset + img_h, x_offset:x_offset + img_w] = image_bgr

        # Prepare text to draw
        true_text = "T: Human" if true_label == 1 else "T: Non-Human"
        pred_text = "P: Human" if pred_label == 1 else "P: Non-Human"
        conf_text = f"C: {confidence:.2f}"

        # Add text to the image
        cv2.putText(grid_image, true_text, (x_offset + 5, y_offset + 15), font, font_scale, font_color, font_thickness)
        cv2.putText(grid_image, pred_text, (x_offset + 5, y_offset + 35), font, font_scale, font_color, font_thickness)
        cv2.putText(grid_image, conf_text, (x_offset + 5, y_offset + 55), font, font_scale, font_color, font_thickness)

    cv2.imwrite(output_path, grid_image)
    print(f"Mistakes collage saved to {output_path}")


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
        os.path.join(data_dir, "test", "human"), label=1, transform=get_test_transforms(), return_numpy=True
    )
    non_human_dataset = ImageClassDataset(
        os.path.join(data_dir, "test", "non_human"), label=0, transform=get_test_transforms(), return_numpy=True
    )
    test_dataset = ConcatDataset([human_dataset, non_human_dataset])
    dataloader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False, num_workers=4)

    correct_predictions = 0
    total_samples = 0
    mistakes = []

    print("Running inference...")
    with torch.no_grad():
        for images_tensor, labels, original_images_numpy in tqdm(dataloader, desc="Inference"):
            images_tensor = images_tensor.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images_tensor)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            misclassified_mask = (preds != labels).squeeze()
            if misclassified_mask.any().item():
                if misclassified_mask.dim() == 0:
                    misclassified_mask = misclassified_mask.unsqueeze(0)

                misclassified_indices = torch.where(misclassified_mask)[0]

                for idx in misclassified_indices:
                    # Convert tensor back to numpy array for cv2
                    original_img = original_images_numpy[idx].numpy()
                    true_label = int(labels[idx].item())
                    pred_label = int(preds[idx].item())
                    prob = probs[idx].item()
                    confidence = prob if pred_label == 1 else 1 - prob
                    mistakes.append((original_img, true_label, confidence, pred_label))

    if total_samples > 0:
        accuracy = correct_predictions / total_samples
        print(f"\n--- Inference Complete ---")
        print(f"Accuracy on Test Set: {accuracy:.4f}")
    else:
        print("No samples found in the test set.")

    visualize_mistakes(mistakes, os.path.join(output_dir, "worst_mistakes.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and visualize model mistakes.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--config_pth", type="str", required=True, help="Path to the config.py file.")
    args = parser.parse_args()

    config = import_vars_from_path(args.config_pth)
    run_inference(args.checkpoint, config)
