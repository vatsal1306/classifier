import argparse
import glob
import logging
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import albumentations as A
import cv2
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import src.config as config
from src.models import get_model

logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def get_test_transforms():
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def save_annotated_prediction(image_path, output_dir, pred_name, probs):
    """
    Reads an image, creates a new canvas with prediction info, and saves it.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        logger.warning(f"Could not read image {image_path}, skipping.")
        return

    h, w, _ = image_bgr.shape
    text_panel_width = 260
    canvas = cv2.copyMakeBorder(image_bgr, 0, 0, 0, text_panel_width, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)
    thickness = 1

    cv2.putText(canvas, f"Pred: {pred_name}", (w + 10, 30), font, font_scale, color, thickness)
    y = 60
    for i, p in enumerate(probs):
        cv2.putText(canvas, f"P[{config.CLASS_NAMES[i]}]: {p:.3f}", (w + 10, y), font, font_scale, color, thickness)
        y += 24

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, canvas)


class Predictor:
    """A helper class to encapsulate the model and prediction logic."""

    def __init__(self, model_name, checkpoint_path, device="cuda"):
        self.device = device
        self.transform = get_test_transforms()

        logger.info(f"Loading model '{model_name}' from {checkpoint_path}")
        self.model = get_model(model_name, pretrained=False, num_classes=config.OUTPUT_FEATURES)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def predict_image(self, image_path):
        """
        Runs inference on a single image.
        Returns (pred_id, pred_name, probs[list]).
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return None, None, None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)  # [1, 3]
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
        pred_name = config.CLASS_NAMES[pred_id]
        return pred_id, pred_name, probs


def main(args):
    """Main function to orchestrate the inference process."""
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        return

    device = args.device

    # --- Get list of image paths ---
    if os.path.isdir(args.input):
        image_paths = []
        image_paths.extend(glob.glob(os.path.join(args.input, '**', '*.[jJ][pP]*[gG]'), recursive=True))
        image_paths.extend(glob.glob(os.path.join(args.input, '**', '*.[pP][nN][gG]'), recursive=True))
        logger.info(f"Found {len(image_paths)} images in directory: {args.input}")
    else:
        image_paths = [args.input]
        logger.info(f"Processing single image: {args.input}")

    if not image_paths:
        logger.warning("No images found to process.")
        return

    if args.limit:
        image_paths = image_paths[:args.limit]

    # --- Initialize Predictor ---
    predictor = Predictor(args.model_name, args.checkpoint, device)

    # --- Run Inference Loop ---
    for img_path in tqdm(image_paths, desc="Running Inference"):
        pred_id, pred_name, probs = predictor.predict_image(img_path)

        # Optionally save annotated preview images
        save_annotated_prediction(img_path, args.output, pred_name, probs)

    logger.info(f"Inference complete. Results saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone inference script for 3-class classification.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to a single image or a directory of images.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the directory where outputs will be saved.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        choices=['vit_b_16', 'vit_l_32', 'efficientnet_v2_l', 'efficientnet_v2_s', 'resnet18',
                                 'resnet34'],
                        help="Name of the model architecture to use.")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-l", "--limit", type=int, help="Limit how many images to process")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
