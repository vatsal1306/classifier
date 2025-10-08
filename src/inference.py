import argparse
import glob
import logging
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import cv2
import torch
from tqdm import tqdm

from src.models import get_model
from src.data.transformations import get_test_transforms

logger = logging.getLogger(__name__)


def save_annotated_prediction(image_path, output_dir, pred_label, confidence):
    """
    Reads an image, creates a new canvas with prediction info, and saves it.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        logger.warning(f"Could not read image {image_path}, skipping.")
        return

    h, w, _ = image_bgr.shape

    # --- Create a new canvas with a white panel on the right for text ---
    text_panel_width = 220
    canvas = cv2.copyMakeBorder(image_bgr, 0, 0, 0, text_panel_width, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # --- Prepare text ---
    pred_lbl_str = "Human" if pred_label == 1 else "Non-Human"
    original_basename = os.path.basename(image_path)

    # --- Add text overlay on the white panel ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)  # Black text
    thickness = 1

    # text1 = f"File: {original_basename[:20]}"  # Truncate long filenames
    text2 = f"Prediction: {pred_lbl_str}"
    text3 = f"Confidence: {confidence:.4f}"

    # cv2.putText(canvas, text1, (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(canvas, text2, (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(canvas, text3, (w + 10, 60), font, font_scale, color, thickness)

    # --- Save the final image ---
    output_path = os.path.join(output_dir, original_basename)
    cv2.imwrite(output_path, canvas)


class Predictor:
    """A helper class to encapsulate the model and prediction logic."""

    def __init__(self, model_name, checkpoint_path, device="cuda"):
        self.device = device
        self.transform = get_test_transforms()

        logger.info(f"Loading model '{model_name}' from {checkpoint_path}")
        self.model = get_model(model_name, pretrained=False, num_classes=1)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def predict_image(self, image_path):
        """
        Runs inference on a single image.
        Returns the predicted label (0 or 1) and the confidence score.
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return None, None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Apply transformations
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prob = torch.sigmoid(output).item()

        pred_label = 1 if prob > 0.5 else 0
        confidence = prob if pred_label == 1 else 1 - prob

        return pred_label, confidence


def main(args):
    """Main function to orchestrate the inference process."""
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # --- Get list of image paths ---
    if os.path.isdir(args.input):
        image_paths = glob.glob(os.path.join(args.input, '**', '*.[jJ][pP]*[gG]'), recursive=True) + \
                      glob.glob(os.path.join(args.input, '**', '*.[pP][nN][gG]'), recursive=True)
        logger.info(f"Found {len(image_paths)} images in directory: {args.input}")
    else:
        image_paths = [args.input]
        logger.info(f"Processing single image: {args.input}")

    if not image_paths:
        logger.warning("No images found to process.")
        return

    # --- Initialize Predictor ---
    predictor = Predictor(args.model_name, args.checkpoint, device)

    # --- Run Inference Loop ---
    for img_path in tqdm(image_paths, desc="Running Inference"):
        pred_label, confidence = predictor.predict_image(img_path)
        if pred_label is not None:
            save_annotated_prediction(img_path, args.output, pred_label, confidence)

    logger.info(f"Inference complete. Results saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone inference script for human/non-human classification.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to a single image or a directory of images.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the directory where output images will be saved.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        choices=['vit_b_16', 'vit_l_32', 'efficientnet_v2_s'],
                        help="Name of the model architecture to use.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
