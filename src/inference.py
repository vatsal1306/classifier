import argparse
import glob
import logging
import os
import json
import shutil

import albumentations as A
import cv2
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torchvision import models
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The target size for the model input
IMAGE_SIZE = 224


def get_test_transforms():
    """
    Returns the data transformation pipeline for the validation/test set using Albumentations.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_efficientnet(model_name, weights, num_classes):
    """Helper function to build and modify an EfficientNet model."""
    if model_name == "efficientnet_v2_l":
        model = models.efficientnet_v2_l(weights=weights)
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=weights)
    else:
        raise ValueError(f"Unsupported EfficientNet variant: {model_name}")

    # The classifier in EfficientNet is the last layer of the `classifier` sequential block
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model


# A mapping from model names to their pre-trained weight enums
WEIGHTS_MAPPING = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "vit_b_16": models.ViT_B_16_Weights.DEFAULT,
    "vit_l_32": models.ViT_L_32_Weights.DEFAULT,
    "efficientnet_v2_l": models.EfficientNet_V2_L_Weights.DEFAULT,
    "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
}

# The main registry mapping model names to their builder functions
MODEL_REGISTRY = {
    "efficientnet_v2_l": _build_efficientnet,
    "efficientnet_v2_s": _build_efficientnet,
}


def get_model(model_name, pretrained=True, num_classes=1):
    """
    Loads a model from the registry, replaces the classification head, and loads pre-trained weights if specified.

    Args:
        model_name (str): The name of the model architecture to load.
        pretrained (bool): Whether to load pre-trained ImageNet weights.
        num_classes (int): The number of output features for the final layer.

    Returns:
        torch.nn.Module: The modified model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(MODEL_REGISTRY.keys())}")

    # 1. Get the correct builder function from the registry
    model_builder = MODEL_REGISTRY[model_name]

    # 2. Determine which weights to use (if any)
    weights = WEIGHTS_MAPPING[model_name] if pretrained else None

    # 3. Build the model
    model = model_builder(model_name=model_name, weights=weights, num_classes=num_classes)

    return model


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
    pred_lbl_str = "Not Safe" if pred_label == 1 else "Safe"
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
    # dest_img_dir = os.path.join(args.output, 'img')
    # dest_json_pth = os.path.join(args.output, 'data.json')
    # os.makedirs(dest_img_dir, exist_ok=True)
    device = args.device
    
    results_dict = {}

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
    
    if args.limit:
        image_paths = image_paths[:args.limit]

    # --- Initialize Predictor ---
    predictor = Predictor(args.model_name, args.checkpoint, device)

    # --- Run Inference Loop ---
    for img_path in tqdm(image_paths, desc="Running Inference"):
        pred_label, confidence = predictor.predict_image(img_path)
        if pred_label is not None and pred_label == 0:
            save_annotated_prediction(img_path, args.output, pred_label, confidence)
    #     results_dict[os.path.basename(img_path)] = {"label": pred_label, "reviewed": False}
    #     shutil.copy(img_path, os.path.join(dest_img_dir, os.path.basename(img_path)))
    #
    # # Load existing data if file exists
    # if os.path.exists(dest_json_pth):
    #     with open(dest_json_pth, "r") as f:
    #         existing = json.load(f)
    # else:
    #     existing = {}
    #
    # # Merge (existing keys will be updated)
    # existing.update(results_dict)
    #
    # # Save back
    # with open(dest_json_pth, "w") as f:
    #     json.dump(existing, f, indent=4)
        
    logger.info(f"Inference complete. Results saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone inference script for Binary NSFW classification.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to a single image or a directory of images.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the directory where output images will be saved.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        choices=['vit_b_16', 'vit_l_32', 'efficientnet_v2_l', 'efficientnet_v2_s'],
                        help="Name of the model architecture to use.")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-l", "--limit", type=int, help="Limit how many images to save")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
