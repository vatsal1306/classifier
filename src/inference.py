import argparse
import glob
import logging
import os
from typing import List, Tuple, Optional

import albumentations as A
import cv2
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
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


def _build_vit(model_name, weights, num_classes):
    """Helper to build and modify a ViT model."""
    if model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
    elif model_name == "vit_l_32":
        model = models.vit_l_32(weights=weights)
    else:
        raise ValueError(f"Unsupported ViT variant: {model_name}")

    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    return model


def _build_resnet(model_name, weights, num_classes):
    """Helper to build and modify a ResNet model."""
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
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
    "vit_b_16": _build_vit,
    "vit_l_32": _build_vit,
    "resnet18": _build_resnet,
    "resnet34": _build_resnet,
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
    weights = WEIGHTS_MAPPING.get(model_name) if pretrained else None

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

    text2 = f"Prediction: {pred_lbl_str}"
    text3 = f"Confidence: {confidence:.4f}"

    cv2.putText(canvas, text2, (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(canvas, text3, (w + 10, 60), font, font_scale, color, thickness)

    # --- Save the final image ---
    output_path = os.path.join(output_dir, original_basename)
    cv2.imwrite(output_path, canvas)


class Predictor:
    """A helper class to encapsulate the model and prediction logic."""

    def __init__(self, model_name: str, checkpoint_path: str, device: str = "cuda", *,
                 use_amp: bool = False, channels_last: bool = False, compile_model: bool = False):
        # Resolve device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = device

        self.use_amp = use_amp and device.startswith("cuda") and torch.cuda.is_available()
        self.channels_last = channels_last and device.startswith("cuda") and torch.cuda.is_available()

        # Backend & precision tweaks
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                # Improves TF32 matmul perf on Ampere+
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.transform = get_test_transforms()

        logger.info(f"Loading model '{model_name}' from {checkpoint_path}")
        self.model = get_model(model_name, pretrained=False, num_classes=1)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.model.to(self.device)

        if compile_model:
            try:
                self.model = torch.compile(self.model)  # PyTorch 2.0+
                logger.info("Model compiled with torch.compile().")
            except Exception as e:
                logger.warning(f"torch.compile unavailable or failed: {e}")

        self.model.eval()
        logger.info("Model loaded successfully.")

    def predict_batch(self, images: torch.Tensor) -> List[float]:
        """
        Runs inference on a batch of images.
        Returns a list of probabilities (NSFW = 1 class probability) aligned with input order.
        """
        if images is None:
            return []

        # Move to device (non_blocking for pinned memory)
        images = images.to(self.device, non_blocking=True)

        if self.channels_last and images.dim() == 4:
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(images)
            else:
                output = self.model(images)

            probs = torch.sigmoid(output).squeeze(1)
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)
            return probs.detach().cpu().tolist()


class ImageDataset(Dataset):
    """Dataset that reads images from a list of paths and applies Albumentations transforms."""

    def __init__(self, image_paths: List[str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str]]:
        path = self.image_paths[idx]
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            logger.warning(f"Could not read image {path}, skipping.")
            return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image']
        return image_tensor, path


def _collate_fn(batch):
    """Custom collate that filters out failed reads while preserving order within the batch."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None
    images, paths = zip(*batch)
    return torch.stack(images, dim=0), list(paths)


def main(args):
    """Main function to orchestrate the inference process."""
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        return

    # Resolve device string
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    os.makedirs(args.output, exist_ok=True)

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

    logger.info(
        f"Device: {device} | Batch size: {args.batch_size} | Workers: {args.workers} | "
        f"AMP: {args.amp} | Channels-last: {args.channels_last} | Compile: {args.compile}"
    )

    # --- Initialize Predictor ---
    predictor = Predictor(
        args.model_name, args.checkpoint, device=device,
        use_amp=args.amp, channels_last=args.channels_last, compile_model=args.compile
    )

    # --- Build DataLoader (order preserved: shuffle=False) ---
    dataset = ImageDataset(image_paths, predictor.transform)
    pin_memory = device.startswith("cuda") and torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
        drop_last=False,
    )

    # --- Run Inference Loop (batched, ordered) ---
    saved = 0
    for batch_images, batch_paths in tqdm(loader, desc="Running Inference", total=len(loader)):
        if batch_images is None:
            continue
        probs = predictor.predict_batch(batch_images)
        for path, prob in zip(batch_paths, probs):
            pred_label = 1 if prob > 0.5 else 0
            confidence = prob if pred_label == 1 else 1 - prob
            save_annotated_prediction(path, args.output, pred_label, confidence)
            saved += 1

    logger.info(f"Inference complete. Saved {saved} annotated results to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone inference script for Binary NSFW classification (batched).")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to a single image or a directory of images.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the directory where output images will be saved.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        choices=['vit_b_16', 'vit_l_32', 'efficientnet_v2_l', 'efficientnet_v2_s'],
                        help="Name of the model architecture to use.")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Device to run on, e.g. 'cuda', 'cuda:0', or 'cpu'.")
    parser.add_argument("-l", "--limit", type=int, help="Limit how many images to save")

    # --- Performance-related flags ---
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of DataLoader workers for preprocessing.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA Automatic Mixed Precision (faster on GPUs).")
    parser.add_argument("--channels_last", action="store_true", help="Use channels_last memory format on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile (PyTorch 2.0+).")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
