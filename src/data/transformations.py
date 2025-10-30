import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The target size for the model input
IMAGE_SIZE = 224


def old_transforms():
    """
    Returns the data augmentation pipeline for the training set using Albumentations.
    This pipeline works directly with OpenCV images (NumPy arrays).
    """
    return A.Compose([
        A.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), ratio=(1, 1)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
        A.Erasing(scale=[0.02, 0.1], ratio=[0.3, 2.0], fill=0, p=0.25),
        A.ShiftScaleRotate(shift_limit=[-0.1, 0.1], scale_limit=0, rotate_limit=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_train_transforms():
    """
    Augmentations for NSFW multi-class (safe / not_safe / kiss).
    Goals:
      - Keep semantics (faces/body context) -> avoid extreme crops/rotations.
      - Add mild color/lighting jitter without large hue shifts (avoid skin-tone drift).
      - Be robust to web artifacts (JPEG) and mild blur/noise.
      - Small occlusions to reduce reliance on a single region.
    """
    return A.Compose([
        # Crop with modest zoom/aspect jitter to avoid chopping away key regions
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.80, 1.00),  # do not go too small; preserve context
            ratio=(1, 1),  # square
            interpolation=cv2.INTER_CUBIC
        ),

        # Horizontal left/right is fine; vertical flips are unrealistic -> off
        A.HorizontalFlip(p=0.5),

        # Small geometric jitter; avoid big rotations
        A.ShiftScaleRotate(
            shift_limit=0.1,  # ±10% translation
            scale_limit=0,  # ±0% zoom
            rotate_limit=10,  # ±10° rotate
            p=0.4
        ),

        # Light color/illumination changes; keep hue small to avoid skin-tone artifacts
        A.OneOf([
            A.ColorJitter(brightness=0.15, contrast=0.20, saturation=0.15, hue=0.03),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=12),
        ], p=0.35),

        # Web-like degradations: small blur/noise/compression
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.20),

        A.OneOf([
            A.GaussNoise(std_range=(0.0088, 0.0175)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.20)),
        ], p=0.30),

        A.ImageCompression(quality_range=(60, 95), p=0.30),  # simulate JPEG artifacts

        # Small occlusions to reduce over-reliance on one patch (e.g., small badge/emoji/hand)
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(16, 32), hole_width_range=(16, 32), fill=0, p=0.2),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_test_transforms():
    """
    Returns the data transformation pipeline for the validation/test set using Albumentations.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
