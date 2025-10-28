import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The target size for the model input
IMAGE_SIZE = 224


def get_train_transforms():
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


def get_test_transforms():
    """
    Returns the data transformation pipeline for the validation/test set using Albumentations.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
