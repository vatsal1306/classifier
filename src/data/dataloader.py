import glob
import logging
import os
import random
from itertools import cycle

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset

from src.data.transformations import get_train_transforms, get_test_transforms

# Get a logger instance for this module. It will inherit the root logger's configuration.
logger = logging.getLogger(__name__)


class ImageClassDataset(Dataset):
    """
    A PyTorch Dataset for loading images using OpenCV from a single class folder.
    """

    def __init__(self, root_dir, label, transform=None, return_path=False):
        """
        Args:
            root_dir (str): Directory with all the images for one class.
            label (int): The label to assign to all images in this dataset.
            transform (callable, optional): Albumentations transform to be applied.
            return_path (bool): If True, __getitem__ returns the image path.
        """
        self.image_paths = glob.glob(os.path.join(root_dir, '**', '*.[jJ][pP]*[gG]'), recursive=True) + \
                           glob.glob(os.path.join(root_dir, '**', '*.[pP][nN][gG]'), recursive=True)
        self.label = label
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Read image with OpenCV
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            logger.warning(f"Could not read image {img_path}. Returning a dummy tensor.")
            # Return a dummy sample, which can be filtered out later if needed
            return torch.randn(3, 224, 224), self.label

        # Convert from BGR to RGB color space
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            # Albumentations expects a dictionary
            transformed = self.transform(image=image_rgb)
            image_tensor = transformed['image']
        else:
            # Fallback: convert numpy array to tensor and normalize
            image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float().div(255)

        if self.return_path:
            return image_tensor, self.label, img_path
        else:
            return image_tensor, self.label


class BalancedBatchSampler(Sampler):
    """
    A custom PyTorch Sampler to create balanced batches.
    It ensures each batch has an equal number of samples from two datasets.
    It oversamples the minority class to match the majority class.
    """

    def __init__(self, majority_indices, minority_indices, batch_size):
        """
        Args:
            majority_indices (list): List of indices for the majority class dataset.
            minority_indices (list): List of indices for the minority class dataset.
            batch_size (int): The total batch size. Must be an even number.
        """
        super().__init__()
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be an even number for balanced sampling.")

        self.majority_indices = majority_indices
        self.minority_indices = minority_indices
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

    def __iter__(self):
        # Shuffle both lists of indices
        random.shuffle(self.majority_indices)
        random.shuffle(self.minority_indices)

        # Use itertools.cycle to endlessly loop over the minority indices
        minority_iterator = cycle(self.minority_indices)

        # Iterate through the majority indices in chunks of half_batch
        for i in range(0, len(self.majority_indices), self.half_batch):
            majority_batch_indices = self.majority_indices[i: i + self.half_batch]

            # If the last majority chunk is smaller than half_batch, we skip it.
            if len(majority_batch_indices) < self.half_batch:
                break

            minority_batch_indices = [next(minority_iterator) for _ in range(self.half_batch)]

            # Combine and shuffle the batch indices
            batch_indices = majority_batch_indices + minority_batch_indices
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        # The number of batches is determined by the majority class
        return len(self.majority_indices) // self.half_batch


def build_dataloader(split, config):
    """
    Builds a standard PyTorch DataLoader. For the 'train' split, it uses
    a custom BalancedBatchSampler to ensure 50/50 class distribution in each batch.
    """
    if split not in ['train', 'test']:
        raise ValueError(f"Invalid split name: {split}. Must be 'train' or 'test'.")

    # for human, 0 -> non_human, 1 -> human
    # for nsfw, 0 -> safe, 1 -> not_safe
    not_safe_path = os.path.join(config.DATA_DIR, split, "not_safe")
    safe_path = os.path.join(config.DATA_DIR, split, "safe")

    # Create datasets for each class
    transform = get_train_transforms() if split == 'train' else get_test_transforms()
    not_safe_dataset = ImageClassDataset(not_safe_path, label=1, transform=transform)
    safe_dataset = ImageClassDataset(safe_path, label=0, transform=transform)

    if split == 'train':
        # Determine majority and minority classes
        if len(not_safe_dataset) >= len(safe_dataset):
            majority_ds, minority_ds = not_safe_dataset, safe_dataset
        else:
            majority_ds, minority_ds = safe_dataset, not_safe_dataset

        # The sampler needs indices relative to the concatenated dataset
        majority_indices = list(range(len(majority_ds)))
        minority_indices = list(range(len(majority_ds), len(majority_ds) + len(minority_ds)))

        full_dataset = ConcatDataset([majority_ds, minority_ds])

        batch_sampler = BalancedBatchSampler(majority_indices, minority_indices, config.TRAIN_BATCH_SIZE)

        dataloader = DataLoader(
            full_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:  # 'test' split
        # For validation, a standard shuffled dataloader is fine.
        full_dataset = ConcatDataset([not_safe_dataset, safe_dataset])
        dataloader = DataLoader(
            full_dataset,
            batch_size=config.TEST_BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    return dataloader
