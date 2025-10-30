import glob
import logging
import os
import random
from itertools import cycle
from typing import Iterator, List, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset

from src.data.transformations import get_train_transforms, get_test_transforms

# Get a logger instance for this module. It will inherit the root logger's configuration.
logger = logging.getLogger(__name__)

IMG_PATTERNS = [
    "**/*.[jJ][pP]*[gG]",  # jpg / jpeg variants
    "**/*.[pP][nN][gG]",  # png
    "**/*.[wW][eE][bB][pP]",  # webp
    "**/*.[bB][mM][pP]",  # bmp
    "**/*.[tT][iI][fF]*",  # tif / tiff
]


def _list_images(root_dir: str) -> List[str]:
    paths = []
    for pat in IMG_PATTERNS:
        paths.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))
    return sorted(paths)


class ImageClassDataset(Dataset):
    """
    A PyTorch Dataset for loading images using OpenCV from a single class folder.
    """

    def __init__(self, root_dir, label, transform=None, return_path=False):
        """
        Args:
            root_dir (str): Directory with all the images for one class.
            label (int): Class id (0..NUM_CLASSES-1).
            transform (callable, optional): Albumentations transform to be applied.
            return_path (bool): If True, __getitem__ returns the image path.
        """
        self.image_paths = _list_images(root_dir) if os.path.isdir(root_dir) else []
        if not self.image_paths:
            logger.warning(f"No images found in: {root_dir}")
        self.label = int(label)
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
            return torch.randn(3, 224, 224), self.label if not self.return_path else (torch.randn(3, 224, 224),
                                                                                      self.label, img_path)

        # Convert from BGR to RGB color space
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image_rgb)
            image_tensor = transformed['image']
        else:
            image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float().div(255)

        if self.return_path:
            return image_tensor, self.label, img_path
        else:
            return image_tensor, self.label


# ---------------- Anchored Balanced Sampler ---------------- #

class AnchoredBalancedBatchSampler(Sampler[List[int]]):
    """
    Anchored balanced batching for multi-class classification.

    Rules:
      - Determine anchor class (largest cardinality or user-specified).
      - Build batches of size `batch_size` (as balanced as possible across classes).
      - Anchor indices are consumed WITHOUT replacement.
      - Other classes are sampled WITH replacement (cycled).
      - Epoch ends immediately when the anchor class is exhausted.
    """

    def __init__(
            self,
            labels: Sequence[int],
            batch_size: int,
            num_classes: int,
            anchor_class: int | str = "auto",
            drop_last: bool = True,
            seed: int = 42,
            balanced_per_batch: bool = True,
    ):
        super().__init__(None)
        assert batch_size >= 1, "batch_size must be >= 1"
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.drop_last = drop_last
        self.seed = seed
        self.balanced_per_batch = balanced_per_batch

        self.labels = np.asarray(labels, dtype=np.int64)
        classes = list(range(num_classes))
        # class buckets
        buckets = {c: np.where(self.labels == c)[0].tolist() for c in classes}
        counts = {c: len(buckets[c]) for c in classes}

        # choose anchor
        if anchor_class == "auto":
            self.anchor = max(counts, key=counts.get)
        else:
            self.anchor = int(anchor_class)

        rng = random.Random(seed)
        for c in classes:
            rng.shuffle(buckets[c])

        self.anchor_indices = buckets[self.anchor]
        self.minority_cycles = {
            c: cycle(buckets[c] if buckets[c] else [None]) for c in classes if c != self.anchor
        }
        self.minority_nonempty = [c for c in classes if c != self.anchor and counts[c] > 0]
        self.classes = classes

        # per-batch allocation
        if balanced_per_batch:
            base = batch_size // num_classes
            rem = batch_size % num_classes
            class_order = [self.anchor] + [c for c in classes if c != self.anchor]
            alloc = {c: base for c in classes}
            for k in range(rem):
                alloc[class_order[k % len(class_order)]] += 1
            # ensure anchor appears
            if alloc[self.anchor] == 0:
                alloc[self.anchor] = 1
                for c in class_order[1:]:
                    if alloc[c] > 0:
                        alloc[c] -= 1
                        break
            self.alloc = alloc
        else:
            self.alloc = {c: 0 for c in classes}
            self.alloc[self.anchor] = 1
            others = [c for c in classes if c != self.anchor]
            self.alloc[others[0]] = batch_size - 1

        # number of batches in an epoch: anchor exhaustion
        self._batches = (len(self.anchor_indices) + self.alloc[self.anchor] - 1) // max(1, self.alloc[self.anchor])

    def __iter__(self) -> Iterator[List[int]]:
        anchor_ptr = 0
        produced = 0

        for _ in range(self._batches):
            batch: List[int] = []

            for c in self.classes:
                need = self.alloc[c]
                if need <= 0:
                    continue

                if c == self.anchor:
                    available = len(self.anchor_indices) - anchor_ptr
                    take = min(need, available)
                    if take > 0:
                        batch.extend(self.anchor_indices[anchor_ptr:anchor_ptr + take])
                        anchor_ptr += take

                    deficit = need - take
                    for k in range(deficit):
                        if self.minority_nonempty:
                            cc = self.minority_nonempty[(produced + k) % len(self.minority_nonempty)]
                            nxt = next(self.minority_cycles[cc])
                            if nxt is not None:
                                batch.append(nxt)
                else:
                    for _k in range(need):
                        if c in self.minority_cycles:
                            nxt = next(self.minority_cycles[c])
                            if nxt is not None:
                                batch.append(nxt)

            # enforce exact batch size when dropping last
            if len(batch) > self.batch_size:
                batch = batch[: self.batch_size]

            if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                produced += 1
                yield batch

            # stop epoch once anchor exhausted
            if anchor_ptr >= len(self.anchor_indices):
                break

    def __len__(self) -> int:
        return self._batches


# ---------------- Builder ---------------- #

def build_dataloader(split, cfg):
    """
    Builds a standard PyTorch DataLoader.

    Train:
      - Uses AnchoredBalancedBatchSampler to implement your epoch rule
        (exhaust majority class per epoch; minority sampled with replacement).
    Test:
      - Standard DataLoader (shuffle=True to match your prior style).
    """
    if split not in ['train', 'test']:
        raise ValueError(f"Invalid split name: {split}. Must be 'train' or 'test'.")

    split_dir = os.path.join(cfg.DATA_DIR, split)

    # class folders
    safe_dir = os.path.join(split_dir, "safe")
    not_safe_dir = os.path.join(split_dir, "not_safe")
    kiss_dir = os.path.join(split_dir, "kiss")

    # transforms
    transform = get_train_transforms() if split == 'train' else get_test_transforms()

    # datasets
    safe_ds = ImageClassDataset(safe_dir, label=0, transform=transform, return_path=(split != 'train'))
    not_safe_ds = ImageClassDataset(not_safe_dir, label=1, transform=transform, return_path=(split != 'train'))
    kiss_ds = ImageClassDataset(kiss_dir, label=2, transform=transform, return_path=(split != 'train'))

    full_dataset = ConcatDataset([safe_ds, not_safe_ds, kiss_ds])
    lengths = [len(safe_ds), len(not_safe_ds), len(kiss_ds)]

    if split == 'train':
        # labels list aligned with ConcatDataset ordering
        labels = [0] * lengths[0] + [1] * lengths[1] + [2] * lengths[2]

        batch_sampler = AnchoredBalancedBatchSampler(
            labels=labels,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            num_classes=cfg.NUM_CLASSES,
            anchor_class=cfg.ANCHOR_CLASS,
            drop_last=cfg.DROP_LAST,
            seed=cfg.SEED,
            balanced_per_batch=cfg.BALANCED_PER_BATCH,
        )

        dataloader = DataLoader(
            full_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            full_dataset,
            batch_size=cfg.TEST_BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    return dataloader
