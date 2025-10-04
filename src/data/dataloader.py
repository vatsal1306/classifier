import os

import torch
import webdataset as wds

from src.data.transformations import get_train_transforms, get_test_transforms


def build_dataloader(split, config):
    """
    Builds a WebDataset-based DataLoader with balanced sampling for the 'train' split.

    Args:
        split (str): The dataset split to load ('train' or 'test').
        config: The configuration object.

    Returns:
        torch.utils.data.DataLoader: The configured DataLoader.
    """
    if split not in ['train', 'test']:
        raise ValueError(f"Invalid split name: {split}. Must be 'train' or 'test'.")

    # Define paths to the class-specific shard directories
    human_path = os.path.join(config.DATA_DIR, split, "human", "shard-*.tar")
    non_human_path = os.path.join(config.DATA_DIR, split, "non_human", "shard-*.tar")

    if split == 'train':
        # --- Balanced Sampling for Training ---
        # Create separate datasets for each class
        human_dataset = wds.WebDataset(human_path).shuffle(1000)
        non_human_dataset = wds.WebDataset(non_human_path).shuffle(1000)

        # Combine them using 'Shorter' to ensure they yield samples in lockstep
        # and 'zip' to create pairs (human_sample, non_human_sample)
        # 'concat' then flattens these pairs into a single stream for the dataloader
        combined_dataset = wds.Shorter([human_dataset, non_human_dataset]).zip().concat()

        # Apply the training transformations
        dataset = combined_dataset.decode("pil").to_tuple("jpg;png", "cls").map_tuple(get_train_transforms(),
                                                                                      lambda x: torch.tensor(
                                                                                          int(x.decode()))).shuffle(
            2000)

        batch_size = config.TRAIN_BATCH_SIZE
        # Ensure batch size is even for perfect 50/50 split
        if batch_size % 2 != 0:
            raise ValueError("TRAIN_BATCH_SIZE must be an even number for balanced sampling.")

    else:  # 'test' split
        # --- Standard Sampling for Testing ---
        # For validation, we can just combine the datasets
        dataset_urls = [human_path, non_human_path]
        dataset = wds.WebDataset(dataset_urls).shuffle(1000).decode("pil").to_tuple("jpg;png", "cls").map_tuple(
            get_test_transforms(), lambda x: torch.tensor(int(x.decode())))
        batch_size = config.TEST_BATCH_SIZE

    # Create the DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    return dataloader
