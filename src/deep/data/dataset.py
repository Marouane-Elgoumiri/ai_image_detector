"""
CIFAKE dataset loader for PyTorch.

CIFAKE is a dataset containing 100K training images and 20K test images
for AI-generated image detection (real vs fake).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple
from pathlib import Path


class CIFAKEDataset(Dataset):
    """
    CIFAKE dataset wrapper for PyTorch.

    Loads images from HuggingFace datasets or from local cache.

    Args:
        split: 'train' or 'test'
        transform: Optional transform to apply to images
        max_samples: Maximum number of samples to load (None for all)
        cache_dir: Directory to cache the dataset
        seed: Random seed for reproducible sampling
    """

    def __init__(
        self,
        split: str = 'train',
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.seed = seed

        # Load dataset
        self._load_dataset(cache_dir)

    def _load_dataset(self, cache_dir: Optional[str]):
        """Load the CIFAKE dataset from HuggingFace."""
        try:
            from datasets import load_dataset

            print(f"Loading CIFAKE {self.split} split...")
            dataset = load_dataset(
                'dragonintelligence/CIFAKE-image-dataset',
                split=self.split,
                cache_dir=cache_dir,
            )

            # Shuffle for reproducibility
            dataset = dataset.shuffle(seed=self.seed)

            # Limit samples if specified
            if self.max_samples is not None:
                dataset = dataset.select(range(min(self.max_samples, len(dataset))))

            self.data = dataset
            print(f"Loaded {len(self.data)} samples for {self.split}")

        except ImportError:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install with: pip install datasets"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            tuple: (image_tensor, label)
                   - image_tensor: Transformed image (C, H, W)
                   - label: 0 for real, 1 for fake/AI-generated
        """
        item = self.data[idx]

        # Get image and label
        image = item['image']
        label = int(item['label'])

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_label_distribution(self) -> dict:
        """Get the distribution of labels in the dataset."""
        labels = [int(item['label']) for item in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}


class IndexedDataset(Dataset):
    """Wrapper that returns (index, image, label) for tracking."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return idx, image, label


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        max_train_samples: Maximum training samples (None for all)
        max_val_samples: Maximum validation samples (None for all)
        train_transform: Training transforms (uses default if None)
        val_transform: Validation transforms (uses default if None)
        cache_dir: Dataset cache directory
        seed: Random seed

    Returns:
        tuple: (train_loader, val_loader)
    """
    from .augmentation import get_training_transforms, get_validation_transforms

    # Use default transforms if not provided
    if train_transform is None:
        train_transform = get_training_transforms()
    if val_transform is None:
        val_transform = get_validation_transforms()

    # Create datasets
    train_dataset = CIFAKEDataset(
        split='train',
        transform=train_transform,
        max_samples=max_train_samples,
        cache_dir=cache_dir,
        seed=seed,
    )

    val_dataset = CIFAKEDataset(
        split='test',
        transform=val_transform,
        max_samples=max_val_samples,
        cache_dir=cache_dir,
        seed=seed,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader