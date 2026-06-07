"""
GenBench dataset loader for PyTorch.

Combines GenBench AI-generated images (from lrzpellegrini/AI-GenBench-fake_part)
with CIFAKE real images for a more diverse training set.

GenBench includes 30+ generators: FLUX, DALL-E 3, Midjourney, SD variants,
ProGAN, StyleGAN, and more. Only fake images are available; real images
are paired from source datasets (COCO, LAION-400M) but not included.

Strategy: Use GenBench fakes + CIFAKE reals for balanced training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple, List
from pathlib import Path


class GenBenchDataset(Dataset):
    """
    Dataset combining GenBench AI images with CIFAKE real images.

    Creates a balanced dataset:
    - Real images: from CIFAKE training split
    - Fake images: from GenBench (30+ generators)

    Args:
        split: 'train' or 'validation' (GenBench splits)
        transform: Optional transform to apply to images
        max_samples: Max fake samples from GenBench (None for all)
        real_max_samples: Max real samples from CIFAKE (None = match fake count)
        seed: Random seed
    """

    def __init__(
        self,
        split: str = 'train',
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        real_max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.split = split
        self.transform = transform
        self.seed = seed

        self.images: List[Image.Image] = []
        self.labels: List[int] = []
        self.generators: List[str] = []

        self._load_data(split, max_samples, real_max_samples)

    def _load_data(self, split: str, max_fake: Optional[int], max_real: Optional[int]):
        """Load GenBench fake images and CIFAKE real images."""
        from datasets import load_dataset

        # Load GenBench fake images
        print(f"Loading GenBench fake images ({split})...")
        hf_split = 'validation' if split == 'validation' else 'train'
        genbench = load_dataset(
            'lrzpellegrini/AI-GenBench-fake_part',
            split=hf_split,
        )
        genbench = genbench.shuffle(seed=self.seed)

        if max_fake is not None:
            genbench = genbench.select(range(min(max_fake, len(genbench))))

        fake_count = 0
        for item in genbench:
            try:
                img = item['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                self.images.append(img)
                self.labels.append(1)  # 1 = AI-generated
                self.generators.append(item.get('generator', 'unknown'))
                fake_count += 1
            except Exception:
                continue

        print(f"  Loaded {fake_count} GenBench fake images")

        # Load CIFAKE real images to balance
        print(f"Loading CIFAKE real images to balance...")
        cifake_split = 'test' if split == 'validation' else 'train'
        cifake = load_dataset(
            'dragonintelligence/CIFAKE-image-dataset',
            split=cifake_split,
        )
        cifake = cifake.shuffle(seed=self.seed)

        # Filter to real images only (label=0)
        real_indices = [i for i, item in enumerate(cifake) if item['label'] == 0]

        n_real = max_real or fake_count
        real_indices = real_indices[:min(n_real, len(real_indices))]

        real_count = 0
        for idx in real_indices:
            try:
                img = cifake[idx]['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                self.images.append(img)
                self.labels.append(0)  # 0 = Real
                self.generators.append('real')
                real_count += 1
            except Exception:
                continue

        print(f"  Loaded {real_count} CIFAKE real images")
        print(f"  Total: {len(self.images)} images (Real={real_count}, Fake={fake_count})")

        # Print generator distribution
        generators, counts = np.unique(self.generators, return_counts=True)
        print(f"  Generator distribution:")
        for gen, count in sorted(zip(generators, counts), key=lambda x: -x[1]):
            print(f"    {gen}: {count}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_label_distribution(self) -> dict:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def get_generator_distribution(self) -> dict:
        generators, counts = np.unique(self.generators, return_counts=True)
        return {str(k): int(v) for k, v in zip(generators, counts)}


def create_genbench_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders from GenBench + CIFAKE real images.

    Args:
        batch_size: Batch size
        num_workers: Data loading workers
        max_train_samples: Max fake samples for training (None for all)
        max_val_samples: Max fake samples for validation (None for all)
        train_transform: Training transforms
        val_transform: Validation transforms
        seed: Random seed

    Returns:
        tuple: (train_loader, val_loader)
    """
    from .augmentation import get_training_transforms, get_validation_transforms

    if train_transform is None:
        train_transform = get_training_transforms()
    if val_transform is None:
        val_transform = get_validation_transforms()

    train_dataset = GenBenchDataset(
        split='train',
        transform=train_transform,
        max_samples=max_train_samples,
        seed=seed,
    )

    val_dataset = GenBenchDataset(
        split='validation',
        transform=val_transform,
        max_samples=max_val_samples,
        seed=seed,
    )

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
