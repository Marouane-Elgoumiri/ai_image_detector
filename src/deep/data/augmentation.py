"""
Data augmentation pipeline for AI image detection.

Provides strong augmentation for training and minimal augmentation
for validation/testing.
"""

import torch
from torchvision import transforms
from typing import Optional, Tuple


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_training_transforms(
    image_size: int = 224,
    strong_augmentation: bool = True,
) -> transforms.Compose:
    """
    Get training data transforms with augmentation.

    Args:
        image_size: Target image size
        strong_augmentation: Use strong augmentation pipeline

    Returns:
        Composed transform pipeline
    """
    if strong_augmentation:
        # Strong augmentation for better generalization
        transform_list = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ]
    else:
        # Light augmentation
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    return transforms.Compose(transform_list)


def get_validation_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size

    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size + 8, image_size + 8)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transforms(
    image_size: int = 224,
    strong_augmentation: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get both training and validation transforms.

    Args:
        image_size: Target image size
        strong_augmentation: Use strong augmentation for training

    Returns:
        tuple: (train_transforms, val_transforms)
    """
    train_transforms = get_training_transforms(image_size, strong_augmentation)
    val_transforms = get_validation_transforms(image_size)
    return train_transforms, val_transforms


class MixUp:
    """
    MixUp augmentation for regularization.

    Implements the mixup data augmentation strategy:
    x = lambda * x_i + (1 - lambda) * x_j
    y = lambda * y_i + (1 - lambda) * y_j

    Args:
        alpha: Beta distribution parameter for sampling lambda
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation.

        Args:
            images: Batch of images (B, C, H, W)
            labels: One-hot encoded labels (B, num_classes)

        Returns:
            tuple: (mixed_images, mixed_labels)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_labels


class CutMix:
    """
    CutMix augmentation for regularization.

    Args:
        alpha: Beta distribution parameter
        prob: Probability of applying cutmix
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cutmix augmentation.

        Args:
            images: Batch of images (B, C, H, W)
            labels: One-hot encoded labels (B, num_classes)

        Returns:
            tuple: (mixed_images, mixed_labels)
        """
        import numpy as np

        if np.random.random() > self.prob:
            return images, labels

        batch_size, _, h, w = images.size()
        lam = np.random.beta(self.alpha, self.alpha)

        # Random box coordinates
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply cutmix
        index = torch.randperm(batch_size)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust labels
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return images, mixed_labels


# Import numpy for MixUp/CutMix
import numpy as np