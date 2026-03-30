"""
EfficientNet-V2 model for AI-generated image detection.

This module provides fine-tuned EfficientNet models with:
- Transfer learning from ImageNet
- Layer-wise unfreezing strategy
- Optional pretrained weights
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Literal


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-V2 model for binary classification of AI-generated images.

    Supports EfficientNet-V2-S (small) and EfficientNet-V2-M (medium) variants.
    Uses transfer learning with optional layer-wise fine-tuning.

    Args:
        variant: Model size - 'small' or 'medium'
        num_classes: Number of output classes (default: 2 for real/fake)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate before final classifier
        freeze_backbone: If True, freeze all backbone layers initially
    """

    def __init__(
        self,
        variant: Literal['small', 'medium'] = 'small',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.variant = variant
        self.num_classes = num_classes

        # Load pretrained EfficientNet-V2
        if variant == 'small':
            if pretrained:
                self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_v2_s(weights=None)
            feature_dim = 1280
        elif variant == 'medium':
            if pretrained:
                self.backbone = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_v2_m(weights=None)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'small' or 'medium'.")

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = -1):
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end.
                        -1 means unfreeze all layers.
        """
        # Get all parameters
        params = list(self.backbone.parameters())

        if num_layers == -1:
            # Unfreeze all
            for param in params:
                param.requires_grad = True
        else:
            # Unfreeze last num_layers parameter groups
            for param in params[-num_layers:]:
                param.requires_grad = True

    def unfreeze_by_block(self, block_idx: int):
        """
        Unfreeze backbone from a specific block index onwards.

        EfficientNet-V2 has 7 feature blocks (0-6).
        Use this for gradual unfreezing during training.

        Args:
            block_idx: Block index to start unfreezing (0-6)
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then unfreeze from block_idx onwards
        # features is a Sequential of blocks
        features = self.backbone.features

        for i in range(block_idx, len(features)):
            for param in features[i].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
               Expected size: 224x224 or 384x384

        Returns:
            Logits of shape (batch, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.

        Args:
            x: Input tensor

        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        return self.backbone(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_efficientnet_small(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> EfficientNetDetector:
    """Create EfficientNet-V2-S model."""
    return EfficientNetDetector(
        variant='small',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


def create_efficientnet_medium(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> EfficientNetDetector:
    """Create EfficientNet-V2-M model."""
    return EfficientNetDetector(
        variant='medium',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )