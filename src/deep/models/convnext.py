"""
ConvNeXt model for AI-generated image detection.

ConvNeXt is a modernized CNN architecture that incorporates
design principles from Vision Transformers while maintaining
the efficiency of convolutional networks.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


class ConvNeXtDetector(nn.Module):
    """
    ConvNeXt model for binary classification of AI-generated images.

    Args:
        variant: Model size - 'tiny', 'small', or 'base'
        num_classes: Number of output classes (default: 2)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate before final classifier
        freeze_backbone: If True, freeze all backbone layers
    """

    def __init__(
        self,
        variant: Literal['tiny', 'small', 'base'] = 'tiny',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.variant = variant
        self.num_classes = num_classes

        # Map variant to torchvision models
        model_map = {
            'tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
            'small': (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
            'base': (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
        }

        if variant not in model_map:
            raise ValueError(f"Unknown variant: {variant}. Use 'tiny', 'small', or 'base'.")

        model_fn, weights, feature_dim = model_map[variant]

        # Load pretrained model
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)

        # Get classifier input dimension
        self.feature_dim = feature_dim

        # Replace classifier
        self.backbone.classifier = nn.Identity()

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = -1):
        """Unfreeze backbone layers for fine-tuning."""
        params = list(self.backbone.parameters())

        if num_layers == -1:
            for param in params:
                param.requires_grad = True
        else:
            for param in params[-num_layers:]:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone."""
        return self.backbone(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_convnext_tiny(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> ConvNeXtDetector:
    """Create ConvNeXt-Tiny model."""
    return ConvNeXtDetector(
        variant='tiny',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )