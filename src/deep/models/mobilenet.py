"""
MobileNet-V3 model for AI-generated image detection.

MobileNet-V3 is optimized for mobile/embedded devices with limited memory.
Ideal for GPUs with 4GB or less VRAM.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


class MobileNetDetector(nn.Module):
    """
    MobileNet-V3 model for binary classification of AI-generated images.

    Much smaller than EfficientNet (2-5M parameters vs 20M+).
    Ideal for GPUs with limited VRAM (<4GB).

    Args:
        variant: 'small' (~2.5M params) or 'large' (~5.4M params)
        num_classes: Number of output classes (default: 2)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate before final classifier
        freeze_backbone: If True, freeze all backbone layers
    """

    def __init__(
        self,
        variant: Literal['small', 'large'] = 'small',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.variant = variant
        self.num_classes = num_classes

        # Load pretrained MobileNet-V3
        if variant == 'small':
            if pretrained:
                self.backbone = models.mobilenet_v3_small(
                    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                )
            else:
                self.backbone = models.mobilenet_v3_small(weights=None)
            feature_dim = 576
        elif variant == 'large':
            if pretrained:
                self.backbone = models.mobilenet_v3_large(
                    weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
                )
            else:
                self.backbone = models.mobilenet_v3_large(weights=None)
            feature_dim = 960
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'small' or 'large'.")

        # Replace classifier
        self.backbone.classifier = nn.Identity()

        # Custom lightweight classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes),
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


def create_mobilenet_small(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> MobileNetDetector:
    """Create MobileNet-V3-Small (~2.5M params)."""
    return MobileNetDetector(
        variant='small',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


def create_mobilenet_large(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> MobileNetDetector:
    """Create MobileNet-V3-Large (~5.4M params)."""
    return MobileNetDetector(
        variant='large',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )