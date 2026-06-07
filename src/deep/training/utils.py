"""
Training utilities: loss functions, optimizers, schedulers, and helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Literal


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for class balance (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model logits (batch, num_classes)
            targets: Ground truth labels (batch,)

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Helps prevent overconfidence and improves generalization.

    Args:
        smoothing: Smoothing factor (default: 0.1)
        reduction: 'mean' or 'sum'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            inputs: Model logits (batch, num_classes)
            targets: Ground truth labels (batch,)

        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -torch.sum(log_probs * smooth_targets, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def get_optimizer(
    model: nn.Module,
    optimizer_type: Literal['adamw', 'sgd', 'adam'] = 'adamw',
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.9,  # Only for SGD
    layerwise_lr: bool = False,
    backbone_lr_mult: float = 0.1,
) -> torch.optim.Optimizer:
    """
    Create an optimizer with optional layer-wise learning rates.

    Args:
        model: The model to optimize
        optimizer_type: Optimizer type
        lr: Learning rate
        weight_decay: Weight decay coefficient
        momentum: Momentum for SGD
        layerwise_lr: Use different LR for backbone vs classifier
        backbone_lr_mult: Multiplier for backbone LR

    Returns:
        Configured optimizer
    """
    if layerwise_lr:
        # Different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': lr * backbone_lr_mult},
            {'params': classifier_params, 'lr': lr},
        ]
    else:
        param_groups = model.parameters()

    if optimizer_type == 'adamw':
        return AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Literal['cosine', 'cosine_warmup', 'step', 'plateau'] = 'cosine',
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs (for cosine_warmup)
        min_lr: Minimum learning rate

    Returns:
        Configured scheduler
    """
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr,
        )

    elif scheduler_type == 'cosine_warmup':
        # Warmup + Cosine decay
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )

    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=min_lr,
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple = (1,),
) -> list:
    """
    Compute top-k accuracy.

    Args:
        outputs: Model outputs (batch, num_classes)
        targets: Ground truth labels (batch,)
        topk: Tuple of k values

    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    scheduler: Optional[Any] = None,
    best_metric: Optional[float] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model
        optimizer: The optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        path: Save path
        scheduler: Optional scheduler
        best_metric: Optional best metric value
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if best_metric is not None:
        checkpoint['best_metric'] = best_metric

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to map tensors to

    Returns:
        Dictionary with epoch, metrics, and best_metric
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
        'best_metric': checkpoint.get('best_metric'),
    }