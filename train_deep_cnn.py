#!/usr/bin/env python3
"""
Train AI image detector using end-to-end deep learning.

Supports multiple architectures optimized for different GPU memory:
- MobileNet-V3 (2.5M params): Best for <4GB VRAM
- EfficientNet-V2-S (20M params): For 4-8GB VRAM
- EfficientNet-V2-M (54M params): For 8+GB VRAM

Usage:
    # MobileNet (recommended for low-memory GPUs)
    python train_deep_cnn.py --model mobilenet --epochs 50 --batch-size 16

    # EfficientNet-S with gradient accumulation for limited memory
    python train_deep_cnn.py --model efficientnet --batch-size 8 --accumulation-steps 4

    # Quick test
    python train_deep_cnn.py --model mobilenet --max-train 5000 --max-val 1000 --epochs 3
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train deep CNN for AI-generated image detection'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenet',
        choices=['mobilenet', 'efficientnet', 'convnext'],
        help='Model architecture (mobilenet for low-memory GPUs)'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='small',
        help='Model variant (small/large for MobileNet, small/medium for EfficientNet)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size (reduce for lower memory)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (reduce for lower memory)'
    )
    parser.add_argument(
        '--accumulation-steps',
        type=int,
        default=2,
        help='Gradient accumulation steps (increases effective batch size)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay coefficient'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=3,
        help='Number of warmup epochs'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate'
    )

    # Data arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifake',
        choices=['cifake', 'genbench'],
        help='Dataset to use (cifake=balanced real/fake, genbench=30+ AI generators + CIFAKE reals)'
    )
    parser.add_argument(
        '--max-train',
        type=int,
        default=None,
        help='Maximum training samples (None for all 100K)'
    )
    parser.add_argument(
        '--max-val',
        type=int,
        default=None,
        help='Maximum validation samples (None for all 20K)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of data loading workers (reduce for lower memory)'
    )

    # Training strategy
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze backbone initially (saves memory)'
    )
    parser.add_argument(
        '--unfreeze-schedule',
        type=str,
        default=None,
        help='Unfreeze schedule as JSON, e.g., \'{"10": 2, "20": 4, "30": -1}\''
    )
    parser.add_argument(
        '--layerwise-lr',
        action='store_true',
        help='Use different learning rates for backbone and classifier'
    )
    parser.add_argument(
        '--backbone-lr-mult',
        type=float,
        default=0.1,
        help='Backbone LR multiplier when using layerwise LR'
    )

    # Loss function
    parser.add_argument(
        '--loss',
        type=str,
        default='label_smooth',
        choices=['ce', 'focal', 'label_smooth'],
        help='Loss function'
    )
    parser.add_argument(
        '--focal-alpha',
        type=float,
        default=0.25,
        help='Focal loss alpha parameter'
    )
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Focal loss gamma parameter'
    )
    parser.add_argument(
        '--label-smoothing',
        type=float,
        default=0.1,
        help='Label smoothing factor'
    )

    # Augmentation
    parser.add_argument(
        '--strong-augmentation',
        action='store_true',
        default=True,
        help='Use strong data augmentation'
    )
    parser.add_argument(
        '--no-strong-augmentation',
        action='store_false',
        dest='strong_augmentation',
        help='Disable strong augmentation'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./models/deep',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Early stopping patience'
    )

    # Low memory mode
    parser.add_argument(
        '--low-memory',
        action='store_true',
        help='Optimize for low-memory GPU (<4GB VRAM)'
    )

    # Other
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to train on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Apply low-memory optimizations
    if args.low_memory:
        args.model = 'mobilenet'
        args.batch_size = 8
        args.accumulation_steps = 4
        args.num_workers = 2
        args.freeze_backbone = True
        print("[Low Memory Mode] Applied optimizations for <4GB VRAM")

    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(args):
    """Create model based on arguments."""
    from src.deep.models import EfficientNetDetector, ConvNeXtDetector, MobileNetDetector

    if args.model == 'mobilenet':
        model = MobileNetDetector(
            variant=args.variant,
            num_classes=2,
            pretrained=True,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
        )
    elif args.model == 'efficientnet':
        model = EfficientNetDetector(
            variant=args.variant,
            num_classes=2,
            pretrained=True,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
        )
    elif args.model == 'convnext':
        model = ConvNeXtDetector(
            variant=args.variant,
            num_classes=2,
            pretrained=True,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def create_loss_function(args):
    """Create loss function based on arguments."""
    from src.deep.training.utils import FocalLoss, LabelSmoothingCrossEntropy

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss == 'label_smooth':
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    return criterion


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # GPU memory check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        if gpu_memory < 4 and not args.low_memory:
            print("WARNING: Low GPU memory detected. Consider using --low-memory flag.")
            print("         Or use: --model mobilenet --batch-size 8")

    print("=" * 60)
    print("END-TO-END DEEP LEARNING TRAINING")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} ({args.variant})")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Loss: {args.loss}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print("=" * 60)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print("\n[1/5] Creating data loaders...")
    from src.deep.data import get_transforms

    train_transform, val_transform = get_transforms(
        image_size=args.image_size,
        strong_augmentation=args.strong_augmentation,
    )

    if args.dataset == 'genbench':
        from src.deep.data import create_genbench_dataloaders
        print("Using GenBench dataset (30+ AI generators + CIFAKE reals)")
        train_loader, val_loader = create_genbench_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_samples=args.max_train,
            max_val_samples=args.max_val,
            train_transform=train_transform,
            val_transform=val_transform,
            seed=args.seed,
        )
    else:
        from src.deep.data import create_dataloaders
        print("Using CIFAKE dataset")
        train_loader, val_loader = create_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_samples=args.max_train,
            max_val_samples=args.max_val,
            train_transform=train_transform,
            val_transform=val_transform,
            seed=args.seed,
        )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\n[2/5] Creating model...")
    model = create_model(args)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    from src.deep.training.utils import get_optimizer, get_scheduler

    optimizer = get_optimizer(
        model,
        optimizer_type='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,
        layerwise_lr=args.layerwise_lr,
        backbone_lr_mult=args.backbone_lr_mult,
    )

    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine_warmup',
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=1e-6,
    )

    # Create loss
    criterion = create_loss_function(args)

    # Parse unfreeze schedule
    unfreeze_schedule = None
    if args.unfreeze_schedule:
        import json
        unfreeze_schedule = {int(k): v for k, v in json.loads(args.unfreeze_schedule).items()}
        print(f"Unfreeze schedule: {unfreeze_schedule}")

    # Create trainer
    print("\n[3/5] Setting up trainer...")
    from src.deep.training import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=args.device,
        config={
            'use_amp': True,
            'accumulation_steps': args.accumulation_steps,
            'checkpoint_dir': str(checkpoint_dir),
            'early_stopping_patience': args.early_stopping,
        },
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train
    print("\n[4/5] Training...")
    history = trainer.train(
        num_epochs=args.epochs,
        unfreeze_schedule=unfreeze_schedule,
        eval_every=1,
        save_best_only=True,
    )

    # Save final results
    print("\n[5/5] Saving results...")

    # Load best model
    best_model_path = checkpoint_dir / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    # Final evaluation
    print("\nFinal Evaluation:")
    from sklearn.metrics import classification_report, roc_auc_score

    predictions, labels, probabilities = trainer.predict(val_loader)

    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Real', 'AI-Generated']))

    roc_auc = roc_auc_score(labels, [p[1] for p in probabilities])
    print(f"ROC AUC: {roc_auc:.4f}")

    # Save training summary
    summary = {
        'model': args.model,
        'variant': args.variant,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'effective_batch_size': args.batch_size * args.accumulation_steps,
        'learning_rate': args.lr,
        'best_val_acc': float(max(history['val_acc'])) if history['val_acc'] else 0,
        'best_val_loss': float(min(history['val_loss'])) if history['val_loss'] else 0,
        'final_train_acc': float(history['train_acc'][-1]) if history['train_acc'] else 0,
        'final_val_acc': float(history['val_acc'][-1]) if history['val_acc'] else 0,
        'roc_auc': float(roc_auc),
        'timestamp': datetime.now().isoformat(),
    }

    with open(checkpoint_dir / 'training_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Results saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()