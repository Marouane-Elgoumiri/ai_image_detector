"""
Training loop implementation for deep learning models.

Supports:
- Mixed precision training (AMP)
- Gradient accumulation
- Layer-wise unfreezing
- Checkpoint saving
- Early stopping
- TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Callable, Any
import time
import json


class Trainer:
    """
    Trainer for deep learning models.

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Training configuration
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: str = 'auto',
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config or {}

        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Training on device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        # Mixed precision
        self.use_amp = self.config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.autocast_device = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Gradient accumulation
        self.accumulation_steps = self.config.get('accumulation_steps', 1)

        # Checkpoint settings
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.best_metric = 0.0
        self.patience_counter = 0

        # Training state
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }

    def train(
        self,
        num_epochs: int,
        unfreeze_schedule: Optional[Dict[int, int]] = None,
        eval_every: int = 1,
        save_best_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
            unfreeze_schedule: Dict mapping epoch to number of backbone blocks to unfreeze
                               e.g., {5: 2, 10: 4, 15: -1} means:
                               - At epoch 5, unfreeze last 2 blocks
                               - At epoch 10, unfreeze last 4 blocks
                               - At epoch 15, unfreeze all (-1)
            eval_every: Evaluate every N epochs
            save_best_only: Only save when validation accuracy improves

        Returns:
            Training history
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Apply unfreeze schedule
            if unfreeze_schedule and epoch in unfreeze_schedule:
                num_blocks = unfreeze_schedule[epoch]
                print(f"\n[Epoch {epoch}] Unfreezing {num_blocks} backbone blocks...")
                self._unfreeze_backbone(num_blocks)

            # Train one epoch
            train_loss, train_acc = self._train_epoch(epoch)

            # Evaluate
            if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
                val_loss, val_acc = self._validate(epoch)

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_acc)
                    else:
                        self.scheduler.step()

                # Get current LR
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['lr'].append(current_lr)

                # Print progress
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                      f"LR: {current_lr:.2e}")

                # Save checkpoint
                is_best = val_acc > self.best_metric
                if is_best:
                    self.best_metric = val_acc
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if save_best_only and is_best:
                    self._save_checkpoint(epoch, val_acc, is_best=True)
                elif not save_best_only:
                    self._save_checkpoint(epoch, val_acc, is_best=False)

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {self.early_stopping_patience} epochs)")
                    break
            else:
                # Just log training metrics
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_metric:.2f}%")

        # Save final history
        self._save_history()

        return self.history

    def _train_epoch(self, epoch: int) -> tuple:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(self.autocast_device):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total,
            })

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp:
                with autocast(self.autocast_device):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': total_loss / (len(pbar) + 1),
                'acc': 100.0 * correct / total,
            })

        return total_loss / len(self.val_loader), 100.0 * correct / total

    def _unfreeze_backbone(self, num_blocks: int):
        """Unfreeze backbone layers."""
        if hasattr(self.model, 'unfreeze_backbone'):
            self.model.unfreeze_backbone(num_blocks)
        elif hasattr(self.model, 'unfreeze_by_block'):
            self.model.unfreeze_by_block(num_blocks)
        else:
            # Generic unfreezing
            params = list(self.model.backbone.parameters())
            for param in params[-num_blocks:] if num_blocks != -1 else params:
                param.requires_grad = True

    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def _save_history(self):
        """Save training history."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_metric = checkpoint.get('metric', 0.0)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best metric: {self.best_metric:.2f}%")

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> tuple:
        """
        Generate predictions for a dataloader.

        Returns:
            tuple: (predictions, labels, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp:
                with autocast(self.autocast_device):
                    outputs = self.model(images)
            else:
                outputs = self.model(images)

            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        return all_preds, all_labels, all_probs