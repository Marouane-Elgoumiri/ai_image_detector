#!/usr/bin/env python3
"""
Diagnose trained model: inspect checkpoint, test predictions, check for bias.

Usage:
    python diagnose_model.py
    python diagnose_model.py --checkpoint models/deep/best_model.pt
    python diagnose_model.py --test-samples 20
"""

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def inspect_checkpoint(path: str):
    """Inspect checkpoint contents."""
    print("=" * 60)
    print("CHECKPOINT INSPECTION")
    print("=" * 60)

    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    print(f"Keys: {list(ckpt.keys())}")
    if 'epoch' in ckpt:
        print(f"Saved at epoch: {ckpt['epoch']}")
    if 'metric' in ckpt:
        print(f"Best metric: {ckpt['metric']:.4f}")
    if 'config' in ckpt:
        print(f"Config: {ckpt['config']}")

    state_dict = ckpt['model_state_dict']
    print(f"\nState dict keys ({len(state_dict)} total):")

    # Detect architecture
    classifier_weights = None
    classifier_bias = None
    for k, v in state_dict.items():
        if 'classifier' in k and 'weight' in k and v.dim() == 2:
            classifier_weights = v
            print(f"\nClassifier layer: {k}")
            print(f"  Shape: {v.shape}")
            if k.replace('weight', 'bias') in state_dict:
                classifier_bias = state_dict[k.replace('weight', 'bias')]
                print(f"  Bias: {classifier_bias}")

    # Check backbone features
    first_conv = state_dict.get('backbone.features.0.0.weight')
    if first_conv is not None:
        print(f"\nFirst conv: {first_conv.shape[0]} channels, {first_conv.shape[1]} input channels")

    # Analyze classifier bias for class imbalance
    if classifier_bias is not None:
        print(f"\n--- CLASS BIAS ANALYSIS ---")
        print(f"  Bias values: {classifier_bias.tolist()}")
        if classifier_bias[0] > classifier_bias[1] + 0.5:
            print(f"  WARNING: Strong bias toward class 0 (Real)")
            print(f"  The model will preferentially predict 'Real'")
        elif classifier_bias[1] > classifier_bias[0] + 0.5:
            print(f"  WARNING: Strong bias toward class 1 (AI-Generated)")
        else:
            print(f"  Bias is relatively balanced")

    # Analyze classifier weights magnitude
    if classifier_weights is not None:
        w_norm = classifier_weights.norm(dim=1)
        print(f"\n--- WEIGHT MAGNITUDES ---")
        print(f"  Class 0 (Real) weight norm: {w_norm[0]:.4f}")
        print(f"  Class 1 (AI) weight norm:   {w_norm[1]:.4f}")
        if w_norm.max() < 0.1:
            print(f"  WARNING: Very small weights - model may not have learned meaningful features")
        elif w_norm.max() < 0.5:
            print(f"  WARNING: Small weights - model is still early in training")

    return ckpt


def test_on_cifake(ckpt_path: str, num_samples: int = 20):
    """Test model on CIFAKE samples to check if it works on training domain."""
    print("\n" + "=" * 60)
    print("CIFAKE DOMAIN TEST")
    print("=" * 60)

    from src.models.classifier import DeepLearningClassifier
    from src.deep.data.dataset import CIFAKEDataset
    from src.deep.data.augmentation import get_validation_transforms

    # Load model
    classifier = DeepLearningClassifier.load(ckpt_path, image_size=224)

    # Load CIFAKE test samples
    val_transform = get_validation_transforms(224)
    dataset = CIFAKEDataset(split='test', transform=val_transform, max_samples=num_samples, seed=42)

    # Collect predictions per class
    real_preds = []
    fake_preds = []
    real_probs = []
    fake_probs = []

    for i in range(len(dataset)):
        image, label = dataset[i]

        # Convert tensor back to numpy for predict_image
        img_np = image.permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        prediction, confidence, _ = classifier.predict_image(img_np)

        if label == 0:  # Real
            real_preds.append(prediction)
            real_probs.append(confidence)
        else:  # Fake
            fake_preds.append(prediction)
            fake_probs.append(confidence)

    # Report
    n_real = len(real_preds)
    n_fake = len(fake_preds)

    print(f"\nTested on {n_real} real + {n_fake} fake CIFAKE images")
    print(f"\nReal images:")
    print(f"  Correctly classified as Real: {real_preds.count(0)}/{n_real} ({100*real_preds.count(0)/max(n_real,1):.1f}%)")
    print(f"  Misclassified as AI:          {real_preds.count(1)}/{n_real} ({100*real_preds.count(1)/max(n_real,1):.1f}%)")
    if real_probs:
        print(f"  Avg AI probability: {np.mean(real_probs):.4f}")

    print(f"\nFake (AI) images:")
    print(f"  Correctly classified as AI: {fake_preds.count(1)}/{n_fake} ({100*fake_preds.count(1)/max(n_fake,1):.1f}%)")
    print(f"  Misclassified as Real:      {fake_preds.count(0)}/{n_fake} ({100*fake_preds.count(0)/max(n_fake,1):.1f}%)")
    if fake_probs:
        print(f"  Avg AI probability: {np.mean(fake_probs):.4f}")

    # Overall
    all_preds = real_preds + fake_preds
    print(f"\n--- SUMMARY ---")
    print(f"  Always predicts Real? {all(p == 0 for p in all_preds)}")
    print(f"  Always predicts AI?   {all(p == 1 for p in all_preds)}")
    print(f"  Prediction distribution: Real={all_preds.count(0)}, AI={all_preds.count(1)}")


def test_raw_logits(ckpt_path: str, num_samples: int = 10):
    """Examine raw logits and softmax outputs."""
    print("\n" + "=" * 60)
    print("RAW LOGIT ANALYSIS")
    print("=" * 60)

    from src.deep.data.dataset import CIFAKEDataset
    from src.deep.data.augmentation import get_validation_transforms
    from src.deep.models import EfficientNetDetector, MobileNetDetector, ConvNeXtDetector

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    # Auto-detect architecture
    from src.models.classifier import DeepLearningClassifier
    model_name, variant = DeepLearningClassifier._detect_architecture(state_dict)
    print(f"Architecture: {model_name} ({variant})")

    if model_name == 'efficientnet':
        model = EfficientNetDetector(variant=variant, num_classes=2, pretrained=False)
    elif model_name == 'mobilenet':
        model = MobileNetDetector(variant=variant, num_classes=2, pretrained=False)
    else:
        model = ConvNeXtDetector(variant=variant, num_classes=2, pretrained=False)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get samples
    val_transform = get_validation_transforms(224)
    dataset = CIFAKEDataset(split='test', transform=val_transform, max_samples=num_samples, seed=42)

    all_logits = []
    all_probs = []
    all_labels = []

    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)

        all_logits.append(logits.cpu().numpy()[0])
        all_probs.append(probs.cpu().numpy()[0])
        all_labels.append(label)

    all_logits = np.array(all_logits)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"\nLogit statistics:")
    print(f"  Mean logit (class 0 - Real): {all_logits[:, 0].mean():.4f}")
    print(f"  Mean logit (class 1 - AI):   {all_logits[:, 1].mean():.4f}")
    print(f"  Std logit (class 0): {all_logits[:, 0].std():.4f}")
    print(f"  Std logit (class 1): {all_logits[:, 1].std():.4f}")

    print(f"\nProbability statistics:")
    print(f"  Mean P(AI): {all_probs[:, 1].mean():.4f}")
    print(f"  Std P(AI):  {all_probs[:, 1].std():.4f}")
    print(f"  Min P(AI):  {all_probs[:, 1].min():.4f}")
    print(f"  Max P(AI):  {all_probs[:, 1].max():.4f}")

    print(f"\nPer-sample breakdown:")
    for i in range(len(all_labels)):
        label_str = "Real" if all_labels[i] == 0 else "Fake"
        pred_str = "Real" if all_probs[i, 1] < 0.5 else "AI"
        correct = "OK" if (all_labels[i] == 0 and all_probs[i, 1] < 0.5) or \
                          (all_labels[i] == 1 and all_probs[i, 1] >= 0.5) else "WRONG"
        print(f"  [{label_str:>4}] P(AI)={all_probs[i,1]:.4f} -> {pred_str:>4} {correct}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose trained AI image detector')
    parser.add_argument('--checkpoint', type=str, default='models/deep/best_model.pt')
    parser.add_argument('--test-samples', type=int, default=20)
    parser.add_argument('--skip-cifake', action='store_true')
    parser.add_argument('--skip-logits', action='store_true')
    args = parser.parse_args()

    ckpt_path = Path(__file__).parent / args.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Diagnosing: {ckpt_path}\n")

    # Phase 1: Inspect checkpoint
    inspect_checkpoint(str(ckpt_path))

    # Phase 2: Test on CIFAKE
    if not args.skip_cifake:
        test_on_cifake(str(ckpt_path), args.test_samples)

    # Phase 3: Raw logit analysis
    if not args.skip_logits:
        test_raw_logits(str(ckpt_path), min(args.test_samples, 20))

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
