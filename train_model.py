#!/usr/bin/env python3
"""
Training script for AI Image Classifier.

This script:
1. Loads the combined dataset
2. Extracts features from all images
3. Trains an SVM classifier with PCA
4. Saves the trained model to disk
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dataset import load_combined_dataset, train_test_split
from src.features.pca_features import build_feature_matrix, extract_all_features
from src.models.classifier import AIImageClassifier
from src.models.metrics import evaluate_model, plot_roc_curve, plot_confusion_matrix
from src.utils.visualization import plot_pca_scatter


def parse_args():
    parser = argparse.ArgumentParser(description='Train AI Image Classifier')
    parser.add_argument('--data-path', type=str, 
                        default=str(Path(__file__).parent.parent),
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent / 'models'),
                        help='Directory to save trained model')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Target image size (width and height)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Fraction of data for testing')
    parser.add_argument('--pca-components', type=int, default=None,
                        help='Number of PCA components (None for auto)')
    parser.add_argument('--pca-variance', type=float, default=0.95,
                        help='Target variance ratio for PCA auto mode')
    parser.add_argument('--svm-kernel', type=str, default='rbf',
                        choices=['linear', 'poly', 'rbf'],
                        help='SVM kernel type')
    parser.add_argument('--svm-c', type=float, default=1.0,
                        help='SVM regularization parameter')
    parser.add_argument('--class-weight', type=str, default=None,
                        choices=['balanced', None],
                        help='Class weight strategy (balanced for imbalanced data)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum images per class (None for all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("AI IMAGE CLASSIFIER TRAINING")
    print("="*60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/5] Loading dataset from {args.data_path}")
    print(f"      Image size: {args.image_size}x{args.image_size}")
    print(f"      Test ratio: {args.test_ratio}")
    
    images, labels, paths = load_combined_dataset(
        base_path=args.data_path,
        target_size=(args.image_size, args.image_size),
        max_per_class=args.max_images
    )
    
    if len(images) == 0:
        print("ERROR: No images loaded!")
        sys.exit(1)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"      Loaded {len(images)} images")
    for label, count in zip(unique_labels, counts):
        class_name = "Real" if label == 0 else "AI-Generated"
        print(f"        - {class_name}: {count} images")
    
    print(f"\n[2/5] Splitting dataset (test_ratio={args.test_ratio})")
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    print(f"      Train: {len(train_images)} images")
    print(f"      Test: {len(test_images)} images")
    
    print(f"\n[3/5] Extracting features from training set...")
    X_train, y_train = build_feature_matrix(train_images, train_labels, verbose=True)
    
    print(f"\n[4/5] Training classifier...")
    classifier = AIImageClassifier(
        kernel=args.svm_kernel,
        C=args.svm_c,
        gamma='scale',
        class_weight=args.class_weight,
        n_pca_components=args.pca_components,
        pca_variance_ratio=args.pca_variance,
        random_state=args.seed
    )
    
    classifier.fit(X_train, y_train, verbose=True)
    
    print(f"      Support vectors: {classifier.get_support_vectors_count()}")
    
    print(f"\n[5/5] Evaluating on test set...")
    X_test, y_test = build_feature_matrix(test_images, test_labels, verbose=True)
    
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    
    metrics = evaluate_model(y_test, y_pred, y_proba, verbose=True)
    
    print(f"\nSaving model to {output_dir}")
    classifier.save(str(output_dir))
    
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        X_train_pca, _, _ = classifier.pca.transform(
            classifier.scaler.transform(X_train)
        ), classifier.scaler, classifier.pca
        X_train_scaled = classifier.scaler.transform(X_train)
        X_train_pca = classifier.pca.transform(X_train_scaled)
        
        plot_pca_scatter(
            X_train_pca[:, :2], 
            np.array(train_labels),
            title="PCA: Training Data Distribution",
            save_path=str(vis_dir / 'pca_scatter.png')
        )
        
        plot_roc_curve(
            y_test,
            y_proba[:, 1],
            title="ROC Curve - AI Image Detector",
            save_path=str(vis_dir / 'roc_curve.png')
        )
        
        plot_confusion_matrix(
            y_test,
            y_pred,
            title="Confusion Matrix - Test Set",
            save_path=str(vis_dir / 'confusion_matrix.png')
        )
        
        print(f"      Visualizations saved to {vis_dir}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {output_dir}")
    print(f"\nTo use the model in the web app, run:")
    print(f"  streamlit run app.py -- --model-path {output_dir}")
    
    return metrics


if __name__ == '__main__':
    main()