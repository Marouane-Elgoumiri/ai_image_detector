#!/usr/bin/env python3
"""
Enhanced training script for AI Image Classifier using CIFAKE dataset.

This script:
1. Loads the CIFAKE dataset from HuggingFace (100K train + 20K test)
2. Extracts enhanced features from images
3. Trains multiple classifiers (SVM, RF, GradientBoosting)
4. Compares models and saves the best one
"""

import os
import sys
import argparse
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models.classifier import AIImageClassifier
from src.models.metrics import evaluate_model, plot_roc_curve, plot_confusion_matrix
from src.utils.visualization import plot_pca_scatter


def load_cifake_dataset(
    max_train: int = None,
    max_test: int = None
):
    """Load CIFAKE dataset from HuggingFace with balanced classes."""
    from datasets import load_dataset
    
    print("Loading CIFAKE dataset from HuggingFace...")
    train_ds = load_dataset('dragonintelligence/CIFAKE-image-dataset', split='train')
    test_ds = load_dataset('dragonintelligence/CIFAKE-image-dataset', split='test')
    
    # Shuffle to ensure balanced classes
    train_ds = train_ds.shuffle(seed=42)
    test_ds = test_ds.shuffle(seed=42)
    
    if max_train:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_test:
        test_ds = test_ds.select(range(min(max_test, len(test_ds))))
    
    print(f"Train: {len(train_ds)} samples")
    print(f"Test: {len(test_ds)} samples")
    
    return train_ds, test_ds


def extract_image_features(img):
    """
    Extract enhanced feature vector from a single image.
    
    Includes:
    1. Luminance statistics (5 features)
    2. Gradient statistics (21 features)  
    3. Frequency domain features (10 features)
    4. Texture features (5 features)
    
    Total: 41 features
    """
    from PIL import Image
    import numpy as np
    from src.preprocessing.luminance import extract_luminance
    from src.preprocessing.gradients import compute_gradients, gradient_magnitude, gradient_direction
    from src.features.frequency import compute_fft, extract_frequency_features
    from src.features.texture import extract_texture_features, extract_gradient_histogram_features
    
    # Ensure image is numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Resize to 64x64 for better feature extraction
    if img.shape[0] != 64 or img.shape[1] != 64:
        img = np.array(Image.fromarray(img).resize((64, 64), Image.Resampling.LANCZOS))
    
    # Ensure RGB
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    
    # Extract luminance
    luminance = extract_luminance(img)
    
    # Compute gradients
    gradient_x, gradient_y = compute_gradients(luminance, operator='sobel')
    mag = gradient_magnitude(gradient_x, gradient_y)
    direction = gradient_direction(gradient_x, gradient_y)
    
    # FFT analysis
    _, power_spectrum = compute_fft(luminance)
    
    # Extract all feature types
    from scipy import stats
    
    features = {}
    
    # 1. Luminance statistics
    features['lum_mean'] = np.mean(luminance)
    features['lum_std'] = np.std(luminance)
    features['lum_kurtosis'] = stats.kurtosis(luminance.flatten(), fisher=True)
    features['lum_skewness'] = stats.skew(luminance.flatten())
    
    # 2. Gradient statistics
    mag_flat = mag.flatten()
    dir_flat = direction.flatten()
    
    features['mag_mean'] = np.mean(mag_flat)
    features['mag_std'] = np.std(mag_flat)
    features['mag_kurtosis'] = stats.kurtosis(mag_flat, fisher=True)
    features['mag_skewness'] = stats.skew(mag_flat)
    
    features['dir_mean'] = np.mean(dir_flat)
    features['dir_std'] = np.std(dir_flat)
    
    # Gradient energy
    features['energy_x'] = np.sum(gradient_x**2) / gradient_x.size
    features['energy_y'] = np.sum(gradient_y**2) / gradient_y.size
    features['energy_ratio'] = features['energy_x'] / (features['energy_y'] + 1e-10)
    features['anisotropy'] = abs(features['energy_x'] - features['energy_y']) / (features['energy_x'] + features['energy_y'] + 1e-10)
    
    # 3. Frequency features
    freq_feats = extract_frequency_features(power_spectrum, luminance.shape)
    features.update(freq_feats)
    
    # 4. Texture features
    tex_feats = extract_texture_features(gradient_x, gradient_y)
    for k, v in tex_feats.items():
        features[f'tex_{k}'] = v
    
    # 5. Histogram features
    hist_feats = extract_gradient_histogram_features(gradient_x, gradient_y)
    for k, v in hist_feats.items():
        features[f'hist_{k}'] = v
    
    # Convert to numpy array
    feature_vector = np.array(list(features.values()), dtype=np.float64)
    
    # Handle NaN/Inf
    if not np.isfinite(feature_vector).all():
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vector


def extract_features_batch(dataset, batch_size=1000, desc="Extracting features"):
    """Extract features from a dataset in batches."""
    all_features = []
    all_labels = []
    
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[batch_end - batch_size:batch_end] if i + batch_size < len(dataset) else dataset[i:batch_end]
        
        images = batch['image']
        labels = batch['label']
        
        for img, label in zip(images, labels):
            try:
                features = extract_image_features(img)
                all_features.append(features)
                all_labels.append(int(label))
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
    
    return np.array(all_features), np.array(all_labels)


def parse_args():
    parser = argparse.ArgumentParser(description='Train AI Image Classifier on CIFAKE Dataset')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent / 'models'),
                        help='Directory to save trained models')
    parser.add_argument('--max-train', type=int, default=10000,
                        help='Maximum training samples (None for all)')
    parser.add_argument('--max-test', type=int, default=5000,
                        help='Maximum test samples (None for all)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip feature extraction if already done')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--models', nargs='+', default=['svm', 'rf', 'gb'],
                        choices=['svm', 'rf', 'gb', 'all'],
                        help='Models to train')
    return parser.parse_args()


def train_svm(X_train, y_train, X_test, y_test, output_dir, visualize=False):
    """Train and evaluate SVM classifier."""
    print("\n" + "="*60)
    print("Training SVM Classifier")
    print("="*60)
    
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.calibration import CalibratedClassifierCV
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} components")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # SVM with balanced weights
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    svm.fit(X_train_pca, y_train)
    
    y_pred = svm.predict(X_test_pca)
    y_proba = svm.predict_proba(X_test_pca)
    
    metrics = evaluate_model(y_test, y_pred, y_proba, verbose=True)
    
    # Save
    model_dir = Path(output_dir) / 'svm'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(svm, model_dir / 'model.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    joblib.dump(pca, model_dir / 'pca.pkl')
    
    if visualize:
        plot_roc_curve(y_test, y_proba[:, 1], save_path=str(model_dir / 'roc_curve.png'))
        plot_confusion_matrix(y_test, y_pred, save_path=str(model_dir / 'confusion_matrix.png'))
    
    return metrics, svm, scaler, pca


def train_random_forest(X_train, y_train, X_test, y_test, output_dir, visualize=False):
    """Train and evaluate Random Forest classifier."""
    print("\n" + "="*60)
    print("Training Random Forest Classifier")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    rf.fit(X_train_scaled, y_train)
    
    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)
    
    metrics = evaluate_model(y_test, y_pred, y_proba, verbose=True)
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
    
    # Save
    model_dir = Path(output_dir) / 'random_forest'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf, model_dir / 'model.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    
    if visualize:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Random Forest')
        fig.savefig(str(model_dir / 'roc_curve.png'))
        plt.close()
        
        plot_confusion_matrix(y_test, y_pred, save_path=str(model_dir / 'confusion_matrix.png'))
    
    return metrics, rf, scaler


def train_gradient_boosting(X_train, y_train, X_test, y_test, output_dir, visualize=False):
    """Train and evaluate Gradient Boosting classifier."""
    print("\n" + "="*60)
    print("Training Gradient Boosting Classifier")
    print("="*60)
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    print("Training Gradient Boosting...")
    gb.fit(X_train_scaled, y_train)
    
    y_pred = gb.predict(X_test_scaled)
    y_proba = gb.predict_proba(X_test_scaled)
    
    metrics = evaluate_model(y_test, y_pred, y_proba, verbose=True)
    
    # Save
    model_dir = Path(output_dir) / 'gradient_boosting'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(gb, model_dir / 'model.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    
    if visualize:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Gradient Boosting')
        fig.savefig(str(model_dir / 'roc_curve.png'))
        plt.close()
        
        plot_confusion_matrix(y_test, y_pred, save_path=str(model_dir / 'confusion_matrix.png'))
    
    return metrics, gb, scaler


def main():
    args = parse_args()
    
    print("="*60)
    print("AI IMAGE CLASSIFIER TRAINING (CIFAKE Dataset)")
    print("="*60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if features already extracted
    feature_path = Path(__file__).parent / 'data' / 'cifake_features.npz'
    
    if args.skip_extraction and feature_path.exists():
        print("\n[1/3] Loading pre-extracted features...")
        data = np.load(feature_path)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        print("\n[1/3] Loading and extracting features...")
        train_ds, test_ds = load_cifake_dataset(
            max_train=args.max_train,
            max_test=args.max_test
        )
        
        print("\nExtracting training features...")
        X_train, y_train = extract_features_batch(train_ds, batch_size=500, desc="Train")
        
        print("\nExtracting test features...")
        X_test, y_test = extract_features_batch(test_ds, batch_size=500, desc="Test")
        
        # Save features
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(feature_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        print(f"\nFeatures saved to {feature_path}")
    
    print(f"\nTraining data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test data: {X_test.shape[0]} samples")
    print(f"Label distribution - Train: Real={np.sum(y_train==1)}, Fake={np.sum(y_train==0)}")
    print(f"Label distribution - Test: Real={np.sum(y_test==1)}, Fake={np.sum(y_test==0)}")
    
    print("\n[2/3] Training models...")
    
    results = {}
    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['svm', 'rf', 'gb']
    
    if 'svm' in models_to_train:
        try:
            results['svm'] = train_svm(X_train, y_train, X_test, y_test, output_dir, args.visualize)
        except Exception as e:
            print(f"Error training SVM: {e}")
            results['svm'] = None
    
    if 'rf' in models_to_train:
        try:
            results['rf'] = train_random_forest(X_train, y_train, X_test, y_test, output_dir, args.visualize)
        except Exception as e:
            print(f"Error training RF: {e}")
            results['rf'] = None
    
    if 'gb' in models_to_train:
        try:
            results['gb'] = train_gradient_boosting(X_train, y_train, X_test, y_test, output_dir, args.visualize)
        except Exception as e:
            print(f"Error training GB: {e}")
            results['gb'] = None
    
    print("\n[3/3] Model Comparison")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'AI Recall':<12} {'ROC AUC':<12}")
    print("-"*60)
    
    best_model = None
    best_f1 = 0
    
    for name, result in results.items():
        if result:
            metrics, model, scaler = result[0], result[1], result[2]
            f1 = metrics['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_model = name
            
            print(f"{name.upper():<25} {metrics['accuracy']:<12.4f} {f1:<12.4f} {metrics['recall']:<12.4f} {metrics.get('roc_auc', 0):<12.4f}")
    
    print("-"*60)
    print(f"Best model: {best_model.upper()} (F1: {best_f1:.4f})")
    
    # Save summary
    summary_path = output_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("AI Image Classifier Training Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: CIFAKE (HuggingFace)\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Features: {X_train.shape[1]}\n\n")
        
        for name, result in results.items():
            if result:
                metrics = result[0]
                f.write(f"{name.upper()}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  AI Recall: {metrics['recall']:.4f}\n")
                if 'roc_auc' in metrics:
                    f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write("\n")
        
        f.write(f"Best model: {best_model.upper()} (F1: {best_f1:.4f})\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()