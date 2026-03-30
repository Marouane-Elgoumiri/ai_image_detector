#!/usr/bin/env python3
"""
Fast training script using our actual datasets (Dataset_AI-Real_images and Dataset_Real-Fake_images).

This script uses optimized features and multiple models for best accuracy.
"""

import os
import sys
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dataset import load_combined_dataset, train_test_split
from src.preprocessing.luminance import extract_luminance
from src.preprocessing.gradients import compute_gradients, gradient_magnitude, gradient_direction
from src.features.frequency import compute_fft, extract_frequency_features
from src.features.texture import extract_texture_features, extract_gradient_histogram_features
from src.models.metrics import evaluate_model


def extract_optimized_features(image):
    """
    Extract optimized feature vector from an image.
    Focus on most discriminative features for better accuracy.
    """
    from scipy import stats
    from PIL import Image
    
    if hasattr(image, 'mode'):  # PIL Image
        image = np.array(image)
    
    # Resize to 64x64 for better feature extraction
    if image.shape[0] != 64 or image.shape[1] != 64:
        image = np.array(Image.fromarray(image).resize((64, 64), Image.Resampling.LANCZOS))
    
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    
    # Extract luminance
    luminance = extract_luminance(image)
    
    # Compute gradients
    gradient_x, gradient_y = compute_gradients(luminance, operator='sobel')
    mag = gradient_magnitude(gradient_x, gradient_y)
    direction = gradient_direction(gradient_x, gradient_y)
    
    # FFT analysis
    _, power_spectrum = compute_fft(luminance)
    
    features = []
    
    # 1. Luminance statistics (5 features)
    features.extend([
        np.mean(luminance),
        np.std(luminance),
        stats.kurtosis(luminance.flatten(), fisher=True),
        stats.skew(luminance.flatten()),
        np.var(luminance)
    ])
    
    # 2. Gradient statistics (10 features)
    mag_flat = mag.flatten()
    dir_flat = direction.flatten()
    
    features.extend([
        np.mean(mag_flat),
        np.std(mag_flat),
        stats.kurtosis(mag_flat, fisher=True),
        stats.skew(mag_flat),
        np.median(mag_flat),
        np.mean(dir_flat),
        np.std(dir_flat),
        np.sum(gradient_x**2) / gradient_x.size,  # energy_x
        np.sum(gradient_y**2) / gradient_y.size,  # energy_y
        np.abs(np.sum(gradient_x**2) - np.sum(gradient_y**2)) / (np.sum(gradient_x**2) + np.sum(gradient_y**2) + 1e-10)  # anisotropy
    ])
    
    # 3. Frequency features (10 features)
    freq_feats = extract_frequency_features(power_spectrum, luminance.shape)
    features.extend([
        freq_feats['low_freq_energy'],
        freq_feats['mid_freq_energy'],
        freq_feats['high_freq_energy'],
        freq_feats['low_high_ratio'],
        freq_feats['radial_entropy'],
        freq_feats['spectrum_mean'],
        freq_feats['spectrum_std'],
        freq_feats['spectrum_max'],
        freq_feats['spectral_flatness'],
        freq_feats['total_energy_normalized']
    ])
    
    # 4. Texture features (7 features)
    tex_feats = extract_texture_features(gradient_x, gradient_y)
    features.extend([
        tex_feats['kurtosis_mag'],
        tex_feats['entropy_mag'],
        tex_feats['skewness_mag'],
        tex_feats['uniform_kl'],
        tex_feats['energy_x'],
        tex_feats['energy_y'],
        tex_feats['anisotropy']
    ])
    
    # 5. Histogram features (4 features)
    hist_feats = extract_gradient_histogram_features(gradient_x, gradient_y)
    features.extend([
        hist_feats['hist_max_value'],
        hist_feats['hist_entropy'],
        hist_feats['hist_smoothness'],
        hist_feats['dominant_orientation']
    ])
    
    feature_vector = np.array(features, dtype=np.float64)
    
    # Handle NaN/Inf
    if not np.isfinite(feature_vector).all():
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vector


def extract_dataset_features(images, labels, desc="Extracting"):
    """Extract features from a dataset."""
    all_features = []
    all_labels = []
    
    for img, label in tqdm(zip(images, labels), total=len(images), desc=desc):
        try:
            features = extract_optimized_features(img)
            all_features.append(features)
            all_labels.append(label)
        except Exception as e:
            continue
    
    return np.array(all_features), np.array(all_labels)


def main():
    print("="*60)
    print("FAST TRAINING ON ORIGINAL DATASETS")
    print("="*60)
    
    output_dir = Path('./models/optimized')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load our actual datasets
    print("\n[1/4] Loading datasets...")
    images, labels, _ = load_combined_dataset(
        base_path='..',
        target_size=(64, 64)
    )
    
    print(f"Total images: {len(images)}")
    print(f"Real: {sum(l == 0 for l in labels)}, AI: {sum(l == 1 for l in labels)}")
    
    # Split
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        images, labels, test_ratio=0.2, random_state=42
    )
    
    print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")
    
    # Extract features
    print("\n[2/4] Extracting features...")
    X_train, y_train = extract_dataset_features(train_imgs, train_labels, "Train")
    X_test, y_test = extract_dataset_features(test_imgs, test_labels, "Test")
    
    print(f"Features shape: {X_train.shape}")
    
    # Save features
    np.savez(output_dir / 'features.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    # Train multiple models
    print("\n[3/4] Training models...")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Gradient Boosting with tuned hyperparameters
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    y_pred_gb = gb.predict(X_test_scaled)
    y_proba_gb = gb.predict_proba(X_test_scaled)
    results['gb'] = evaluate_model(y_test, y_pred_gb, y_proba_gb, verbose=False)
    models['gb'] = gb
    print(f"GB Accuracy: {results['gb']['accuracy']:.4f}, F1: {results['gb']['f1_score']:.4f}")
    
    # Random Forest with tuned hyperparameters
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_proba_rf = rf.predict_proba(X_test_scaled)
    results['rf'] = evaluate_model(y_test, y_pred_rf, y_proba_rf, verbose=False)
    models['rf'] = rf
    print(f"RF Accuracy: {results['rf']['accuracy']:.4f}, F1: {results['rf']['f1_score']:.4f}")
    
    # SVM with probability calibration
    print("\nTraining SVM...")
    svm = SVC(
        kernel='rbf',
        C=100,
        gamma='auto',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    y_proba_svm = svm.predict_proba(X_test_scaled)
    results['svm'] = evaluate_model(y_test, y_pred_svm, y_proba_svm, verbose=False)
    models['svm'] = svm
    print(f"SVM Accuracy: {results['svm']['accuracy']:.4f}, F1: {results['svm']['f1_score']:.4f}")
    
    # Ensemble (soft voting)
    print("\nTraining Ensemble...")
    ensemble_proba = (y_proba_gb + y_proba_rf + y_proba_svm) / 3
    y_pred_ensemble = (ensemble_proba[:, 1] > 0.5).astype(int)
    results['ensemble'] = evaluate_model(y_test, y_pred_ensemble, ensemble_proba, verbose=False)
    print(f"Ensemble Accuracy: {results['ensemble']['accuracy']:.4f}, F1: {results['ensemble']['f1_score']:.4f}")
    
    # Find best model
    print("\n[4/4] Results Summary")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'AI Recall':<12} {'ROC AUC':<12}")
    print("-"*60)
    
    best_model_name = None
    best_f1 = 0
    
    for name, res in results.items():
        f1 = res['f1_score']
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
        
        print(f"{name.upper():<20} {res['accuracy']:<12.4f} {f1:<12.4f} {res['recall']:<12.4f} {res.get('roc_auc', 0):<12.4f}")
    
    print("-"*60)
    print(f"Best model: {best_model_name.upper()} (F1: {best_f1:.4f})")
    
    # Save best model
    if best_model_name == 'ensemble':
        # Save all models for ensemble
        ensemble_dir = output_dir / 'ensemble'
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, ensemble_dir / f'{name}.pkl')
        joblib.dump(scaler, ensemble_dir / 'scaler.pkl')
        print(f"\nEnsemble saved to {ensemble_dir}")
    else:
        # Save best single model
        model_dir = output_dir / best_model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models[best_model_name], model_dir / 'model.pkl')
        joblib.dump(scaler, model_dir / 'scaler.pkl')
        print(f"\nBest model ({best_model_name}) saved to {model_dir}")
    
    # Save training summary
    summary = {
        'dataset': 'AI-Real + Real-Fake (combined)',
        'train_samples': len(train_imgs),
        'test_samples': len(test_imgs),
        'features': X_train.shape[1],
        'best_model': best_model_name,
        'results': {name: {'accuracy': r['accuracy'], 'f1': r['f1_score'], 'recall': r['recall']} for name, r in results.items()}
    }
    joblib.dump(summary, output_dir / 'summary.pkl')
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()