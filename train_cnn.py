#!/usr/bin/env python3
"""
Train AI image detector using PyTorch pre-trained CNN features.
"""

import os
import sys
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class FeatureExtractor(nn.Module):
    """Extract features from pre-trained ResNet."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x.squeeze(-1).squeeze(-1)


def preprocess_image(img):
    """Preprocess image for ResNet."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        img = Image.fromarray(img.astype(np.uint8))
    
    return transform(img)


def extract_features_batch(images, model, batch_size=32, device='cpu'):
    """Extract features from a batch of images."""
    model.eval()
    model.to(device)
    
    all_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i + batch_size]
            
            tensors = []
            for img in batch:
                try:
                    tensor = preprocess_image(img)
                    tensors.append(tensor)
                except Exception as e:
                    continue
            
            if tensors:
                batch_tensor = torch.stack(tensors).to(device)
                features = model(batch_tensor)
                all_features.append(features.cpu().numpy())
    
    return np.vstack(all_features)


def main():
    print("="*60)
    print("TRAINING WITH PyTorch ResNet50 FEATURES")
    print("="*60)
    
    output_dir = Path('./models/cnn')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n[1/4] Loading datasets...")
    from src.utils.dataset import load_combined_dataset, train_test_split
    
    images, labels, _ = load_combined_dataset(
        base_path='..',
        target_size=(256, 256)
    )
    
    print(f"Total images: {len(images)}")
    print(f"Real: {sum(l == 0 for l in labels)}, AI: {sum(l == 1 for l in labels)}")
    
    # Split
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        images, labels, test_ratio=0.2, random_state=42
    )
    
    print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")
    
    # Extract features
    print("\n[2/4] Extracting ResNet50 features...")
    
    feature_cache = output_dir / 'resnet_features.npz'
    if feature_cache.exists():
        print("Loading cached features...")
        data = np.load(feature_cache)
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    else:
        model = FeatureExtractor()
        
        print("Extracting training features...")
        X_train = extract_features_batch(train_imgs, model, batch_size=64, device=device)
        y_train = np.array(train_labels)
        
        print("Extracting test features...")
        X_test = extract_features_batch(test_imgs, model, batch_size=64, device=device)
        y_test = np.array(test_labels)
        
        np.savez(feature_cache, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    print(f"CNN features shape: {X_train.shape}")
    
    # Train classifiers
    print("\n[3/4] Training classifiers...")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models_dict = {}
    results = {}
    
    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    y_pred_gb = gb.predict(X_test_scaled)
    y_proba_gb = gb.predict_proba(X_test_scaled)[:, 1]
    results['gb'] = {
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'f1': f1_score(y_test, y_pred_gb),
        'roc_auc': roc_auc_score(y_test, y_proba_gb)
    }
    models_dict['gb'] = gb
    print(f"GB: Acc={results['gb']['accuracy']:.4f}, F1={results['gb']['f1']:.4f}, AUC={results['gb']['roc_auc']:.4f}")
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    results['rf'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_proba_rf)
    }
    models_dict['rf'] = rf
    print(f"RF: Acc={results['rf']['accuracy']:.4f}, F1={results['rf']['f1']:.4f}, AUC={results['rf']['roc_auc']:.4f}")
    
    # SVM
    print("\nTraining SVM...")
    svm = SVC(kernel='rbf', C=10, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    y_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]
    results['svm'] = {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'f1': f1_score(y_test, y_pred_svm),
        'roc_auc': roc_auc_score(y_test, y_proba_svm)
    }
    models_dict['svm'] = svm
    print(f"SVM: Acc={results['svm']['accuracy']:.4f}, F1={results['svm']['f1']:.4f}, AUC={results['svm']['roc_auc']:.4f}")
    
    # Find best
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model = models_dict[best_model_name]
    best_f1 = results[best_model_name]['f1']
    
    print("\n[4/4] Summary")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'ROC AUC':<12}")
    print("-"*60)
    for name, res in results.items():
        print(f"{name.upper():<20} {res['accuracy']:<12.4f} {res['f1']:<12.4f} {res['roc_auc']:<12.4f}")
    print("-"*60)
    print(f"Best model: {best_model_name.upper()} (F1: {best_f1:.4f})")
    
    # Save best
    model_dir = output_dir / best_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / 'model.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    
    # Print classification report
    print(f"\nClassification Report for best model ({best_model_name.upper()}):")
    y_pred_best = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_best, target_names=['Real', 'AI-Generated']))
    
    print(f"\nModel saved to {model_dir}")
    
    # Test on sample images
    print("\n=== Sample Predictions ===")
    test_images = [
        ('../Dataset_AI-Real_images/real_dataset/nature/nature_0.jpg', 'Real'),
        ('../Dataset_AI-Real_images/Ai_generated_dataset/nature/image-10.png', 'AI'),
        ('../Dataset_Real-Fake_images/fake/easy_115_0010.jpg', 'Fake'),
    ]
    
    for path, true_label in test_images:
        if os.path.exists(path):
            from src.utils.dataset import load_single_image
            img = load_single_image(path, target_size=(256, 256))
            
            # Extract features
            model = FeatureExtractor()
            features = extract_features_batch([img], model, batch_size=1, device=device)
            features_scaled = scaler.transform(features)
            
            pred = best_model.predict(features_scaled)[0]
            proba = best_model.predict_proba(features_scaled)[0]
            
            pred_label = 'AI' if pred == 1 else 'Real'
            confidence = proba[1]
            
            filename = path.split('/')[-1]
            print(f"{filename} (true: {true_label}): Pred={pred_label} (conf: {confidence:.1%})")


if __name__ == '__main__':
    main()