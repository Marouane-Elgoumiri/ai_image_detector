import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from pathlib import Path

from ..features.pca_features import extract_all_features, save_preprocessors, load_preprocessors


class AIImageClassifier:
    """
    SVM-based classifier for detecting AI-generated images.
    
    Uses features extracted from:
    - Luminance statistics
    - Gradient field analysis
    - Frequency domain (FFT)
    - Texture features (kurtosis, entropy, skewness)
    
    The classifier works well in this setting because:
    - SVM handles high-dimensional, low-sample scenarios well
    - RBF kernel captures non-linear decision boundaries
    - Feature space separation is based on fundamental image statistics
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        class_weight: Optional[str] = None,
        n_pca_components: Optional[int] = None,
        pca_variance_ratio: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialize the classifier.
        
        Args:
            kernel: SVM kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            class_weight: Class weights ('balanced' or None)
            n_pca_components: Number of PCA components (None for auto)
            pca_variance_ratio: Target variance ratio if n_pca_components is None
            random_state: Random seed for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.n_pca_components = n_pca_components
        self.pca_variance_ratio = pca_variance_ratio
        self.random_state = random_state
        
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.svm: Optional[SVC] = None
        self.is_fitted: bool = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> 'AIImageClassifier':
        """
        Fit the classifier on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=real, 1=AI-generated)
            verbose: Print training progress
        
        Returns:
            self
        """
        if verbose:
            print(f"Fitting classifier on {X.shape[0]} samples with {X.shape[1]} features...")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if self.n_pca_components is not None:
            self.pca = PCA(n_components=self.n_pca_components)
        else:
            self.pca = PCA(n_components=self.pca_variance_ratio, svd_solver='full')
        
        X_pca = self.pca.fit_transform(X_scaled)
        
        if verbose:
            print(f"PCA reduced dimensions: {X.shape[1]} -> {X_pca.shape[1]}")
            print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        self.svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=True,
            random_state=self.random_state
        )
        
        self.svm.fit(X_pca, y)
        self.is_fitted = True
        
        if verbose:
            print("Training complete.")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        return self.svm.predict(X_pca)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability estimates (n_samples, 2)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        return self.svm.predict_proba(X_pca)
    
    def predict_image(self, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict whether a single image is AI-generated.
        
        Args:
            image: RGB image array (H, W, 3)
        
        Returns:
            Tuple of (prediction, confidence, features)
            prediction: 0=real, 1=AI-generated
            confidence: Probability of AI-generated class
            features: Extracted feature vector
        """
        features = extract_all_features(image)
        features = features.reshape(1, -1)
        
        prediction = self.predict(features)[0]
        proba = self.predict_proba(features)[0]
        
        confidence = proba[1]
        
        return int(prediction), float(confidence), features.flatten()
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Perform cross-validation on training data.
        
        Args:
            X: Feature matrix
            y: Labels
            n_folds: Number of cross-validation folds
            verbose: Print results
        
        Returns:
            Dictionary of cross-validation metrics
        """
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        X_pca = self.pca.fit_transform(X_scaled) if self.pca else X_scaled
        
        scores = cross_val_score(self.svm, X_pca, y, cv=cv, scoring='accuracy')
        
        if verbose:
            print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'folds': n_folds
        }
    
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Directory path to save model files
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.svm, save_dir / 'svm_model.pkl')
        joblib.dump(self.scaler, save_dir / 'scaler.pkl')
        joblib.dump(self.pca, save_dir / 'pca.pkl')
        
        config = {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'class_weight': self.class_weight,
            'n_pca_components': self.n_pca_components,
            'pca_variance_ratio': self.pca_variance_ratio,
            'random_state': self.random_state
        }
        joblib.dump(config, save_dir / 'config.pkl')
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AIImageClassifier':
        """
        Load a trained model from disk.
        
        Args:
            path: Directory path containing model files
        
        Returns:
            Loaded AIImageClassifier instance
        """
        load_dir = Path(path)
        
        config = joblib.load(load_dir / 'config.pkl')
        
        classifier = cls(
            kernel=config['kernel'],
            C=config['C'],
            gamma=config['gamma'],
            class_weight=config.get('class_weight', None),
            n_pca_components=config['n_pca_components'],
            pca_variance_ratio=config['pca_variance_ratio'],
            random_state=config['random_state']
        )
        
        classifier.scaler = joblib.load(load_dir / 'scaler.pkl')
        classifier.pca = joblib.load(load_dir / 'pca.pkl')
        classifier.svm = joblib.load(load_dir / 'svm_model.pkl')
        classifier.is_fitted = True
        
        print(f"Model loaded from {path}")
        return classifier
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from PCA components.
        
        For SVM with RBF kernel, we use the PCA explained variance
        to show which transformed features are most important.
        
        Returns:
            Array of feature importances (in PCA space)
        """
        if not self.is_fitted:
            return None
        
        return self.pca.explained_variance_ratio_
    
    def get_support_vectors_count(self) -> Optional[int]:
        """
        Get number of support vectors.
        
        Returns:
            Number of support vectors
        """
        if not self.is_fitted:
            return None
        return len(self.svm.support_vectors_)


class CIFAKEClassifier:
    """
    Classifier wrapper for models trained on CIFAKE dataset.
    
    Supports loading SVM, Random Forest, and Gradient Boosting models
    that were trained with train_cifake.py.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.is_fitted = False
        self.model_type = None
    
    @classmethod
    def load(cls, model_dir: str) -> 'CIFAKEClassifier':
        """
        Load a trained model from the specified directory.
        
        Args:
            model_dir: Directory containing model files (e.g., models/gradient_boosting/)
        
        Returns:
            Loaded CIFAKEClassifier instance
        """
        model_path = Path(model_dir)
        
        classifier = cls()
        
        # Try loading different model types
        if (model_path / 'model.pkl').exists():
            classifier.model = joblib.load(model_path / 'model.pkl')
            classifier.scaler = joblib.load(model_path / 'scaler.pkl')
            
            # Determine model type by class name
            model_class_name = type(classifier.model).__name__
            
            if model_class_name == 'SVC':
                classifier.model_type = 'svm'
                # SVM uses PCA
                if (model_path / 'pca.pkl').exists():
                    classifier.pca = joblib.load(model_path / 'pca.pkl')
            elif model_class_name == 'GradientBoostingClassifier':
                classifier.model_type = 'gradient_boosting'
            elif model_class_name == 'RandomForestClassifier':
                classifier.model_type = 'random_forest'
            else:
                classifier.model_type = 'unknown'
            
            classifier.is_fitted = True
            print(f"Loaded {classifier.model_type} model from {model_dir}")
        else:
            raise FileNotFoundError(f"No model found in {model_dir}")
        
        return classifier
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return self.model.predict_proba(X_scaled)
    
    def predict_image(self, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict whether a single image is AI-generated.
        
        Args:
            image: RGB image array (H, W, 3)
        
        Returns:
            Tuple of (prediction, confidence, features)
            prediction: 0=fake/AI, 1=real
            confidence: Probability of being AI-generated
            features: Extracted feature vector
        """
        # Import the feature extraction function from train_cifake
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from train_cifake import extract_image_features
        
        features = extract_image_features(image)
        features = features.reshape(1, -1)
        
        prediction = self.predict(features)[0]
        proba = self.predict_proba(features)[0]
        
        # Note: In CIFAKE, 0=FAKE, 1=REAL
        # So confidence of being AI is proba[0]
        confidence = proba[0]
        
        return int(prediction), float(confidence), features.flatten()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances if available."""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif self.pca is not None:
            return self.pca.explained_variance_ratio_
        else:
            return None


class DeepLearningClassifier:
    """
    End-to-end deep learning classifier for AI image detection.

    Loads a PyTorch checkpoint (.pt) from train_deep_cnn.py
    and provides the same predict_image() interface as other classifiers.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.is_fitted = False
        self.model_name = None
        self.image_size = 224

    @staticmethod
    def _detect_architecture(state_dict: dict) -> Tuple[str, str]:
        """Detect model architecture and variant from checkpoint state dict keys."""
        keys = list(state_dict.keys())

        # Get first conv channels and classifier feature dimension
        first_conv = state_dict.get('backbone.features.0.0.weight')
        first_channels = first_conv.shape[0] if first_conv is not None else 24

        # Find 2D classifier weight (Linear layer, not LayerNorm)
        feature_dim = 1280
        for k in keys:
            if 'classifier' in k and 'weight' in k:
                w = state_dict[k]
                if w.dim() == 2:
                    feature_dim = w.shape[1]
                    break

        # Check for ConvNeXt (has layer_scale keys, unique to ConvNeXt)
        if any('layer_scale' in k for k in keys):
            return 'convnext', 'tiny'

        # Classify by feature dimension (most reliable)
        if feature_dim >= 1280:
            return 'efficientnet', 'small'
        elif feature_dim >= 960:
            return 'mobilenet', 'large'
        else:
            return 'mobilenet', 'small'

    @classmethod
    def load(cls, checkpoint_path: str, model_name: str = None,
             variant: str = None, image_size: int = 224) -> 'DeepLearningClassifier':
        """
        Load a trained deep learning model from a .pt checkpoint.

        Args:
            checkpoint_path: Path to best_model.pt
            model_name: Architecture name ('efficientnet', 'mobilenet', 'convnext'). Auto-detected if None.
            variant: Model variant ('small', 'medium', 'large', etc.). Auto-detected if None.
            image_size: Input image size (must match training)
        """
        import torch
        from torchvision import transforms

        from src.deep.models import EfficientNetDetector, MobileNetDetector, ConvNeXtDetector

        classifier = cls()
        classifier.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier.image_size = image_size

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=classifier.device, weights_only=False)
        state_dict = ckpt['model_state_dict']

        # Auto-detect architecture if not specified
        if model_name is None or variant is None:
            detected_name, detected_variant = cls._detect_architecture(state_dict)
            model_name = model_name or detected_name
            variant = variant or detected_variant
            print(f"Auto-detected: {model_name} ({variant})")

        classifier.model_name = model_name

        # Build model architecture
        if model_name == 'efficientnet':
            classifier.model = EfficientNetDetector(
                variant=variant, num_classes=2, pretrained=False, dropout=0.2
            )
        elif model_name == 'mobilenet':
            classifier.model = MobileNetDetector(
                variant=variant, num_classes=2, pretrained=False, dropout=0.2
            )
        elif model_name == 'convnext':
            classifier.model = ConvNeXtDetector(
                variant=variant, num_classes=2, pretrained=False, dropout=0.2
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load weights
        classifier.model.load_state_dict(state_dict)
        classifier.model.to(classifier.device)
        classifier.model.eval()

        # Preprocessing transform (same as validation during training)
        classifier.transform = transforms.Compose([
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        classifier.is_fitted = True
        print(f"Loaded {model_name} ({variant}) from {checkpoint_path}")
        return classifier

    def predict_image(self, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict whether a single image is AI-generated.

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            Tuple of (prediction, confidence, features)
            prediction: 0=real, 1=AI-generated
            confidence: Probability of being AI-generated
            features: Empty array (no handcrafted features for deep model)
        """
        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)

        probs = probs.cpu().numpy()[0]
        prediction = int(np.argmax(probs))
        confidence = float(probs[1])  # P(AI-generated)

        return prediction, confidence, np.array([])


class CNNClassifier:
    """
    Classifier that uses pre-trained CNN (ResNet50) features.
    
    This gives much better accuracy than hand-crafted features.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_extractor = None
        self.is_fitted = False
        self.device = None
    
    @classmethod
    def load(cls, model_dir: str) -> 'CNNClassifier':
        """
        Load a trained CNN classifier.
        
        Args:
            model_dir: Directory containing model.pkl and scaler.pkl
        """
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        
        model_path = Path(model_dir)
        
        classifier = cls()
        
        if (model_path / 'model.pkl').exists():
            classifier.model = joblib.load(model_path / 'model.pkl')
            classifier.scaler = joblib.load(model_path / 'scaler.pkl')
            
            # Load ResNet feature extractor
            classifier.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            classifier.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            classifier.feature_extractor.to(classifier.device)
            classifier.feature_extractor.eval()
            
            classifier.is_fitted = True
            print(f"CNN classifier loaded from {model_dir}")
        else:
            raise FileNotFoundError(f"No model found in {model_dir}")
        
        return classifier
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract ResNet features from a single image."""
        import torch
        from torchvision import transforms
        from PIL import Image
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            image = Image.fromarray(image.astype(np.uint8))
        
        tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(tensor)
        
        return features.cpu().numpy().flatten().reshape(1, -1)
    
    def predict_image(self, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict whether an image is AI-generated.
        
        Args:
            image: RGB image array (H, W, 3)
        
        Returns:
            Tuple of (prediction, confidence, features)
            prediction: 0=real, 1=AI-generated
            confidence: Probability of being AI-generated
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be loaded first")
        
        # Extract CNN features
        features = self.extract_features(image)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        
        # Return AI confidence
        confidence = proba[1]  # Probability of class 1 (AI)
        
        return int(prediction), float(confidence), features.flatten()