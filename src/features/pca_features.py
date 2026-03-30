import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from ..preprocessing.luminance import extract_luminance
from ..preprocessing.gradients import compute_gradients, gradient_magnitude, gradient_direction
from .frequency import compute_fft, extract_frequency_features
from .texture import extract_texture_features, extract_gradient_histogram_features


def extract_all_features(image: np.ndarray) -> np.ndarray:
    """
    Extract complete feature vector from a single image.
    
    Combines all features:
    1. Luminance statistics
    2. Gradient statistics
    3. Frequency domain features
    4. Texture features
    
    Args:
        image: RGB image as numpy array (H, W, 3)
    
    Returns:
        1D feature array
    """
    luminance = extract_luminance(image)
    
    gradient_x, gradient_y = compute_gradients(luminance, operator='sobel')
    
    _, power_spectrum = compute_fft(luminance)
    
    texture_features = extract_texture_features(gradient_x, gradient_y)
    
    freq_features = extract_frequency_features(power_spectrum, luminance.shape)
    
    hist_features = extract_gradient_histogram_features(gradient_x, gradient_y)
    
    lum_mean = np.mean(luminance)
    lum_std = np.std(luminance)
    lum_kurtosis = float(_kurtosis(luminance.flatten()))
    lum_skewness = float(_skewness(luminance.flatten()))
    lum_entropy = _compute_entropy_1d(luminance.flatten(), bins=64)
    
    features = {}
    
    features['lum_mean'] = lum_mean
    features['lum_std'] = lum_std
    features['lum_kurtosis'] = lum_kurtosis
    features['lum_skewness'] = lum_skewness
    features['lum_entropy'] = lum_entropy
    
    features.update(texture_features)
    
    features.update(freq_features)
    
    features.update(hist_features)
    
    feature_vector = np.array(list(features.values()))
    
    if not np.isfinite(feature_vector).all():
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vector


def build_feature_matrix(
    images: List[np.ndarray],
    labels: List[int],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix from list of images.
    
    Args:
        images: List of RGB images
        labels: List of labels (0=real, 1=AI)
        verbose: Print progress
    
    Returns:
        Tuple of (feature_matrix, label_array)
        feature_matrix: (n_samples, n_features)
        label_array: (n_samples,)
    """
    n_samples = len(images)
    
    if verbose:
        print(f"Extracting features from {n_samples} images...")
    
    feature_list = []
    
    for i, img in enumerate(images):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples} images")
        
        features = extract_all_features(img)
        feature_list.append(features)
    
    feature_matrix = np.array(feature_list)
    label_array = np.array(labels)
    
    if verbose:
        print(f"Feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix, label_array


def apply_pca(
    X: np.ndarray,
    n_components: Optional[int] = None,
    variance_ratio: float = 0.95,
    scaler: Optional[StandardScaler] = None,
    pca: Optional[PCA] = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler, PCA]:
    """
    Apply PCA for dimensionality reduction.
    
    PCA helps separate real and AI images in a reduced feature space
    by finding directions of maximum variance.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_components: Number of PCA components (None for auto)
        variance_ratio: Target variance ratio if n_components is None
        scaler: Pre-fitted scaler (if not fitting)
        pca: Pre-fitted PCA (if not fitting)
        fit: Whether to fit new scaler and PCA
    
    Returns:
        Tuple of (X_transformed, scaler, pca)
    """
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if n_components is None:
            pca = PCA(n_components=variance_ratio, svd_solver='full')
        else:
            pca = PCA(n_components=n_components)
        
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        if scaler is None or pca is None:
            raise ValueError("scaler and pca must be provided when fit=False")
        
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
    
    return X_pca, scaler, pca


def get_feature_names() -> List[str]:
    """
    Get names of all features extracted.
    
    Returns:
        List of feature names in order
    """
    names = [
        'lum_mean',
        'lum_std',
        'lum_kurtosis',
        'lum_skewness',
        'lum_entropy',
        
        'kurtosis_mag',
        'kurtosis_dir',
        'entropy_mag',
        'entropy_dir',
        'skewness_mag',
        'skewness_dir',
        'mean_mag',
        'std_mag',
        'var_mag',
        'median_mag',
        'mad_mag',
        'uniform_kl',
        'moment_2_mag',
        'moment_3_mag',
        'moment_4_mag',
        'energy_x',
        'energy_y',
        'energy_ratio',
        'cross_energy',
        'anisotropy',
        'homogeneity',
        
        'low_freq_energy',
        'mid_freq_energy',
        'high_freq_energy',
        'low_high_ratio',
        'radial_entropy',
        'spectrum_mean',
        'spectrum_std',
        'spectrum_max',
        'spectral_flatness',
        'total_energy_normalized',
        
        'hist_max_bin',
        'hist_max_value',
        'hist_entropy',
        'hist_smoothness',
        'dominant_orientation'
    ]
    return names


def save_preprocessors(
    scaler: StandardScaler,
    pca: PCA,
    save_path: str
) -> None:
    """
    Save fitted scaler and PCA to disk.
    
    Args:
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        save_path: Directory path to save
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, save_dir / 'scaler.pkl')
    joblib.dump(pca, save_dir / 'pca.pkl')


def load_preprocessors(
    load_path: str
) -> Tuple[StandardScaler, PCA]:
    """
    Load fitted scaler and PCA from disk.
    
    Args:
        load_path: Directory path containing saved models
    
    Returns:
        Tuple of (scaler, pca)
    """
    load_dir = Path(load_path)
    
    scaler = joblib.load(load_dir / 'scaler.pkl')
    pca = joblib.load(load_dir / 'pca.pkl')
    
    return scaler, pca


def _kurtosis(data: np.ndarray) -> float:
    from scipy.stats import kurtosis
    return kurtosis(data, fisher=True)


def _skewness(data: np.ndarray) -> float:
    from scipy.stats import skew
    return skew(data)


def _compute_entropy_1d(data: np.ndarray, bins: int = 64) -> float:
    data_min, data_max = np.min(data), np.max(data)
    if data_max - data_min < 1e-10:
        return 0.0
    # Manual histogram to avoid NumPy bugs
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    indices = np.digitize(data, bin_edges[1:-1])
    hist = np.bincount(indices, minlength=bins).astype(np.float64)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    hist_normalized = hist / hist.sum()
    return -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))