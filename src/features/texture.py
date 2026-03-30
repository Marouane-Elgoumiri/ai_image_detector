import numpy as np
from scipy import stats
from typing import Dict
from ..preprocessing.gradients import (
    gradient_magnitude,
    gradient_direction,
    compute_gradient_histogram
)


def extract_texture_features(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    num_bins: int = 32
) -> Dict[str, float]:
    """
    Extract statistical texture features from gradient fields.
    
    Real photos have specific statistical signatures in gradient histograms
    that diffusion models often deviate from:
    - Kurtosis: measures "peakedness" vs "flatness" of distribution
    - Entropy: measures randomness/uniformity
    - Skewness: measures asymmetry of distribution
    
    Args:
        gradient_x: Horizontal gradient component
        gradient_y: Vertical gradient component
        num_bins: Number of bins for gradient histogram
    
    Returns:
        Dictionary of texture features
    """
    magnitude = gradient_magnitude(gradient_x, gradient_y)
    direction = gradient_direction(gradient_x, gradient_y)
    
    mag_flat = magnitude.flatten()
    dir_flat = direction.flatten()
    
    kurtosis_mag = stats.kurtosis(mag_flat, fisher=True)
    kurtosis_dir = stats.kurtosis(dir_flat, fisher=True)
    
    entropy_mag = _compute_entropy(mag_flat, bins=num_bins)
    entropy_dir = _compute_entropy(dir_flat, bins=num_bins)
    
    skewness_mag = stats.skew(mag_flat)
    skewness_dir = stats.skew(dir_flat)
    
    mean_mag = np.mean(mag_flat)
    std_mag = np.std(mag_flat)
    var_mag = np.var(mag_flat)
    
    # Manual histogram to avoid NumPy bugs
    bin_edges = np.linspace(np.min(mag_flat), np.max(mag_flat), num_bins + 1)
    indices = np.digitize(mag_flat, bin_edges[1:-1])
    mag_hist = np.bincount(indices, minlength=num_bins).astype(np.float64)
    mag_hist = mag_hist / mag_hist.sum()
    uniform_kl = _kl_divergence_uniform(mag_hist)
    
    median_mag = np.median(mag_flat)
    mad_mag = np.median(np.abs(mag_flat - median_mag))
    
    moments = {
        'moment_2_mag': np.mean((mag_flat - mean_mag) ** 2),
        'moment_3_mag': np.mean((mag_flat - mean_mag) ** 3),
        'moment_4_mag': np.mean((mag_flat - mean_mag) ** 4),
    }
    
    features = {
        'kurtosis_mag': kurtosis_mag,
        'kurtosis_dir': kurtosis_dir,
        'entropy_mag': entropy_mag,
        'entropy_dir': entropy_dir,
        'skewness_mag': skewness_mag,
        'skewness_dir': skewness_dir,
        'mean_mag': mean_mag,
        'std_mag': std_mag,
        'var_mag': var_mag,
        'median_mag': median_mag,
        'mad_mag': mad_mag,
        'uniform_kl': uniform_kl,
        'moment_2_mag': moments['moment_2_mag'],
        'moment_3_mag': moments['moment_3_mag'],
        'moment_4_mag': moments['moment_4_mag'],
    }
    
    energy_features = _compute_energy_features(gradient_x, gradient_y)
    features.update(energy_features)
    
    homogeneity = _compute_homogeneity(magnitude)
    features['homogeneity'] = homogeneity
    
    return features


def _compute_entropy(data: np.ndarray, bins: int = 32) -> float:
    """
    Compute Shannon entropy of the data distribution.
    
    Args:
        data: Input data array
        bins: Number of histogram bins
    
    Returns:
        Entropy value in bits
    """
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
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    
    return entropy


def _kl_divergence_uniform(hist: np.ndarray) -> float:
    """
    Compute KL divergence from uniform distribution.
    
    Higher values indicate more deviation from uniform (more structure).
    
    Args:
        hist: Histogram values
    
    Returns:
        KL divergence
    """
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    
    hist_normalized = hist / hist.sum()
    uniform = np.ones_like(hist_normalized) / len(hist_normalized)
    
    kl = np.sum(hist_normalized * np.log2((hist_normalized + 1e-10) / (uniform + 1e-10)))
    
    return kl


def _compute_energy_features(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray
) -> Dict[str, float]:
    """
    Compute energy-based features from gradient fields.
    
    Args:
        gradient_x: Horizontal gradient
        gradient_y: Vertical gradient
    
    Returns:
        Dictionary of energy features
    """
    energy_x = np.sum(gradient_x ** 2)
    energy_y = np.sum(gradient_y ** 2)
    total_energy = energy_x + energy_y
    
    energy_ratio = energy_x / (energy_y + 1e-10)
    
    cross_energy = np.sum(gradient_x * gradient_y)
    
    anisotropy = np.abs(energy_x - energy_y) / (total_energy + 1e-10)
    
    return {
        'energy_x': energy_x / gradient_x.size,
        'energy_y': energy_y / gradient_y.size,
        'energy_ratio': energy_ratio,
        'cross_energy': cross_energy / gradient_x.size,
        'anisotropy': anisotropy
    }


def _compute_homogeneity(magnitude: np.ndarray, block_size: int = 8) -> float:
    """
    Compute local homogeneity measure.
    
    AI-generated images often have more uniform local regions.
    
    Args:
        magnitude: Gradient magnitude
        block_size: Size of local blocks
    
    Returns:
        Homogeneity score
    """
    h, w = magnitude.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    if h_blocks == 0 or w_blocks == 0:
        return 0.0
    
    local_stds = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = magnitude[i*block_size:(i+1)*block_size, 
                            j*block_size:(j+1)*block_size]
            local_stds.append(np.std(block))
    
    local_stds = np.array(local_stds)
    
    homogeneity = 1.0 / (1.0 + np.mean(local_stds))
    
    return homogeneity


def extract_gradient_histogram_features(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    num_bins: int = 16
) -> Dict[str, float]:
    """
    Extract features from weighted gradient orientation histogram.
    
    The distribution of gradient orientations captures texture patterns
    that differ between real and AI images.
    
    Args:
        gradient_x: Horizontal gradient
        gradient_y: Vertical gradient
        num_bins: Number of histogram bins
    
    Returns:
        Dictionary of histogram-based features
    """
    magnitude = gradient_magnitude(gradient_x, gradient_y)
    direction = gradient_direction(gradient_x, gradient_y)
    
    weighted_hist = compute_gradient_histogram(magnitude, direction, bins=num_bins)
    
    max_bin = np.argmax(weighted_hist)
    max_bin_value = weighted_hist[max_bin]
    
    entropy = _compute_entropy(direction.flatten(), bins=num_bins)
    
    smoothness = np.sum(np.abs(np.diff(weighted_hist)))
    
    dominant_orientation = (max_bin / num_bins) * np.pi
    
    return {
        'hist_max_bin': max_bin,
        'hist_max_value': max_bin_value,
        'hist_entropy': entropy,
        'hist_smoothness': smoothness,
        'dominant_orientation': dominant_orientation
    }