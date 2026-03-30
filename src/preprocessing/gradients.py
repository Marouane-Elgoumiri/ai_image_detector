import numpy as np
import cv2
from typing import Tuple
from scipy import ndimage


def compute_gradients(
    image: np.ndarray,
    operator: str = 'sobel'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel or Scharr operators.
    
    Gradient computation is fundamental for detecting AI-generated images because:
    - Diffusion models produce overly smooth gradients
    - AI images often have periodic/structured gradient patterns
    - Real photos have natural gradient distributions
    
    Args:
        image: Grayscale or luminance image (H, W)
        operator: 'sobel' or 'scharr' for gradient computation
    
    Returns:
        Tuple of (gradient_x, gradient_y) as float arrays
    """
    if image.dtype != np.float64:
        image = image.astype(np.float64)
    
    if operator == 'sobel':
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float64)
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]], dtype=np.float64)
    elif operator == 'scharr':
        kernel_x = np.array([[-3, 0, 3],
                             [-10, 0, 10],
                             [-3, 0, 3]], dtype=np.float64)
        kernel_y = np.array([[-3, -10, -3],
                             [0, 0, 0],
                             [3, 10, 3]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown operator: {operator}. Use 'sobel' or 'scharr'")
    
    gradient_x = ndimage.convolve(image, kernel_x)
    gradient_y = ndimage.convolve(image, kernel_y)
    
    return gradient_x, gradient_y


def gradient_magnitude(gradient_x: np.ndarray, gradient_y: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude from x and y components.
    
    Magnitude = sqrt(gradient_x² + gradient_y²)
    
    Args:
        gradient_x: Horizontal gradient component
        gradient_y: Vertical gradient component
    
    Returns:
        Gradient magnitude array
    """
    return np.sqrt(gradient_x**2 + gradient_y**2)


def gradient_direction(gradient_x: np.ndarray, gradient_y: np.ndarray) -> np.ndarray:
    """
    Compute gradient direction (angle) from x and y components.
    
    Direction = arctan2(gradient_y, gradient_x)
    Values are in radians [-π, π]
    
    Args:
        gradient_x: Horizontal gradient component
        gradient_y: Vertical gradient component
    
    Returns:
        Gradient direction array in radians
    """
    return np.arctan2(gradient_y, gradient_x)


def compute_gradient_histogram(
    magnitude: np.ndarray,
    direction: np.ndarray,
    bins: int = 16,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute histogram of gradient directions weighted by magnitude.
    
    This histogram captures the distribution of edge orientations,
    which differs significantly between real photos and AI-generated images.
    
    Args:
        magnitude: Gradient magnitude array
        direction: Gradient direction array (in radians)
        bins: Number of histogram bins
        normalize: Whether to normalize histogram to sum to 1
    
    Returns:
        Gradient orientation histogram
    """
    flat_mag = magnitude.flatten()
    flat_dir = direction.flatten()
    
    # Manual histogram to avoid NumPy bugs
    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
    indices = np.digitize(flat_dir, bin_edges[1:-1])
    hist = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        hist[i] = np.sum(flat_mag[indices == i]) if np.any(indices == i) else 0.0
    
    if normalize:
        total = hist.sum()
        if total > 0:
            hist = hist / total
    
    return hist


def compute_gradient_stats(gradient_x: np.ndarray, gradient_y: np.ndarray) -> dict:
    """
    Compute statistical measures from gradient fields.
    
    Returns statistics that capture the texture characteristics:
    - Mean/Std of gradient magnitudes
    - Kurtosis of gradient distribution
    - Entropy of gradient histogram
    
    Args:
        gradient_x: Horizontal gradient component
        gradient_y: Vertical gradient component
    
    Returns:
        Dictionary of gradient statistics
    """
    magnitude = gradient_magnitude(gradient_x, gradient_y)
    direction = gradient_direction(gradient_x, gradient_y)
    
    flat_mag = magnitude.flatten()
    
    mean_mag = np.mean(flat_mag)
    std_mag = np.std(flat_mag)
    
    from scipy.stats import kurtosis
    
    kurt = kurtosis(flat_mag, fisher=True)
    
    skew = 0.0
    try:
        from scipy.stats import skew
        skew = skew(flat_mag)
    except:
        pass
    
    # Manual histogram to avoid NumPy bugs
    mag_min, mag_max = np.min(flat_mag), np.max(flat_mag)
    if mag_max - mag_min > 1e-10:
        bin_edges = np.linspace(mag_min, mag_max, 257)
        indices = np.digitize(flat_mag, bin_edges[1:-1])
        hist = np.bincount(indices, minlength=256).astype(np.float64)
        hist_normalized = hist / hist.sum()
        hist_normalized = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized))
    else:
        entropy = 0.0
    
    return {
        'mean': mean_mag,
        'std': std_mag,
        'kurtosis': kurt,
        'skewness': skew,
        'entropy': entropy
    }