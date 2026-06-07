import numpy as np
from typing import Tuple, Dict
from scipy import fft


def compute_fft(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D FFT of an image and return the power spectrum.
    
    FFT analysis reveals high-frequency artifacts in AI-generated images:
    - Diffusion models produce periodic grid patterns in frequency domain
    - Real photos have natural 1/f frequency falloff
    - Spectral peaks indicate AI upscaling or generation artifacts
    
    Args:
        image: Grayscale or luminance image (H, W)
    
    Returns:
        Tuple of (fft_shifted, power_spectrum)
        fft_shifted: Shifted FFT coefficients (low frequencies in center)
        power_spectrum: Power spectrum (magnitude squared)
    """
    if image.dtype != np.float64:
        image = image.astype(np.float64)
    
    fft_result = fft.fft2(image)
    fft_shifted = fft.fftshift(fft_result)
    power_spectrum = np.abs(fft_shifted) ** 2
    
    return fft_shifted, power_spectrum


def extract_frequency_features(
    power_spectrum: np.ndarray,
    image_shape: Tuple[int, int]
) -> Dict[str, float]:
    """
    Extract discriminative frequency domain features.
    
    These features capture AI-specific frequency characteristics:
    - Energy distribution across frequency bands
    - Spectral peaks (indicating periodic artifacts)
    - High-frequency energy ratios
    - Radial frequency statistics
    
    Args:
        power_spectrum: Power spectrum from compute_fft
        image_shape: Original image shape (H, W) for normalization
    
    Returns:
        Dictionary of frequency features
    """
    h, w = image_shape
    center_h, center_w = h // 2, w // 2
    
    log_power = np.log1p(power_spectrum)
    
    total_energy = np.sum(power_spectrum)
    
    y_coords, x_coords = np.ogrid[:h, :w]
    radii = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
    max_radius = min(center_h, center_w)
    
    low_freq_mask = radii < max_radius * 0.2
    mid_freq_mask = (radii >= max_radius * 0.2) & (radii < max_radius * 0.5)
    high_freq_mask = radii >= max_radius * 0.5
    
    low_energy = np.sum(power_spectrum[low_freq_mask])
    mid_energy = np.sum(power_spectrum[mid_freq_mask])
    high_energy = np.sum(power_spectrum[high_freq_mask])
    
    low_ratio = low_energy / total_energy if total_energy > 0 else 0
    mid_ratio = mid_energy / total_energy if total_energy > 0 else 0
    high_ratio = high_energy / total_energy if total_energy > 0 else 0
    
    num_radial_bins = 20
    radial_profile = np.zeros(num_radial_bins)
    for i in range(num_radial_bins):
        r_inner = i * max_radius / num_radial_bins
        r_outer = (i + 1) * max_radius / num_radial_bins
        mask = (radii >= r_inner) & (radii < r_outer)
        if np.sum(mask) > 0:
            radial_profile[i] = np.mean(power_spectrum[mask])
    
    radial_entropy = 0.0
    radial_normalized = radial_profile / radial_profile.sum() if radial_profile.sum() > 0 else radial_profile
    for val in radial_normalized:
        if val > 0:
            radial_entropy -= val * np.log2(val)
    
    mean_spectrum = np.mean(log_power)
    std_spectrum = np.std(log_power)
    max_spectrum = np.max(log_power)
    
    spectral_flatness = np.exp(np.mean(np.log(log_power + 1e-10))) / (np.mean(log_power) + 1e-10)
    
    features = {
        'low_freq_energy': low_ratio,
        'mid_freq_energy': mid_ratio,
        'high_freq_energy': high_ratio,
        'low_high_ratio': low_ratio / (high_ratio + 1e-10),
        'radial_entropy': radial_entropy,
        'spectrum_mean': mean_spectrum,
        'spectrum_std': std_spectrum,
        'spectrum_max': max_spectrum,
        'spectral_flatness': spectral_flatness,
        'total_energy_normalized': total_energy / (h * w)
    }
    
    return features


def detect_spectral_peaks(
    power_spectrum: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, int]:
    """
    Detect periodic spectral peaks that indicate AI generation artifacts.
    
    Diffusion models often produce grid-like patterns in the frequency domain
    due to their denoising process operating in discrete steps.
    
    Args:
        power_spectrum: Power spectrum from compute_fft
        threshold: Threshold for peak detection (fraction of max)
    
    Returns:
        Dictionary with count of detected peaks and their characteristics
    """
    log_power = np.log1p(power_spectrum)
    
    normalized = (log_power - log_power.min()) / (log_power.max() - log_power.min())
    
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(normalized, size=5)
    peaks = (normalized == local_max) & (normalized > threshold)
    
    num_peaks = np.sum(peaks)
    
    center_h, center_w = power_spectrum.shape[0] // 2, power_spectrum.shape[1] // 2
    center_mask = np.zeros_like(peaks)
    center_mask[max(0, center_h-10):center_h+10, max(0, center_w-10):center_w+10] = True
    
    peaks[center_mask] = False
    peripheral_peaks = np.sum(peaks)
    
    return {
        'total_peaks': int(num_peaks),
        'peripheral_peaks': int(peripheral_peaks)
    }


def get_radial_power_profile(
    power_spectrum: np.ndarray,
    num_bins: int = 50
) -> np.ndarray:
    """
    Compute radial power profile for frequency analysis.
    
    Args:
        power_spectrum: Power spectrum from compute_fft
        num_bins: Number of radial bins
    
    Returns:
        1D array of radial power values
    """
    h, w = power_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    y_coords, x_coords = np.ogrid[:h, :w]
    radii = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
    max_radius = min(center_h, center_w)
    
    radial_profile = np.zeros(num_bins)
    for i in range(num_bins):
        r_inner = i * max_radius / num_bins
        r_outer = (i + 1) * max_radius / num_bins
        mask = (radii >= r_inner) & (radii < r_outer)
        if np.sum(mask) > 0:
            radial_profile[i] = np.mean(power_spectrum[mask])
    
    return radial_profile