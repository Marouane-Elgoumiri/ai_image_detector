import numpy as np
import cv2
from typing import Tuple


def rgb_to_ycbcr(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB image to YCbCr color space.
    
    The Y (luminance) channel isolates brightness information from color,
    which is essential for detecting structural artifacts in AI-generated images.
    
    Args:
        image: RGB image as numpy array (H, W, 3) with values in [0, 255]
    
    Returns:
        Tuple of (Y, Cb, Cr) channels as separate arrays
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    y_channel = ycbcr[:, :, 0].astype(np.float64)
    cb_channel = ycbcr[:, :, 1].astype(np.float64)
    cr_channel = ycbcr[:, :, 2].astype(np.float64)
    
    return y_channel, cb_channel, cr_channel


def extract_luminance(image: np.ndarray) -> np.ndarray:
    """
    Extract the luminance (Y) channel from an RGB image.
    
    Working in luminance separates structural information from color noise,
    making gradient and frequency analysis more effective for detecting
    AI-generated artifacts.
    
    Args:
        image: RGB image as numpy array (H, W, 3) with values in [0, 255] or [0, 1]
    
    Returns:
        Luminance channel as float array
    """
    y_channel, _, _ = rgb_to_ycbcr(image)
    return y_channel


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminance-preserving formula.
    
    Uses standard OpenCV conversion which applies:
    Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: RGB image as numpy array (H, W, 3)
    
    Returns:
        Grayscale image as float array
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float64)


def normalize_luminance(luminance: np.ndarray) -> np.ndarray:
    """
    Normalize luminance values to [0, 1] range.
    
    Args:
        luminance: Luminance channel array
    
    Returns:
        Normalized luminance array
    """
    min_val = luminance.min()
    max_val = luminance.max()
    
    if max_val - min_val < 1e-10:
        return np.zeros_like(luminance)
    
    return (luminance - min_val) / (max_val - min_val)