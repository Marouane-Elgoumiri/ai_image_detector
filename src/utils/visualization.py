import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_gradient_field(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    magnitude: np.ndarray,
    title: str = "Gradient Field",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gradient_x, cmap='RdBu')
    axes[0].set_title('Gradient X (∂I/∂x)')
    axes[0].axis('off')
    
    axes[1].imshow(gradient_y, cmap='RdBu')
    axes[1].set_title('Gradient Y (∂I/∂y)')
    axes[1].axis('off')
    
    axes[2].imshow(magnitude, cmap='gray')
    axes[2].set_title('Gradient Magnitude')
    axes[2].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_fft_spectrum(
    power_spectrum: np.ndarray,
    title: str = "FFT Power Spectrum",
    log_scale: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    display_spectrum = power_spectrum.copy()
    if log_scale:
        display_spectrum = np.log1p(display_spectrum)
    
    ax.imshow(display_spectrum, cmap='inferno')
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pca_scatter(
    X_pca: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA: Real vs AI-Generated Images",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    real_mask = labels == 0
    ai_mask = labels == 1
    
    ax.scatter(X_pca[real_mask, 0], X_pca[real_mask, 1], 
               c='blue', alpha=0.6, label='Real', s=30)
    ax.scatter(X_pca[ai_mask, 0], X_pca[ai_mask, 1], 
               c='red', alpha=0.6, label='AI-Generated', s=30)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison_grid(
    original: np.ndarray,
    luminance: np.ndarray,
    gradient_mag: np.ndarray,
    fft_spectrum: np.ndarray,
    prediction: Optional[str] = None,
    confidence: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(luminance, cmap='gray')
    axes[0, 1].set_title('Luminance (Y Channel)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gradient_mag, cmap='gray')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')
    
    fft_display = np.log1p(fft_spectrum)
    axes[1, 1].imshow(fft_display, cmap='inferno')
    axes[1, 1].set_title('FFT Power Spectrum')
    axes[1, 1].axis('off')
    
    if prediction and confidence is not None:
        fig.suptitle(f'Prediction: {prediction} (Confidence: {confidence:.2%})', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig