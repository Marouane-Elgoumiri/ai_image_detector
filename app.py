#!/usr/bin/env python3
"""
Streamlit Web App for AI-Generated Image Detection.

This app provides a user-friendly interface to:
1. Upload images for analysis
2. Visualize feature extraction (luminance, gradients, FFT)
3. Get predictions with confidence scores
4. Understand why an image is classified as real or AI-generated
"""

import os
import sys
from pathlib import Path
import tempfile
from typing import Tuple, Optional

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dataset import load_single_image
from src.preprocessing.luminance import extract_luminance
from src.preprocessing.gradients import compute_gradients, gradient_magnitude, gradient_direction
from src.features.frequency import compute_fft, extract_frequency_features
from src.features.texture import extract_texture_features
from src.features.pca_features import extract_all_features
from src.models.classifier import AIImageClassifier, CIFAKEClassifier, CNNClassifier, DeepLearningClassifier
from src.utils.visualization import plot_comparison_grid


st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_default_model_path() -> str:
    return str(Path(__file__).parent / 'models' / 'deep')


@st.cache_resource
def load_model(model_path: str, model_type: str = 'deep_learning') -> Optional[object]:
    """
    Load the trained classifier model.

    Args:
        model_path: Path to the model directory or checkpoint
        model_type: Type of model to load ('deep_learning', 'cnn', 'svm', 'gradient_boosting', 'random_forest')

    Returns:
        Loaded classifier or None if not found
    """
    try:
        model_dir = Path(model_path)

        # Load deep learning model (best accuracy)
        if model_type == 'deep_learning':
            ckpt_path = model_dir / 'best_model.pt'
            if ckpt_path.exists():
                classifier = DeepLearningClassifier.load(
                    str(ckpt_path),
                    image_size=224,
                )
                return classifier

        # Try loading CNN model (ResNet features + sklearn)
        if model_type == 'cnn':
            cnn_dir = model_dir / 'cnn' / 'gb'
            if cnn_dir.exists() and (cnn_dir / 'model.pkl').exists():
                classifier = CNNClassifier.load(str(cnn_dir))
                return classifier

        # Try loading CIFAKE sklearn models
        if model_type in ['gradient_boosting', 'random_forest']:
            gb_dir = model_dir / model_type
            if gb_dir.exists() and (gb_dir / 'model.pkl').exists():
                classifier = CIFAKEClassifier.load(str(gb_dir))
                return classifier

        # Fall back to old SVM model
        if (model_dir / 'svm_model.pkl').exists():
            classifier = AIImageClassifier.load(str(model_dir))
            return classifier

        # Try loading from subdirectory
        if model_type == 'svm':
            svm_dir = model_dir / 'svm'
            if svm_dir.exists() and (svm_dir / 'model.pkl').exists():
                classifier = CIFAKEClassifier.load(str(svm_dir))
                return classifier

        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def process_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Process an uploaded image file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        RGB image as numpy array (resized to 256x256)
    """
    img = Image.open(uploaded_file)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    
    target_size = (256, 256)
    from PIL import Image as PILImage
    img_resized = PILImage.fromarray(img_array).resize(target_size, PILImage.Resampling.LANCZOS)
    
    return np.array(img_resized)


def create_visualization(
    image: np.ndarray,
    luminance: np.ndarray,
    gradient_mag: np.ndarray,
    fft_spectrum: np.ndarray,
    prediction: int,
    confidence: float
) -> plt.Figure:
    """
    Create comparison visualization for the uploaded image.
    
    Args:
        image: Original RGB image
        luminance: Luminance channel
        gradient_mag: Gradient magnitude
        fft_spectrum: Power spectrum
        prediction: 0 for real, 1 for AI
        confidence: Prediction confidence
    
    Returns:
        Matplotlib figure
    """
    pred_text = "AI-Generated" if prediction == 1 else "Real"
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(luminance, cmap='gray')
    axes[0, 1].set_title('Luminance (Y Channel)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gradient_mag, cmap='gray')
    axes[1, 0].set_title('Gradient Magnitude', fontsize=12)
    axes[1, 0].axis('off')
    
    fft_display = np.log1p(fft_spectrum)
    axes[1, 1].imshow(fft_display, cmap='inferno')
    axes[1, 1].set_title('FFT Power Spectrum', fontsize=12)
    axes[1, 1].axis('off')
    
    fig.suptitle(f'Prediction: {pred_text} (Confidence: {confidence:.1%})',
                fontsize=14, fontweight='bold', 
                color='red' if prediction == 1 else 'green')
    
    plt.tight_layout()
    return fig


def main():
    st.title("🔍 AI-Generated Image Detector")
    st.markdown("""
    This tool detects images generated by AI models (Stable Diffusion, DALL-E, etc.)
    using frequency domain analysis and statistical texture features.
    
    **Upload an image to analyze whether it's real or AI-generated.**
    """)
    
    sidebar_info = """
    ### About This Tool

    **Primary Model:**
    Deep learning (EfficientNet-V2-S) trained on CIFAKE dataset
    (~97% accuracy on 100K images, end-to-end CNN)

    **Legacy Models (sklearn):**
    1. **Luminance Analysis**: Extracts Y channel from YCbCr
    2. **Gradient Field**: Computes Sobel gradients to detect edge patterns
    3. **FFT Analysis**: Identifies spectral artifacts from AI generation
    4. **Texture Features**: Analyzes kurtosis, entropy, and skewness
    5. **Ensemble Classification**: Gradient Boosting combines features for final prediction
    """
    
    with st.sidebar:
        st.markdown(sidebar_info)
        
        model_path_input = st.text_input(
            "Model Path",
            value=get_default_model_path(),
            help="Path to trained model directory"
        )
        
        model_type = st.selectbox(
            "Model Type",
            options=['deep_learning', 'cnn', 'gradient_boosting', 'random_forest', 'svm'],
            index=0,
            help="Deep learning uses EfficientNet-V2-S (best accuracy). Others use sklearn."
        )
        
        if st.button("Reload Model"):
            st.cache_resource.clear()
            st.rerun()
    
    classifier = load_model(model_path_input, model_type)
    
    if classifier is None:
        st.warning("""
        ⚠️ No trained model found. Please train the model first:

        ```
        cd ai_image_detector
        python train_deep_cnn.py --model efficientnet --batch-size 16 --epochs 30
        ```

        Or specify a different model path in the sidebar.
        """)
        
        st.info("""
        **Alternatively, you can still explore the feature visualizations
        by uploading an image below. The prediction will not be available
        until a model is trained.**
        """)
    
    uploaded_file = st.file_uploader(
        "Upload an image to analyze",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported formats: JPG, PNG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        try:
            image = process_uploaded_image(uploaded_file)
            
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded image (resized to 256x256)", width=300)
            
            with st.spinner("Analyzing image features..."):
                luminance = extract_luminance(image)
                
                gradient_x, gradient_y = compute_gradients(luminance, operator='sobel')
                gradient_mag = gradient_magnitude(gradient_x, gradient_y)
                
                _, power_spectrum = compute_fft(luminance)
                
                texture_features = extract_texture_features(gradient_x, gradient_y)
                freq_features = extract_frequency_features(power_spectrum, luminance.shape)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Gradients")
                st.markdown("""
                **What to look for:**
                - AI images often have smoother/blurrier gradients
                - Real photos have sharper, more natural edges
                - Look for unnatural periodic patterns
                """)
            
            with col2:
                st.subheader("FFT Spectrum")
                st.markdown("""
                **What to look for:**
                - Bright spots in corners/edges = periodic artifacts
                - AI upscaling leaves grid patterns
                - Real photos have natural 1/f falloff
                """)
            
            with col3:
                st.subheader("Statistics")
                st.markdown("""
                **Key metrics:**
                - **Kurtosis**: Peakedness of gradient distribution
                - **Entropy**: Randomness in textures
                - **Spectral flatness**: Uniformity of frequency distribution
                """)
            
            fig = create_visualization(
                image, luminance, gradient_mag, power_spectrum,
                prediction=0, confidence=0.0
            )
            st.pyplot(fig)
            plt.close(fig)
            
            if classifier is not None:
                st.subheader("🔍 Classification Result")
                
                with st.spinner("Classifying..."):
                    prediction, confidence, features = classifier.predict_image(image)
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    if prediction == 1:
                        st.error(f"🤖 **AI-Generated**\n\nConfidence: {confidence:.1%}")
                    else:
                        st.success(f"📷 **Real Photo**\n\nConfidence: {1-confidence:.1%}")
                
                with result_col2:
                    st.metric(
                        label="AI Probability",
                        value=f"{confidence:.1%}",
                        delta=f"{'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low'} confidence"
                    )
                
                st.subheader("📊 Key Features Analysis")
                
                feature_names = [
                    'Gradient Kurtosis', 'Gradient Entropy', 'Gradient Skewness',
                    'High Freq Energy', 'Spectral Flatness', 'Radial Entropy'
                ]
                feature_values = [
                    texture_features['kurtosis_mag'],
                    texture_features['entropy_mag'],
                    texture_features['skewness_mag'],
                    freq_features['high_freq_energy'],
                    freq_features['spectral_flatness'],
                    freq_features['radial_entropy']
                ]
                
                feature_df = {
                    'Feature': feature_names,
                    'Value': feature_values
                }
                
                import pandas as pd
                st.dataframe(pd.DataFrame(feature_df), use_container_width=True)
                
                st.markdown("""
                **Interpretation Tips:**
                - **High kurtosis**: Sharp peaks in gradient distribution (often real photos)
                - **High entropy**: More randomness in texture (typically real)
                - **High spectral flatness**: Energy spread across frequencies (AI artifacts)
                """)
            else:
                st.info("Train the model to get predictions. Currently showing feature visualizations only.")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.exception(e)
    
    else:
        st.info("👆 Upload an image to begin analysis")
        
        st.markdown("""
        ---

        ### How It Works

        **Deep Learning Model (primary):**
        1. **Preprocessing**: Resize to 224x224, normalize with ImageNet stats
        2. **Feature Extraction**: EfficientNet-V2-S learns features automatically
        3. **Classification**: 2-class output (Real vs AI-generated)
        4. **Confidence**: Softmax probability of AI-generated class

        **Legacy sklearn models:**
        1. Extract 41 handcrafted features (luminance, gradients, FFT, texture)
        2. Scale + PCA dimensionality reduction
        3. SVM/GradientBoosting/RandomForest classification
        """


if __name__ == '__main__':
    main()