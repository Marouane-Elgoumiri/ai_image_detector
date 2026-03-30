# AI-Generated Image Detector

A machine learning system for detecting AI-generated images using deep learning and frequency domain analysis, with a Streamlit web interface.

## Features

- **CIFAKE Dataset**: Loads directly from HuggingFace (100K train + 20K test)
- **Multiple Training Approaches**:
  - Deep learning (EfficientNet, MobileNet, ConvNeXt)
  - CNN feature extraction (ResNet50)
  - Frequency domain analysis + SVM
- **Fast training**: Feature extraction runs once, then trains lightweight classifiers
- **Streamlit Web App**: Upload images and get instant AI/Real predictions

## Quick Start (Google Colab)

The easiest way to train! Open the notebook:

```bash
# Option 1: Open colab_quickstart.ipynb in Google Colab
# Runtime → Change runtime type → GPU (optional)
# Run all cells

# Option 2: Command line
git clone https://github.com/YOUR_USERNAME/AI_generated_images_detector.git
cd AI_generated_images_detector/ai_image_detector
pip install -r requirements.txt
python train_cifake.py --max-train 10000
```

## Installation (Local)

```bash
cd ai_image_detector
pip install -r requirements.txt
```

## Training Scripts

| Script | Description | Best For |
|--------|-------------|----------|
| `train_cifake.py` | CIFAKE from HuggingFace + feature extraction | Quick experiments |
| `train_cnn.py` | ResNet50 features + sklearn classifiers | Balanced approach |
| `train_deep_cnn.py` | Full deep learning training | Best accuracy |
| `train_model.py` | Frequency analysis + SVM | Traditional approach |

### Train with CIFAKE (Recommended)

```bash
python train_cifake.py --max-train 10000 --max-test 2000
```

### Train with Deep Learning

```bash
# MobileNet (lighter, better for limited VRAM)
python train_deep_cnn.py --model mobilenet --batch-size 16 --image-size 224

# EfficientNet (higher accuracy, needs more memory)
python train_deep_cnn.py --model efficientnet --batch-size 8 --image-size 224
```

## Project Structure

```
ai_image_detector/
├── app.py                    # Streamlit web app
├── train_cifake.py           # Main training script (CIFAKE)
├── train_cnn.py              # ResNet feature extraction
├── train_deep_cnn.py         # Deep learning training
├── train_model.py            # Frequency analysis + SVM
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── models/                   # Saved model files
└── src/
    ├── deep/                 # Deep learning modules
    ├── preprocessing/        # Luminance, gradients
    ├── features/             # Frequency, texture, PCA
    ├── models/               # Classifier, metrics
    └── utils/                # Dataset handling, visualization
```

## Running the Web App

```bash
streamlit run app.py
```

Or with a custom model:

```bash
streamlit run app.py -- --model-path ./models/best_model.pkl
```

## Detection Methodology

### Deep Learning Approach
- Pre-trained CNN (ResNet50, EfficientNet, MobileNet)
- Transfer learning with fine-tuning
- Binary classification (Real vs AI-generated)

### Frequency Analysis Approach
- **Luminance**: Extract Y channel from YCbCr
- **Gradient Analysis**: Sobel/Scharr operators
- **FFT**: Detect periodic artifacts in spectral domain
- **SVM**: RBF kernel classifier

## Dataset

### CIFAKE (from HuggingFace)
- 100,000 training images (50K real + 50K AI-generated)
- 20,000 test images (10K real + 10K AI-generated)
- Auto-downloaded via `datasets` library

### Custom Dataset Format
```
dataset/
├── real/           # Real images
└── ai/             # AI-generated images
```

## License

MIT License