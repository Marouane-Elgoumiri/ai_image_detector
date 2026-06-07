# AI-Generated Image Detector

A deep learning system for detecting AI-generated images using the CIFAKE dataset, with a Streamlit web interface.

## Features

- **CIFAKE Dataset**: Loads directly from HuggingFace (100K train + 20K test)
- **Deep Learning Training**: End-to-end training with EfficientNet, MobileNet, ConvNeXt
- **Google Colab Ready**: T4 GPU optimized with mixed precision and gradient accumulation
- **Streamlit Web App**: Upload images and get instant AI/Real predictions

## Quick Start (Google Colab)

The easiest way to train:

```bash
# 1. Open colab_quickstart.ipynb in Google Colab
# 2. Runtime → Change runtime type → T4 GPU
# 3. Run all cells
```

Or command line:

```bash
git clone https://github.com/YOUR_USERNAME/AI_generated_images_detector.git
cd AI_generated_images_detector/ai_image_detector
pip install -r requirements.txt
python train_deep_cnn.py --model efficientnet --batch-size 16 --image-size 224 --epochs 30
```

## Installation (Local)

```bash
cd ai_image_detector
pip install -r requirements.txt
```

## Training

`train_deep_cnn.py` is the only training script. It does end-to-end deep learning on the CIFAKE dataset.

### EfficientNet-V2-S (Recommended — best accuracy)

```bash
# Quick test (2 min)
python train_deep_cnn.py --model efficientnet --batch-size 16 --max-train 5000 --max-val 1000 --epochs 2

# Full training (~30-60 min on T4)
python train_deep_cnn.py --model efficientnet --batch-size 16 --image-size 224 --epochs 30 --accumulation-steps 2
```

### MobileNet-V3-Large (Faster, lower accuracy)

```bash
python train_deep_cnn.py --model mobilenet --variant large --batch-size 32 --image-size 224 --epochs 30
```

### Model Options

| Model | Params | VRAM (224px) | Accuracy | Speed |
|-------|--------|-------------|----------|-------|
| EfficientNet-V2-S | 20M | ~10GB (batch 16) | Best | Moderate |
| MobileNet-V3-Large | 5.4M | ~4GB (batch 32) | Good | Fast |
| ConvNeXt-Small | 50M | ~14GB (batch 16) | High | Slow |

## Project Structure

```
ai_image_detector/
├── app.py                    # Streamlit web app
├── train_deep_cnn.py         # Deep learning training (GPU required)
├── colab_quickstart.ipynb    # Colab notebook for training
├── requirements.txt          # Python dependencies
├── models/                   # Saved model checkpoints
└── src/
    ├── deep/                 # Deep learning modules
    │   ├── models.py         # Model architectures
    │   ├── data/             # Dataset & augmentation
    │   └── training/         # Trainer, losses, schedulers
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
streamlit run app.py -- --model-path ./models/deep/best_model.pt
```

## Dataset

### CIFAKE (from HuggingFace)
- 100,000 training images (50K real + 50K AI-generated)
- 20,000 test images (10K real + 10K AI-generated)
- Auto-downloaded via `datasets` library

## License

MIT License
