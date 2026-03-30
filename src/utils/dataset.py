import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import cv2


def load_images_from_directory(
    directory: str,
    label: int,
    target_size: Tuple[int, int] = (256, 256),
    max_images: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    images = []
    labels = []
    
    directory_path = Path(directory)
    if not directory_path.exists():
        return images, labels
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    count = 0
    
    for file_path in directory_path.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            if max_images and count >= max_images:
                break
            
            try:
                img = cv2.imread(str(file_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    images.append(img)
                    labels.append(label)
                    count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    return images, labels


def load_combined_dataset(
    base_path: str,
    target_size: Tuple[int, int] = (256, 256),
    max_per_class: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load combined dataset from both Dataset_AI-Real_images and Dataset_Real-Fake_images.
    
    Args:
        base_path: Base directory containing both dataset folders
        target_size: Target size for image resizing
        max_per_class: Maximum images per class (None for all)
    
    Returns:
        Tuple of (images, labels, image_paths)
        labels: 0 for real, 1 for AI-generated/fake
    """
    all_images = []
    all_labels = []
    all_paths = []
    
    dataset_ai_real = Path(base_path) / "Dataset_AI-Real_images"
    dataset_real_fake = Path(base_path) / "Dataset_Real-Fake_images"
    
    real_dirs = []
    fake_dirs = []
    
    if dataset_ai_real.exists():
        real_dirs.append(dataset_ai_real / "real_dataset")
        fake_dirs.append(dataset_ai_real / "Ai_generated_dataset")
    
    if dataset_real_fake.exists():
        real_dirs.append(dataset_real_fake / "real")
        fake_dirs.append(dataset_real_fake / "fake")
    
    for real_dir in real_dirs:
        if real_dir.exists():
            images, labels = load_images_from_directory(
                str(real_dir), 
                label=0, 
                target_size=target_size,
                max_images=max_per_class
            )
            all_images.extend(images)
            all_labels.extend(labels)
            all_paths.extend([str(p) for p in real_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}][:len(images)])
    
    for fake_dir in fake_dirs:
        if fake_dir.exists():
            images, labels = load_images_from_directory(
                str(fake_dir), 
                label=1, 
                target_size=target_size,
                max_images=max_per_class
            )
            all_images.extend(images)
            all_labels.extend(labels)
            all_paths.extend([str(p) for p in fake_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}][:len(images)])
    
    return all_images, all_labels, all_paths


def load_single_image(
    image_path: str,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Load a single image for inference.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for image resizing
    
    Returns:
        Preprocessed image as numpy array (RGB, resized)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    return img


def train_test_split(
    images: List[np.ndarray],
    labels: List[int],
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    """
    Split dataset into train and test sets with stratification.
    
    Args:
        images: List of images
        labels: List of labels
        test_ratio: Fraction of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_images, test_images, train_labels, test_labels)
    """
    from sklearn.model_selection import train_test_split as sk_train_test_split
    
    indices = np.arange(len(images))
    train_idx, test_idx = sk_train_test_split(
        indices, 
        test_size=test_ratio, 
        random_state=random_state,
        stratify=labels
    )
    
    train_images = [images[i] for i in train_idx]
    test_images = [images[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    
    return train_images, test_images, train_labels, test_labels