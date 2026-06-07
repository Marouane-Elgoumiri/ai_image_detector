import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: tuple = ('Real', 'AI-Generated'),
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for the classifier.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (n_samples, 2)
        class_names: Names for each class
        verbose: Print results
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    if verbose:
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              {class_names[0]:12} {class_names[1]:12}")
        print(f"Actual {class_names[0]:6} {tn:12} {fp:12}")
        print(f"       {class_names[1]:6} {fn:12} {tp:12}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve - AI Image Detector",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve for the classifier.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple = ('Real', 'AI-Generated'),
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    ax.set_ylim(len(cm) - 0.5, -0.5)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def get_classification_report_str(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple = ('Real', 'AI-Generated')
) -> str:
    """
    Get classification report as string.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
    
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=class_names)