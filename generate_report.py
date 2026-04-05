#!/usr/bin/env python3
"""Generate detailed PDF report for training results on Vast.ai RTX 4090."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from pathlib import Path

RESULTS_DIR = Path("results_from_vast_4090")

# Load training history
with open(RESULTS_DIR / "training_history.json") as f:
    history = json.load(f)

with open(RESULTS_DIR / "training_summary.json") as f:
    summary = json.load(f)

epochs = list(range(1, len(history["train_loss"]) + 1))

# ── Generate plots ──────────────────────────────────────────────────────────

# 1. Loss curves
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, history["train_loss"], 'b-o', markersize=4, label="Train Loss")
ax.plot(epochs, history["val_loss"], 'r-s', markersize=4, label="Val Loss")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training & Validation Loss", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
plt.close()

# 2. Accuracy curves
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, history["train_acc"], 'b-o', markersize=4, label="Train Acc")
ax.plot(epochs, history["val_acc"], 'r-s', markersize=4, label="Val Acc")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(75, 100.5)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "accuracy_curves.png", dpi=150)
plt.close()

# 3. Learning rate schedule
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs, [lr * 1000 for lr in history["lr"]], 'g-o', markersize=4)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Learning Rate (x1e-3)", fontsize=12)
ax.set_title("Learning Rate Schedule (Cosine with Warmup)", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "lr_schedule.png", dpi=150)
plt.close()

# 4. Val accuracy progression with best markers
fig, ax = plt.subplots(figsize=(10, 5))
val_accs = history["val_acc"]
ax.bar(epochs, val_accs, color=['#2ecc71' if v == max(val_accs) else '#3498db' for v in val_accs], alpha=0.8)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
ax.set_title("Validation Accuracy per Epoch", fontsize=14, fontweight='bold')
ax.set_ylim(89, 100)
ax.axhline(y=max(val_accs), color='red', linestyle='--', alpha=0.7, label=f"Best: {max(val_accs):.2f}%")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(RESULTS_DIR / "val_accuracy_bar.png", dpi=150)
plt.close()

print("Plots saved.")

# ── Generate PDF ────────────────────────────────────────────────────────────

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "AI Image Detector - Training Report (Vast.ai RTX 4090)", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.cell(70, 6, key + ":", new_x="END")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ── Page 1: Overview ────────────────────────────────────────────────────────
pdf.add_page()
pdf.ln(3)

pdf.section_title("1. Executive Summary")
pdf.body_text(
    "This report documents the training of an EfficientNet-V2-S deep learning model for "
    "AI-generated image detection on a Vast.ai server equipped with an NVIDIA RTX 4090 GPU (24GB VRAM). "
    "The model was trained on the CIFAKE dataset (100,000 training images, 20,000 validation images) "
    "for 30 epochs, achieving a best validation accuracy of 98.97% and ROC AUC of 0.9974."
)

pdf.section_title("2. Hardware & Environment")
pdf.key_value("GPU", "NVIDIA GeForce RTX 4090 (24 GB VRAM)")
pdf.key_value("CUDA Version", "12.7")
pdf.key_value("Driver Version", "565.57.01")
pdf.key_value("PyTorch", "2.x with CUDA + AMP (mixed precision)")
pdf.key_value("Platform", "Linux (Vast.ai)")
pdf.key_value("Total Training Time", "80.7 minutes")
pdf.ln(3)

pdf.section_title("3. Training Configuration")
pdf.key_value("Model", "EfficientNet-V2-S (20.8M parameters)")
pdf.key_value("Dataset", "CIFAKE (dragonintelligence/CIFAKE-image-dataset)")
pdf.key_value("Train Samples", "100,000")
pdf.key_value("Val Samples", "20,000")
pdf.key_value("Image Size", "224 x 224")
pdf.key_value("Batch Size", "64")
pdf.key_value("Gradient Accumulation", "1 (effective batch = 64)")
pdf.key_value("Optimizer", "AdamW (weight_decay=0.01)")
pdf.key_value("Learning Rate", "3e-4 (cosine schedule with 3-epoch warmup)")
pdf.key_value("Loss Function", "Label Smoothing Cross-Entropy (smoothing=0.1)")
pdf.key_value("Data Augmentation", "Strong (RandomResizedCrop, ColorJitter, GaussianBlur, RandomErasing)")
pdf.key_value("Mixed Precision (AMP)", "Enabled")
pdf.key_value("Early Stopping Patience", "7 epochs")
pdf.key_value("Epochs Completed", "30 / 30 (no early stopping triggered)")

# ── Page 2: Results ─────────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("4. Final Results")
pdf.key_value("Best Validation Accuracy", f"{summary['best_val_acc']:.2f}% (Epoch 29)")
pdf.key_value("Best Validation Loss", f"{summary['best_val_loss']:.6f}")
pdf.key_value("Final Train Accuracy", f"{summary['final_train_acc']:.2f}%")
pdf.key_value("Final Val Accuracy", f"{summary['final_val_acc']:.2f}%")
pdf.key_value("ROC AUC Score", f"{summary['roc_auc']:.4f}")
pdf.ln(3)

pdf.section_title("5. Classification Report (on 20K validation set)")
pdf.set_font("Courier", "", 9)
report_text = (
    "                precision    recall  f1-score   support\n"
    "\n"
    "        Real       0.99      0.99      0.99     10000\n"
    "AI-Generated       0.99      0.99      0.99     10000\n"
    "\n"
    "    accuracy                           0.99     20000\n"
    "   macro avg       0.99      0.99      0.99     20000\n"
    "weighted avg       0.99      0.99      0.99     20000\n"
)
pdf.multi_cell(0, 4.5, report_text)
pdf.ln(3)

pdf.section_title("6. Epoch-by-Epoch Summary")
pdf.set_font("Courier", "", 7.5)
header = f"{'Ep':>3} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>10} | Best"
pdf.cell(0, 4, header, new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 4, "-" * 82, new_x="LMARGIN", new_y="NEXT")

best_val = 0
for i in range(len(epochs)):
    tl = history["train_loss"][i]
    ta = history["train_acc"][i]
    vl = history["val_loss"][i]
    va = history["val_acc"][i]
    lr = history["lr"][i]
    is_best = va > best_val
    if is_best:
        best_val = va
    marker = " *" if is_best else ""
    line = f"{epochs[i]:>3} | {tl:>10.6f} | {ta:>9.2f}% | {vl:>10.6f} | {va:>8.2f}% | {lr:>10.2e} |{marker}"
    pdf.cell(0, 4, line, new_x="LMARGIN", new_y="NEXT")

pdf.set_font("Helvetica", "I", 8)
pdf.ln(2)
pdf.cell(0, 4, "* = new best validation accuracy", new_x="LMARGIN", new_y="NEXT")

# ── Page 3: Charts ──────────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("7. Training Curves")

pdf.image(str(RESULTS_DIR / "loss_curves.png"), x=10, w=190)
pdf.ln(5)
pdf.image(str(RESULTS_DIR / "accuracy_curves.png"), x=10, w=190)

pdf.add_page()
pdf.image(str(RESULTS_DIR / "lr_schedule.png"), x=10, w=190)
pdf.ln(5)
pdf.image(str(RESULTS_DIR / "val_accuracy_bar.png"), x=10, w=190)

# ── Page 5: Errors & Code Changes ──────────────────────────────────────────
pdf.add_page()
pdf.section_title("8. Errors Encountered")
pdf.body_text(
    "No errors were encountered during the training run. The entire pipeline "
    "(data loading from HuggingFace, model creation, training loop, evaluation, "
    "and checkpoint saving) executed successfully without any modifications required.\n\n"
    "Minor notes:\n"
    "- A HuggingFace warning about unauthenticated requests was displayed (non-blocking).\n"
    "- numpy was downgraded from 2.4.3 to 1.26.4 during pip install to satisfy the "
    "requirements.txt constraint (numpy>=1.24.0,<2.0.0). This was handled automatically by pip.\n"
    "- No CUDA out-of-memory issues with batch_size=64 on the RTX 4090."
)

pdf.section_title("9. Code Changes Made")
pdf.body_text(
    "No code changes were necessary. The repository's training pipeline ran cleanly "
    "out of the box on the Vast.ai RTX 4090 server. The following command was used:\n\n"
    "  python train_deep_cnn.py \\\n"
    "    --model efficientnet --variant small \\\n"
    "    --batch-size 64 --accumulation-steps 1 \\\n"
    "    --epochs 30 --lr 3e-4 \\\n"
    "    --num-workers 8 \\\n"
    "    --loss label_smooth --label-smoothing 0.1 \\\n"
    "    --early-stopping 7 \\\n"
    "    --strong-augmentation \\\n"
    "    --checkpoint-dir ./models/deep\n\n"
    "Key differences from default settings:\n"
    "- batch-size increased from 16 to 64 (RTX 4090 has 24GB VRAM)\n"
    "- accumulation-steps reduced from 2 to 1 (not needed with large batch)\n"
    "- num-workers increased from 2 to 8 (server has more CPU cores)"
)

pdf.section_title("10. Key Observations")
pdf.body_text(
    "1. Rapid Convergence: The model reached 90% val accuracy after just 1 epoch and "
    "97%+ after epoch 2, demonstrating the effectiveness of transfer learning from "
    "ImageNet-pretrained EfficientNet-V2-S.\n\n"
    "2. Consistent Improvement: Validation accuracy improved steadily from 90.07% (epoch 1) "
    "to 98.97% (epoch 29), with the cosine LR schedule providing smooth convergence.\n\n"
    "3. No Overfitting: The gap between train accuracy (99.91%) and val accuracy (98.97%) "
    "remained small (~1%), indicating good generalization aided by label smoothing "
    "and strong data augmentation.\n\n"
    "4. Balanced Performance: The model achieves 0.99 precision and recall for BOTH classes "
    "(Real and AI-Generated), with no class bias.\n\n"
    "5. Training Speed: The RTX 4090 completed all 30 epochs over 100K images in ~81 minutes, "
    "averaging ~2.7 minutes per epoch."
)

# Save
output_path = RESULTS_DIR / "training_report_vast_4090.pdf"
pdf.output(str(output_path))
print(f"PDF report saved to: {output_path}")
