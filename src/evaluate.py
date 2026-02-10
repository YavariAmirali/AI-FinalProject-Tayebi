import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, recall_score, accuracy_score
from data_loader import get_data_loaders, DATA_DIR

# Configuration
BATCH_SIZE = 32
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Paths to BOTH models for comparison
MODEL_BASELINE_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet_model.h5')
MODEL_FINETUNED_PATH = os.path.join(BASE_DIR, 'models', 'finetuned_resnet.h5')


def get_metrics(model, test_gen, y_true):
    """Calculates metrics for a single model"""
    print(f"   Running predictions...")
    predictions = model.predict(test_gen, verbose=0)
    y_pred_probs = predictions.flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # Sensitivity

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Accuracy": acc,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "False Positives": fp,
        "Confusion Matrix": cm,
        "Predictions": y_pred,
        "Probs": y_pred_probs
    }


def evaluate_model():
    print(f"🚀 Starting Week 7 Evaluation (Before vs. After)...")

    # 1. Load Test Data
    _, _, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    y_true = np.array(test_gen.get_labels())
    class_names = list(test_gen.class_to_idx.keys())

    if not os.path.exists(MODEL_FINETUNED_PATH):
        print(f"❌ Error: Fine-tuned model not found at {MODEL_FINETUNED_PATH}")
        print("   Run 'python src/train_finetune.py' first.")
        return

    # 2. Evaluate Baseline (Week 6)
    print(f"\n📊 Evaluating Baseline Model (Week 6)...")
    if os.path.exists(MODEL_BASELINE_PATH):
        model_base = tf.keras.models.load_model(MODEL_BASELINE_PATH)
        metrics_base = get_metrics(model_base, test_gen, y_true)
    else:
        print("   Warning: Baseline model not found. Skipping comparison.")
        metrics_base = None

    # 3. Evaluate Fine-Tuned (Week 7)
    print(f"\n📊 Evaluating Fine-Tuned Model (Week 7)...")
    model_finetuned = tf.keras.models.load_model(MODEL_FINETUNED_PATH)
    metrics_finetuned = get_metrics(model_finetuned, test_gen, y_true)

    # 4. Print Comparison Table (Task 15)
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} | {'Baseline (Frozen)':<20} | {'Fine-Tuned (Week 7)':<20}")
    print("-" * 65)

    metrics = ["Accuracy", "Recall (Sensitivity)", "Specificity", "False Positives"]
    for m in metrics:
        val_base = f"{metrics_base[m]:.4f}" if metrics_base else "N/A"
        val_fine = f"{metrics_finetuned[m]:.4f}"
        print(f"{m:<20} | {val_base:<20} | {val_fine:<20}")
    print("=" * 60)

    # 5. Save Results for Report (Fine-Tuned Version)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Save Confusion Matrix (Fine-Tuned)
    plt.figure(figsize=(6, 5))
    sns.heatmap(metrics_finetuned["Confusion Matrix"], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Fine-Tuned)')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_finetuned.png'))
    print(f"\n✅ Saved Fine-Tuned Confusion Matrix to results/")

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_true, metrics_finetuned["Probs"])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('ROC Curve (Fine-Tuned)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_finetuned.png'))
    print(f"✅ Saved ROC Curve to results/")


if __name__ == "__main__":
    evaluate_model()