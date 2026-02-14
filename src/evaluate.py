import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from data_loader import get_data_loaders, DATA_DIR

# Configuration
BATCH_SIZE = 32
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Paths to BOTH models for comparison
MODEL_BASELINE_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet_model.h5')
MODEL_FINETUNED_PATH = os.path.join(BASE_DIR, 'models', 'finetuned_resnet.h5')


def calculate_metrics(y_true, y_pred_probs, threshold):
    y_pred = (y_pred_probs > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        fp = 0
        specificity = 0

    return {
        "Accuracy": acc,
        "Recall": recall,
        "Specificity": specificity,
        "False Positives": fp,
        "Confusion Matrix": cm,
        "Predictions": y_pred
    }


def find_optimal_threshold(y_true_calib, probs_calib, target_recall=0.95):
    print(f"\n🔍 Tuning threshold on 20% Calibration Set to maintain Recall >= {target_recall}...")
    best_thresh = 0.50
    best_spec = -1.0

    thresholds = np.arange(0.50, 1.00, 0.01)
    for thresh in thresholds:
        metrics = calculate_metrics(y_true_calib, probs_calib, thresh)
        if metrics["Recall"] >= target_recall and metrics["Specificity"] >= best_spec:
            best_spec = metrics["Specificity"]
            best_thresh = thresh

    print(f"   ✅ Optimal Threshold Found: {best_thresh:.2f} (Estimated Specificity: {best_spec:.4f})")
    return best_thresh


def evaluate_model():
    print(f"🚀 Starting Strict Unseen Evaluation (80/20 Split)...")

    # 1. Load All Test Data
    _, _, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    y_true_all = np.array(test_gen.get_labels())
    class_names = list(test_gen.class_to_idx.keys())
    filepaths_all = np.array(test_gen.get_filepaths())

    if not os.path.exists(MODEL_FINETUNED_PATH):
        print(f"❌ Error: Fine-tuned model not found at {MODEL_FINETUNED_PATH}")
        return

    # 2. Get Raw Probabilities
    print(f"\n🧠 Generating raw probabilities from models...")

    # Fine-Tuned Model
    model_finetuned = tf.keras.models.load_model(MODEL_FINETUNED_PATH)
    probs_fine_all = model_finetuned.predict(test_gen, verbose=0).flatten()

    # Baseline Model (if it exists)
    if os.path.exists(MODEL_BASELINE_PATH):
        model_base = tf.keras.models.load_model(MODEL_BASELINE_PATH)
        probs_base_all = model_base.predict(test_gen, verbose=0).flatten()
    else:
        probs_base_all = None
        print("   Warning: Baseline model not found.")

    # Split
    indices = np.arange(len(y_true_all))
    idx_calib, idx_test = train_test_split(
        indices,
        test_size=0.80,
        stratify=y_true_all,
        random_state=42
    )

    y_true_calib = y_true_all[idx_calib]
    probs_fine_calib = probs_fine_all[idx_calib]

    y_true_test = y_true_all[idx_test]
    probs_fine_test = probs_fine_all[idx_test]
    filepaths_test = filepaths_all[idx_test]

    # Find the Optimal Threshold ONLY on the 20% Calibration Data
    optimal_thresh = find_optimal_threshold(y_true_calib, probs_fine_calib, target_recall=0.95)

    # Evaluate BOTH models on the 80% Unseen Test Data using the optimal threshold
    print(
        f"\n📊 Evaluating models on 80% Unseen Test Data ({len(y_true_test)} images) at threshold {optimal_thresh:.2f}...")
    metrics_fine_test = calculate_metrics(y_true_test, probs_fine_test, optimal_thresh)

    if probs_base_all is not None:
        probs_base_test = probs_base_all[idx_test]
        metrics_base_test = calculate_metrics(y_true_test, probs_base_test, optimal_thresh)
    else:
        metrics_base_test = None

    # Print Comparison Table
    print("\n" + "=" * 70)
    print(
        f"{'Metric':<25} | {'Frozen (' + str(optimal_thresh)[:4] + ')':<20} | {'Fine-Tuned (' + str(optimal_thresh)[:4] + ')':<20}")
    print("-" * 75)

    metrics_list = ["Accuracy", "Recall", "Specificity", "False Positives"]
    for m in metrics_list:
        val_base = f"{metrics_base_test[m]:.4f}" if metrics_base_test else "N/A"
        if m == "False Positives" and metrics_base_test:
            val_base = f"{metrics_base_test[m]}"
            val_fine = f"{metrics_fine_test[m]}"
        else:
            val_fine = f"{metrics_fine_test[m]:.4f}"
        print(f"{m:<25} | {val_base:<20} | {val_fine:<20}")
    print("=" * 70)

    # Print Classification Report for Fine-Tuned Model
    print("\nClassification Report (Fine-Tuned Model on Unseen Data):")
    print(classification_report(y_true_test, metrics_fine_test["Predictions"], target_names=class_names))

    # Save Results
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(metrics_fine_test["Confusion Matrix"], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Unseen Data, Thresh={optimal_thresh:.2f})')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_finetuned.png'))
    print(f"\n✅ Saved Confusion Matrix to results/")

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_true_test, probs_fine_test)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('ROC Curve (Unseen Test Data)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_finetuned.png'))
    print(f"✅ Saved ROC Curve to results/")

    # Generate Error Analysis CSV
    errors = []
    y_pred_fine = metrics_fine_test["Predictions"]

    for i in range(len(y_true_test)):
        if y_pred_fine[i] != y_true_test[i]:
            errors.append({
                "Filename": os.path.basename(filepaths_test[i]),
                "True_Label": class_names[y_true_test[i]],
                "Predicted": class_names[y_pred_fine[i]],
                "Confidence": f"{probs_fine_test[i]:.4f}",
                "Type": "False Positive" if y_pred_fine[i] == 1 else "False Negative"
            })

    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(RESULTS_DIR, 'error_analysis.csv'), index=False)
        print(f"✅ Error analysis saved to results/error_analysis.csv with {len(errors)} errors.")
    else:
        print("✅ No errors found. Error analysis CSV not created.")


if __name__ == "__main__":
    evaluate_model()