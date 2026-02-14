import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import roc_curve, auc

# --- REUSE YOUR EXISTING MODULES ---
from data_loader import MedicalDataGenerator, DATA_DIR
from evaluate import calculate_metrics

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PHASE1_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet_model.h5')  # The frozen baseline
PHASE2_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'finetuned_resnet.h5')  # The fine-tuned final
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_phase1_predictions(model):
    print(f"   [Phase 1] Creating Standard Generator (No Crop)...")
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"   [Phase 1] Predicting...")
    probs = model.predict(generator, verbose=1).flatten()
    return generator.classes, probs


def get_phase2_predictions(model):
    print(f"   [Phase 2] Creating MedicalGenerator (With Crop)...")
    generator = MedicalDataGenerator(
        directory=TEST_DIR,
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )

    print(f"   [Phase 2] Predicting...")
    probs = model.predict(generator, verbose=1).flatten()
    labels = np.array(generator.get_labels())

    min_len = min(len(probs), len(labels))
    return labels[:min_len], probs[:min_len]


def plot_comparison(p1_metrics, p2_metrics, p1_data, p2_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confusion Matrix Phase 1
    sns.heatmap(p1_metrics["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f"Phase 1: Baseline (Uncropped)\nSpecificity: {p1_metrics['Specificity']:.2f}")

    # Confusion Matrix Phase 2
    sns.heatmap(p2_metrics["Confusion Matrix"], annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
    axes[0, 1].set_title(f"Phase 2: Fine-Tuned (Cropped)\nSpecificity: {p2_metrics['Specificity']:.2f}")

    # ROC Curves
    fpr1, tpr1, _ = roc_curve(p1_data[0], p1_data[1])
    fpr2, tpr2, _ = roc_curve(p2_data[0], p2_data[1])
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)

    axes[1, 0].plot(fpr1, tpr1, linestyle='--', label=f'Phase 1 (AUC={auc1:.3f})')
    axes[1, 0].plot(fpr2, tpr2, linewidth=2, color='green', label=f'Phase 2 (AUC={auc2:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 0].set_title("ROC Curve Comparison")
    axes[1, 0].legend(loc='lower right')

    # Metric Improvement Bar Chart
    metrics = ['Accuracy', 'Recall', 'Specificity']
    x = np.arange(len(metrics))
    width = 0.35

    p1_vals = [p1_metrics[m] for m in metrics]
    p2_vals = [p2_metrics[m] for m in metrics]

    axes[1, 1].bar(x - width / 2, p1_vals, width, label='Phase 1', color='#a6cee3')
    axes[1, 1].bar(x + width / 2, p2_vals, width, label='Phase 2', color='#1f78b4')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].set_title("Metric Improvement")
    axes[1, 1].legend()

    # Annotate bars for Phase 1 (Baseline)
    for i, v in enumerate(p1_vals):
        axes[1, 1].text(i - width / 2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f4e79')

    # Annotate bars for Phase 2 (Fine-Tuned)
    for i, v in enumerate(p2_vals):
        axes[1, 1].text(i + width / 2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'phase1_vs_phase2_comparison.png')
    plt.savefig(save_path)
    print(f"\n✅ Comparison plot saved to: {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("🧪 PHASE 1 vs PHASE 2 COMPARISON")
    print("=" * 60)

    # Load Models
    if not os.path.exists(PHASE1_MODEL_PATH) or not os.path.exists(PHASE2_MODEL_PATH):
        print("❌ Error: One or both models are missing.")
        return

    model_p1 = tf.keras.models.load_model(PHASE1_MODEL_PATH)
    model_p2 = tf.keras.models.load_model(PHASE2_MODEL_PATH)

    # Get Predictions
    y_true_p1, probs_p1 = get_phase1_predictions(model_p1)
    y_true_p2, probs_p2 = get_phase2_predictions(model_p2)

    # Calculate Metrics (Using evaluate.py logic)
    p1_metrics = calculate_metrics(y_true_p1, probs_p1, threshold=0.99)
    p2_metrics = calculate_metrics(y_true_p2, probs_p2, threshold=0.99)

    # Print Table
    print("\n" + "=" * 80)
    print(f"{'METRIC':<20} | {'PHASE 1 (Baseline)':<20} | {'PHASE 2 (Fine-Tuned)':<20} | {'CHANGE':<10}")
    print("-" * 80)

    for m in ['Accuracy', 'Recall', 'Specificity']:
        v1 = p1_metrics[m]
        v2 = p2_metrics[m]
        diff = v2 - v1
        print(f"{m:<20} | {v1:.4f}               | {v2:.4f}                 | {diff:+.4f}")

    print("-" * 80)
    cm1 = p1_metrics['Confusion Matrix']
    cm2 = p2_metrics['Confusion Matrix']
    if cm1.shape == (2, 2) and cm2.shape == (2, 2):
        print(f"{'True Negatives':<20} | {cm1[0, 0]:<20} | {cm2[0, 0]:<20} | {(cm2[0, 0] - cm1[0, 0]):+d}")
        print(f"{'False Positives':<20} | {cm1[0, 1]:<20} | {cm2[0, 1]:<20} | {(cm2[0, 1] - cm1[0, 1]):+d}")
    print("=" * 80)

    # Plot
    plot_comparison(p1_metrics, p2_metrics, (y_true_p1, probs_p1), (y_true_p2, probs_p2))


if __name__ == "__main__":
    main()