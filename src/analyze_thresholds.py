import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score
from data_loader import get_data_loaders, DATA_DIR

# Configuration
BATCH_SIZE = 32
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FINETUNED_PATH = os.path.join(BASE_DIR, 'models', 'finetuned_resnet.h5')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def analyze_thresholds():
    print("🚀 Starting Threshold Analysis on TE Data...")

    # Load TEST Data instead of Validation Data
    _, _, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    y_true = np.array(test_gen.get_labels())

    print(f"   Running predictions on Test Set...")

    if not os.path.exists(MODEL_FINETUNED_PATH):
        print(f"❌ Error: Fine-tuned model not found at {MODEL_FINETUNED_PATH}")
        return

    # Load Model & Predict ONCE (to save time)
    print(f"\n📊 Loading Fine-Tuned Model...")
    model = tf.keras.models.load_model(MODEL_FINETUNED_PATH)

    print(f"   Running predictions on Validation Set...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred_probs = predictions.flatten()

    # Sweep Thresholds (0.50 to 0.95 with step 0.05)
    thresholds = np.arange(0.5, 1.00, 0.05)
    recalls = []
    specificities = []

    print("\n" + "=" * 50)
    print(f"{'Threshold':<15} | {'Recall':<15} | {'Specificity':<15}")
    print("-" * 50)

    for thresh in thresholds:
        y_pred = (y_pred_probs > thresh).astype(int)

        # Calculate Recall (Sensitivity)
        recall = recall_score(y_true, y_pred)

        # Calculate Specificity
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = 0

        recalls.append(recall)
        specificities.append(specificity)

        print(f"{thresh:<15.2f} | {recall:<15.4f} | {specificity:<15.4f}")

    print("=" * 50)

    # 4. Plot the results
    plt.figure(figsize=(10, 6))

    # Plot the lines
    plt.plot(thresholds, recalls, marker='o', label='Recall (Sensitivity)', color='blue', linewidth=2)
    plt.plot(thresholds, specificities, marker='s', label='Specificity (TN Rate)', color='green', linewidth=2)

    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Project Minimum Recall (0.95)')

    # Formatting
    plt.title('Threshold Tuning: Specificity vs. Recall (Validation Set)')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.xticks(thresholds)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left')

    # Save and Show
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    save_path = os.path.join(RESULTS_DIR, 'threshold_analysis.png')
    plt.savefig(save_path)
    print(f"\n✅ Saved Threshold Analysis plot to {save_path}")

    # Opens the window to show you the graph immediately
    plt.show()


if __name__ == "__main__":
    analyze_thresholds()