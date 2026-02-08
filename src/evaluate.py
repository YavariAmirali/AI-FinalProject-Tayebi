import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data_loaders, DATA_DIR

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_resnet_model.h5')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def evaluate_model():
    print("=" * 50)
    print(f"Starting Evaluation Pipeline - Phase 2")
    print("=" * 50)

    # Load Data
    print(f"[Data Loader] Loading Test Data...")
    _, _, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Check your paths!")
        return

    print(f"[Model] Loading best model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Run Predictions
    print("[Evaluation] Running predictions on Test Set...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = (predictions > 0.5).astype(int).reshape(-1)

    y_true = np.array(test_gen.labels)

    class_names = list(test_gen.class_to_idx.keys())

    # Generate Classification Report
    print("\n" + "=" * 30)
    print("   CLASSIFICATION REPORT")
    print("=" * 30)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix & Specificity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate Specificity: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "=" * 30)
    print("   MEDICAL METRICS (BONUS)")
    print("=" * 30)
    print(f"✅ Specificity (True Negative Rate): {specificity:.4f}")
    print(f"✅ Sensitivity (Recall):            {sensitivity:.4f}")
    print(f"✅ Confusion Matrix: \n{cm}")

    # Save Confusion Matrix Plot
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (ResNet50)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"\n[Output] Confusion Matrix saved to: {save_path}")


if __name__ == "__main__":
    evaluate_model()