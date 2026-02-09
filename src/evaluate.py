import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data_loaders, DATA_DIR

# Configuration
BATCH_SIZE = 32
# THIS POINTS TO YOUR EXISTING TRAINED MODEL
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_resnet_model.h5')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def evaluate_model():
    print("=" * 50)
    print(f"Starting Evaluation Pipeline - Phase 2 (NO RETRAINING)")
    print("=" * 50)

    # Load Data (TEST SET ONLY)
    print(f"[Data Loader] Loading Test Data...")
    _, _, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Load EXISTING Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. You need to have trained the model once.")
        return

    print(f"[Model] Loading best model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Run Predictions
    print("[Evaluation] Running predictions on Test Set...")
    # Get raw probabilities
    predictions = model.predict(test_gen, verbose=1)

    # Convert to binary classes (0 or 1)
    y_pred = (predictions > 0.5).astype(int).reshape(-1)

    # Get True labels from the generator
    y_true = np.array(test_gen.labels)

    # Get Filepaths using the NEW method
    try:
        filepaths = test_gen.get_filepaths()
    except AttributeError:
        print("❌ ERROR: Your data_loader.py is still old! Copy the new code provided.")
        return

    class_names = list(test_gen.class_to_idx.keys())

    # Generate Standard Metrics
    print("\n" + "=" * 30)
    print("   CLASSIFICATION REPORT")
    print("=" * 30)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Medical Metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "=" * 30)
    print("   MEDICAL METRICS")
    print("=" * 30)
    print(f"✅ Specificity (TN Rate):       {specificity:.4f}")
    print(f"✅ Sensitivity (Recall):        {sensitivity:.4f}")
    print(f"✅ False Positives (Healthy->Sick): {fp}")
    print(f"✅ False Negatives (Sick->Healthy): {fn}")

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

    # ERROR ANALYSIS
    print("\n" + "=" * 30)
    print("   GENERATING ERROR REPORT")
    print("=" * 30)

    errors = []
    # Identify mismatches
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            filename = os.path.basename(filepaths[i])
            true_class = class_names[y_true[i]]
            pred_class = class_names[y_pred[i]]
            confidence = predictions[i][0]

            errors.append({
                "Filename": filename,
                "True_Label": true_class,
                "Predicted_Label": pred_class,
                "Model_Confidence": f"{confidence:.4f}",
                "Error_Type": "False Positive" if y_pred[i] == 1 else "False Negative"
            })

    # Save errors to CSV
    if errors:
        error_df = pd.DataFrame(errors)
        error_csv_path = os.path.join(RESULTS_DIR, 'error_analysis.csv')
        error_df.to_csv(error_csv_path, index=False)
        print(f"✅ Found {len(errors)} errors.")
        print(f"✅ Error report saved to: {error_csv_path}")
        print("   -> Send this CSV to Person 1 for the report.")
    else:
        print("🎉 0 Errors found! (Or check if data is correct).")


if __name__ == "__main__":
    evaluate_model()
