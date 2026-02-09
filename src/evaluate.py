import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from data_loader import get_data_loaders, DATA_DIR

# Configuration
BATCH_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_resnet_model.h5')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

def evaluate_model():
    print(f"Starting Evaluation Pipeline...")

    # Load test data
    _, _, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    # Run predictions
    print("Running predictions on Test Set...")
    predictions = model.predict(test_gen, verbose=1)
    
    # Flatten for ROC and threshold for binary classification
    y_pred_probs = predictions.flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Get ground truth labels and filepaths
    try:
        y_true = np.array(test_gen.get_labels())
        filepaths = test_gen.get_filepaths()
    except AttributeError:
        print("Error: data_loader.py is outdated. Missing get_filepaths method.")
        return

    class_names = list(test_gen.class_to_idx.keys())

    # Print standard classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Calculate medical metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"Specificity (TN Rate): {specificity:.4f}")
    print(f"Sensitivity (Recall):  {sensitivity:.4f}")

    # Create results directory if needed
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()
    print("Confusion Matrix saved.")

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
    plt.close()
    print(f"ROC Curve saved (AUC: {roc_auc:.2f})")

    # Generate Error Analysis CSV
    errors = []
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            errors.append({
                "Filename": os.path.basename(filepaths[i]),
                "True_Label": class_names[y_true[i]],
                "Predicted": class_names[y_pred[i]],
                "Confidence": f"{predictions[i][0]:.4f}",
                "Type": "False Positive" if y_pred[i] == 1 else "False Negative"
            })
    
    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(RESULTS_DIR, 'error_analysis.csv'), index=False)
        print(f"Error analysis saved with {len(errors)} errors.")
    else:
        print("No errors found.")

if __name__ == "__main__":
    evaluate_model()
