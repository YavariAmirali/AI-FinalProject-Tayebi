import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc
from data_loader import get_data_loaders, DATA_DIR

BATCH_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'resnet_model.h5')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_roc_curve(y_true, y_pred_probs):
    """
    رسم منحنی ROC و ذخیره آن بدون باز کردن پنجره گرافیکی
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ ROC Curve saved at: {save_path} (AUC: {roc_auc:.2f})")


def main():
    # 1. Load Data (Using your custom MedicalDataGenerator)
    print("[Data] Loading Test Data...")
    _, _, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

    test_gen.shuffle = False
    test_gen.on_epoch_end()
    y_true = np.array(test_gen.labels)

    # 2. Load Model
    print(f"[Model] Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Predict
    print("[Prediction] Running predictions on test set...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    if y_pred_probs.ndim > 1 and y_pred_probs.shape[1] == 1:
        y_pred_probs = y_pred_probs.flatten()
    elif y_pred_probs.ndim > 1 and y_pred_probs.shape[1] == 2:
        y_pred_probs = y_pred_probs[:, 1]
    y_pred_classes = (y_pred_probs > 0.5).astype(int)

    min_len = min(len(y_true), len(y_pred_classes))
    y_true = y_true[:min_len]
    y_pred_classes = y_pred_classes[:min_len]
    y_pred_probs = y_pred_probs[:min_len]

    # 4. Generate Reports
    class_names = test_gen.class_names  # ['NORMAL', 'PNEUMONIA']

    print("\n" + "=" * 30)
    print("CLASSIFICATION REPORT")
    print("=" * 30)

    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)

    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 5. Visualizations
    plot_roc_curve(y_true, y_pred_probs)

    print("\n✅ Evaluation finished! Check the 'results' folder.")


if __name__ == "__main__":
    main()
