import tensorflow as tf
import os
import json
import numpy as np
from collections import Counter

from src.preprocessing import preprocess_image
from src.model import build_baseline_model

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 1          # Smoke test = 1 epoch
DATA_DIR = "data/train"
MODEL_SAVE_PATH = "models/baseline_model.h5"

# Utility: Class Weights
def compute_class_weights(dataset):
    labels = []

    for _, y in dataset:
        labels.extend(y.numpy())

    counter = Counter(labels)
    total = sum(counter.values())

    class_weights = {
        0: total / (2 * counter[0]),
        1: total / (2 * counter[1])
    }

    print("Class weights:", class_weights)
    return class_weights

def main():
    print("Starting training pipeline")

    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU detected")
    else:
        print("⚠️ Running on CPU")

    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset directory '{DATA_DIR}' not found")
        return

    print("Loading dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    # Preprocessing
    train_ds = train_ds.map(lambda x, y: (preprocess_image(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_image(x), y))

    # Smoke test
    train_ds_smoke = train_ds.take(2)
    val_ds_smoke = val_ds.take(1)

    print("Building model...")
    model = build_baseline_model()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    # Class weighting
    class_weights = compute_class_weights(train_ds_smoke)

    print("Starting training (Smoke Test)...")
    history = model.fit(
        train_ds_smoke,
        validation_data=val_ds_smoke,
        epochs=EPOCHS,
        class_weight=class_weights
    )

    # Save outputs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model.save(MODEL_SAVE_PATH)

    with open("logs/history.json", "w") as f:
        json.dump(history.history, f)

    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
    print("✅ Training pipeline completed successfully")


if __name__ == "__main__":
    main()
