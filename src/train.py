import tensorflow as tf
import os
from preprocessing import tf_preprocess_image

# Try to import the model, otherwise use a dummy one (for safety)
try:
    from model import build_baseline_model
except ImportError:
    print("Warning: 'src/model.py' not found. Using dummy model for DevOps test.")
    from tensorflow.keras import layers, models


    def build_baseline_model():
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

# === Configuration ===
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 2
DATA_DIR = "../data/train"
MODEL_SAVE_PATH = "../models/baseline_model.h5"  # Adjusted to save outside src


def main():
    print("=== Starting Smoke Test Pipeline ===")

    if tf.config.list_physical_devices('GPU'):
        print("GPU detected.")
    else:
        print("Running on CPU.")

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found.")
        return

    print("Loading dataset...")
    # This loads images and resizes them to 224x224
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # === Modular Preprocessing Step ===
    print("Applying preprocessing from preprocessing.py...")
    train_ds = train_ds.map(tf_preprocess_image)
    val_ds = val_ds.map(tf_preprocess_image)

    print("Building model...")
    model = build_baseline_model()

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Starting training (Smoke Test)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Ensure models directory exists
    if not os.path.exists('../models'):
        os.makedirs('../models')

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()