import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

import wandb
from wandb.integration.keras import WandbCallback

# Import your custom modules
import model_builder
from data_loader import get_data_loaders, DATA_DIR

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15  # Increased slightly since we have EarlyStopping
LEARNING_RATE = 0.001

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet_model.h5')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def plot_training_history(history, save_dir):
    """
    Plots Accuracy and Loss curves locally for the Final PDF Report.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 8))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Save plot
    plot_path = os.path.join(save_dir, 'training_plot.png')
    plt.savefig(plot_path)
    print(f"📊 Training plots saved at: {plot_path}")
    plt.close()


def train_model():
    print("=" * 50)
    print(f"Starting Training Pipeline - Phase 2 (ResNet50 + WandB)")
    print("=" * 50)

    wandb.init(
        project="pneumonia-detection-v2", mode='disabled',
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "ResNet50",
            "dataset": "ChestXRay",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }
    )

    # 1. LOAD DATA
    try:
        print(f"[Data Loader] Loading from: {DATA_DIR}")
        train_gen, val_gen, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # HANDLE IMBALANCE
    print("\n[Class Weights] Calculating weights...")
    train_labels = train_gen.labels
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights_array))
    print(f"✅ Class Weights: {class_weights}")

    # BUILD MODEL
    print("\n[Model] Building ResNet50 architecture...")
    model = model_builder.build_resnet50_model(input_shape=IMG_SIZE + (3,))

    print("[Model] Compiling...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
    )

    # CALLBACKS
    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1),

        CSVLogger(os.path.join(LOG_DIR, 'training_log.csv')),
        # WandBCallback(save_model=False)
    ]

    # TRAIN
    print(f"\n[Training] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # SAVE PLOTS LOCALLY
    plot_training_history(history, LOG_DIR)

    # Finish WandB run
    # wandb.finish()

    print(f"\n✅ Training Finished. Model saved at: {MODEL_SAVE_PATH}")
    print(f"✅ Logs saved at: {LOG_DIR}")


if __name__ == "__main__":
    train_model()

