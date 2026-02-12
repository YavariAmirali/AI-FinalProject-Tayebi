import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
from sklearn.utils import class_weight
from data_loader import get_data_loaders, DATA_DIR

import wandb
from wandb.keras import WandbCallback

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def build_model(input_shape):
    """
    Builds the model architecture (ResNet50 based).
    Kept exactly as previous logic.
    """
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model layers
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def train_model():
    # Initialize WandB run
    wandb.init(
        project="pneumonia-detection-phase2",
        entity="YOUR_WANDB_USERNAME",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "architecture": "ResNet50",
            "dataset": "ChestXRay"
        }
    )

    print("🚀 Loading Data...")
    train_gen, val_gen, _ = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Calculate Class Weights
    print("⚖️ Calculating Class Weights...")
    train_labels = train_gen.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f" Class Weights: {class_weight_dict}")

    # Build Model
    print("🧠 Building Model...")
    model = build_model(input_shape=IMG_SIZE + (3,))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )

    # --- CALLBACKS ---
    print("jj Setting up Callbacks (Logs, WandB, Checkpoints)...")

    # 1. Model Checkpoint (Standard)
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, 'finetuned_resnet.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # 2. Early Stopping (Standard)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # 3. Reduce LR (Standard)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # 4. TensorBoard (Standard Logging)
    tensorboard_callback = TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)

    # Saves epoch-by-epoch results to a CSV file
    csv_logger = CSVLogger(os.path.join(LOGS_DIR, 'training_log.csv'), append=True)

    # Automatically logs metrics and system stats to wandb.ai
    wandb_callback = WandbCallback(
        save_model=False,  # We use our own ModelCheckpoint for saving
        monitor='val_loss',
        mode='min'
    )

    # Combine all callbacks
    callbacks_list = [
        checkpoint,
        early_stopping,
        reduce_lr,
        tensorboard_callback,
        csv_logger,
        wandb_callback
    ]

    # Train
    print("🔥 Starting Training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1
    )

    wandb.finish()

    print("✅ Training Complete.")


if __name__ == "__main__":
    train_model()

