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
# Import the builder you already have to be consistent
from model_builder import build_resnet50_model

import wandb
from wandb.integration.keras import WandbCallback
# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3  # Standard LR for frozen training
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def train_model():
    wandb.init(
        project="pneumonia-detection-phase2",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "architecture": "ResNet50_Frozen"
        }
    )

    print("🚀 Loading Data...")
    train_gen, val_gen, _ = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Calculate Class Weights
    print("⚖️ Calculating Class Weights...")
    train_labels = train_gen.get_labels()
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f" Class Weights: {class_weight_dict}")

    # Build Model using your model_builder.py
    print("🧠 Building Frozen Model...")
    model = build_resnet50_model(input_shape=IMG_SIZE + (3,))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )

    # 1. Model Checkpoint -> SAVING AS best_resnet_model.h5 (CRITICAL FIX)
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_resnet_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    tensorboard_callback = TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)
    csv_logger = CSVLogger(os.path.join(LOGS_DIR, 'training_log.csv'), append=True)
    wandb_callback = WandbCallback(save_model=False, monitor='val_loss', mode='min')

    callbacks_list = [checkpoint, early_stopping, reduce_lr, tensorboard_callback, csv_logger, wandb_callback]

    # Train
    print("🔥 Starting Phase 1 Training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1
    )

    wandb.finish()
    print("✅ Phase 1 Complete. Saved to models/best_resnet_model.h5")


if __name__ == "__main__":
    train_model()