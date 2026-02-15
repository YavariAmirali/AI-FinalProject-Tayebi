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
from model_builder import build_resnet50_model
import wandb
from wandb.integration.keras import WandbCallback
import argparse

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def train_model(epochs, batch_size, use_wandb):
    # Loading Data
    print("🚀 Loading Data...")
    train_gen, val_gen, _ = get_data_loaders(DATA_DIR, batch_size=batch_size)

    # Calculating Class Weights
    print("⚖️ Calculating Class Weights...")
    try:
        train_labels = train_gen.labels
    except AttributeError:
        train_labels = train_gen.get_labels()
        
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights_array))
    print(f" Class Weights Dictionary: {class_weight_dict}")

    # Build Model
    print("🧠 Building Frozen Model...")
    model = build_resnet50_model(input_shape=IMG_SIZE + (3,))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_resnet_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    csv_logger = CSVLogger(os.path.join(LOGS_DIR, 'training_log.csv'), append=True)
    
    callbacks_list = [checkpoint, early_stopping, reduce_lr, csv_logger]

    if use_wandb:
        wandb_callback = WandbCallback(save_model=False, monitor='val_loss', mode='min')
        callbacks_list.append(wandb_callback)

    # Train
    print(f"\n[Training] Starting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1 
    )

    if use_wandb:
        wandb.finish()
    
    print("✅ Phase 1 Complete. Saved to models/best_resnet_model.h5")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--wandb", action='store_true', help="Enable WandB logging")
    
    args = parser.parse_args()
    if args.wandb:
        wandb.init(
            project="pneumonia-detection-phase2",
            config={
                "learning_rate": LEARNING_RATE,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": "ResNet50_Frozen"
            }
        )

    train_model(epochs=args.epochs, batch_size=args.batch_size, use_wandb=args.wandb)
