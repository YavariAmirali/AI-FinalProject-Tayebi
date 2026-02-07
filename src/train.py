import os
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import datetime

import model_builder
from data_loader import get_data_loaders, DATA_DIR

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models',
                               'best_resnet_model.h5')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs',
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def train_model():
    print("=" * 50)
    print(f"Starting Training Pipeline - Phase 2 (ResNet50) - WEEK 6 FULL RUN")
    print("=" * 50)

    # 1. LOAD DATA
    try:
        print(f"[Data Loader] Loading from: {DATA_DIR}")
        train_gen, val_gen, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # HANDLE IMBALANCE (Class Weights)
    print("\n[Class Weights] Calculating weights for imbalance handling...")
    train_labels = train_gen.labels
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights_array))
    print(f"✅ Class Weights Computed: {class_weights}")

    # BUILD MODEL
    print("\n[Model] Building ResNet50 architecture...")
    model = model_builder.build_resnet50_model(input_shape=IMG_SIZE + (3,))

    print("[Model] Compiling with Optimizer=Adam, Loss=BinaryCrossentropy...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)

    # CALLBACKS
    callbacks = [
        # UPDATE: Increased patience to 10 so model has time to learn
        EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1),

        # Save the BEST model (not just the last one)
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),

        # Reduce learning rate if stuck
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),

        # UPDATE: Added TensorBoard for logging history [cite: 1001]
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
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

    print(f"\n✅ Training Finished. Best model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()