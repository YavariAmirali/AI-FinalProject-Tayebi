import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

try:
    import model_builder
except ImportError:
    raise ImportError("Could not find 'model_builder.py'. Make sure it is in the same directory.")

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = '../data'
MODEL_SAVE_PATH = os.path.join('../models', 'resnet_model.h5')


# TEMPORARY DATA LOADER
def get_temporary_data_generators(data_dir, target_size, batch_size):
    """
    Creates data generators. Mimics what 'data_loader.py' should do.
    Handles 'train' folder and splits validation automatically if needed.
    """
    train_dir = os.path.join(data_dir, 'train')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory not found: {train_dir}")

    print(f"[Data Loader] Creating generators from: {train_dir}")

    # Data Augmentation & Preprocessing
    # Rescale is mandatory for Neural Networks (0-255 -> 0-1)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,  # Using 20% of training data for validation
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Generator for Training
    print("[Data Loader] Found images for TRAINING:")
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    # Generator for Validation
    print("[Data Loader] Found images for VALIDATION:")
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen


# MAIN TRAINING PIPELINE
def train_model():
    print("=" * 50)
    print(f"Starting Training Pipeline - Phase 2 (ResNet50)")
    print("=" * 50)

    try:
        train_gen, val_gen = get_temporary_data_generators(
            data_dir=DATA_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    print("\n[Class Weights] Calculating weights for imbalance handling...")

    # Compute weights:
    train_labels = train_gen.classes
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = dict(enumerate(class_weights_array))

    print(f"✅ Class Weights Computed: {class_weights}")
    print(f"   (Class 0: Normal, Class 1: Pneumonia)")

    print("\n[Model] Building ResNet50 architecture...")
    model = model_builder.build_resnet50_model(input_shape=IMG_SIZE + (3,))

    print("[Model] Compiling with Optimizer=Adam, Loss=BinaryCrossentropy...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

    if not os.path.exists('../models'):
        os.makedirs('../models')

    # Callbacks
    callbacks = [
        EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
    ]

    # Train the model
    print(f"\n[Training] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print(f"\n✅ Training Finished. Model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
