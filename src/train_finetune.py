import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import datetime
from sklearn.utils import class_weight

# Import your modules
import model_builder
from data_loader import get_data_loaders, DATA_DIR

# --- CONFIGURATION FOR FINE-TUNING ---
BATCH_SIZE = 32
EPOCHS = 10  # Fewer epochs for fine-tuning [cite: 453]
FINE_TUNE_LR = 1e-5  # CRITICAL: Very low learning rate

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOAD_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet_model.h5')
SAVE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'finetuned_resnet.h5') # New name [cite: 453]
LOG_DIR = os.path.join(BASE_DIR, 'logs', 'finetune_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def run_fine_tuning():
    print("=" * 50)
    print(f"🚀 Starting Phase 2: Fine-Tuning Pipeline (Week 7)")
    print("=" * 50)

    # 1. Check if previous model exists
    if not os.path.exists(LOAD_MODEL_PATH):
        print(f"❌ Error: Could not find {LOAD_MODEL_PATH}")
        print("   You must run Week 6 training first.")
        return

    # 2. Load Data
    print(f"[Data] Loading data generators...")
    train_gen, val_gen, test_gen = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Calculate Class Weights (Still needed)
    train_labels = train_gen.labels
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights_array))

    # 3. Load the Week 6 Model
    print(f"[Model] Loading Best Model from Week 6: {LOAD_MODEL_PATH}")
    model = tf.keras.models.load_model(LOAD_MODEL_PATH)

    # 4. Unfreeze layers
    print("[Model] Unfreezing last 20 layers for fine-tuning...")
    model = model_builder.unfreeze_model(model, num_layers_to_unfreeze=30)

    # 5. Re-compile with LOW Learning Rate
    print(f"[Model] Re-compiling with Learning Rate = {FINE_TUNE_LR}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )

    # 6. Callbacks
    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True, verbose=1),
        ModelCheckpoint(SAVE_MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    ]

    # 7. Train (Fine-Tune)
    print(f"\n[Training] Starting Fine-Tuning for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("\n" + "="*50)
    print(f"✅ Fine-Tuning Complete!")
    print(f"✅ New Model Saved to: {SAVE_MODEL_PATH}")
    print("="*50)

if __name__ == "__main__":
    run_fine_tuning()