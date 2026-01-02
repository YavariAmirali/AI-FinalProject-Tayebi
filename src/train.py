import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint

# --- تلاش برای ایمپورت کردن کدهای هم‌گروهی‌ها ---
try:
    # طبق دستور عکس: تابع build_baseline_model باید از نفر دوم ایمپورت شود
    from model import build_baseline_model
except ImportError:
    # اگر هم‌گروهی هنوز کد را نپوش نکرده، این مدل موقت اجرا می‌شود تا کار تو نخوابد
    print("⚠️ Warning: 'src/model.py' not found. Using dummy model for DevOps test.")
    from tensorflow.keras import layers, models
    def build_baseline_model():
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid') # خروجی ۱ برای باینری
        ])
        return model

# --- تنظیمات (Configurations) ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
# طبق دستور عکس: برای اسموک تست فقط ۱ یا ۲ اپوک کافی است
EPOCHS = 2 
DATA_DIR = "dataset/raw" 
MODEL_SAVE_PATH = "models/baseline_model.h5"

def main():
    print("🚀 Starting Smoke Test Pipeline...")

    # 1. چک کردن GPU
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU detected.")
    else:
        print("⚠️ Running on CPU.")

    # 2. لود کردن دیتاها (طبق دستور عکس: استفاده از image_dataset_from_directory)
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset directory '{DATA_DIR}' not found.")
        return

    print("📂 Loading dataset...")
    # نکته فنی: چون لاس فانکشن binary_crossentropy است، label_mode باید 'binary' باشد
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

    # طبق دستور عکس: اعمال تابع preprocessing روی داده‌ها
    # نکته: معمولاً در مدل‌های مدرن preprocessing لایه اول مدل است، 
    # اما اگر تابع جدا دارید، اینجا استانداردسازی (Rescaling) را انجام می‌دهیم:
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 3. ساخت و کامپایل مدل
    print("🏗️ Building model...")
    model = build_baseline_model()
    
    # طبق دستور عکس: کامپایل با adam و binary_crossentropy
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. اجرای آموزش (Smoke Test)
    print("🔥 Starting training (Smoke Test)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 5. ذخیره مدل (طبق دستور عکس: فرمت .h5 در پوشه models)
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save(MODEL_SAVE_PATH)
    print(f"💾 Model saved successfully at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
