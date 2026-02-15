import os
import shutil
import random

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
LOCK_FILE = os.path.join(DATA_DIR, '.split_fixed')

CLASSES = ['NORMAL', 'PNEUMONIA']
SPLIT_RATIO = 0.10


def fix_split():
    # CHECK 1: Lock File Check
    if os.path.exists(LOCK_FILE):
        print("🛑 SPLIT ALREADY FIXED! The lock file exists.")
        return

    print(f"🔧 Starting Validation Split Fix...")
    total_moved = 0

    for class_name in CLASSES:
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)

        images = os.listdir(train_class_dir)
        num_images = len(images)

        val_images_count = len(os.listdir(val_class_dir))
        if val_images_count > 50:
            print(f"⚠️ Warning: {class_name} val folder has {val_images_count} images. Skipping.")
            continue

        num_to_move = int(num_images * SPLIT_RATIO)

        random.seed(42)
        random.shuffle(images)
        images_to_move = images[:num_to_move]

        for img in images_to_move:
            src = os.path.join(train_class_dir, img)
            dst = os.path.join(val_class_dir, img)

            if not os.path.exists(dst):
                shutil.move(src, dst)
                total_moved += 1

    # CREATE LOCK FILE
    if total_moved > 0:
        with open(LOCK_FILE, 'w') as f:
            f.write("Split fixed. Do not delete.")
        print(f"🔒 Lock file created at {LOCK_FILE}")

    print(f"✅ DONE! Moved {total_moved} images total.")


if __name__ == "__main__":
    if os.path.exists(LOCK_FILE):
        print("🛑 This script has already been run. Aborting.")
    else:
        confirm = input("⚠️ Move files? (yes/no): ")
        if confirm.lower() == "yes":
            fix_split()