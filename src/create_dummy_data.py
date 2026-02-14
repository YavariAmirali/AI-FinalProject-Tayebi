import os
import cv2
import numpy as np

structure = [
    'data/train/NORMAL',
    'data/train/PNEUMONIA',
    'data/test/NORMAL',
    'data/test/PNEUMONIA',
    'data/val/NORMAL',
    'data/val/PNEUMONIA'
]

def create_dummy_data():
    print("🛠 Creating dummy data for CI/CD smoke test...")
    
    for folder in structure:
        os.makedirs(folder, exist_ok=True)
        for i in range(2):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(folder, f'dummy_{i}.jpg'), img)
            
    print("✅ Dummy data created successfully at data/")

if __name__ == "__main__":
    create_dummy_data()

