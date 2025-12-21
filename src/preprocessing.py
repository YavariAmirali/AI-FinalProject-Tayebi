import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = cv2.imread(image_path)
        
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        
        return img
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
