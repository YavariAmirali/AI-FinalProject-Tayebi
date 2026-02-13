import numpy as np
import tensorflow as tf
import cv2
import os
import pydicom


def make_gradcam_heatmap(img_array, model, pred_index=None):
    # Find ResNet base
    base_model = None
    for layer in model.layers:
        if "resnet50" in layer.name:
            base_model = layer
            break

    # Connect input to last conv layer output AND model output
    head_layers = model.layers[2:]

    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_array)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in head_layers:
            x = layer(x)
        preds = x

        if pred_index is None: pred_index = 0
        class_channel = preds[:, pred_index]

    # Calculate Gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and Normalize
    heatmap = tf.maximum(heatmap, 0)
    if tf.math.reduce_max(heatmap) > 0:
        heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def load_and_preprocess_image(img_path):

    # 1. Load Image
    if img_path.lower().endswith('.dcm'):
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array
        # Normalize 16-bit to 8-bit
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
        img = img.astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. CRITICAL: 10% Crop (Consistency with Training)
    h, w = img.shape[:2]
    crop_fraction = 0.10
    start_y = int(h * crop_fraction)
    end_y = int(h * (1 - crop_fraction))
    start_x = int(w * crop_fraction)
    end_x = int(w * (1 - crop_fraction))
    img = img[start_y:end_y, start_x:end_x]

    return img


if __name__ == "__main__":
    # Settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'finetuned_resnet.h5')

    # Example: Test on a Pneumonia case
    TEST_DIR = os.path.join(BASE_DIR, 'data', 'test', 'PNEUMONIA')

    if os.path.exists(TEST_DIR):
        img_name = os.listdir(TEST_DIR)[1]
        img_path = os.path.join(TEST_DIR, img_name)

        print(f"Loading Model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)

        # Load & Preprocess
        img = load_and_preprocess_image(img_path)
        img_resized = cv2.resize(img, (224, 224))
        img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_resized.astype(float))
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # Generate Heatmap
        print(f"Generating heatmap for {img_name}...")
        heatmap = make_gradcam_heatmap(img_batch, model)

        # Display/Save
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)
        save_path = os.path.join(BASE_DIR, 'results', 'gradcam_result.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved to {save_path}")