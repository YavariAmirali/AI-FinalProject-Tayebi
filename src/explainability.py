import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os


def make_gradcam_heatmap(img_array, model, pred_index=None):
    base_model = None
    for layer in model.layers:
        if "resnet50" in layer.name:
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find resnet50 base layer in the model!")

    # Extract the classification head
    head_layers = model.layers[2:]

    # Compute Gradients
    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_array)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in head_layers:
            x = layer(x)
        preds = x

        if pred_index is None:
            pred_index = 0
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load and CROP the original image for display to match the model input
    img = cv2.imread(img_path)

    # --- CRITICAL FIX: Match the cropping used in training ---
    h, w = img.shape[:2]
    crop_fraction = 0.10
    start_y = int(h * crop_fraction)
    end_y = int(h * (1 - crop_fraction))
    start_x = int(w * crop_fraction)
    end_x = int(w * (1 - crop_fraction))
    img = img[start_y:end_y, start_x:end_x]
    # ---------------------------------------------------------

    # Resize heatmap to be the same size as the cropped image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img

    cv2.imwrite(cam_path, superimposed_img)
    return cam_path


if __name__ == "__main__":
    # Settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'finetuned_resnet.h5')

    # Let's pick a PNEUMONIA image to see the disease pattern
    TEST_IMG_DIR = os.path.join(BASE_DIR, 'data', 'test', 'PNEUMONIA')

    if not os.path.exists(MODEL_PATH):
        print("⚠️ Fine-tuned model not found. Using best_resnet_model.h5 instead.")
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet_model.h5')

    if os.path.exists(TEST_IMG_DIR) and len(os.listdir(TEST_IMG_DIR)) > 0:
        # Pick a random image or a specific one
        img_name = os.listdir(TEST_IMG_DIR)[0]
        img_path = os.path.join(TEST_IMG_DIR, img_name)

        print(f"Loading Model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)

        # Preprocess Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- CRITICAL FIX: APPLY SAME CROP AS TRAINING ---
        h, w = img.shape[:2]
        crop_fraction = 0.10  # 10% crop
        start_y = int(h * crop_fraction)
        end_y = int(h * (1 - crop_fraction))
        start_x = int(w * crop_fraction)
        end_x = int(w * (1 - crop_fraction))
        img = img[start_y:end_y, start_x:end_x]
        # -------------------------------------------------

        img = cv2.resize(img, (224, 224))
        img = tf.keras.applications.resnet50.preprocess_input(img)
        img_array = np.expand_dims(img, axis=0)

        print(f"Generating Grad-CAM for {img_name}...")

        # Get the prediction score
        preds = model.predict(img_array)
        print(f"Prediction: {preds[0][0]:.4f} (0=Normal, 1=Pneumonia)")

        heatmap = make_gradcam_heatmap(img_array, model)

        save_path = os.path.join(BASE_DIR, 'results', 'gradcam_test.jpg')
        save_and_display_gradcam(img_path, heatmap, save_path)
        print(f"✅ Grad-CAM saved to {save_path}")
    else:
        print("❌ Could not find a test image to run Grad-CAM demo.")