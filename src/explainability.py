import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    """
    # 1. We need to access the inner ResNet50 model because our main model wraps it.
    # Find the nested 'resnet50' layer
    base_model = None
    for layer in model.layers:
        if "resnet50" in layer.name:
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find resnet50 base layer in the model!")

    # 2. Create a model that maps the input image to the activations of the last conv layer
    #    AND the output predictions.
    #    We create a new model using the Functional API inputs/outputs.
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [base_model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 3. Compute the Gradient
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel in the feature map array
    #    by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU (we only care about positive influence)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Superimposes the heatmap on the original image.
    """
    # Load the original image
    img = cv2.imread(img_path)

    # Resize heatmap to be the same size as the original image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Use jet colormap to colorize heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img

    # Save
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path


# --- TEST FUNCTION (Run this to verify it works) ---
if __name__ == "__main__":
    # Settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'finetuned_resnet.h5')  # Use the fine-tuned model

    # Pick a random Pneumonia image from test set to test
    TEST_IMG_DIR = os.path.join(BASE_DIR, 'data', 'test', 'PNEUMONIA')

    if not os.path.exists(MODEL_PATH):
        print("⚠️ Fine-tuned model not found. Using best_resnet_model.h5 instead.")
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet_model.h5')

    if os.path.exists(TEST_IMG_DIR) and len(os.listdir(TEST_IMG_DIR)) > 0:
        img_name = os.listdir(TEST_IMG_DIR)[0]
        img_path = os.path.join(TEST_IMG_DIR, img_name)

        print(f"Loading Model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)

        # Preprocess Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = tf.keras.applications.resnet50.preprocess_input(img)
        img_array = np.expand_dims(img, axis=0)

        # "conv5_block3_out" is the standard last layer name in ResNet50
        print("Generating Grad-CAM...")
        heatmap = make_gradcam_heatmap(img_array, model, "conv5_block3_out")

        save_path = os.path.join(BASE_DIR, 'results', 'gradcam_test.jpg')
        save_and_display_gradcam(img_path, heatmap, save_path)
        print(f"✅ Grad-CAM saved to {save_path}")
    else:
        print("❌ Could not find a test image to run Grad-CAM demo.")

