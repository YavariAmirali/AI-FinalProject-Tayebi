import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pydicom
from tensorflow.keras.applications.resnet50 import preprocess_input

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, 'models', 'finetuned_resnet.h5')


# --------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------------------

def process_image_for_model(pil_image):
    """
    Prepares the image exactly as the model saw it during training.
    """
    # Convert PIL to Numpy Array
    img_array = np.array(pil_image)

    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    if img_array.shape[-1] == 4:  # Drop Alpha channel if PNG
        img_array = img_array[..., :3]

    # 10% Border Crop (Mitigate Shortcut Learning)
    # This removes hospital markers/text that confuse the model
    h, w = img_array.shape[:2]
    crop_fraction = 0.10
    start_y = int(h * crop_fraction)
    end_y = int(h * (1 - crop_fraction))
    start_x = int(w * crop_fraction)
    end_x = int(w * (1 - crop_fraction))

    img_cropped = img_array[start_y:end_y, start_x:end_x]

    # Resize to 224x224 (ResNet Standard)
    img_resized = cv2.resize(img_cropped, (224, 224))

    # ResNet Specific Preprocessing (Zero-centers the data)
    # Note: Do NOT divide by 255.0 manually; this function handles scaling.
    img_preprocessed = preprocess_input(img_resized.copy())

    # Add Batch Dimension
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Return display image (cropped) and model input
    img_display = Image.fromarray(img_resized.astype('uint8'))
    return img_display, img_batch


def make_gradcam_heatmap(img_array, model):
    """
    Robust Grad-CAM generator for Nested ResNet Models.
    Fixes 'Invalid reduction dimension' error.
    """
    # 1. Find the ResNet backbone layer inside the model
    backbone = None
    backbone_idx = 0
    for idx, layer in enumerate(model.layers):
        if "resnet50" in layer.name:
            backbone = layer
            backbone_idx = idx
            break

    if backbone is None:
        return None

    # 2. Identify Head Layers (Everything after the backbone)
    head_layers = model.layers[backbone_idx + 1:]

    # 3. Compute Gradients
    with tf.GradientTape() as tape:
        # Get feature maps from the backbone (4D tensor)
        conv_outputs = backbone(img_array)
        tape.watch(conv_outputs)

        # Pass features through the rest of the model (Head)
        x = conv_outputs
        for layer in head_layers:
            x = layer(x)
        preds = x

        class_channel = preds[:, 0]

    # 4. Process Gradients
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


def apply_heatmap_overlay(original_img_pil, heatmap, alpha=0.4):
    """
    Overlays heatmap on image.
    CRITICAL FIX: Converts OpenCV BGR to RGB to prevent weird colors.
    """
    img = np.array(original_img_pil)

    # Resize heatmap to match image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply JET colormap (Returns BGR)
    heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # --- FIX: Convert BGR to RGB ---
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Superimpose
    superimposed_img = heatmap_rgb * alpha + img * (1 - alpha)
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)


@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------------------------------
# 3. UI LOGIC
# --------------------------------------------------------------------------
model = load_prediction_model()

st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload X-Ray (JPG/PNG/DICOM)", type=["jpg", "png", "jpeg", "dcm"])

# --- SLIDER (Max set to 0.99 as requested) ---
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 0.99, 0.50)

st.title("🫁 Intelligent Pneumonia Detection System")

if uploaded_file and model:
    # --- DICOM & IMAGE LOADING ---
    if uploaded_file.name.lower().endswith('.dcm'):
        ds = pydicom.dcmread(uploaded_file)
        pixel_array = ds.pixel_array

        # Normalize DICOM (16-bit to 8-bit for display)
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
        pil_image = Image.fromarray(pixel_array.astype('uint8')).convert('RGB')

        st.sidebar.success("✅ DICOM Metadata Extracted")
        with st.sidebar.expander("Patient Details"):
            st.write(f"ID: {getattr(ds, 'PatientID', 'N/A')}")
            st.write(f"Sex: {getattr(ds, 'PatientSex', 'N/A')}")
            st.write(f"Modality: {getattr(ds, 'Modality', 'N/A')}")
    else:
        pil_image = Image.open(uploaded_file).convert('RGB')

    # --- PROCESS & PREDICT ---
    img_display, img_batch = process_image_for_model(pil_image)

    preds = model.predict(img_batch)
    score = preds[0][0]

    # --- GRAD-CAM ---
    heatmap = make_gradcam_heatmap(img_batch, model)

    if heatmap is not None:
        overlay = apply_heatmap_overlay(img_display, heatmap)
    else:
        st.warning("Could not generate heatmap.")
        overlay = np.array(img_display)

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Analyzed Region")
        st.image(img_display, caption="Cropped Input (Grayscale)", use_column_width=True)
    with col2:
        st.subheader("AI Attention Map")
        st.image(overlay, caption="Red = Suspicious Areas", use_column_width=True)

    # --- DIAGNOSIS ---
    is_pneumonia = score > threshold
    label = "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL"
    color = "#ff4b4b" if is_pneumonia else "#09ab3b"

    st.markdown(f"""
    <div style="background-color:{color}20; border: 2px solid {color}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
        <h2 style="color:{color}; margin:0;">{label}</h2>
        <p style="margin-top:10px; font-size:18px;">Confidence: <b>{score:.2%}</b></p>
    </div>
    """, unsafe_allow_html=True)