import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pydicom
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils import crop_center_10_percent
from explainability import make_gradcam_heatmap

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

def preprocess_image(img_array):
    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    if img_array.shape[-1] == 4:  # Drop Alpha channel if PNG
        img_array = img_array[..., :3]

    img_cropped = crop_center_10_percent(img_array)

    # 2. Resize to 224x224 (ResNet Standard)
    img_resized = cv2.resize(img_cropped, (224, 224))

    # 3. Model Input Preparation (ResNet Specific)
    img_preprocessed = preprocess_input(img_resized.copy())
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Return display image (cropped) and model input
    return img_resized, img_batch


def load_dicom_file(uploaded_file):
    ds = pydicom.dcmread(uploaded_file)
    pixel_array = ds.pixel_array

    # Normalize 16-bit to 8-bit for display
    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
    img_array = pixel_array.astype(np.uint8)

    # Metadata extraction
    metadata = {
        "ID": getattr(ds, 'PatientID', 'N/A'),
        "Sex": getattr(ds, 'PatientSex', 'N/A'),
        "Modality": getattr(ds, 'Modality', 'N/A')
    }
    return img_array, metadata


def apply_heatmap_overlay(original_img_array, heatmap, alpha=0.4):
    # Resize heatmap to match image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))

    # Apply JET colormap (Returns BGR)
    heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert BGR to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Superimpose
    superimposed_img = heatmap_rgb * alpha + original_img_array * (1 - alpha)
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

# --- SLIDER ---
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 0.99, 0.99)

st.title("🫁 Intelligent Pneumonia Detection System")

if uploaded_file and model:
    # --- 1. LOAD IMAGE ---
    if uploaded_file.name.lower().endswith('.dcm'):
        img_array, metadata = load_dicom_file(uploaded_file)
        st.sidebar.success("✅ DICOM Metadata Extracted")
        with st.sidebar.expander("Patient Details"):
            st.write(metadata)
    else:
        # Standard Image Loading
        pil_image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(pil_image)

    # --- 2. PREPROCESS ---
    # Now using the unified logic that matches explainability.py
    img_display, img_batch = preprocess_image(img_array)

    # --- 3. PREDICT ---
    preds = model.predict(img_batch)
    score = preds[0][0]

    # --- 4. EXPLAIN (Grad-CAM) ---
    heatmap = make_gradcam_heatmap(img_batch, model)

    if heatmap is not None:
        overlay = apply_heatmap_overlay(img_display, heatmap)
    else:
        st.warning("Could not generate heatmap.")
        overlay = img_display

    # --- 5. DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Analyzed Region")
        st.image(img_display, caption="Cropped Input (224x224)", use_column_width=True)
    with col2:
        st.subheader("AI Attention Map")
        st.image(overlay, caption="Red = Suspicious Areas", use_column_width=True)

    # --- 6. DIAGNOSIS BOX ---
    is_pneumonia = score > threshold
    label = "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL"
    color = "#ff4b4b" if is_pneumonia else "#09ab3b"

    st.markdown(f"""
    <div style="background-color:{color}20; border: 2px solid {color}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
        <h2 style="color:{color}; margin:0;">{label}</h2>
        <p style="margin-top:10px; font-size:18px;">Confidence: <b>{score:.2%}</b></p>
    </div>
    """, unsafe_allow_html=True)