import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import os
import pydicom

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, 'models', 'best_resnet_model_v2.h5')


# --------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------------------

def process_image_for_model(pil_image):
    width, height = pil_image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2

    img_cropped = pil_image.crop((left, top, right, bottom))

    # 2. Resize
    img_resized = img_cropped.resize((224, 224))

    # 3. Convert to Array & Normalize (CRITICAL FIX: Divide by 255.0 only)
    img_array = np.array(img_resized)

    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Convert to float and Normalize
    img_normalized = img_array.astype(np.float32) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_cropped, img_batch


def find_last_conv_layer(model):
    """
    Search for the last convolutional layer in the model automatically.
    Works for both Simple CNN and ResNet/VGG.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        if isinstance(layer, tf.keras.Model):
            return find_last_conv_layer(layer)

    if "resnet" in model.name.lower(): return "conv5_block3_out"
    if "vgg" in model.name.lower(): return "block5_conv3"

    raise ValueError("Could not auto-detect a Convolutional layer. Please specify 'last_conv_layer_name'.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Generates Grad-CAM heatmap for ANY model (Universal Version).
    """
    # 1. Auto-detect the target layer if not provided
    if last_conv_layer_name is None:
        try:
            last_conv_layer_name = find_last_conv_layer(model)
            # print(f"DEBUG: Using layer '{last_conv_layer_name}' for Grad-CAM") # Uncomment for debugging
        except ValueError as e:
            st.error(f"Grad-CAM Error: {e}")
            return None

    # 2. Create Grad-Model (Safe Method)
    # Instead of reconstructing the model manually, we access the layer directly.
    # This avoids the "Graph Disconnected" and "Rank 2 vs Rank 4" errors.
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
    except Exception as e:
        # If accessing by name fails (e.g. nested model), usually passing the layer object works
        st.warning(f"Complex model structure detected. Grad-CAM might vary. Error: {e}")
        return None

    # 3. Compute Gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        # Handle binary vs multi-class output
        if preds.shape[-1] == 1:
            # Binary classification (Sigmoid)
            class_channel = preds[0][0]
        else:
            # Multi-class (Softmax) - take the winning class
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

    # 4. Process Gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    # 5. Generate Heatmap
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def apply_heatmap_overlay(original_img_pil, heatmap, alpha=0.4):
    """
    Overlays heatmap on image with CORRECT colors (Red=Hot).
    """
    img = np.array(original_img_pil)
    # Ensure RGB
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap_color * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img


@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------------------------------
# 3. UI
# --------------------------------------------------------------------------
model = load_prediction_model()

st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg", "dcm"])
threshold = 0.95

st.title("🫁 Intelligent Pneumonia Detection")

if uploaded_file and model:
    # --- 1. Load Image & Extract DICOM Metadata ---
    if uploaded_file.name.lower().endswith('.dcm'):
        ds = pydicom.dcmread(uploaded_file)
        pixel_array = ds.pixel_array

        # Normalize DICOM to 0-255 for display
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
        pil_image = Image.fromarray(pixel_array.astype('uint8')).convert('RGB')

        # --- Enhanced DICOM Sidebar Info ---
        st.sidebar.markdown("## 📋 DICOM Metadata")


        # Helper function to safely get DICOM tags
        def get_dcm_tag(tag, default="N/A"):
            val = getattr(ds, tag, default)
            return str(val) if val else default


        # Group 1: Patient Information
        with st.sidebar.expander("👤 Patient Information", expanded=True):
            st.markdown(f"**Name:** {get_dcm_tag('PatientName')}")
            st.markdown(f"**ID:** {get_dcm_tag('PatientID')}")
            st.markdown(f"**Sex:** {get_dcm_tag('PatientSex')}")
            st.markdown(f"**Age:** {get_dcm_tag('PatientAge')}")
            st.markdown(f"**Birth Date:** {get_dcm_tag('PatientBirthDate')}")

        # Group 2: Study/Scan Information
        with st.sidebar.expander("☢️ Scan Details", expanded=True):
            st.markdown(f"**Modality:** {get_dcm_tag('Modality')}")
            st.markdown(f"**Body Part:** {get_dcm_tag('BodyPartExamined')}")
            st.markdown(f"**View Position:** {get_dcm_tag('ViewPosition')}")  # e.g., PA or AP
            st.markdown(f"**Study Date:** {get_dcm_tag('StudyDate')}")
            st.markdown(f"**Description:** {get_dcm_tag('StudyDescription')}")

        # Group 3: Image Technicals
        with st.sidebar.expander("⚙️ Technical Specs", expanded=False):
            st.markdown(f"**Rows/Cols:** {ds.Rows} x {ds.Columns}")
            st.markdown(f"**Photometric:** {get_dcm_tag('PhotometricInterpretation')}")

    else:
        # Standard Image (JPG/PNG)
        pil_image = Image.open(uploaded_file).convert('RGB')
        st.sidebar.info("Standard Image Format (No DICOM tags available)")

    # --- 2. Process (Crop & Preprocess) ---
    # This function handles the Center Crop and normalization
    img_display, img_batch = process_image_for_model(pil_image)

    # --- 3. Predict ---
    preds = model.predict(img_batch)
    score = preds[0][0]

    # --- 4. Grad-CAM ---
    heatmap = make_gradcam_heatmap(img_batch, model)

    # --- 5. Display ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Processed Input")
        st.image(img_display, caption="Center Cropped (Input to Model)", use_column_width=True)

    with col2:
        st.subheader("AI Attention Map")
        if heatmap is not None:
            overlay = apply_heatmap_overlay(img_display, heatmap)
            st.image(overlay, caption="Red = High Attention", use_column_width=True)
        else:
            st.warning("Could not generate heatmap.")

    # --- 6. Diagnosis Box ---
    is_pneumonia = score > threshold
    label = "PNEUMONIA" if is_pneumonia else "NORMAL"
    color = "#ff4b4b" if is_pneumonia else "#09ab3b"  # Red vs Green

    st.markdown(f"""
    <div style="background-color:{color}20; padding:20px; border-radius:10px; border: 2px solid {color}; text-align:center; margin-top:20px;">
        <h1 style="color:{color}; margin:0;">{label}</h1>
        <h3 style="margin:10px 0; color: #333;">Confidence: {score:.4f}</h3>
        <p style="color: #666;">Threshold: {threshold}</p>
    </div>
    """, unsafe_allow_html=True)

