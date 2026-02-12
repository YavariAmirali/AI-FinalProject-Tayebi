import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pydicom
import matplotlib.cm as cm
import os

# ==========================================
# 1.PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="Pneumonia Diagnosis (ResNet)",
    page_icon="🫁",
    layout="wide"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, 'models', 'finetuned_resnet.h5')


# ==========================================
# 2.FUNCTIONS
# ==========================================

@st.cache_resource
def load_model_clean():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_standard(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ==========================================
# 3.CALCULATING DICOM
# ==========================================

def process_dicom(file):
    try:
        ds = pydicom.dcmread(file)
        pixel_array = ds.pixel_array.astype(float)
        scaled_image = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        scaled_image = np.uint8(scaled_image)

        img_pil = Image.fromarray(scaled_image).convert('RGB')
        return img_pil, ds
    except Exception as e:
        st.error(f"Error reading DICOM: {e}")
        return None, None


def get_dicom_info(ds):
    data = {}
    tags = ['PatientID', 'PatientName', 'PatientSex', 'Age', 'Modality', 'BodyPartExamined']
    for tag in tags:
        if hasattr(ds, tag):
            data[tag] = str(getattr(ds, tag))
    return pd.DataFrame.from_dict(data, orient='index', columns=['Value'])


# ==========================================
# 4. Grad-CAM (ResNet)
# ==========================================

def get_gradcam(model, img_array, layer_name="conv5_block3_out"):
    target_layer = None
    resnet_base = None

    try:
        target_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [target_layer.output, model.output]
        )
    except:
        for layer in model.layers:
            if 'resnet' in layer.name.lower():
                resnet_base = layer
                break

        if resnet_base:
            try:
                target_layer = resnet_base.get_layer(layer_name)
                grad_model = tf.keras.models.Model(
                    [resnet_base.inputs],
                    [target_layer.output, resnet_base.output]
                )
            except:
                return np.zeros((224, 224))
        else:
            return np.zeros((224, 224))

    # GRADIENT EVALUATE
    with tf.GradientTape() as tape:
        if resnet_base:
            inputs = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
        else:
            conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def apply_heatmap(image, heatmap):
    img = np.array(image.resize((224, 224)))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_img)


# ==========================================
# 5.Main
# ==========================================

def main():
    st.title("🫁 AI Detection Pneumonia")

    # SIDEBAR
    with st.sidebar:
        st.header("Upload Info")
        dicom_container = st.container()

    uploaded_file = st.file_uploader("Upload X-Ray (JPG, PNG, DCM)", type=['jpg', 'png', 'jpeg', 'dcm'])

    if uploaded_file:
        model = load_model_clean()
        if not model:
            st.stop()

        file_ext = uploaded_file.name.split('.')[-1].lower()
        image = None

        if file_ext == 'dcm':
            image, ds = process_dicom(uploaded_file)
            if ds:
                dicom_container.write("📋 DICOM Metadata:")
                dicom_container.dataframe(get_dicom_info(ds))
        else:
            image = Image.open(uploaded_file)
            dicom_container.info("Standard Image (No Metadata)")

        if image:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input Image")
                st.image(image, use_column_width=True)

            processed_img = preprocess_standard(image)

            with st.spinner("Analyzing with ResNet50..."):
                prediction = model.predict(processed_img, verbose=0)
                score = prediction[0][0]

                # Grad-CAM
                try:
                    heatmap = get_gradcam(model, processed_img)
                    if np.max(heatmap) > 0:
                        cam_img = apply_heatmap(image, heatmap)
                        with col2:
                            st.subheader("Grad-CAM Analysis")
                            st.image(cam_img, use_column_width=True, caption="Model Attention")
                except Exception as e:
                    print(f"GradCAM Error: {e}")

            st.divider()
            st.markdown("### Diagnosis Result")

            if score > 0.8:
                st.error(f"## 🦠 PNEUMONIA DETECTED")
                st.write(f"Confidence: **{score * 100:.2f}%**")
            else:
                st.success(f"## ✅ NORMAL")
                st.write(f"Confidence (Health): **{(1 - score) * 100:.2f}%**")

            st.progress(int(score * 100))


if __name__ == "__main__":
    main()

