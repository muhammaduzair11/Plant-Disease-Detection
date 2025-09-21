# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import pandas as pd
import os

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="🌱 Plant Disease Detection", layout="wide")
st.title("🌱 Plant Disease Detection Dashboard")


# ---------------------------
# Load Model & Metadata
# ---------------------------
@st.cache_resource
def load_model_and_metadata():
    model = tf.keras.models.load_model("plant_model.h5")
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, class_indices, idx_to_class


model, class_indices, idx_to_class = load_model_and_metadata()


# ---------------------------
# Grad-CAM Functions
# ---------------------------
def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img), 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(superimposed_img)


# ---------------------------
# Upload & Prediction
# ---------------------------
st.sidebar.header("🔍 Make a Prediction")
uploaded = st.sidebar.file_uploader("Upload a leaf image", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    arr = np.expand_dims(preprocess_input(np.array(img)), 0)

    pred = model.predict(arr, verbose=0)[0]
    cls_idx = int(np.argmax(pred))
    cls = idx_to_class[cls_idx].replace("_", " ")

    st.subheader(f"✅ Prediction: **{cls}** ({np.max(pred)*100:.2f}%)")

    # Probability Distribution
    st.markdown("### 📊 Prediction Probabilities")
    probs = {
        idx_to_class[i].replace("_", " "): float(pred[i]) for i in range(len(pred))
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(probs.values()), y=list(probs.keys()), ax=ax, palette="viridis")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Class")
    st.pyplot(fig)

    # Grad-CAM Visualization
    st.subheader("👁️ Grad-CAM Visualization")
    heatmap = get_gradcam_heatmap(
        arr, model, last_conv_layer_name="Conv_1", pred_index=cls_idx
    )
    gradcam_img = overlay_gradcam(img, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Leaf", use_column_width=True)
    with col2:
        st.image(gradcam_img, caption="Grad-CAM", use_column_width=True)

    # Download Grad-CAM
    gradcam_img.save("gradcam_result.jpg")
    with open("gradcam_result.jpg", "rb") as f:
        st.download_button("📥 Download Grad-CAM", f, file_name="gradcam.jpg")

# ---------------------------
# Training Metrics
# ---------------------------
if os.path.exists("metrics.json"):
    st.header("📈 Training Performance")
    with open("metrics.json") as f:
        metrics = json.load(f)
    epochs = range(1, len(metrics["accuracy"]) + 1)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(epochs, metrics["accuracy"], label="Train Acc", marker="o")
        ax.plot(epochs, metrics["val_accuracy"], label="Val Acc", marker="s")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(epochs, metrics["loss"], label="Train Loss", marker="o")
        ax.plot(epochs, metrics["val_loss"], label="Val Loss", marker="s")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

# ---------------------------
# Confusion Matrix
# ---------------------------
if os.path.exists("confusion_matrix.npy"):
    st.header("🧾 Confusion Matrix")
    cm = np.load("confusion_matrix.npy")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_indices.keys()),
        yticklabels=list(class_indices.keys()),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

# ---------------------------
# Classification Report
# ---------------------------
if os.path.exists("report.json"):
    st.header("📊 Classification Report")
    with open("report.json") as f:
        report = json.load(f)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="YlGnBu"))

    if "accuracy" in report:
        st.metric("Overall Accuracy", f"{report['accuracy']*100:.2f}%")
