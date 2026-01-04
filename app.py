import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Pulak Image Classifier", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model(model_path: str = "MobileNetV2_best_model.h5"):
    return tf.keras.models.load_model(model_path)

@st.cache_data
def parse_labels(text: str, num_classes: int):
    labels = [l.strip() for l in text.splitlines() if l.strip()]
    if len(labels) != num_classes:
        labels = [f"class_{i}" for i in range(num_classes)]
    return labels

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def normalize_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits).reshape(-1)
    s = logits.sum()
    if s < 0.99 or s > 1.01 or (logits < 0).any() or (logits > 1).any():
        return tf.nn.softmax(logits).numpy()
    return logits

def predict(img: Image.Image, model) -> np.ndarray:
    inp = preprocess_image(img)
    preds = model.predict(inp, verbose=0)[0]
    probs = normalize_probs(preds)
    return probs

def main():
    st.title("Pulak Image Classifier")

    model_path = "MobileNetV2_best_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return

    with st.spinner("Loading model..."):
        model = load_model(model_path)

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=3)
    label_text = st.sidebar.text_area(
        "Class labels (optional)",
        help="One label per line; leave empty to use generic names.",
    )

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Input Image", use_column_width=True)
        probs = predict(img, model)
        n = len(probs)
        labels = parse_labels(label_text, n)
        k = min(top_k, n)
        idxs = np.argsort(probs)[::-1][:k]

        st.subheader("Predictions")
        rows = [{"label": labels[i], "probability": float(probs[i])} for i in idxs]
        st.dataframe(rows, use_container_width=True)
        st.bar_chart({labels[i]: float(probs[i]) for i in idxs})

    st.caption("Powered by MobileNetV2")

if __name__ == "__main__":
    main()
