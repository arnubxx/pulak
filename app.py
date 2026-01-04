import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Pulak Image Classifier", page_icon="🧠", layout="centered")

# Fixed class names provided by user
CLASS_NAMES = [
    "Leaf_Spot",
    "Abnormal",
    "Anthracnose",
    "Leaf_Blight",
    "Powdery_Mildew",
    "Dry_Leaf",
    "Healthy",
    "Black_Spot",
]

@st.cache_resource
def load_model(model_path: str = "MobileNetV2_best_model.h5"):
    # compile=False avoids needing optimizer/loss during load and helps
    # compatibility across TF/Keras versions.
    return tf.keras.models.load_model(model_path, compile=False)

def get_labels(num_classes: int):
    if len(CLASS_NAMES) == num_classes:
        return CLASS_NAMES
    return [f"class_{i}" for i in range(num_classes)]

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
        labels = get_labels(n)
        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        confidence = float(probs[top_idx])

        # Show only the predicted class name (no percentages/graphs)
        st.write(top_label)

    st.caption("Powered by MobileNetV2")

if __name__ == "__main__":
    main()
