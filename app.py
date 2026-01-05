import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Classes fixed as requested
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
def load_model(model_path: str = "DenseNet201_full_model.h5"):
    return tf.keras.models.load_model(model_path, compile=False)

def get_labels(n: int):
    return CLASS_NAMES if len(CLASS_NAMES) == n else [f"class_{i}" for i in range(n)]

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.densenet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(img: Image.Image, model) -> np.ndarray:
    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]
    # Ensure probabilities
    if np.any(preds < 0) or np.any(preds > 1) or not np.isclose(preds.sum(), 1.0):
        preds = tf.nn.softmax(preds).numpy()
    return preds

def main():
    model_path = "DenseNet201_full_model.h5"
    if not os.path.exists(model_path):
        st.write("Model file not found: DenseNet201_full_model.h5")
        return

    model = load_model(model_path)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False)
    if uploaded:
        img = Image.open(uploaded)
        probs = predict(img, model)
        idx = int(np.argmax(probs))
        labels = get_labels(len(probs))
        st.write(labels[idx])

if __name__ == "__main__":
    main()
