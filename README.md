# Pulak Streamlit App

A minimal Streamlit app to run an image classifier using `MobileNetV2_best_model.h5`.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

1. Push this folder to a GitHub repo (include `app.py`, `requirements.txt`, and `MobileNetV2_best_model.h5`).
2. Create a new Streamlit Cloud app pointing to the repo and `app.py`.
3. Optional: Provide class labels via the sidebar text area (one per line). If omitted, generic names like `class_0` will be used.

## Notes
- The app expects `MobileNetV2_best_model.h5` in the project root.
- Images are resized to 224x224 and preprocessed with MobileNetV2.
- If the model outputs logits, the app normalizes them with softmax.
