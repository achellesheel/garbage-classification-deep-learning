"""Streamlit app: upload a garbage image, get predicted category + confidence.
Also includes an About tab documenting the data flow, ML approach, and results.

Usage: streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))
from datasets import IMG_SIZE, PREPROCESS_FNS  # noqa: E402

BEST_MODEL_DIR = PROJECT_ROOT / "models" / "best_model"
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
EVAL_DIR = PROJECT_ROOT / "reports" / "evaluation"

MODEL_LABELS = {
    "baseline_cnn": "Baseline CNN (trained from scratch)",
    "mobilenetv2": "MobileNetV2 (transfer learning)",
    "efficientnetb0": "EfficientNetB0 (transfer learning)",
}


@st.cache_resource
def load_model_and_metadata():
    model = tf.keras.models.load_model(BEST_MODEL_DIR / "model.keras")
    with open(BEST_MODEL_DIR / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    with open(BEST_MODEL_DIR / "model_info.json") as f:
        model_info = json.load(f)
    return model, label_map, model_info


def preprocess_image(pil_image, model_key):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    preprocessed = PREPROCESS_FNS[model_key](arr)
    return preprocessed.numpy() if hasattr(preprocessed, "numpy") else preprocessed


def render_classify_tab():
    if not (BEST_MODEL_DIR / "model.keras").exists():
        st.error(
            f"No trained model found at {BEST_MODEL_DIR}. "
            "Run the training + evaluation pipeline first "
            "(src/training/train.py, src/evaluation/evaluate.py, "
            "src/evaluation/select_best_model.py)."
        )
        return

    model, label_map, model_info = load_model_and_metadata()
    st.caption(
        f"Using **{model_info['model_key']}** "
        f"(test accuracy: {model_info['accuracy']:.1%}, macro F1: {model_info['macro_f1']:.3f})"
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is None:
        return

    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded image", width="stretch")

    with st.spinner("Classifying..."):
        x = preprocess_image(pil_image, model_info["model_key"])
        probs = model.predict(x, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:3]
    best_idx = top_indices[0]
    st.subheader(f"Prediction: **{label_map[best_idx]}** ({probs[best_idx]:.1%} confidence)")

    st.write("Top-3 predictions:")
    for idx in top_indices:
        st.progress(float(probs[idx]), text=f"{label_map[idx]} — {probs[idx]:.1%}")


def render_about_tab():
    st.header("Data Flow and Architecture")
    st.markdown(
        """
1. **Raw garbage image datasets** — 3 Kaggle sources (trashnet, garbage-classification
   12-class, garbage-classification-v2) combined and remapped to a shared taxonomy.
2. **Data preprocessing** — dedup by file hash, drop corrupt/unsupported files, resize to
   224×224, per-backbone normalization, on-the-fly augmentation (flip, rotation, zoom,
   contrast) applied only to the training split.
3. **Train/val/test split** — stratified 70/15/15 per class.
4. **Model training** — a from-scratch baseline CNN plus two transfer-learning models
   (MobileNetV2, EfficientNetB0), each with a frozen-backbone phase followed by
   fine-tuning the top ~20% of backbone layers at a lower learning rate.
5. **Model evaluation** — accuracy, precision/recall/F1 (per-class + macro), confusion
   matrices.
6. **Best model selection** — highest macro-F1 model staged for the app.
7. **Streamlit app** — upload an image, get a predicted category + top-3 confidence.
8. **Deployment** — run locally, or on Streamlit Community Cloud.
        """
    )

    st.header("Technical Details")

    st.subheader("Dataset")
    class_counts = {
        cls_dir.name: len(list(cls_dir.iterdir()))
        for cls_dir in sorted(UNIFIED_DIR.iterdir())
    } if UNIFIED_DIR.exists() else {}
    total_images = sum(class_counts.values())
    st.markdown(
        f"""
- **3 datasets merged** into one unified taxonomy of **{len(class_counts)} classes**:
  cardboard, glass, metal, paper, plastic, trash, biological.
- `battery`, `clothes`, `shoes` were dropped — present in only 2 of 3 source datasets and
  outside the stated target categories (plastic/metal/glass/paper/organic).
- Glass sub-colors (green/brown/white-glass) from two sources were collapsed into a
  single `glass` class to match the third source.
- **{total_images:,} images** after deduplication (exact-hash duplicates across sources —
  the two 12-class-style datasets shared thousands of identical photos) and filtering out
  corrupt/mislabeled files (e.g. `.jpg`-named WebP files).
        """
    )
    if class_counts:
        st.bar_chart(pd.Series(class_counts, name="images"))

    st.subheader("Models compared")
    st.markdown(
        """
| Model | Type | Notes |
|---|---|---|
| Baseline CNN | Trained from scratch | 4 Conv2D/MaxPool blocks + dense head — establishes a lower bound |
| MobileNetV2 | Transfer learning (ImageNet) | Lightweight, fast to train on Apple Silicon (no CUDA available) |
| EfficientNetB0 | Transfer learning (ImageNet) | Best accuracy/compute tradeoff of the three |

All models: 224×224 input, `GlobalAveragePooling2D` + `Dropout(0.3)` + softmax head,
`class_weight` applied during training to counter class imbalance (~2.6× between the
largest and smallest classes), `EarlyStopping` + `ReduceLROnPlateau` callbacks, best
checkpoint kept by validation loss.
        """
    )

    st.header("Results & Insights")
    comparison_path = EVAL_DIR / "comparison_table.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path).sort_values("macro_f1", ascending=False)
        df["model"] = df["model"].map(lambda k: MODEL_LABELS.get(k, k))
        display_df = df[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]].copy()
        for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, hide_index=True, width="stretch")
        st.markdown(
            "Both transfer-learning models clear the 85% accuracy target from the "
            "evaluation criteria; the from-scratch baseline CNN falls well short, "
            "confirming ImageNet-pretrained features transfer well to this domain "
            "even with a moderately sized (~11k image), imbalanced dataset."
        )
    else:
        st.info("Run `src/evaluation/evaluate.py` to populate results here.")

    per_class_path = EVAL_DIR / "per_class_metrics_efficientnetb0.csv"
    if per_class_path.exists():
        st.subheader("Best model (EfficientNetB0) — per-class metrics")
        pc_df = pd.read_csv(per_class_path)
        st.dataframe(pc_df, hide_index=True, width="stretch")
        st.markdown(
            """
**Key insights from the confusion matrix and per-class metrics:**
- **Glass** has the highest precision (~97%) but its recall dips to ~87% — most
  misclassifications are clear/translucent glass items predicted as `plastic`.
- **Metal** shows the opposite pattern: highest recall (~93%) but the lowest precision
  (~75%) — the model rarely misses a real metal item, but shiny/reflective plastic or
  other objects are sometimes misclassified *as* metal.
- **Biological** waste is the most reliably classified class (F1 ~0.98) — organic
  matter is visually distinct from the other 6 categories.
- **Paper vs. cardboard** confusion is the next largest source of error, consistent
  with the two materials' visual similarity in some lighting/framing.
            """
        )

    st.header("Deployment Notes")
    st.markdown(
        """
- Run locally: `streamlit run app/streamlit_app.py`.
- For Streamlit Community Cloud (or any non-macOS host), swap `tensorflow-macos` +
  `tensorflow-metal` in `requirements.txt` for plain `tensorflow` — the Metal-accelerated
  packages are Apple-Silicon-only.
        """
    )


def main():
    st.set_page_config(page_title="Garbage Classifier", page_icon="♻️", layout="centered")
    st.title("♻️ Garbage Image Classifier")
    st.write(
        "Upload a photo of a waste item and the model will predict which "
        "recycling category it belongs to: cardboard, glass, metal, paper, "
        "plastic, trash, or biological."
    )

    classify_tab, about_tab = st.tabs(["Classify", "About"])
    with classify_tab:
        render_classify_tab()
    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
