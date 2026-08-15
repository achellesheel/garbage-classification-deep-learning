"""Evaluates one or more trained models on the test split: metrics, confusion matrix,
misclassified examples, and a cross-model comparison table.

Usage: python3 src/evaluation/evaluate.py [model_key ...]   (default: all trained models found)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))
from datasets import get_datasets  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "evaluation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODEL_KEYS = ["baseline_cnn", "mobilenetv2", "efficientnetb0"]


def evaluate_model(model_key):
    model_dir = MODELS_DIR / model_key
    model_path = model_dir / "model.keras"
    if not model_path.exists():
        print(f"Skipping {model_key}: no trained model at {model_path}")
        return None

    model = tf.keras.models.load_model(model_path)
    with open(model_dir / "class_names.json") as f:
        class_names = json.load(f)

    _, _, test_ds, ds_class_names = get_datasets(model_key)
    assert ds_class_names == class_names, f"class order mismatch for {model_key}"

    y_true, y_pred_probs = [], []
    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred_probs.extend(probs.tolist())
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = y_pred_probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    per_class_df = pd.DataFrame({
        "class": class_names, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    })
    per_class_df.to_csv(REPORTS_DIR / f"per_class_metrics_{model_key}.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_key}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{model_key}.png", dpi=150)
    plt.close()

    print(f"{model_key}: accuracy={acc:.4f} macro_f1={macro_f1:.4f}")

    return {
        "model": model_key,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "n_test": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_keys", nargs="*", default=None)
    args = parser.parse_args()
    model_keys = args.model_keys or ALL_MODEL_KEYS

    results = []
    for key in model_keys:
        r = evaluate_model(key)
        if r is not None:
            results.append(r)

    if not results:
        print("No models evaluated.")
        return

    comparison_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    comparison_df.to_csv(REPORTS_DIR / "comparison_table.csv", index=False)
    print("\n=== Comparison table (sorted by macro F1) ===")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
