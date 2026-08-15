"""Picks the model with highest macro-F1 from reports/evaluation/comparison_table.csv
and copies it (+ its label map) into models/best_model/.

Usage: python3 src/evaluation/select_best_model.py
"""
import json
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
COMPARISON_CSV = PROJECT_ROOT / "reports" / "evaluation" / "comparison_table.csv"


def main():
    df = pd.read_csv(COMPARISON_CSV)
    best_row = df.sort_values("macro_f1", ascending=False).iloc[0]
    best_model_key = best_row["model"]
    print(f"Best model: {best_model_key} (macro_f1={best_row['macro_f1']:.4f}, "
          f"accuracy={best_row['accuracy']:.4f})")

    src_dir = MODELS_DIR / best_model_key
    dest_dir = MODELS_DIR / "best_model"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    shutil.copy2(src_dir / "model.keras", dest_dir / "model.keras")
    shutil.copy2(src_dir / "class_names.json", dest_dir / "class_names.json")

    with open(dest_dir / "class_names.json") as f:
        class_names = json.load(f)
    label_map = {i: name for i, name in enumerate(class_names)}
    with open(dest_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    with open(dest_dir / "model_info.json", "w") as f:
        json.dump({
            "model_key": best_model_key,
            "macro_f1": float(best_row["macro_f1"]),
            "accuracy": float(best_row["accuracy"]),
        }, f, indent=2)

    print(f"Copied best model into {dest_dir}")


if __name__ == "__main__":
    main()
