"""Stratified 70/15/15 split of data/unified/ into data/splits/{train,val,test}/<class>/.

Usage: python3 src/data_prep/split_dataset.py
"""
import random
import shutil
from pathlib import Path

from class_mapping import UNIFIED_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining ~0.15 goes to test
SEED = 42


def main():
    random.seed(SEED)
    if SPLITS_DIR.exists():
        shutil.rmtree(SPLITS_DIR)

    for split in ("train", "val", "test"):
        for cls in UNIFIED_CLASSES:
            (SPLITS_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {}
    for cls in UNIFIED_CLASSES:
        paths = sorted((UNIFIED_DIR / cls).iterdir())
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        splits = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:],
        }
        for split, split_paths in splits.items():
            for p in split_paths:
                shutil.copy2(p, SPLITS_DIR / split / cls / p.name)
        summary[cls] = {k: len(v) for k, v in splits.items()}

    print(f"{'class':12s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for cls in UNIFIED_CLASSES:
        s = summary[cls]
        print(f"{cls:12s} {s['train']:6d} {s['val']:6d} {s['test']:6d}")


if __name__ == "__main__":
    main()
