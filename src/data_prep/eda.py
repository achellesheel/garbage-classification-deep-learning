"""EDA on data/unified/: class distribution, sample grid, pixel-intensity histograms.

Usage: python3 src/data_prep/eda.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from class_mapping import UNIFIED_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS_FOR_STATS = 150


def class_image_paths(cls):
    return sorted((UNIFIED_DIR / cls).iterdir())


def plot_class_distribution():
    counts = [len(class_image_paths(cls)) for cls in UNIFIED_CLASSES]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(UNIFIED_CLASSES, counts, color="#4C72B0")
    plt.title("Images per class (unified dataset)")
    plt.ylabel("Image count")
    plt.xticks(rotation=30)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = REPORTS_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_sample_grid(n_per_class=5):
    fig, axes = plt.subplots(len(UNIFIED_CLASSES), n_per_class,
                              figsize=(n_per_class * 2, len(UNIFIED_CLASSES) * 2))
    for row, cls in enumerate(UNIFIED_CLASSES):
        paths = class_image_paths(cls)[:n_per_class]
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(paths):
                img = Image.open(paths[col]).convert("RGB")
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(cls, fontsize=10)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
    plt.tight_layout()
    out = REPORTS_DIR / "sample_grid.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_pixel_intensity():
    fig, axes = plt.subplots(1, len(UNIFIED_CLASSES), figsize=(len(UNIFIED_CLASSES) * 3, 3), sharey=True)
    for ax, cls in zip(axes, UNIFIED_CLASSES):
        paths = class_image_paths(cls)[:SAMPLES_PER_CLASS_FOR_STATS]
        means = {"R": [], "G": [], "B": []}
        for p in paths:
            arr = np.asarray(Image.open(p).convert("RGB").resize((64, 64)), dtype=np.float32)
            means["R"].append(arr[:, :, 0].mean())
            means["G"].append(arr[:, :, 1].mean())
            means["B"].append(arr[:, :, 2].mean())
        for channel, color in zip("RGB", ["red", "green", "blue"]):
            ax.hist(means[channel], bins=20, alpha=0.5, color=color, label=channel)
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("mean channel intensity")
    axes[0].set_ylabel("image count")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    out = REPORTS_DIR / "pixel_intensity.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    plot_class_distribution()
    plot_sample_grid()
    plot_pixel_intensity()


if __name__ == "__main__":
    main()
