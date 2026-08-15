"""tf.data pipelines for train/val/test splits, with per-backbone preprocessing + augmentation."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Each backbone expects its own preprocessing. "none" means plain /255 rescale (baseline CNN).
PREPROCESS_FNS = {
    "baseline_cnn": lambda x: x / 255.0,
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
}

_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
], name="augmentation")


def _load_raw_dataset(split, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        SPLITS_DIR / split,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )
    class_names = ds.class_names
    return ds, class_names


def get_datasets(model_key):
    """Returns (train_ds, val_ds, test_ds, class_names) preprocessed for the given model_key."""
    preprocess = PREPROCESS_FNS[model_key]

    train_ds, class_names = _load_raw_dataset("train", shuffle=True)
    val_ds, _ = _load_raw_dataset("val", shuffle=False)
    test_ds, _ = _load_raw_dataset("test", shuffle=False)

    train_ds = train_ds.map(lambda x, y: (_augmentation(x, training=True), y),
                             num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(class_names):
    """Inverse-frequency class weights computed from the train split's file counts."""
    counts = np.array([
        len(list((SPLITS_DIR / "train" / cls).iterdir())) for cls in class_names
    ], dtype=np.float32)
    total = counts.sum()
    n_classes = len(class_names)
    weights = total / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}
