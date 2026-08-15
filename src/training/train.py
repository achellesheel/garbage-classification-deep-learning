"""Trains one model (baseline_cnn | mobilenetv2 | efficientnetb0) and saves it under models/<name>/.

Usage: python3 src/training/train.py <model_key> [--epochs N] [--fine-tune-epochs N]
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from datasets import get_datasets, compute_class_weights
from models import build_baseline_cnn, build_mobilenetv2, build_efficientnetb0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

BUILDERS = {
    "baseline_cnn": lambda n: (build_baseline_cnn(n), None),
    "mobilenetv2": build_mobilenetv2,
    "efficientnetb0": build_efficientnetb0,
}


def merge_histories(*histories):
    merged = {}
    for h in histories:
        if h is None:
            continue
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key", choices=list(BUILDERS.keys()))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    args = parser.parse_args()

    out_dir = MODELS_DIR / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = get_datasets(args.model_key)
    class_weight = compute_class_weights(class_names)
    print(f"Classes: {class_names}")
    print(f"Class weights: {class_weight}")

    model, base = BUILDERS[args.model_key](len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    checkpoint_path = out_dir / "model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weight, callbacks=callbacks,
    )

    history2 = None
    if base is not None and args.fine_tune_epochs > 0:
        print("Fine-tuning: unfreezing top of base model...")
        base.trainable = True
        # Freeze all but the last ~20% of layers to limit overfitting risk on a modest dataset.
        n_layers = len(base.layers)
        freeze_until = int(n_layers * 0.8)
        for layer in base.layers[:freeze_until]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history2 = model.fit(
            train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
            class_weight=class_weight, callbacks=callbacks,
        )

    # ModelCheckpoint already saved the best-val-loss weights to checkpoint_path.
    merged_history = merge_histories(history1, history2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(merged_history, f, indent=2)
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy ({args.model_key}): {test_acc:.4f}")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_acc)}, f, indent=2)


if __name__ == "__main__":
    main()
