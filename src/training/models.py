"""Model builders: baseline CNN + MobileNetV2 and EfficientNetB0 transfer-learning heads."""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

IMG_SHAPE = (224, 224, 3)


def build_baseline_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=IMG_SHAPE),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="baseline_cnn")
    return model


def _build_transfer_model(base_model_cls, num_classes, name):
    base = base_model_cls(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name=name)
    return model, base


def build_mobilenetv2(num_classes):
    return _build_transfer_model(MobileNetV2, num_classes, "mobilenetv2")


def build_efficientnetb0(num_classes):
    return _build_transfer_model(EfficientNetB0, num_classes, "efficientnetb0")
