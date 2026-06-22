from dataclasses import dataclass

import h5py
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["Negative_Class", "Gunshot", "Screaming"]

SUBTIPO_MAP = {
    "Negative_Class": "otro",
    "Gunshot": "disparo",
    "Screaming": "grito",
}


@dataclass
class ClassificationResult:
    class_name: str
    subtipo: str
    confidence_pct: float
    confidence: float
    index: int
    is_alert: bool


def load_classifier(path: str) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1024,)),
        tf.keras.layers.Dense(512, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.5, name="dropout"),
        tf.keras.layers.Dense(256, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.3, name="dropout_1"),
        tf.keras.layers.Dense(3, activation="softmax", name="dense_2"),
    ])
    with h5py.File(path, "r") as f:
        model.get_layer("dense").set_weights([
            np.array(f["model_weights/dense/sequential/dense/kernel"]),
            np.array(f["model_weights/dense/sequential/dense/bias"]),
        ])
        model.get_layer("dense_1").set_weights([
            np.array(f["model_weights/dense_1/sequential/dense_1/kernel"]),
            np.array(f["model_weights/dense_1/sequential/dense_1/bias"]),
        ])
        model.get_layer("dense_2").set_weights([
            np.array(f["model_weights/dense_2/sequential/dense_2/kernel"]),
            np.array(f["model_weights/dense_2/sequential/dense_2/bias"]),
        ])
    return model
