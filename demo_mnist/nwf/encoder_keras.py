# -*- coding: utf-8 -*-
"""Keras CNN encoder for MNIST (Windows-compatible alternative to Flax)."""
import numpy as np


def build_encoder(output_dim: int = 64):
    """Build Keras CNN encoder."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("tensorflow required for encoder_keras")

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(output_dim),
    ])
    return model


def build_classifier(output_dim: int = 64, num_classes: int = 10):
    """Encoder + classifier for training."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("tensorflow required")

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(output_dim),
        layers.Dense(num_classes),
    ])
    return model
