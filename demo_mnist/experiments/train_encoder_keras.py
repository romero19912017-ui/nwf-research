# -*- coding: utf-8 -*-
"""Train CNN encoder on MNIST using Keras (Windows-compatible)."""
import argparse
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from nwf.encoder_keras import build_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=str, default="encoder_keras.keras")
    args = parser.parse_args()

    (ds_train, ds_test), info = tfds.load(
        "mnist",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    def normalize(img, label):
        return tf.cast(img, tf.float32) / 255.0, label

    ds_train = ds_train.map(normalize).batch(args.batch_size).prefetch(1)
    ds_test = ds_test.map(normalize).batch(args.batch_size).prefetch(1)

    full_model = build_classifier(output_dim=64, num_classes=10)
    full_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    full_model.fit(ds_train, epochs=args.epochs, validation_data=ds_test)
    # Extract encoder (all layers except last Dense(10))
    encoder = tf.keras.Model(full_model.input, full_model.layers[-2].output)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(out_path)
    print(f"Saved encoder to {out_path}")


if __name__ == "__main__":
    main()
