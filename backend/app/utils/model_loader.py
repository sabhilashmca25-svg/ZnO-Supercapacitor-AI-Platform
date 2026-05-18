"""Utility helpers for loading model files from disk."""
from pathlib import Path


def load_joblib(path: Path):
    import joblib
    return joblib.load(path)


def load_keras(path: Path):
    import tensorflow as tf
    return tf.keras.models.load_model(str(path))
