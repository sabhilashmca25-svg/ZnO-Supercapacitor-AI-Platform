"""
GRU predictor wrapper.

The GRU expects input shape (1, 651, 10) — a padded sequence tensor.
It outputs shape (1, 651, 1). We unpad to only the actual_len real points.
"""
import numpy as np
import pandas as pd
from app.services.ml.base_predictor import BasePredictor
from app.services.preprocessing.sequence_builder import build_padded_sequence
from app.core.model_registry import model_registry


class GRUPredictor(BasePredictor):
    model_id = "gru"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        model = model_registry.get("gru")
        if model is None:
            raise RuntimeError("GRU model is not loaded. Check ENABLED_MODELS in .env")

        seq, actual_len = build_padded_sequence(features)   # (1, 651, 10)
        pred = model.predict(seq, verbose=0)                 # (1, 651, 1)
        return pred[0, :actual_len, 0]                       # (actual_len,)
