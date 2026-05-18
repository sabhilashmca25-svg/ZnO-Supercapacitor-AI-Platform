"""Stacked LSTM predictor wrapper — handles sequence packing/unpacking."""
import numpy as np
import pandas as pd
from app.services.ml.base_predictor import BasePredictor
from app.services.preprocessing.sequence_builder import build_padded_sequence
from app.core.model_registry import model_registry


class LSTMPredictor(BasePredictor):
    model_id = "lstm"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        model = model_registry.get("lstm")
        if model is None:
            raise RuntimeError("LSTM model is not loaded. Check ENABLED_MODELS in .env")
        seq, L = build_padded_sequence(features)   # (1, 651, 10)
        pred   = model.predict(seq, verbose=0)     # (1, 651, 1)
        return pred[0, :L, 0]                      # (L,) — unpad
