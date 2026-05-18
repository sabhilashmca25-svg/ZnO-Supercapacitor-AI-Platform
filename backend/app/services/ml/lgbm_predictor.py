"""LightGBM predictor wrapper."""
import numpy as np
import pandas as pd
from app.services.ml.base_predictor import BasePredictor
from app.core.model_registry import model_registry


class LGBMPredictor(BasePredictor):
    model_id = "lightgbm"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        model = model_registry.get("lightgbm")
        if model is None:
            raise RuntimeError("LightGBM model is not loaded. Check ENABLED_MODELS in .env")
        return model.predict(features.values)
