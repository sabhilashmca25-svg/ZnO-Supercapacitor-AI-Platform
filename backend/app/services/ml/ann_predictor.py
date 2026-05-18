"""Dense ANN predictor wrapper (Keras)."""
import numpy as np
import pandas as pd
from app.services.ml.base_predictor import BasePredictor
from app.core.model_registry import model_registry


class ANNPredictor(BasePredictor):
    model_id = "ann"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        model = model_registry.get("ann")
        if model is None:
            raise RuntimeError("ANN model is not loaded. Check ENABLED_MODELS in .env")
        return model.predict(features.values, verbose=0).flatten()
