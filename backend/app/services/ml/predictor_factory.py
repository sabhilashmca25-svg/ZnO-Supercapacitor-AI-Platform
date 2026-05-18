"""
predictor_factory.py — Returns the correct BasePredictor for a given model_id.
Add a new model: import its class and add it to REGISTRY.
"""
from app.services.ml.rf_predictor       import RFPredictor
from app.services.ml.xgboost_predictor  import XGBoostPredictor
from app.services.ml.lgbm_predictor     import LGBMPredictor
from app.services.ml.ann_predictor      import ANNPredictor
from app.services.ml.lstm_predictor     import LSTMPredictor
from app.services.ml.gru_predictor      import GRUPredictor

REGISTRY = {
    "rf":        RFPredictor(),
    "xgboost":   XGBoostPredictor(),
    "lightgbm":  LGBMPredictor(),
    "ann":       ANNPredictor(),
    "lstm":      LSTMPredictor(),
    "gru":       GRUPredictor(),
}

def get_predictor(model_id: str):
    return REGISTRY.get(model_id.lower())
