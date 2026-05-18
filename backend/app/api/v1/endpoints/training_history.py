"""
GET /api/v1/training-history/{model_name}

Serves per-epoch training curves for deep learning models (ANN, LSTM, GRU).
Tree models (RF, XGBoost, LightGBM) do not have epoch-based training histories.

Response includes:
  - loss / val_loss per epoch (MSE on normalised targets)
  - mae / val_mae per epoch
  - learning_rate schedule
  - epochs_ran, best_epoch, best_epoch_1indexed
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

_DEEP_MODELS = {"ann", "lstm", "gru"}
_HISTORY_FILES = {
    "ann":  "ann_training_history.json",
    "lstm": "lstm_training_history.json",
    "gru":  "gru_training_history.json",
}


_HIST_DIR: Path = settings.METRICS_DIR.parent / "training_histories"


@lru_cache(maxsize=None)
def _load_history(model_id: str) -> dict | None:
    fname = _HISTORY_FILES.get(model_id)
    if not fname:
        return None
    path = _HIST_DIR / fname
    if not path.exists():
        logger.warning("training_history: file not found — %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("training_history: parse error — %s", exc)
        return None


@router.get(
    "/{model_name}",
    summary="Training history for a deep learning model",
)
def get_training_history(model_name: str) -> dict:
    """
    Return per-epoch training curves for ANN, LSTM, or GRU.
    Tree/boosting models (RF, XGBoost, LightGBM) return 404 — they have no epoch history.
    """
    model_name = model_name.lower().strip()
    if model_name not in _DEEP_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Training history only available for deep learning models: {', '.join(sorted(_DEEP_MODELS))}. '{model_name}' is a tree/boosting model.",
        )

    data = _load_history(model_name)
    if data is None:
        raise HTTPException(
            status_code=503,
            detail=f"Training history file not found for '{model_name}'. Check research/training_histories/.",
        )

    return {
        "model_id": model_name,
        **data,
    }


@router.get(
    "",
    summary="Training histories for all deep learning models",
)
def get_all_training_histories() -> dict:
    """Return training histories for ANN, LSTM, and GRU in a single call."""
    result = {}
    for model_id in _DEEP_MODELS:
        data = _load_history(model_id)
        if data is not None:
            result[model_id] = {"model_id": model_id, **data}
    if not result:
        raise HTTPException(
            status_code=503,
            detail="No training history files found. Check research/training_histories/.",
        )
    return result
