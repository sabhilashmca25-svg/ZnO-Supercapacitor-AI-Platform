"""
model_loader.py — Convenience wrapper around the ModelRegistry singleton.

WHY THIS FILE EXISTS:
  model_registry.py is the low-level registry (loads + stores models).
  This file provides simpler, beginner-friendly functions that any part
  of the codebase can call without knowing the registry internals.

USAGE EXAMPLE (in any endpoint or service):
  from app.services.model_loader import get_model, is_model_ready

  model = get_model("rf")      # returns the RandomForestRegressor object
  ready = is_model_ready("gru")  # True / False
"""
from app.core.model_registry import model_registry
from app.core.exceptions import ModelNotFoundError, ModelNotLoadedError

# All model IDs the platform supports (even if not all are loaded)
ALL_SUPPORTED_MODELS = ["rf", "lightgbm", "gru", "xgboost", "ann", "lstm"]


def get_model(model_id: str):
    """
    Retrieve a loaded model object by its ID.

    Raises:
        ModelNotFoundError  — if the model_id is not a recognised name
        ModelNotLoadedError — if the model exists but wasn't loaded at startup
    """
    model_id = model_id.lower().strip()

    if model_id not in ALL_SUPPORTED_MODELS:
        raise ModelNotFoundError(model_id)

    model = model_registry.get(model_id)
    if model is None:
        raise ModelNotLoadedError(model_id)

    return model


def is_model_ready(model_id: str) -> bool:
    """Return True if a model is loaded and ready for inference."""
    return model_registry.is_loaded(model_id.lower())


def list_loaded_models() -> list[str]:
    """Return the list of model IDs currently in memory."""
    return model_registry.loaded_models()
