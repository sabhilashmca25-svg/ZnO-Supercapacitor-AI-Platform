"""
GET /api/v1/models         — list all available models with tier info
GET /api/v1/models/status  — loading status of all 6 models
GET /api/v1/models/{name}  — single model metadata
POST /api/v1/models/{name}/load — on-demand load a Tier-2 model
"""
from fastapi import APIRouter, HTTPException
from app.core.model_registry import model_registry, HOT_MODELS, LAZY_MODELS
from app.services.model_loader import ALL_SUPPORTED_MODELS

router = APIRouter()

_MODEL_DISPLAY = {
    "rf": "Random Forest",
    "lightgbm": "LightGBM",
    "gru": "Stacked GRU",
    "xgboost": "XGBoost",
    "ann": "Dense ANN",
    "lstm": "Stacked LSTM",
}


@router.get("")
def list_models() -> dict:
    """List all 6 models with their tier and loading status."""
    result = {}
    for model_id in ALL_SUPPORTED_MODELS:
        result[model_id] = {
            "model_id":    model_id,
            "display":     _MODEL_DISPLAY.get(model_id, model_id.upper()),
            "tier":        "hot" if model_id in HOT_MODELS else "lazy",
            "loaded":      model_registry.is_loaded(model_id),
            "metadata":    model_registry.get_metadata(model_id) or {},
        }
    return result


@router.get("/status")
def models_status() -> dict:
    """Return load status for all models."""
    return {
        "loaded": model_registry.loaded_models(),
        "hot_models": list(HOT_MODELS),
        "lazy_models": list(LAZY_MODELS),
        "all_loaded": all(model_registry.is_loaded(m) for m in ALL_SUPPORTED_MODELS),
    }


@router.get("/{model_name}")
def get_model(model_name: str) -> dict:
    """Return metadata for a single model."""
    model_name = model_name.lower().strip()
    if model_name not in ALL_SUPPORTED_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    meta = model_registry.get_metadata(model_name) or {}
    return {
        "model_id": model_name,
        "display":  _MODEL_DISPLAY.get(model_name, model_name.upper()),
        "tier":     "hot" if model_name in HOT_MODELS else "lazy",
        "loaded":   model_registry.is_loaded(model_name),
        "metadata": meta,
    }


@router.post("/{model_name}/load")
def load_model(model_name: str) -> dict:
    """
    On-demand load a Tier-2 (lazy) research model.

    Tier-1 models are always loaded; this endpoint is primarily for
    triggering lazy-load of XGBoost, ANN, or LSTM before comparison runs.
    """
    model_name = model_name.lower().strip()
    if model_name not in ALL_SUPPORTED_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    if model_registry.is_loaded(model_name):
        return {"model_id": model_name, "status": "already_loaded", "loaded": True}

    success = model_registry.ensure_loaded(model_name)
    if not success:
        raise HTTPException(status_code=503, detail=f"Failed to load model: {model_name}")

    return {
        "model_id": model_name,
        "status":   "loaded",
        "loaded":   True,
        "message":  f"Model '{model_name}' loaded and ready for inference.",
    }
