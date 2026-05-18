"""
GET /api/v1/metrics              — All models' pre-computed evaluation metrics
GET /api/v1/metrics/{model_name} — One model's metrics

RESPONSE FORMAT:
  Returns a dict keyed by model_id. Each entry includes:
    - model_id:      API id (e.g. "rf")
    - model_display: CSV display name (e.g. "RF")
    - metrics:       list of {partition, rmse_uA, r2, mae_uA}
    - feature_importance: top features for tree models (optional)
    - params:        training hyperparameters
    - raw:           the full original metrics JSON (for advanced use)

  Field names are normalised to camelCase/lowercase for frontend compatibility.

PARTITIONS:
  train     — training data (in-sample fit)
  val       — scan rate 30 mV/s interpolation
  test_SR   — scan rate 50 mV/s unseen scan rate
  test_MAT  — NM4 material extrapolation (hardest)
"""
import logging
from fastapi import APIRouter, HTTPException

from app.services.analytics.metrics_loader import load_metrics, load_all_metrics, known_model_ids

logger = logging.getLogger(__name__)
router = APIRouter()

_MODEL_DISPLAY = {
    "rf": "RF", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "gru": "GRU", "lstm": "LSTM", "ann": "ANN",
}


def _normalize_metrics(model_id: str, raw: dict) -> dict:
    """
    Transform raw metrics JSON (uppercase keys from training notebooks)
    into the normalized format expected by the frontend TypeScript interface.
    """
    # Normalize per-partition metrics
    partitions_out = []
    for p in raw.get("metrics", []):
        partitions_out.append({
            "partition": p.get("partition", ""),
            "rmse_uA":   p.get("RMSE_uA") or p.get("rmse_uA") or 0.0,
            "r2":        p.get("R2")      or p.get("r2")      or 0.0,
            "mae_uA":    p.get("MAE_uA")  or p.get("mae_uA"),
            "max_err_uA": p.get("MaxErr_uA") or p.get("max_err_uA"),
            "n_samples": p.get("n_samples"),
        })

    # Extract feature importance — key names differ by training notebook
    # RF: feature_importance_mdi | LightGBM: gain_importance | XGBoost: feature_importance_gain
    feat_imp = (
        raw.get("feature_importance_mdi")           # RF
        or raw.get("gain_importance")               # LightGBM
        or raw.get("feature_importance_gain")       # XGBoost
        or raw.get("feature_importance_permutation")# GRU / ANN / fallback
        or raw.get("permutation_importance")        # LightGBM fallback
        or {}
    )

    return {
        "model_id":           model_id,
        "model_display":      _MODEL_DISPLAY.get(model_id, model_id.upper()),
        "metrics":            partitions_out,
        "feature_importance": feat_imp,
        "params":             raw.get("params", {}),
        "oob_r2":             raw.get("oob_R2"),
    }


@router.get(
    "",
    summary="Get metrics for all models",
)
def all_metrics() -> dict:
    """Return normalized evaluation metrics for all 6 trained models."""
    data = load_all_metrics()
    if not data:
        raise HTTPException(
            status_code=503,
            detail="No metrics files found. Check research/metrics/ directory.",
        )
    return {
        model_id: _normalize_metrics(model_id, raw)
        for model_id, raw in data.items()
    }


@router.get(
    "/{model_name}",
    summary="Get metrics for one model",
)
def get_metrics(model_name: str) -> dict:
    """Return normalized evaluation metrics for a single model."""
    model_name = model_name.lower().strip()
    data = load_metrics(model_name)
    if data is None:
        valid = ", ".join(known_model_ids())
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for '{model_name}'. Valid: {valid}",
        )
    return _normalize_metrics(model_name, data)
