"""
metrics_loader.py — Loads and caches per-model *_metrics.json files.

Each JSON file was produced by the training notebook and contains:
  - model: model class name
  - params: hyperparameters used
  - oob_R2: out-of-bag R² (RF only)
  - metrics: list of dicts, one per evaluation partition
      - partition: "train", "val  (SR=30 interpolation)", etc.
      - n_samples: number of CV data points in this split
      - RMSE_norm: RMSE in normalised [0,1] units
      - RMSE_uA, MAE_uA, MaxErr_uA: real µA errors
      - R2: coefficient of determination
  - per_group: per-{material, scan_rate} breakdown for val/test_sr/test_mat
  - feature_importance_mdi / feature_importance_permutation (RF/tree models)

Results are cached with @lru_cache — each file is read from disk exactly ONCE
per server process, then served from memory for all subsequent requests.
"""
import json
import logging
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)

# Maps the API model_id (lowercase) → filename in research/metrics/
_MODEL_TO_FILE: dict[str, str] = {
    "rf":       "rf_baseline_metrics.json",
    "xgboost":  "xgboost_metrics.json",
    "lightgbm": "lightgbm_metrics.json",
    "ann":      "ann_metrics.json",
    "lstm":     "lstm_metrics.json",
    "gru":      "gru_metrics.json",
}


@lru_cache(maxsize=None)
def load_metrics(model_id: str) -> dict | None:
    """
    Load and cache the metrics JSON for a single model.

    Args:
        model_id: lowercase model identifier, e.g. "rf", "gru", "lightgbm"

    Returns:
        The parsed JSON dict, or None if model_id is unknown / file is missing.
    """
    fname = _MODEL_TO_FILE.get(model_id.lower())
    if not fname:
        logger.warning("metrics_loader: unknown model_id '%s'", model_id)
        return None

    path = settings.METRICS_DIR / fname
    if not path.exists():
        logger.warning("metrics_loader: file not found — %s", path)
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("metrics_loader: loaded %s (%d bytes)", fname, path.stat().st_size)
        return data
    except Exception as exc:
        logger.error("metrics_loader: failed to parse %s — %s", fname, exc)
        return None


def load_all_metrics() -> dict[str, dict]:
    """
    Load metrics for all known models. Missing files are silently skipped.

    Returns:
        Dict mapping model_id → metrics dict (only models whose files exist).
    """
    result: dict[str, dict] = {}
    for model_id in _MODEL_TO_FILE:
        data = load_metrics(model_id)
        if data is not None:
            result[model_id] = data
    return result


def known_model_ids() -> list[str]:
    """Return the list of model IDs that have a metrics file mapping."""
    return list(_MODEL_TO_FILE.keys())
