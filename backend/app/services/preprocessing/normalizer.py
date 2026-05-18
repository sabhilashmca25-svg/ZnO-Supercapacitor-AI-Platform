"""
normalizer.py — Apply min-max normalisation using scalers.json.

WHAT THIS MODULE DOES:
  The models were trained on normalised features.
  This module replicates the exact same normalisation at inference time.

HOW MIN-MAX NORMALISATION WORKS:
  normalised = (raw_value - min) / (max - min)
  Result is always in [0, 1] for values within the training range.

THE COLUMN NAME MISMATCH PROBLEM (and how we fix it):
  The DataFrame columns use "_norm" suffix (e.g. "potential_V_norm")
  but scalers.json stores keys WITHOUT the suffix (e.g. "potential_V").

  Why? Column names describe the DESIRED state (normalised), but before
  this function runs, they hold RAW values. The "_norm" suffix is added
  during feature engineering to match training column names.

  Solution: the COL_TO_SCALER_KEY dict maps each column name to its
  corresponding key in scalers.json.

NOT NORMALISED (pass-through columns):
  - sweep_direction : already 0.0 or 1.0 (binary, no scaling needed)
  - sweep_position  : already in [0, 1]  (naturally normalised)
"""
import json
import logging
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Column name → scalers.json key mapping ────────────────────────────────────
# DataFrame column   →   key in scalers.json["global_features"]
COL_TO_SCALER_KEY = {
    "potential_V_norm":           "potential_V",
    "scan_rate_mVs_norm":         "scan_rate_mVs",
    "log_scan_rate_norm":         "log_scan_rate",
    "sqrt_scan_rate_norm":        "sqrt_scan_rate",
    "potential_from_lower_norm":  "potential_from_lower",
    "potential_from_upper_norm":  "potential_from_upper",
    "sr_x_potential_norm":        "sr_x_potential",
    "direction_x_potential_norm": "direction_x_potential",
    # "sweep_direction" and "sweep_position" are intentionally absent
    # — they are NOT normalised (pass through unchanged).
}


# ── Lazy-loaded scalers (loaded once on first call) ───────────────────────────
_scalers_cache: dict | None = None


def _load_scalers() -> dict:
    """
    Load scalers.json from disk (cached after first load).

    Returns the full scalers dict, e.g.:
      {
        "global_features": {"potential_V": {"min": -0.65, "max": 0.0}, ...},
        "current_per_group": {"NM1_10": {"min": ..., "max": ..., "range": ...}, ...},
        ...
      }
    """
    global _scalers_cache
    if _scalers_cache is None:
        path = settings.SCALERS_PATH
        if not path.exists():
            raise FileNotFoundError(f"scalers.json not found at: {path}")
        _scalers_cache = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Scalers loaded from: %s", path)
    return _scalers_cache


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalisation to the raw feature DataFrame.

    Args:
        df: DataFrame with shape (N, 10), columns = FEATURE_NAMES,
            holding RAW (un-normalised) values.

    Returns:
        DataFrame with the same shape and columns, but normalised values.
        sweep_direction and sweep_position are passed through unchanged.
    """
    scalers = _load_scalers()
    global_features = scalers.get("global_features", {})

    out = df.copy()

    for col, scaler_key in COL_TO_SCALER_KEY.items():
        if col not in df.columns:
            logger.warning("Column '%s' not found in DataFrame — skipping.", col)
            continue

        if scaler_key not in global_features:
            logger.warning("Scaler key '%s' not found in scalers.json — skipping.", scaler_key)
            continue

        mn  = global_features[scaler_key]["min"]
        mx  = global_features[scaler_key]["max"]
        rng = mx - mn

        if rng == 0.0:
            # Edge case: constant feature, set to 0
            out[col] = 0.0
        else:
            out[col] = (df[col] - mn) / rng

    return out
