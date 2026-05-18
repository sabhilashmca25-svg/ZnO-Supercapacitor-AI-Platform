"""
denormalizer.py — Convert normalised model predictions back to real µA values.

WHAT THIS MODULE DOES:
  ML models predict "current_normalised" (a value roughly in [0, 1]).
  This module converts those normalised predictions back to physical
  current in microamperes (µA).

HOW THE CURRENT NORMALISATION WORKS:
  The training pipeline used PER-GROUP normalisation:
    - Each (material, scan_rate) combination was normalised independently.
    - "Group" key = f"{material_id}_{int(scan_rate)}"  e.g. "NM1_30"
    - Formula: current_norm = (current_A - min_A) / (max_A - min_A)

  This module inverts that formula:
    current_A = current_norm × (max_A - min_A) + min_A

  Then converts Amperes → microamperes:
    current_µA = current_A × 1_000_000

WHY PER-GROUP SCALING?
  Different materials and scan rates produce very different current magnitudes
  (e.g. NM1 at 100 mV/s can be 5× higher than NM2 at 10 mV/s).
  Per-group scaling ensures the model learns the SHAPE of the CV curve
  rather than being dominated by absolute current magnitudes.

SCALERS.JSON STRUCTURE (current_per_group section):
  {
    "current_per_group": {
      "NM1_10": {"min": -0.0003133, "max": 0.0001727, "range": 0.000486},
      "NM1_20": {...},
      ...
      "NM4_100": {...}
    }
  }
  Values are in AMPERES. Output of this module is in MICROAMPERES (µA).
"""
import json
import logging
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Lazy-loaded scalers cache ─────────────────────────────────────────────────
_scalers_cache: dict | None = None


def _load_scalers() -> dict:
    """Load and cache scalers.json (loaded once)."""
    global _scalers_cache
    if _scalers_cache is None:
        path = settings.SCALERS_PATH
        if not path.exists():
            raise FileNotFoundError(f"scalers.json not found at: {path}")
        _scalers_cache = json.loads(path.read_text(encoding="utf-8"))
    return _scalers_cache


_VALID_SR = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _nearest_sr(scan_rate_mVs: float) -> int:
    """
    Find the nearest valid training scan rate for denormalization.

    Models were trained at integer scan rates [10, 20, ..., 100].
    Decimal inputs are rounded to the nearest available scaler key.
    """
    sr = int(round(scan_rate_mVs))
    if sr in _VALID_SR:
        return sr
    # Find nearest valid scan rate
    return min(_VALID_SR, key=lambda x: abs(x - sr))


def denormalize(
    norm_predictions: np.ndarray,
    material_id:      str,
    scan_rate_mVs:    float,
) -> np.ndarray:
    """
    Convert normalised predictions → physical current in µA.

    Args:
        norm_predictions: np.ndarray of shape (N,) — model output,
                          values roughly in [0, 1]
        material_id:      e.g. "NM1", "NM2", "NM3", "NM4"
        scan_rate_mVs:    e.g. 30.0 (must be a training scan rate: 10–100 in steps of 10)

    Returns:
        np.ndarray of shape (N,) — predicted current in MICROAMPERES (µA)

    Raises:
        KeyError if (material_id, scan_rate) combination is not in scalers.json
    """
    scalers = _load_scalers()
    current_per_group = scalers.get("current_per_group", {})

    # Build the lookup key: e.g. "NM1_30"
    # Decimal scan rates are rounded to the nearest training scan rate
    sr_rounded = _nearest_sr(scan_rate_mVs)
    group_key = f"{material_id}_{sr_rounded}"

    if group_key not in current_per_group:
        available = list(current_per_group.keys())
        raise KeyError(
            f"No scaler found for group '{group_key}'. "
            f"Available groups: {available[:6]}... "
            f"(check material_id and scan_rate_mVs)"
        )

    group_scaler = current_per_group[group_key]
    min_A = group_scaler["min"]     # minimum current in Amperes
    max_A = group_scaler["max"]     # maximum current in Amperes

    # ── Inverse min-max transform ────────────────────────────────────────────
    # current_A = norm × (max - min) + min
    current_A = norm_predictions * (max_A - min_A) + min_A

    # ── Convert Amperes → Microamperes ───────────────────────────────────────
    current_uA = current_A * 1_000_000   # 1 A = 1,000,000 µA

    logger.debug(
        "Denormalized %d points for %s at %.0f mV/s | "
        "range: [%.2f, %.2f] µA",
        len(current_uA), group_key, scan_rate_mVs,
        float(current_uA.min()), float(current_uA.max()),
    )

    return current_uA
