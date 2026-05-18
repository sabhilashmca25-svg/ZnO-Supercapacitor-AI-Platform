"""
experimental_loader.py — Load and serve real experimental CV data from parquet.

DATA SOURCE:
  data/processed/master_long_format.parquet  (104,000 rows × 29 columns)

KEY COLUMNS:
  nm_id           — material identifier (NM1–NM4)
  scan_rate_mVs   — scan rate (10, 20, ..., 100 mV/s)
  potential_V     — electrode potential (V)
  current_A       — measured current in Amperes → multiply × 1e6 for µA
  sweep_direction — 0 = anodic (−0.65 V → 0 V), 1 = cathodic (0 V → −0.65 V)
  step_index      — position within the half-sweep (0 = first point)
  cycle_id        — which measurement cycle (use minimum = most stable)
  split           — train / val / test_SR / test_MAT

STRUCTURE PER (material × scan_rate):
  ~2,600 rows = 2 complete cycles × (651 + 650) points
  Using cycle_id == min gives 1,301 points:
    direction 0: 651 pts  step_index 0–650  potential −0.65 → 0.00 V (ascending)
    direction 1: 650 pts  step_index 0–649  potential −0.001 → −0.65 V (descending)

CACHING:
  The 104K-row parquet is loaded once at first request and kept in memory
  via functools.lru_cache. All subsequent calls are pure in-memory lookups.
"""
import csv
import logging
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# Valid scan rates present in the dataset (integer mV/s)
VALID_SCAN_RATES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
VALID_MATERIALS  = ["NM1", "NM2", "NM3", "NM4"]

# Mapping from CSV model name to API model_id (same as benchmark_ranker.py)
_PER_GROUP_MODEL_NAMES: dict[str, str] = {
    "rf":       "rf",
    "lightgbm": "lightgbm",
    "xgboost":  "xgboost",
    "gru":      "gru",
    "lstm":     "lstm",
    "ann":      "ann",
}


# ── Parquet loading ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_parquet() -> pd.DataFrame:
    """
    Load master_long_format.parquet once and cache it in memory.

    Returns the full 104K-row DataFrame. Called lazily on first request.
    """
    path = settings.PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet not found: {path}. "
            "Ensure data/processed/master_long_format.parquet exists."
        )
    logger.info("Loading experimental parquet from %s …", path)
    df = pd.read_parquet(path, columns=[
        "nm_id", "scan_rate_mVs", "potential_V", "current_A",
        "sweep_direction", "step_index", "cycle_id", "split",
    ])
    logger.info("Parquet loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


# ── Experimental curve extraction ─────────────────────────────────────────

def get_experimental_curve(material_id: str, scan_rate_mVs: float) -> Optional[dict]:
    """
    Return the experimental CV curve for a given material and scan rate.

    Selects the lowest cycle_id for reproducibility and concatenates the
    anodic (direction 0) and cathodic (direction 1) half-sweeps in order.

    Returns None if the requested (material_id, scan_rate_mVs) combination
    does not exist in the dataset.

    Returns a dict with:
      material_id, scan_rate_mVs, n_points,
      potential_V (list[float]), actual_current_uA (list[float]),
      split (str), cycle_id (int)
    """
    sr = round(scan_rate_mVs)
    df = _load_parquet()

    sub = df[(df["nm_id"] == material_id) & (df["scan_rate_mVs"] == float(sr))]
    if sub.empty:
        logger.warning("No experimental data for %s @ %d mV/s", material_id, sr)
        return None

    # Use minimum cycle_id — first complete cycle is most stable
    min_cycle = int(sub["cycle_id"].min())
    sub = sub[sub["cycle_id"] == min_cycle]

    # Sort each direction by step_index so points are in sweep order
    d0 = sub[sub["sweep_direction"] == 0].sort_values("step_index")
    d1 = sub[sub["sweep_direction"] == 1].sort_values("step_index")
    combined = pd.concat([d0, d1], ignore_index=True)

    potential_V        = [round(float(v), 6) for v in combined["potential_V"].tolist()]
    actual_current_uA  = [round(float(c) * 1e6, 4) for c in combined["current_A"].tolist()]
    split              = str(sub["split"].iloc[0])

    return {
        "material_id":       material_id,
        "scan_rate_mVs":     float(sr),
        "n_points":          len(potential_V),
        "potential_V":       potential_V,
        "actual_current_uA": actual_current_uA,
        "split":             split,
        "cycle_id":          min_cycle,
    }


# ── Residual computation ──────────────────────────────────────────────────

def compute_residuals(
    exp_potential: np.ndarray,
    exp_current:   np.ndarray,
    pred_potential: np.ndarray,
    pred_current:   np.ndarray,
) -> dict:
    """
    Compute point-wise residuals between predicted and experimental curves.

    Since the two curves are on different potential grids (651 vs 1,301 pts),
    the experimental curve is interpolated onto the prediction grid via linear
    interpolation, treating each half-sweep separately to handle the V-shaped
    (non-monotonic) potential sweep correctly.

    Args:
        exp_potential:  (1301,) experimental potential array
        exp_current:    (1301,) experimental current array (µA)
        pred_potential: (651,)  predicted potential array
        pred_current:   (651,)  predicted current array (µA)

    Returns dict with:
      residual_uA      — predicted − interpolated_experimental  (651 pts)
      abs_residual_uA  — |residual_uA|                         (651 pts)
      metrics          — rmse_uA, mae_uA, max_error_uA, r2
    """
    n_exp_d0  = 651   # direction 0 has 651 points (anodic: −0.65 → 0 V)
    n_forward = (len(pred_potential) + 1) // 2  # 326 — prediction anodic half

    # ── Split experimental into anodic / cathodic ─────────────────────────
    exp_d0_pot = exp_potential[:n_exp_d0]  # ascending: −0.65 → 0 V
    exp_d0_cur = exp_current[:n_exp_d0]
    exp_d1_pot = exp_potential[n_exp_d0:]  # descending: −0.001 → −0.65 V
    exp_d1_cur = exp_current[n_exp_d0:]

    # ── Split prediction into anodic / cathodic ───────────────────────────
    pred_anodic_pot   = pred_potential[:n_forward]    # ascending: −0.65 → 0 V
    pred_cathodic_pot = pred_potential[n_forward:]    # descending: 0 → −0.65 V

    # ── Interpolate anodic half (both ascending in potential) ─────────────
    interp_anodic = np.interp(pred_anodic_pot, exp_d0_pot, exp_d0_cur)

    # ── Interpolate cathodic half (both descending — flip for np.interp) ──
    # np.interp requires xp to be strictly increasing; flip both arrays,
    # interpolate, then flip the result back to original order.
    if len(exp_d1_pot) > 0 and len(pred_cathodic_pot) > 0:
        interp_cathodic = np.interp(
            pred_cathodic_pot[::-1],   # ascending after flip
            exp_d1_pot[::-1],          # ascending after flip
            exp_d1_cur[::-1],
        )[::-1]                        # flip result back to descending order
    else:
        interp_cathodic = np.zeros(len(pred_cathodic_pot))

    interp_exp = np.concatenate([interp_anodic, interp_cathodic])

    # ── Point-wise residuals ──────────────────────────────────────────────
    residuals     = pred_current - interp_exp
    abs_residuals = np.abs(residuals)

    rmse      = float(np.sqrt(np.mean(residuals ** 2)))
    mae       = float(np.mean(abs_residuals))
    max_error = float(np.max(abs_residuals))
    ss_res    = float(np.sum(residuals ** 2))
    ss_tot    = float(np.sum((interp_exp - np.mean(interp_exp)) ** 2))
    r2        = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "residual_uA":     [round(float(r), 4) for r in residuals.tolist()],
        "abs_residual_uA": [round(float(a), 4) for a in abs_residuals.tolist()],
        "metrics": {
            "rmse_uA":      round(rmse,      4),
            "mae_uA":       round(mae,       4),
            "max_error_uA": round(max_error, 4),
            "r2":           round(r2,        6),
        },
    }


# ── Per-group pre-computed metrics ─────────────────────────────────────────

def get_per_group_metrics(model_name: str) -> Optional[list]:
    """
    Return the pre-computed per-group RMSE / R² metrics for one model.

    Reads from research/per_group_csv/{model_name}/per_group_rmse_{partition}.csv
    for all three evaluation partitions (val, test_SR, test_MAT).

    Returns a list of dicts: [{nm_id, scan_rate_mVs, rmse_uA, r2, partition}, ...]
    Returns None if no CSV files are found for this model.
    """
    base = settings.PER_GROUP_DIR / model_name
    if not base.is_dir():
        logger.warning("Per-group directory not found: %s", base)
        return None

    partitions = {
        "val":      "per_group_rmse_val.csv",
        "test_sr":  "per_group_rmse_test_sr.csv",
        "test_mat": "per_group_rmse_test_mat.csv",
    }

    rows = []
    for partition, filename in partitions.items():
        path = base / filename
        if not path.exists():
            logger.warning("Per-group CSV not found: %s", path)
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "nm_id":         row["nm_id"],
                    "scan_rate_mVs": float(row["scan_rate_mVs"]),
                    "rmse_uA":       round(float(row["RMSE_uA"]), 4),
                    "r2":            round(float(row["R2"]), 6),
                    "partition":     partition,
                })

    return rows if rows else None
