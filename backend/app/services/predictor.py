"""
predictor.py — End-to-end prediction pipeline with lazy model loading
               and LRU result caching.

WHAT THIS MODULE DOES:
  Runs the full pipeline for a single prediction request:
    1. Validate model_id and material_id
    2. Ensure model is loaded (triggers lazy load for Tier-2 if needed)
    3. Build and normalise features
    4. Route to the correct model predictor
    5. Inverse-transform predictions back to µA
    6. Compute CV curve statistics
    7. Return everything needed for the API response

CACHING:
  Results for identical (model_name, material_id, scan_rate_mVs) inputs are
  cached in an LRU cache (size configured by PREDICTION_CACHE_SIZE in .env).
  Since CV predictions are deterministic, repeated calls for the same inputs
  return the cached result instantly — no inference overhead.
  Cache is bypassed when PREDICTION_CACHE_SIZE=0.

SUPPORTED MODELS:
  - Tabular  (RF, LightGBM, XGBoost, ANN): receive features_norm DataFrame
  - Sequence (LSTM, GRU):                  receive padded sequence tensor

SCAN RATE HANDLING:
  The models were trained on integer scan rates [10, 20, ..., 100].
  Decimal scan rates (e.g. 25.5 mV/s) are accepted and rounded to the
  nearest training scan rate for denormalization. The original input
  scan rate is preserved in the response for transparency.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.core.config import settings
from app.core.exceptions import ModelNotFoundError, ModelNotLoadedError, InvalidInputError
from app.core.model_registry import model_registry
from app.services.model_loader import ALL_SUPPORTED_MODELS
from app.services.ml.predictor_factory import get_predictor
from app.services.preprocess import build_inference_inputs
from app.services.postprocessing.denormalizer import denormalize
from app.schemas.prediction import CVStatistics, VALID_MATERIALS

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """All data returned by run_prediction()."""
    model_name:           str
    material_id:          str
    scan_rate_mVs:        float
    potential_V:          list[float]
    predicted_current_uA: list[float]
    n_points:             int
    statistics:           CVStatistics


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_request(model_name: str, material_id: str, scan_rate_mVs: float) -> None:
    """
    Validate inputs and ensure model is loaded (triggers lazy loading if needed).

    For Tier-2 research models (XGBoost, ANN, LSTM), this will transparently
    load the model on first request. Subsequent requests are instant.
    """
    # 1. Check model name is recognised
    if model_name not in ALL_SUPPORTED_MODELS:
        raise ModelNotFoundError(model_name)

    # 2. Ensure model is loaded — triggers on-demand loading for Tier-2 models
    if not model_registry.is_loaded(model_name):
        logger.info("Model '%s' not yet loaded — triggering on-demand load...", model_name)
        success = model_registry.ensure_loaded(model_name)
        if not success:
            raise ModelNotLoadedError(model_name)
        logger.info("Model '%s' successfully loaded on demand.", model_name)

    # 3. Check material ID
    if material_id not in VALID_MATERIALS:
        raise InvalidInputError(
            field="material_id",
            detail=f"'{material_id}' is not recognised. "
                   f"Valid values: {VALID_MATERIALS}",
        )

    # 4. Scan rate range check (10–100 mV/s; decimals accepted, rounded for denorm)
    if not (10.0 <= scan_rate_mVs <= 100.0):
        raise InvalidInputError(
            field="scan_rate_mVs",
            detail=f"{scan_rate_mVs} mV/s is out of range. "
                   f"Valid range: 10–100 mV/s (decimals accepted).",
        )


# ── Statistics computation ────────────────────────────────────────────────────

def _compute_statistics(
    current_uA:  np.ndarray,
    n_forward:   int,
    potential_V: np.ndarray,
) -> CVStatistics:
    """
    Compute summary statistics for a predicted CV curve.

    Args:
        current_uA:  Full predicted current array in µA (651 points)
        n_forward:   Number of points in the anodic (forward) half-sweep (326)
        potential_V: Voltage array in V corresponding to current_uA (651 points)

    Returns:
        CVStatistics with peak currents, range, and CV-loop enclosed area.

    Integral area:
        The physically correct enclosed area of the CV loop is
            ∮ I dV = ∫_anodic I dV  +  ∫_cathodic I dV
        where the cathodic voltage decreases from 0 V to −0.65 V so its
        trapezoid term contributes positively (dV < 0, I < 0).
        Units: µA·V  (proportional to charge stored).
    """
    anodic_half   = current_uA[:n_forward]
    cathodic_half = current_uA[n_forward:]
    V_anodic      = potential_V[:n_forward]
    V_cathodic    = potential_V[n_forward:]

    peak_anodic   = float(np.max(anodic_half))
    peak_cathodic = float(np.min(cathodic_half))
    current_range = float(current_uA.max() - current_uA.min())

    _trapezoid    = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area_anodic   = _trapezoid(anodic_half,   V_anodic)
    area_cathodic = _trapezoid(cathodic_half, V_cathodic)
    integral_area = float(np.abs(area_anodic + area_cathodic))

    return CVStatistics(
        peak_anodic_uA   = round(peak_anodic,   4),
        peak_cathodic_uA = round(peak_cathodic,  4),
        current_range_uA = round(current_range,  4),
        integral_area    = round(integral_area,  6),
    )


# ── Cached inner function ─────────────────────────────────────────────────────
# lru_cache requires hashable arguments — all three are str/float, so this works.
# The cache size is set from settings at module import time.
# Wrapping the heavy computation in a cached inner function means:
#   - First call for a (model, material, sr) triple: full inference pipeline
#   - Subsequent calls: instant in-memory lookup

_CACHE_SIZE = max(1, settings.PREDICTION_CACHE_SIZE) if settings.PREDICTION_CACHE_SIZE > 0 else None


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_predict(
    model_name:    str,
    material_id:   str,
    scan_rate_mVs: float,
) -> PredictionResult:
    """
    Inner prediction function wrapped in LRU cache.

    Called only by run_prediction() after validation. All expensive work
    (feature engineering, model inference, denormalization) happens here.
    """
    import time as _time
    t0 = _time.perf_counter()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    potential_V, features_norm, seq_tensor, actual_len = build_inference_inputs(
        scan_rate_mVs=scan_rate_mVs
    )

    # ── Step 2: Inference ─────────────────────────────────────────────────────
    predictor  = get_predictor(model_name)
    norm_preds = predictor.predict(features_norm)

    # ── Step 3: Denormalisation ───────────────────────────────────────────────
    current_uA = denormalize(norm_preds, material_id, scan_rate_mVs)

    # ── Step 4: Statistics ────────────────────────────────────────────────────
    n_forward  = (len(potential_V) + 1) // 2
    statistics = _compute_statistics(current_uA, n_forward, potential_V)

    elapsed = _time.perf_counter() - t0
    logger.info(
        "  Done: %d points | anodic=%.2f µA | cathodic=%.2f µA | %.3fs",
        len(current_uA),
        statistics.peak_anodic_uA,
        statistics.peak_cathodic_uA,
        elapsed,
    )

    return PredictionResult(
        model_name           = model_name,
        material_id          = material_id,
        scan_rate_mVs        = scan_rate_mVs,
        potential_V          = [round(v, 6) for v in potential_V.tolist()],
        predicted_current_uA = [round(c, 4) for c in current_uA.tolist()],
        n_points             = len(potential_V),
        statistics           = statistics,
    )


# ── Main prediction function ──────────────────────────────────────────────────

def run_prediction(
    model_name:    str,
    material_id:   str,
    scan_rate_mVs: float,
) -> PredictionResult:
    """
    Run the full end-to-end prediction pipeline (with LRU caching).

    Validates inputs, then delegates to _cached_predict() which caches
    results for repeated (model, material, scan_rate) combinations.

    Args:
        model_name:    One of "rf", "lightgbm", "gru", "xgboost", "ann", "lstm"
        material_id:   One of "NM1", "NM2", "NM3", "NM4"
        scan_rate_mVs: Scan rate in mV/s (10.0–100.0; decimals accepted)

    Returns:
        PredictionResult containing the CV curve data and statistics.
    """
    # ── Step 0: Validate + lazy-load if needed ────────────────────────────────
    _validate_request(model_name, material_id, scan_rate_mVs)

    ci = _cached_predict.cache_info() if _CACHE_SIZE else None
    logger.info(
        "Predicting: model=%s  material=%s  sr=%.1f mV/s%s",
        model_name, material_id, scan_rate_mVs,
        f"  [cache hits={ci.hits} misses={ci.misses}]" if ci else "",
    )

    return _cached_predict(model_name, material_id, scan_rate_mVs)
