"""
Validation endpoints — compare model predictions against real experimental data.

POST /api/v1/validation/compare
  Run a prediction and compare it to the measured experimental curve.
  Returns both curves + point-wise residuals + summary metrics.
  This is the primary endpoint for the scientific validation analysis.

GET /api/v1/validation/per-group/{model_name}
  Return the pre-computed per-group RMSE / R² metrics for one model.
  These were computed during training and saved to per_group_csv/.
  Use this to render performance heatmaps and scan-rate trend plots.

SCIENTIFIC CONTEXT:
  - All experimental data is from real ZnO supercapacitor CV measurements
  - Residuals are computed on the prediction's 651-point potential grid
  - Experimental curves are interpolated onto that grid for fair comparison
  - Metrics (RMSE, MAE, R², max error) are computed in physical units (µA)
"""
import logging
import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas.experimental import (
    ValidationRequest,
    ValidationCompareResponse,
    ValidationMetrics,
    PerGroupResponse,
    PerGroupRow,
)
from app.services.analytics.experimental_loader import (
    get_experimental_curve,
    get_per_group_metrics,
    compute_residuals,
    VALID_MATERIALS,
    VALID_SCAN_RATES,
)
from app.services.predictor import run_prediction
from app.core.exceptions import InvalidInputError

logger = logging.getLogger(__name__)

router = APIRouter()

# All supported model IDs
VALID_MODELS = ["rf", "lightgbm", "xgboost", "gru", "lstm", "ann"]


@router.post(
    "/compare",
    response_model=ValidationCompareResponse,
    summary="Compare prediction vs experimental data",
    description=(
        "Runs the ML prediction pipeline for the requested model/material/scan-rate "
        "and compares the result against the real experimental CV curve from the "
        "dataset. Returns both curves, point-wise residuals, and summary metrics. "
        "This is the core endpoint for scientific validation analysis."
    ),
)
def compare_validation(request: ValidationRequest) -> ValidationCompareResponse:
    """
    Full prediction-vs-experimental comparison.

    **Request body:**
    - `model_name`    — which model to validate (rf, lightgbm, gru, etc.)
    - `material_id`   — which material (NM1–NM4)
    - `scan_rate_mVs` — scan rate (10–100 mV/s)

    **Response:**
    - `experimental_*` — actual measured CV curve (≈ 1,301 pts)
    - `predicted_*`    — ML model prediction (651 pts)
    - `residual_*`     — point-wise residuals on prediction grid (651 pts)
    - `metrics`        — RMSE, MAE, R², max error in µA
    - `split`          — dataset partition of this condition

    **Residual sign convention:**  residual = predicted − experimental
    Positive residual → model over-predicts; Negative → under-predicts.
    """
    model_name    = request.model_name
    material_id   = request.material_id
    scan_rate_mVs = request.scan_rate_mVs

    # ── Input validation ─────────────────────────────────────────────────
    if model_name not in VALID_MODELS:
        raise InvalidInputError(
            field="model_name",
            detail=f"'{model_name}' not recognised. Valid: {VALID_MODELS}",
        )
    if material_id not in VALID_MATERIALS:
        raise InvalidInputError(
            field="material_id",
            detail=f"'{material_id}' not recognised. Valid: {VALID_MATERIALS}",
        )
    sr = round(scan_rate_mVs)
    if sr not in VALID_SCAN_RATES:
        raise InvalidInputError(
            field="scan_rate_mVs",
            detail=f"{scan_rate_mVs} mV/s not in dataset. Valid: {VALID_SCAN_RATES}",
        )

    # ── Get experimental curve ────────────────────────────────────────────
    exp = get_experimental_curve(material_id, scan_rate_mVs)
    if exp is None:
        raise HTTPException(
            status_code=404,
            detail=f"No experimental data for {material_id} @ {sr} mV/s.",
        )

    # ── Run prediction ────────────────────────────────────────────────────
    pred = run_prediction(
        model_name    = model_name,
        material_id   = material_id,
        scan_rate_mVs = scan_rate_mVs,
    )

    # ── Compute residuals ─────────────────────────────────────────────────
    residual_data = compute_residuals(
        exp_potential   = np.array(exp["potential_V"]),
        exp_current     = np.array(exp["actual_current_uA"]),
        pred_potential  = np.array(pred.potential_V),
        pred_current    = np.array(pred.predicted_current_uA),
    )

    metrics = ValidationMetrics(**residual_data["metrics"])

    logger.info(
        "Validation compare: %s / %s / %d mV/s  RMSE=%.2f µA  R²=%.4f",
        model_name, material_id, sr,
        metrics.rmse_uA, metrics.r2,
    )

    return ValidationCompareResponse(
        model_name   = model_name,
        material_id  = material_id,
        scan_rate_mVs = float(sr),
        split        = exp["split"],

        experimental_potential_V  = exp["potential_V"],
        experimental_current_uA   = exp["actual_current_uA"],

        predicted_potential_V     = pred.potential_V,
        predicted_current_uA      = pred.predicted_current_uA,

        residual_potential_V      = pred.potential_V,
        residual_uA               = residual_data["residual_uA"],
        abs_residual_uA           = residual_data["abs_residual_uA"],

        metrics = metrics,
    )


@router.get(
    "/per-group/{model_name}",
    response_model=PerGroupResponse,
    summary="Pre-computed per-group validation metrics",
    description=(
        "Returns per-group RMSE and R² metrics for one model, computed during "
        "the training phase from the evaluation partitions (val, test_SR, test_MAT). "
        "Each row corresponds to one (material × scan_rate) group. "
        "Use this to render performance heatmaps and scan-rate trend charts."
    ),
)
def per_group_metrics(model_name: str) -> PerGroupResponse:
    """
    Return pre-computed per-group metrics for one model.

    **Path parameter:**
    - `model_name` — model identifier (rf, lightgbm, gru, xgboost, ann, lstm)

    **Response:**
    - `rows` — list of {nm_id, scan_rate_mVs, rmse_uA, r2, partition}

    **Partitions included:**
    - `val`       — NM1/NM2/NM3 at 30 mV/s (scan rate interpolation)
    - `test_sr`   — NM1/NM2/NM3 at 50 mV/s (unseen scan rate)
    - `test_mat`  — NM4 at all scan rates (unseen material)
    """
    if model_name not in VALID_MODELS:
        raise InvalidInputError(
            field="model_name",
            detail=f"'{model_name}' not recognised. Valid: {VALID_MODELS}",
        )

    rows = get_per_group_metrics(model_name)
    if rows is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Per-group metrics not available for '{model_name}'. "
                f"Check research/per_group_csv/{model_name}/"
            ),
        )

    return PerGroupResponse(
        model_name = model_name,
        rows       = [PerGroupRow(**r) for r in rows],
    )
