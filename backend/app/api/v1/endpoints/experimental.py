"""
GET /api/v1/experimental/{material_id}/{scan_rate_mVs}

Returns the real experimental CV curve from the master parquet dataset for a
given material and scan rate. This is MEASURED data from the actual ZnO
supercapacitor experiments — not predictions.

Use cases:
  - Overlaying experimental data on top of model predictions
  - Validating model accuracy against ground truth
  - Scientific comparison panels in the frontend

NOTES:
  - Returns one complete cycle: anodic (forward) + cathodic (reverse) sweep
  - ~1,301 data points per curve (651 anodic + 650 cathodic)
  - Current is in µA (original current_A × 1e6)
  - `split` field tells you which dataset partition this condition belongs to
"""
import logging
from fastapi import APIRouter, HTTPException, Path as FastPath

from app.schemas.experimental import ExperimentalCurveResponse
from app.services.analytics.experimental_loader import (
    get_experimental_curve,
    VALID_MATERIALS,
    VALID_SCAN_RATES,
)
from app.core.exceptions import InvalidInputError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{material_id}/{scan_rate_mVs}",
    response_model=ExperimentalCurveResponse,
    summary="Get experimental CV curve",
    description=(
        "Returns the real measured cyclic voltammetry curve from the ZnO "
        "supercapacitor experiment dataset. One complete cycle (anodic + cathodic) "
        "is returned as potential_V and actual_current_uA arrays, ready for "
        "overlaying against model predictions."
    ),
)
def get_experimental(
    material_id: str = FastPath(..., description="ZnO material ID (NM1–NM4)"),
    scan_rate_mVs: float = FastPath(..., description="Scan rate in mV/s (10–100)"),
) -> ExperimentalCurveResponse:
    """
    Return the measured experimental CV curve.

    **Path parameters:**
    - `material_id`    — NM1, NM2, NM3, or NM4
    - `scan_rate_mVs`  — 10, 20, 30, ..., 100 (mV/s)

    **Response:**
    - `potential_V`       — electrode potential array (V)
    - `actual_current_uA` — measured current array (µA)
    - `split`             — which partition: train/val/test_SR/test_MAT
    - `n_points`          — total data points (≈ 1,301)

    **Errors:**
    - 422 if material_id or scan_rate_mVs is not valid
    - 404 if the requested combination is not in the dataset
    """
    # Validate inputs
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

    data = get_experimental_curve(material_id, scan_rate_mVs)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No experimental data for {material_id} @ {sr} mV/s.",
        )

    logger.info(
        "Experimental curve served: %s @ %d mV/s  split=%s  n=%d",
        material_id, sr, data["split"], data["n_points"],
    )
    return ExperimentalCurveResponse(**data)
