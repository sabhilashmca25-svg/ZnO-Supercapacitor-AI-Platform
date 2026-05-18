"""
experimental.py — Pydantic schemas for experimental data and validation endpoints.

These schemas serve:
  GET /api/v1/experimental/{material_id}/{scan_rate_mVs}
  POST /api/v1/validation/compare
  GET  /api/v1/validation/per-group/{model_name}
"""
from pydantic import BaseModel, Field
from typing import List


# ── Experimental CV curve ──────────────────────────────────────────────────

class ExperimentalCurveResponse(BaseModel):
    """
    Raw experimental CV curve from the master parquet dataset.

    Data comes directly from the ZnO supercapacitor electrochemical experiments
    (one complete cycle: anodic forward sweep + cathodic reverse sweep).
    """
    material_id:        str         = Field(description="ZnO material ID (NM1–NM4)")
    scan_rate_mVs:      float       = Field(description="Scan rate used (mV/s)")
    n_points:           int         = Field(description="Total data points in the cycle")
    potential_V:        List[float] = Field(description="Electrode potential array (V)")
    actual_current_uA:  List[float] = Field(description="Measured current array (µA)")
    split:              str         = Field(description="Dataset partition: train/val/test_SR/test_MAT")
    cycle_id:           int         = Field(description="Which cycle was used (lowest available)")


# ── Validation comparison ──────────────────────────────────────────────────

class ValidationRequest(BaseModel):
    """Request body for POST /api/v1/validation/compare"""
    model_name:    str   = Field(..., description="ML model to use (rf, lightgbm, gru, etc.)")
    material_id:   str   = Field(..., description="ZnO material (NM1–NM4)")
    scan_rate_mVs: float = Field(..., ge=10, le=100, description="Scan rate (mV/s)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_name": "lightgbm",
                "material_id": "NM1",
                "scan_rate_mVs": 30.0,
            }
        }
    }


class ValidationMetrics(BaseModel):
    """Point-wise error metrics between predicted and experimental curves."""
    rmse_uA:       float = Field(description="Root Mean Squared Error (µA)")
    mae_uA:        float = Field(description="Mean Absolute Error (µA)")
    max_error_uA:  float = Field(description="Maximum absolute error (µA)")
    r2:            float = Field(description="R² coefficient of determination")


class ValidationCompareResponse(BaseModel):
    """
    Full comparison result: experimental + predicted + residuals + metrics.

    Both curves share the same potential range (-0.65 V to 0 V) but
    may be on different grids. Residuals are computed on the prediction grid
    via linear interpolation of the experimental curve.
    """
    model_name:    str = Field(description="Model used")
    material_id:   str = Field(description="Material compared")
    scan_rate_mVs: float = Field(description="Scan rate (mV/s)")
    split:         str = Field(description="Dataset partition of this condition")

    # Experimental curve (raw, 1301 points: full anodic + cathodic cycle)
    experimental_potential_V:   List[float] = Field(description="Experimental potential array (V)")
    experimental_current_uA:    List[float] = Field(description="Experimental current array (µA)")

    # Predicted curve (651 points: -0.65 V → 0 V → -0.65 V)
    predicted_potential_V:      List[float] = Field(description="Predicted potential array (V)")
    predicted_current_uA:       List[float] = Field(description="Predicted current array (µA)")

    # Residuals on prediction grid (predicted - interpolated experimental)
    residual_potential_V:       List[float] = Field(description="Potential grid for residuals (V)")
    residual_uA:                List[float] = Field(description="Signed residuals: predicted − experimental (µA)")
    abs_residual_uA:            List[float] = Field(description="Absolute residuals |predicted − experimental| (µA)")

    # Summary metrics
    metrics: ValidationMetrics = Field(description="RMSE, MAE, R², max error")


# ── Per-group pre-computed metrics ─────────────────────────────────────────

class PerGroupRow(BaseModel):
    """One row from the per-group RMSE CSV (computed during training)."""
    nm_id:         str   = Field(description="Material ID")
    scan_rate_mVs: float = Field(description="Scan rate (mV/s)")
    rmse_uA:       float = Field(description="RMSE in µA for this group")
    r2:            float = Field(description="R² for this group")
    partition:     str   = Field(description="Dataset partition")


class PerGroupResponse(BaseModel):
    """All per-group metrics for one model (all partitions combined)."""
    model_name: str              = Field(description="Model identifier")
    rows:       List[PerGroupRow] = Field(description="One row per (material, scan_rate, partition)")
