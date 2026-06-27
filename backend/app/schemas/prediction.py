"""
prediction.py — Pydantic schemas for the /predict endpoint.

Pydantic schemas do two things:
  1. Validate incoming request data (types, allowed values, ranges)
  2. Define the exact shape of JSON responses

VALID MODELS:  rf, lightgbm, gru
VALID MATERIALS: NM1, NM2, NM3, NM4
VALID SCAN RATES: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100  (mV/s)
"""
from pydantic import BaseModel, Field
from typing import List


# ── Allowed values (used in validation and Swagger examples) ──────────────────

VALID_MODELS    = ["rf", "lightgbm", "gru", "xgboost", "ann", "lstm"]
VALID_MATERIALS = ["NM1", "NM2", "NM3", "NM4"]
VALID_SCAN_RATES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # mV/s


# ── Request schema ────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    What you send to POST /api/v1/predict

    Fields:
      model_name    — which ML model to use
      material_id   — which ZnO material to predict for
      scan_rate_mVs — how fast the voltage sweep runs (mV per second)
    """
    model_name: str = Field(
        ...,
        description="ML model to use. One of: rf, lightgbm, gru",
        examples=["lightgbm"],
    )
    material_id: str = Field(
        ...,
        description="ZnO material identifier. One of: NM1, NM2, NM3, NM4",
        examples=["NM1"],
    )
    scan_rate_mVs: float = Field(
        ...,
        ge=10,
        le=100,
        description="Scan rate in mV/s (10–100). Decimal values accepted; automatically rounded to nearest training scan rate for denormalization.",
        examples=[30.0],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_name": "lightgbm",
                "material_id": "NM1",
                "scan_rate_mVs": 30.0,
            }
        }
    }


# ── Response schema ───────────────────────────────────────────────────────────

class CVStatistics(BaseModel):
    """Summary statistics of the predicted CV curve."""
    peak_anodic_uA:   float = Field(description="Peak current in the anodic (forward) sweep, µA")
    peak_cathodic_uA: float = Field(description="Peak current in the cathodic (reverse) sweep, µA")
    current_range_uA: float = Field(description="Difference between max and min current, µA")
    integral_area:    float = Field(description="Enclosed area of the CV loop (proxy for capacitance)")


class PredictionResponse(BaseModel):
    """
    What you get back from POST /api/v1/predict

    Use potential_V and predicted_current_uA directly for a Plotly graph:
      - x = potential_V
      - y = predicted_current_uA
    """
    model_name:           str       = Field(description="Model used for prediction")
    material_id:          str       = Field(description="Material predicted for")
    scan_rate_mVs:        float     = Field(description="Scan rate used (mV/s)")
    n_points:             int       = Field(description="Number of data points in the curve")
    potential_V:          List[float] = Field(description="Potential array in Volts")
    predicted_current_uA: List[float] = Field(description="Predicted current array in µA")
    statistics:           CVStatistics = Field(description="Summary statistics of the curve")
