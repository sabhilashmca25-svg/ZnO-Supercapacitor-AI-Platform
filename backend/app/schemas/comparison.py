"""
comparison.py — Pydantic schemas for the /compare endpoint.

The compare endpoint runs the SAME input through multiple models and
returns all predictions together — perfect for Plotly overlay charts.
"""
from pydantic import BaseModel, Field
from typing import List, Dict


class ComparisonRequest(BaseModel):
    """
    What you send to POST /api/v1/compare

    Example: compare RF, LightGBM, and GRU on NM1 at 30 mV/s
    """
    models: List[str] = Field(
        ...,
        description="List of model IDs to compare. E.g. ['rf', 'lightgbm', 'gru']",
        examples=[["rf", "lightgbm", "gru"]],
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
        description="Scan rate in mV/s. Must be one of: 10, 20, 30, ..., 100",
        examples=[30.0],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "models": ["rf", "lightgbm", "gru"],
                "material_id": "NM1",
                "scan_rate_mVs": 30.0,
            }
        }
    }


class ModelPrediction(BaseModel):
    """A single model's prediction result within a comparison."""
    predicted_current_uA: List[float]
    peak_anodic_uA:        float
    peak_cathodic_uA:      float
    current_range_uA:      float
    integral_area:         float


class ComparisonResponse(BaseModel):
    """
    What you get back from POST /api/v1/compare

    potential_V is shared across all models (same sweep was used).
    predictions is a dict mapping model_id → its current predictions.

    For a Plotly overlay chart:
      - x = potential_V  (shared for all traces)
      - for each model in predictions:  y = predictions[model].predicted_current_uA
    """
    material_id:   str                          = Field(description="Material predicted for")
    scan_rate_mVs: float                        = Field(description="Scan rate used (mV/s)")
    n_points:      int                          = Field(description="Number of data points")
    potential_V:   List[float]                  = Field(description="Shared potential array (V)")
    predictions:   Dict[str, ModelPrediction]   = Field(description="Per-model predictions")
    models_failed: List[str]                    = Field(
        default=[],
        description="Models that failed (not loaded or errored)",
    )
