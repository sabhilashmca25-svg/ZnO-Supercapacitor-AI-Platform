"""
common.py — Shared Pydantic schemas used across multiple endpoints.
"""
from pydantic import BaseModel, Field
from typing import List, Any


class HealthResponse(BaseModel):
    """Response from GET /api/v1/health"""
    status:        str       = Field(description="'ok' if everything is running")
    version:       str       = Field(description="API version string")
    models_loaded: List[str] = Field(description="Model IDs currently in memory")
    scalers_ok:    bool      = Field(description="True if scalers.json was found on disk")
    environment:   str       = Field(description="'development' or 'production'")


class ErrorResponse(BaseModel):
    """Standard error response shape for all API errors."""
    error:   str       = Field(description="Short error code, e.g. 'model_not_found'")
    message: str       = Field(description="Human-readable explanation")
    details: List[Any] = Field(default=[], description="Additional error details if any")
