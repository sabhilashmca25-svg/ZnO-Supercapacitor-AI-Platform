"""
exceptions.py — Custom exception types + FastAPI exception handlers.

HOW TO USE:
  raise ModelNotFoundError("gru")        → 404 response
  raise InvalidInputError("scan_rate")   → 422 response
  raise ModelNotLoadedError("rf")        → 503 response
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


# ── Custom exception classes ──────────────────────────────────────────────────

class ModelNotFoundError(Exception):
    """Raised when the user requests a model that doesn't exist."""
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' is not recognised.")


class ModelNotLoadedError(Exception):
    """Raised when a recognised model hasn't been loaded at startup."""
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(
            f"Model '{model_id}' exists but is not loaded. "
            f"Check ENABLED_MODELS in your .env file."
        )


class InvalidInputError(Exception):
    """Raised when request parameters fail scientific/domain validation."""
    def __init__(self, field: str, detail: str):
        self.field = field
        self.detail = detail
        super().__init__(f"Invalid value for '{field}': {detail}")


class PreprocessingError(Exception):
    """Raised when the feature engineering or normalization pipeline fails."""
    pass


# ── FastAPI exception handlers ────────────────────────────────────────────────
# These functions are registered in main.py.
# They convert Python exceptions into clean JSON error responses.

async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    """404 — user asked for a model name we don't know about."""
    valid_models = ["rf", "lightgbm", "gru", "xgboost", "ann", "lstm"]
    return JSONResponse(
        status_code=404,
        content={
            "error": "model_not_found",
            "message": str(exc),
            "valid_models": valid_models,
        },
    )


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    """503 — model exists but wasn't started up (not in ENABLED_MODELS)."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "model_not_loaded",
            "message": str(exc),
            "hint": "Add this model to ENABLED_MODELS in your .env file and restart.",
        },
    )


async def invalid_input_handler(request: Request, exc: InvalidInputError):
    """422 — request values are technically valid JSON but scientifically wrong."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_input",
            "field": exc.field,
            "message": exc.detail,
        },
    )


async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422 — Pydantic model validation failed (wrong types, missing fields, etc.)."""
    # Make the pydantic error readable for beginners
    errors = []
    for err in exc.errors():
        loc = " → ".join(str(x) for x in err["loc"])
        errors.append({"field": loc, "message": err["msg"]})
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request body has invalid or missing fields.",
            "details": errors,
        },
    )


async def generic_error_handler(request: Request, exc: Exception):
    """500 — unexpected server error. Check backend logs for the full traceback."""
    import logging
    logging.getLogger(__name__).exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Check the backend logs.",
        },
    )
