"""
main.py — FastAPI application factory.

STARTUP SEQUENCE:
  1. Python imports this module and creates the `app` object.
  2. When uvicorn starts, it calls the `lifespan` context manager.
  3. lifespan calls model_registry.load_all() → loads every Tier-1 (hot) model.
  4. If ENABLE_WARMUP=True, one inference is run per hot model to pre-compile
     TF graphs and warm JIT caches so the first real request is fast.
  5. The server is now ready to handle requests.
  6. When the server shuts down, lifespan calls model_registry.unload_all()
     to free memory cleanly.

EXCEPTION HANDLERS:
  All HTTP error responses are standardised to JSON by the registered handlers:
    - ModelNotFoundError  → 404 {"error": "model_not_found", ...}
    - ModelNotLoadedError → 503 {"error": "model_not_loaded", ...}
    - InvalidInputError   → 422 {"error": "invalid_input", ...}
    - RequestValidationError → 422 {"error": "validation_error", ...}
    - Exception (catch-all) → 500 {"error": "internal_server_error", ...}

HOW TO START:
    cd ZnO_Supercapacitor_AI_Platform/backend
    uvicorn app.main:app --reload --port 8000

SWAGGER UI (interactive docs):
    http://localhost:8000/docs   (disabled in production when DOCS_ENABLED=False)
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.model_registry import model_registry, HOT_MODELS
from app.core.exceptions import (
    ModelNotFoundError,
    ModelNotLoadedError,
    InvalidInputError,
    PreprocessingError,
    model_not_found_handler,
    model_not_loaded_handler,
    invalid_input_handler,
    validation_error_handler,
    generic_error_handler,
)
from app.middleware.timing import TimingMiddleware

# Configure logging for the whole application
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Request size limit middleware ─────────────────────────────────────────────

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Reject requests whose Content-Length exceeds MAX_REQUEST_SIZE_BYTES.
    Prevents oversized payloads from reaching the application layer.
    """

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            if int(content_length) > settings.MAX_REQUEST_SIZE_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "request_too_large",
                        "message": (
                            f"Request body exceeds the "
                            f"{settings.MAX_REQUEST_SIZE_BYTES // 1024} KB limit."
                        ),
                    },
                )
        return await call_next(request)


# ── Warm-up helper ────────────────────────────────────────────────────────────

def _run_warmup() -> None:
    """
    Run one silent inference per hot-loaded model after startup.

    Pre-compiles TensorFlow computation graphs and warms the JIT caches of
    tree-based models so that the first real user request does not incur
    cold-start latency.
    """
    from app.services.predictor import run_prediction

    hot_loaded = HOT_MODELS & set(model_registry.loaded_models())
    if not hot_loaded:
        return

    logger.info("Warm-up: running silent inference for %s", sorted(hot_loaded))
    for model_id in sorted(hot_loaded):
        try:
            t0 = time.perf_counter()
            run_prediction(
                model_name    = model_id,
                material_id   = settings.WARMUP_MATERIAL,
                scan_rate_mVs = float(settings.WARMUP_SCAN_RATE),
            )
            elapsed = time.perf_counter() - t0
            logger.info("  [%s] warm-up done in %.2fs", model_id, elapsed)
        except Exception as exc:
            logger.warning("  [%s] warm-up failed (non-fatal): %s", model_id, exc)

    logger.info("Warm-up complete.")


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup and once at shutdown.

    Startup  → load Tier-1 models → optional warm-up inference.
    Shutdown → release model memory cleanly.
    """
    logger.info("=" * 60)
    logger.info("ZnO Supercapacitor AI Platform — starting up")
    logger.info("Environment : %s", settings.APP_ENV)
    logger.info("Models to load: %s", settings.ENABLED_MODELS)
    logger.info("Docs enabled: %s", settings.DOCS_ENABLED)

    model_registry.load_all()

    loaded = model_registry.loaded_models()
    logger.info("Models loaded: %s", loaded)

    if settings.ENABLE_WARMUP:
        _run_warmup()

    logger.info("Swagger docs  → http://localhost:%s/docs", settings.API_PORT)
    logger.info("Ready probe   → http://localhost:%s/api/v1/health/ready", settings.API_PORT)
    logger.info("=" * 60)

    yield  # ← server runs here, handling requests

    logger.info("Shutting down — releasing model memory...")
    model_registry.unload_all()
    logger.info("Shutdown complete.")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="ZnO Supercapacitor AI Platform",
    description=(
        "## ML-powered Cyclic Voltammetry (CV) prediction for ZnO supercapacitors\n\n"
        "This API lets you:\n"
        "- **Predict** full CV curves (651 data points) using 6 different ML models\n"
        "- **Compare** multiple models side-by-side on the same input\n"
        "- **Benchmark** model performance with pre-computed evaluation metrics\n\n"
        "### Models available\n"
        "| ID | Type | Size | Best for |\n"
        "|----|------|------|----------|\n"
        "| `rf` | Random Forest | 615 MB | Highest accuracy |\n"
        "| `lightgbm` | Gradient Boosting | 1.7 MB | Best accuracy/size trade-off |\n"
        "| `xgboost` | Gradient Boosting | 0.27 MB | Fast & accurate |\n"
        "| `gru` | Recurrent Neural Net | 0.34 MB | Best on new materials |\n"
        "| `lstm` | Recurrent Neural Net | 0.43 MB | Sequence modelling |\n"
        "| `ann` | Dense Neural Net | 0.18 MB | Smallest footprint |\n\n"
        "### Quick start\n"
        "1. Click **POST /api/v1/predict** below\n"
        "2. Click **Try it out**\n"
        "3. Send `{\"model_name\": \"rf\", \"material_id\": \"NM1\", \"scan_rate_mVs\": 20}`\n"
        "4. See the full 651-point CV curve in the response\n"
    ),
    version="2.0.0",
    # Conditionally expose interactive docs — disable in locked-down prod deployments
    docs_url  = "/docs"   if settings.DOCS_ENABLED else None,
    redoc_url = "/redoc"  if settings.DOCS_ENABLED else None,
    lifespan  = lifespan,
)


# ── Middleware (applied in reverse order — last added = outermost) ────────────

# 1. Request size limiter — must be outermost to reject early
app.add_middleware(RequestSizeLimitMiddleware)

# 2. CORS — allows the React frontend to call this API across origins
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["*"],
)

# 3. Timing logger — adds X-Response-Time header + structured access log
app.add_middleware(TimingMiddleware)


# ── Exception handlers ─────────────────────────────────────────────────────────
app.add_exception_handler(ModelNotFoundError,     model_not_found_handler)
app.add_exception_handler(ModelNotLoadedError,    model_not_loaded_handler)
app.add_exception_handler(InvalidInputError,      invalid_input_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception,              generic_error_handler)


# ── API routes ─────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")


# ── Root endpoint ──────────────────────────────────────────────────────────────
@app.get(
    "/",
    tags=["Root"],
    summary="API info",
    description="Returns basic API information and links to documentation.",
    include_in_schema=True,
)
def root() -> dict:
    """Welcome endpoint — confirms the API is running."""
    loaded = model_registry.loaded_models()
    return {
        "name":          "ZnO Supercapacitor AI Platform",
        "version":       "2.0.0",
        "environment":   settings.APP_ENV,
        "status":        "running",
        "models_loaded": loaded,
        "docs": {
            "swagger":  "/docs"    if settings.DOCS_ENABLED else "disabled",
            "redoc":    "/redoc"   if settings.DOCS_ENABLED else "disabled",
            "health":   "/api/v1/health",
            "ready":    "/api/v1/health/ready",
            "system":   "/api/v1/health/system",
            "predict":  "/api/v1/predict",
            "compare":  "/api/v1/compare",
            "metrics":  "/api/v1/metrics",
            "benchmarks": "/api/v1/benchmarks/leaderboard",
        },
    }
