"""
Health, readiness, and system metrics endpoints.

GET /api/v1/health   — Liveness probe: is the server running?
GET /api/v1/ready    — Readiness probe: are hot models loaded?
GET /api/v1/system   — System metrics: uptime, loaded models, memory usage.

Use /health for liveness checks (restart if failing).
Use /ready  for readiness checks (stop routing traffic if failing).
"""
import time
import logging
from fastapi import APIRouter, Response, status

from app.core.model_registry import model_registry, HOT_MODELS
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Server start time (module-level, set once at import) ─────────────────────
_START_TIME = time.time()


# ── GET /health — liveness probe ─────────────────────────────────────────────

@router.get(
    "",
    summary="Liveness check",
    description=(
        "Returns the current status of the backend server. "
        "Check 'models_loaded' to see which models are ready for inference."
    ),
)
def health_check() -> dict:
    """
    Liveness probe — confirms the server process is running.

    Always returns 200 OK while the process is alive, regardless of model
    load status. Use /ready to check model readiness.
    """
    scalers_ok = settings.SCALERS_PATH.exists()
    return {
        "status":        "ok",
        "version":       "1.0.0",
        "models_loaded": settings.ENABLED_MODELS,
        "scalers_ok":    scalers_ok,
        "environment":   settings.APP_ENV,
    }


# ── GET /ready — readiness probe ──────────────────────────────────────────────

@router.get(
    "/ready",
    summary="Readiness check",
    description=(
        "Returns 200 when all hot-tier models are loaded and ready for inference. "
        "Returns 503 if startup loading is still in progress."
    ),
)
def readiness_check(response: Response) -> dict:
    """
    Readiness probe — confirms hot models are loaded and inference is possible.

    Hot models (rf, lightgbm, gru) must all be loaded before this returns 200.
    Used by Render / Kubernetes to delay traffic until the server is ready.
    """
    loaded    = set(model_registry.loaded_models())
    # Only check hot models that are actually enabled in this deployment
    required  = HOT_MODELS & set(settings.ENABLED_MODELS)
    missing   = required - loaded

    if missing:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status":  "not_ready",
            "message": f"Hot models still loading: {sorted(missing)}",
            "loaded":  sorted(loaded),
            "missing": sorted(missing),
        }

    return {
        "status":  "ready",
        "loaded":  sorted(loaded),
        "missing": [],
    }


# ── GET /system — system metrics ──────────────────────────────────────────────

@router.get(
    "/system",
    summary="System metrics",
    description="Returns uptime, loaded models, memory footprint, and cache stats.",
)
def system_metrics() -> dict:
    """
    Basic system metrics for monitoring dashboards.

    Reports uptime, loaded models and their approximate disk sizes,
    and prediction cache hit rate if caching is enabled.
    """
    uptime_s = time.time() - _START_TIME
    loaded   = model_registry.loaded_models()

    # Compute approximate model file sizes
    model_sizes: dict[str, str] = {}
    from app.core.config import PLATFORM_ROOT
    import json
    reg_path = settings.MODEL_DIR / "shared" / "model_registry.json"
    if reg_path.exists():
        registry = json.loads(reg_path.read_text(encoding="utf-8"))
        for model_id in loaded:
            spec = registry.get("models", {}).get(model_id, {})
            if spec:
                fp = PLATFORM_ROOT / spec.get("file", "")
                if fp.exists():
                    mb = fp.stat().st_size / 1e6
                    model_sizes[model_id] = f"{mb:.1f} MB"

    # Prediction cache stats (imported lazily to avoid circular imports)
    cache_info: dict = {}
    try:
        from app.services.predictor import _cached_predict
        ci = _cached_predict.cache_info()
        cache_info = {
            "hits":    ci.hits,
            "misses":  ci.misses,
            "maxsize": ci.maxsize,
            "currsize": ci.currsize,
        }
    except Exception:
        pass

    return {
        "status":       "ok",
        "environment":  settings.APP_ENV,
        "uptime_seconds": round(uptime_s, 1),
        "uptime_human": _format_uptime(uptime_s),
        "models_loaded": loaded,
        "model_sizes":   model_sizes,
        "cache":         cache_info,
        "docs_enabled":  settings.DOCS_ENABLED,
    }


def _format_uptime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
