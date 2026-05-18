"""Aggregates all v1 API route prefixes.

Route map:
  GET  /api/v1/health          → liveness probe
  GET  /api/v1/health/ready    → readiness probe (hot models loaded?)
  GET  /api/v1/health/system   → system metrics (uptime, cache, models)
  POST /api/v1/predict         → single-model CV prediction
  POST /api/v1/compare         → multi-model comparison
  GET  /api/v1/models          → registered model list
  GET  /api/v1/metrics         → evaluation metrics
  GET  /api/v1/benchmarks/...  → leaderboard, comparison, summary
  GET  /api/v1/training-history/... → training curves
  GET  /api/v1/experimental/... → experimental CV data
  POST /api/v1/validation/...  → validation analysis
"""
from fastapi import APIRouter
from app.api.v1.endpoints import (
    health, predict, compare, models, metrics,
    benchmarks, training_history, experimental, validation,
)

api_router = APIRouter()
api_router.include_router(health.router,           prefix="/health",           tags=["Health"])
api_router.include_router(predict.router,          prefix="/predict",          tags=["Prediction"])
api_router.include_router(compare.router,          prefix="/compare",          tags=["Comparison"])
api_router.include_router(models.router,           prefix="/models",           tags=["Models"])
api_router.include_router(metrics.router,          prefix="/metrics",          tags=["Metrics"])
api_router.include_router(benchmarks.router,       prefix="/benchmarks",       tags=["Benchmarks"])
api_router.include_router(training_history.router, prefix="/training-history", tags=["Training History"])
api_router.include_router(experimental.router,     prefix="/experimental",     tags=["Experimental Data"])
api_router.include_router(validation.router,       prefix="/validation",       tags=["Validation"])
