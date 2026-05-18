"""
config.py — Central application configuration.

All paths are computed AUTOMATICALLY from this file's location, so the
backend works correctly no matter which directory you run uvicorn from.

Directory layout assumed:
    ZnO_Supercapacitor_AI_Platform/
    ├── backend/
    │   ├── .env                ← put environment overrides here
    │   └── app/
    │       └── core/
    │           └── config.py  ← YOU ARE HERE
    ├── models/
    │   └── shared/
    │       └── scalers/
    │           └── scalers.json
    └── research/
        ├── metrics/
        └── exports/

IMPORTANT — .env list format:
    pydantic-settings v2 requires JSON array syntax for List[] fields:
        ENABLED_MODELS=["rf","lightgbm","gru"]   ← correct
        ENABLED_MODELS=rf,lightgbm,gru           ← WRONG (JSON parse error)
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Auto-detect the platform root ────────────────────────────────────────────
# config.py lives at: backend/app/core/config.py
# .parents[0] = backend/app/core/
# .parents[1] = backend/app/
# .parents[2] = backend/
# .parents[3] = ZnO_Supercapacitor_AI_Platform/   ← PLATFORM ROOT
PLATFORM_ROOT: Path = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """
    All application settings, read from .env or environment variables.

    List fields must use JSON array syntax in .env:
        ENABLED_MODELS=["rf","lightgbm","gru"]
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Don't crash if extra env vars exist (e.g. system PATH, HOME)
        extra="ignore",
    )

    # ── General ──────────────────────────────────────────────────────────────
    APP_ENV:   str = "development"
    LOG_LEVEL: str = "INFO"
    API_PORT:  int = 8000

    # ── API docs visibility ───────────────────────────────────────────────────
    # Set to False in production to hide /docs and /redoc from the public.
    DOCS_ENABLED: bool = True

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Origins allowed to call the API (React frontend, etc.)
    # Format: JSON array  →  ["http://localhost:5173","http://localhost:3000"]
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # ── Models to load at startup ────────────────────────────────────────────
    # Format: JSON array  →  ["rf","lightgbm","gru"]
    # All 6 options: "rf", "lightgbm", "gru", "xgboost", "ann", "lstm"
    ENABLED_MODELS: List[str] = ["rf", "lightgbm", "gru"]

    # ── Model display order for leaderboard / metrics responses ──────────────
    MODEL_ORDER: List[str] = ["rf", "lightgbm", "xgboost", "gru", "lstm", "ann"]

    # ── Startup warm-up inference ─────────────────────────────────────────────
    # When True, one inference is run per hot-loaded model immediately after
    # startup. This pre-compiles TF graphs and warms JIT caches so the first
    # real request is fast.
    ENABLE_WARMUP: bool = True
    WARMUP_MATERIAL: str = "NM1"
    WARMUP_SCAN_RATE: int = 30

    # ── Prediction result cache ───────────────────────────────────────────────
    # Maximum number of (model, material, scan_rate) results to cache in memory.
    # Set to 0 to disable caching.
    PREDICTION_CACHE_SIZE: int = 128

    # ── Request size limit ────────────────────────────────────────────────────
    # Maximum allowed request body size in bytes (default 1 MB).
    # Prevents oversized payloads from crashing the server.
    MAX_REQUEST_SIZE_BYTES: int = 1 * 1024 * 1024  # 1 MB

    # ── File paths (auto-computed from PLATFORM_ROOT — do NOT change) ────────
    MODEL_DIR:      Path = PLATFORM_ROOT / "models"
    SCALERS_PATH:   Path = PLATFORM_ROOT / "models" / "shared" / "scalers" / "scalers.json"
    METRICS_DIR:    Path = PLATFORM_ROOT / "research" / "metrics"
    EXPORTS_DIR:    Path = PLATFORM_ROOT / "research" / "exports"
    PER_GROUP_DIR:  Path = PLATFORM_ROOT / "research" / "per_group_csv"
    PARQUET_PATH:   Path = PLATFORM_ROOT / "data" / "processed" / "master_long_format.parquet"


settings = Settings()
