"""
model_registry.py — Two-tier ML model loading with on-demand lazy loading.

TWO-TIER SYSTEM:
  TIER 1 — Hot-loaded at startup (primary deployment models):
    ✔ RF       — highest interpolation accuracy
    ✔ LightGBM — best deployment trade-off
    ✔ GRU      — best extrapolation (new materials)

  TIER 2 — Lazy-loaded on first request (research/comparison models):
    ✔ XGBoost  — competitive boosting baseline
    ✔ ANN      — feedforward network for comparison
    ✔ LSTM     — sequence model comparison vs GRU

  Once loaded, ALL models stay in memory for fast subsequent inference.
  This avoids slow server startup while keeping all 6 models accessible.

SUPPORTED LOADERS:
  - joblib  → RF, LightGBM, XGBoost (scikit-learn / tree-based)
  - keras   → ANN, LSTM, GRU (TensorFlow/Keras neural networks)
"""
import json
import logging
import threading
from pathlib import Path

from app.core.config import settings, PLATFORM_ROOT

logger = logging.getLogger(__name__)

# ── Two-tier categorisation ───────────────────────────────────────────────────
HOT_MODELS  = {"rf", "lightgbm", "gru"}      # loaded at startup
LAZY_MODELS = {"xgboost", "ann", "lstm"}     # loaded on first request


class ModelRegistry:
    """
    Singleton container for all trained ML models.

    Tier-1 models are loaded at startup via load_all().
    Tier-2 models are loaded on demand via ensure_loaded().
    All loaded models are cached indefinitely (no eviction).
    """

    def __init__(self):
        self._models:   dict = {}     # model_id → model object
        self._metadata: dict = {}     # model_id → metadata JSON
        self._registry: dict = {}     # model_id → spec (file, loader)
        self._lock = threading.Lock() # thread-safe lazy loading

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """
        Load Tier-1 (hot) models at application startup.
        Tier-2 (lazy) models are NOT loaded here — they load on first request.
        """
        reg_path = settings.MODEL_DIR / "shared" / "model_registry.json"
        if not reg_path.exists():
            logger.error("model_registry.json not found at: %s", reg_path)
            return

        registry = json.loads(reg_path.read_text(encoding="utf-8"))
        self._registry = registry.get("models", {})

        logger.info("=" * 55)
        logger.info("Model Registry — Tier-1 startup load")

        for model_id in settings.ENABLED_MODELS:
            if model_id not in self._registry:
                logger.warning("  [%s] not in model_registry.json — skipping", model_id)
                continue
            if model_id in HOT_MODELS:
                spec = self._registry[model_id]
                self._load_model(model_id, spec)
                self._load_metadata(model_id)
            else:
                logger.info("  [%s] Tier-2 lazy model — will load on first request", model_id)

        logger.info("Tier-1 loaded: %s", list(self._models.keys()))
        logger.info("Tier-2 ready:  %s  (lazy)", [m for m in LAZY_MODELS])
        logger.info("=" * 55)

    def ensure_loaded(self, model_id: str) -> bool:
        """
        Ensure a model is loaded, loading it now if necessary (lazy load).

        Thread-safe — uses a lock so concurrent first-requests don't double-load.

        Returns:
            True if model is loaded and ready, False if loading failed.
        """
        if self.is_loaded(model_id):
            return True

        with self._lock:
            # Double-check inside the lock (another thread may have loaded it)
            if self.is_loaded(model_id):
                return True

            if not self._registry:
                # Registry not yet initialised — load it now
                reg_path = settings.MODEL_DIR / "shared" / "model_registry.json"
                if reg_path.exists():
                    registry = json.loads(reg_path.read_text(encoding="utf-8"))
                    self._registry = registry.get("models", {})

            if model_id not in self._registry:
                logger.error("  [%s] not in model_registry.json — cannot lazy-load", model_id)
                return False

            tier = "Tier-1" if model_id in HOT_MODELS else "Tier-2"
            logger.info("On-demand loading %s model: %s", tier, model_id)
            spec = self._registry[model_id]
            self._load_model(model_id, spec)
            self._load_metadata(model_id)

        return self.is_loaded(model_id)

    def get(self, model_id: str):
        """
        Retrieve a loaded model object by its ID.
        Returns None if not loaded (call ensure_loaded first).
        """
        return self._models.get(model_id)

    def get_metadata(self, model_id: str) -> dict | None:
        return self._metadata.get(model_id)

    def all_metadata(self) -> dict:
        return self._metadata

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def model_tier(self, model_id: str) -> str:
        """Return 'hot', 'lazy', or 'unknown' for a given model_id."""
        if model_id in HOT_MODELS:
            return "hot"
        if model_id in LAZY_MODELS:
            return "lazy"
        return "unknown"

    def unload_all(self) -> None:
        """Release all model objects from memory (called on shutdown)."""
        self._models.clear()
        logger.info("ModelRegistry: all models unloaded.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self, model_id: str, spec: dict) -> None:
        """Load a single model from disk using the spec from model_registry.json."""
        model_path = PLATFORM_ROOT / spec["file"]

        if not model_path.exists():
            logger.error("  [%s] Model file not found: %s", model_id, model_path)
            return

        loader = spec.get("loader", "joblib")
        try:
            if loader == "joblib":
                import joblib
                self._models[model_id] = joblib.load(model_path)
                size_mb = model_path.stat().st_size / 1e6
                logger.info("  [%s] ✔ Loaded via joblib (%.1f MB)", model_id, size_mb)

            elif loader == "keras":
                import os
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                import tensorflow as tf
                self._models[model_id] = tf.keras.models.load_model(
                    str(model_path), compile=False
                )
                size_kb = model_path.stat().st_size / 1e3
                logger.info("  [%s] ✔ Loaded via Keras (%.0f KB)", model_id, size_kb)

            else:
                logger.error("  [%s] Unknown loader: '%s'", model_id, loader)

        except Exception as exc:
            logger.exception("  [%s] ✗ Failed to load: %s", model_id, exc)

    def _load_metadata(self, model_id: str) -> None:
        """Load the metadata.json for a model."""
        meta_path = settings.MODEL_DIR / model_id / "metadata.json"
        if meta_path.exists():
            self._metadata[model_id] = json.loads(
                meta_path.read_text(encoding="utf-8")
            )


# ── Module-level singleton ────────────────────────────────────────────────────
model_registry = ModelRegistry()
