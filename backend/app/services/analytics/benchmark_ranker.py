"""
benchmark_ranker.py — Loads and serves the final model comparison data.

DATA SOURCES (all pre-computed during the research/training phase):
  research/exports/final_leaderboard.csv   — ranked summary (1 row per model)
  research/exports/comparison_metrics.csv  — full metrics table (4 rows per model)
  research/exports/final_summary.json      — key findings + model metadata

WHY CSVs INSTEAD OF RE-COMPUTING?
  The metrics were already computed rigorously during training with proper
  train/val/test splits. Re-computing from the JSON files would require
  re-parsing partition names and could introduce rounding differences.
  Using the pre-exported CSVs keeps the leaderboard 100% reproducible.
"""
import csv
import json
import logging
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── CSV model name → API model_id mapping ────────────────────────────────────
# The CSVs use title case (RF, LightGBM) while the API uses lowercase (rf, lightgbm)
_CSV_NAME_TO_ID: dict[str, str] = {
    "RF":       "rf",
    "XGBoost":  "xgboost",
    "LightGBM": "lightgbm",
    "ANN":      "ann",
    "LSTM":     "lstm",
    "GRU":      "gru",
}


def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return a list of row dicts with auto-typed values."""
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed: dict = {}
            for key, val in row.items():
                try:
                    typed[key] = int(val)
                except ValueError:
                    try:
                        typed[key] = float(val)
                    except ValueError:
                        typed[key] = val
            rows.append(typed)
    return rows


@lru_cache(maxsize=1)
def get_leaderboard() -> list[dict]:
    """
    Return the ranked leaderboard (one entry per model).

    Loads from final_leaderboard.csv. Result is cached after first call.
    Each dict contains: rank, model_id, overall_rank, RMSE metrics, R² metrics,
    model_size_MB, and training_time_min.
    """
    path = settings.EXPORTS_DIR / "final_leaderboard.csv"
    if not path.exists():
        logger.warning("final_leaderboard.csv not found at %s", path)
        return []

    rows = _read_csv(path)
    leaderboard = []
    for row in rows:
        csv_name = row.get("Model", "")
        model_id = _CSV_NAME_TO_ID.get(csv_name, csv_name.lower())
        leaderboard.append({
            "rank":              row.get("Overall"),
            "model_id":          model_id,
            "model_display":     csv_name,
            "composite_rank":    row.get("composite_rank"),
            "rmse_val_uA":       row.get("RMSE_val"),
            "rmse_test_sr_uA":   row.get("RMSE_test_SR"),
            "rmse_test_mat_uA":  row.get("RMSE_test_MAT"),
            "r2_val":            row.get("R2_val"),
            "r2_test_sr":        row.get("R2_test_SR"),
            "r2_test_mat":       row.get("R2_test_MAT"),
            "size_MB":           row.get("size_MB"),
            "train_min":         row.get("train_min"),
        })

    # Sort by composite_rank ascending (already ranked in CSV, but be explicit)
    leaderboard.sort(key=lambda r: r["rank"] or 999)
    return leaderboard


@lru_cache(maxsize=1)
def get_comparison_table() -> list[dict]:
    """
    Return per-partition metrics for all models (4 partitions × 6 models = 24 rows).

    Loads from comparison_metrics.csv. Contains train / val / test_SR / test_MAT
    rows for each model so the frontend can build a detailed comparison table.
    """
    path = settings.EXPORTS_DIR / "comparison_metrics.csv"
    if not path.exists():
        logger.warning("comparison_metrics.csv not found at %s", path)
        return []

    rows = _read_csv(path)
    result = []
    for row in rows:
        csv_name = row.get("Model", "")
        model_id = _CSV_NAME_TO_ID.get(csv_name, csv_name.lower())
        result.append({
            "model_id":    model_id,
            "model_display": csv_name,
            "partition":   row.get("Partition"),
            "rmse_uA":     row.get("RMSE_uA"),
            "mae_uA":      row.get("MAE_uA"),
            "max_err_uA":  row.get("MaxErr_uA"),
            "r2":          row.get("R2"),
            "rmse_norm":   row.get("RMSE_norm"),
        })
    return result


@lru_cache(maxsize=1)
def get_summary() -> dict:
    """
    Return the high-level research summary (key findings + model metadata).

    Loads from final_summary.json. Includes best model per criterion,
    model sizes, training times, and parameter counts.
    """
    path = settings.EXPORTS_DIR / "final_summary.json"
    if not path.exists():
        logger.warning("final_summary.json not found at %s", path)
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "generated":          data.get("generated"),
        "project":            data.get("project"),
        "key_findings":       data.get("key_findings", {}),
        "model_size_MB":      data.get("model_size_mb", {}),
        "train_time_minutes": data.get("approx_training_time_minutes", {}),
        "complexity":         data.get("complexity_params_or_trees", {}),
    }
