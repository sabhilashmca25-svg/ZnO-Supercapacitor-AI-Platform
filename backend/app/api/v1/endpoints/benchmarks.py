"""
GET /api/v1/benchmarks/leaderboard  — Overall ranked comparison of all 6 models.
GET /api/v1/benchmarks/comparison   — Full per-partition metrics table (24 rows).
GET /api/v1/benchmarks/summary      — Key findings + model size / training time.

HOW IT WORKS:
  All data is pre-computed from the research/training phase and stored as CSV
  and JSON files in research/exports/. These endpoints simply load and serve
  that data — no computation happens at request time.

RANKING METHODOLOGY:
  Models are ranked on three independent test sets:
    1. val      — scan rate 30 mV/s interpolation (seen materials)
    2. test_SR  — scan rate 50 mV/s (unseen scan rate)
    3. test_MAT — NM4 material, all scan rates (unseen material — hardest)

  A composite rank is computed as the average of the three individual ranks.
  Lower composite rank = better overall generalisation.

RESULT SUMMARY (from training):
  #1  RF        — Best composite rank; highest val R² (0.985)
  #2  LightGBM  — Near-identical to RF; ~360× smaller file (1.7 MB vs 615 MB)
  #3  XGBoost   — Strong interpolation; slightly weaker extrapolation
  #4  GRU       — Best test_MAT R² (0.975); best at unseen material
  #5  LSTM      — Good generalisation; larger than GRU
  #6  ANN       — Weakest performer; fastest to serve (0.17 MB)
"""
import logging
from fastapi import APIRouter, HTTPException

from app.services.analytics.benchmark_ranker import (
    get_leaderboard,
    get_comparison_table,
    get_summary,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/leaderboard",
    summary="Overall model leaderboard",
    description=(
        "Returns all 6 models ranked by composite performance across three test sets "
        "(val, test_SR, test_MAT). Use this to show users which model performs best "
        "overall and help them choose a model for prediction."
    ),
)
def leaderboard() -> list:
    """
    Return the final ranked model leaderboard.

    **Response fields per model:**
    - `rank` — overall rank (1 = best)
    - `model_id` — API identifier (e.g. "rf", "lightgbm")
    - `composite_rank` — average of individual partition ranks
    - `rmse_val_uA` — RMSE on validation set (µA)
    - `rmse_test_sr_uA` — RMSE on unseen scan rate (µA)
    - `rmse_test_mat_uA` — RMSE on unseen material NM4 (µA)
    - `r2_val`, `r2_test_sr`, `r2_test_mat` — R² for each partition
    - `size_MB` — serialised model file size in megabytes
    - `train_min` — approximate training time in minutes
    """
    data = get_leaderboard()
    if not data:
        raise HTTPException(
            status_code=503,
            detail="Leaderboard data not available. Check research/exports/final_leaderboard.csv",
        )
    return data


@router.get(
    "/comparison",
    summary="Full per-partition comparison table",
    description=(
        "Returns all metrics for all 6 models across all 4 evaluation partitions "
        "(train, val, test_SR, test_MAT). 24 rows total (6 models × 4 partitions). "
        "Use this to build a detailed comparison table or chart in the frontend."
    ),
)
def comparison() -> list:
    """
    Return per-partition metrics for all models.

    **Response fields per row:**
    - `model_id` — API identifier (e.g. "rf")
    - `partition` — train / val / test_SR / test_MAT
    - `rmse_uA` — Root Mean Squared Error in µA
    - `mae_uA` — Mean Absolute Error in µA
    - `max_err_uA` — worst-case error in µA
    - `r2` — R² coefficient of determination
    - `rmse_norm` — RMSE in normalised [0,1] units

    **Partitions:**
    - `train` — in-sample (all scan rates except 30 & 50, materials NM1–NM3)
    - `val` — scan rate 30 mV/s, materials NM1–NM3 (interpolation)
    - `test_SR` — scan rate 50 mV/s, materials NM1–NM3 (unseen scan rate)
    - `test_MAT` — all scan rates, material NM4 (unseen material — hardest)
    """
    data = get_comparison_table()
    if not data:
        raise HTTPException(
            status_code=503,
            detail="Comparison data not available. Check research/exports/comparison_metrics.csv",
        )
    return data


@router.get(
    "/summary",
    summary="Key research findings and model metadata",
    description=(
        "Returns high-level research findings: best model by each criterion, "
        "model file sizes, approximate training times, and complexity (params/trees). "
        "Useful for displaying a research summary card in the frontend."
    ),
)
def summary() -> dict:
    """
    Return the high-level research summary.

    **Response fields:**
    - `key_findings` — best model by val RMSE, test_MAT R², deployment, etc.
    - `model_size_MB` — serialised model file sizes
    - `train_time_minutes` — approximate wall-clock training time
    - `complexity` — number of parameters (ANNs/LSTMs) or trees (RF/boosting)
    - `generated` — ISO timestamp of when this summary was created
    """
    data = get_summary()
    if not data:
        raise HTTPException(
            status_code=503,
            detail="Summary data not available. Check research/exports/final_summary.json",
        )
    return data
