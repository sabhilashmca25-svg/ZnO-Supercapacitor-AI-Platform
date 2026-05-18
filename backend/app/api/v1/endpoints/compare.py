"""
POST /api/v1/compare — Multi-model CV trajectory comparison.

HOW IT WORKS:
  Runs the SAME electrochemical input through multiple models simultaneously.
  Returns ALL predictions on a shared potential axis — perfect for overlay plots.

  The potential sweep is generated ONCE and shared across all models.
  Only the current predictions differ per model.

  Models that fail (not loaded, or errored) are listed in `models_failed`
  instead of crashing the whole request.

RESPONSE USAGE (for Plotly multi-trace overlay):
  x = response["potential_V"]   # shared x-axis for all traces
  for model_id, pred in response["predictions"].items():
      add_trace(x=x, y=pred["predicted_current_uA"], name=model_id)
"""
import logging
from fastapi import APIRouter

from app.schemas.comparison import ComparisonRequest, ComparisonResponse, ModelPrediction
from app.services.predictor import run_prediction
from app.services.preprocessing.feature_engineer import generate_potential_sweep
from app.core.exceptions import ModelNotFoundError, ModelNotLoadedError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=ComparisonResponse,
    summary="Compare multiple models on the same CV input",
    description=(
        "Runs the same electrochemical conditions through multiple ML models at once. "
        "Returns predictions from all requested models on a shared potential axis, "
        "ready for a multi-trace Plotly overlay chart."
    ),
)
def compare_models(request: ComparisonRequest) -> ComparisonResponse:
    """
    Run the same CV sweep through multiple models and return all predictions.

    **Request body:**
    - `models` — list of model IDs, e.g. ["rf", "lightgbm", "gru"]
    - `material_id` — which ZnO material (NM1, NM2, NM3, or NM4)
    - `scan_rate_mVs` — scan rate in mV/s (10, 20, 30, ... 100)

    **Response:**
    - `potential_V` — shared potential array (651 points, Volts)
    - `predictions` — dict mapping model_id → its current predictions + stats
    - `models_failed` — list of models that couldn't be run (not loaded / errored)

    **Tip:** Models that are not loaded are silently skipped and listed in
    `models_failed` — the response still succeeds for the other models.
    """
    predictions:   dict = {}
    models_failed: list = []

    # Deduplicate and normalise model names
    requested_models = list(dict.fromkeys(m.lower().strip() for m in request.models))

    for model_name in requested_models:
        try:
            result = run_prediction(
                model_name    = model_name,
                material_id   = request.material_id,
                scan_rate_mVs = request.scan_rate_mVs,
            )
            predictions[model_name] = ModelPrediction(
                predicted_current_uA = result.predicted_current_uA,
                peak_anodic_uA       = result.statistics.peak_anodic_uA,
                peak_cathodic_uA     = result.statistics.peak_cathodic_uA,
                current_range_uA     = result.statistics.current_range_uA,
                integral_area        = result.statistics.integral_area,
            )
        except (ModelNotFoundError, ModelNotLoadedError) as exc:
            logger.warning("Skipping model '%s': %s", model_name, exc)
            models_failed.append(model_name)
        except Exception as exc:
            logger.exception("Model '%s' raised an unexpected error: %s", model_name, exc)
            models_failed.append(model_name)

    # Generate the shared potential axis
    potential_V = generate_potential_sweep().tolist()
    potential_V = [round(v, 6) for v in potential_V]

    return ComparisonResponse(
        material_id   = request.material_id,
        scan_rate_mVs = request.scan_rate_mVs,
        n_points      = len(potential_V),
        potential_V   = potential_V,
        predictions   = predictions,
        models_failed = models_failed,
    )
