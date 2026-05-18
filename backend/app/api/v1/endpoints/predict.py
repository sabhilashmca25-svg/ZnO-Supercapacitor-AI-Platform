"""
POST /api/v1/predict — Single-model CV trajectory prediction.

HOW IT WORKS:
  1. You send: model name + material ID + scan rate
  2. The backend generates the potential sweep (651 points, -0.65V → 0V → -0.65V)
  3. Computes all 10 electrochemical features
  4. Normalises features using the training scalers
  5. Runs the selected model (RF / LightGBM / GRU / etc.)
  6. Inverse-transforms predictions back to real µA values
  7. Returns the full CV curve + statistics

RESPONSE USAGE (for Plotly graph):
  x = response["potential_V"]            # x-axis
  y = response["predicted_current_uA"]  # y-axis
  title = f"{response['model_name']} prediction for {response['material_id']}"
"""
from fastapi import APIRouter
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.predictor import run_prediction

router = APIRouter()


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict CV trajectory for one model",
    description=(
        "Runs the complete ML prediction pipeline for a single model. "
        "Returns the full cyclic voltammetry curve as potential (V) and "
        "predicted current (µA) arrays, ready for plotting."
    ),
)
def predict_cv(request: PredictionRequest) -> PredictionResponse:
    """
    Predict a cyclic voltammetry curve using one ML model.

    **Request body:**
    - `model_name` — which model to use (rf, lightgbm, or gru)
    - `material_id` — which ZnO material (NM1, NM2, NM3, or NM4)
    - `scan_rate_mVs` — how fast to sweep (10, 20, 30, ... 100 mV/s)

    **Response:**
    - `potential_V` — 651 voltage values from -0.65V to 0V and back
    - `predicted_current_uA` — 651 predicted current values in µA
    - `statistics` — peak anodic/cathodic currents, range, integral area

    **Errors:**
    - 404 if model_name is not recognised
    - 503 if model is recognised but not loaded (not in ENABLED_MODELS)
    - 422 if material_id or scan_rate_mVs is invalid
    """
    result = run_prediction(
        model_name    = request.model_name,
        material_id   = request.material_id,
        scan_rate_mVs = request.scan_rate_mVs,
    )

    return PredictionResponse(
        model_name           = result.model_name,
        material_id          = result.material_id,
        scan_rate_mVs        = result.scan_rate_mVs,
        n_points             = result.n_points,
        potential_V          = result.potential_V,
        predicted_current_uA = result.predicted_current_uA,
        statistics           = result.statistics,
    )
