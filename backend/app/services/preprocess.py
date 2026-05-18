"""
preprocess.py — Full preprocessing pipeline (one function to rule them all).

WHAT THIS MODULE DOES:
  Takes a scan_rate_mVs value and returns everything the models need:
    - A normalised feature DataFrame  (for RF / LightGBM / XGBoost / ANN)
    - A padded sequence tensor        (for LSTM / GRU)
    - The raw potential array         (to include in the API response)

  This is the SINGLE entry point for all preprocessing.
  Internally it calls:
    1. feature_engineer.py  →  generate_potential_sweep + build_raw_features
    2. normalizer.py        →  normalize (apply min-max scaling)
    3. sequence_builder.py  →  build_padded_sequence (for LSTM/GRU only)

PIPELINE DIAGRAM:
  scan_rate_mVs
       │
       ▼
  generate_potential_sweep()   →  V array (651 points)
       │
       ▼
  build_raw_features()         →  DataFrame (651 × 10), raw values
       │
       ▼
  normalize()                  →  DataFrame (651 × 10), normalised values
       │
       ├──► (for RF/LGBM/ANN)  use directly as model.predict(df.values)
       │
       └──► build_padded_sequence()  →  tensor (1, 651, 10) for LSTM/GRU

USAGE EXAMPLE:
  from app.services.preprocess import build_inference_inputs

  V, features_norm, seq_tensor, seq_len = build_inference_inputs(
      scan_rate_mVs=30.0
  )
"""
import numpy as np
import pandas as pd
from typing import Tuple

from app.services.preprocessing.feature_engineer import (
    build_raw_features,
    generate_potential_sweep,
    V_LOWER, V_UPPER, N_POINTS,
)
from app.services.preprocessing.normalizer import normalize
from app.services.preprocessing.sequence_builder import build_padded_sequence


def build_inference_inputs(
    scan_rate_mVs: float,
    v_lower: float = V_LOWER,
    v_upper: float = V_UPPER,
    n_points: int  = N_POINTS,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, int]:
    """
    Run the complete preprocessing pipeline for a given scan rate.

    Args:
        scan_rate_mVs: Scan rate in mV/s (e.g. 30.0)
        v_lower:       Lower potential limit, default -0.65 V
        v_upper:       Upper potential limit, default  0.0  V
        n_points:      Total sweep points,   default  651

    Returns a 4-tuple:
        potential_V   : np.ndarray (n_points,)   — raw potential array in Volts
        features_norm : pd.DataFrame (n_points, 10) — normalised features
                        → pass directly to tabular models (RF / LightGBM / ANN)
        seq_tensor    : np.ndarray (1, 651, 10) float32 — padded sequence tensor
                        → pass directly to sequence models (LSTM / GRU)
        actual_len    : int — actual number of real rows (= n_points)
                        → use to unpad the GRU/LSTM output
    """
    # Step 1 — Generate the potential sweep array
    potential_V = generate_potential_sweep(v_lower, v_upper, n_points)

    # Step 2 — Build all 10 raw features
    features_raw = build_raw_features(scan_rate_mVs, v_lower, v_upper, n_points)

    # Step 3 — Normalise using scalers.json
    features_norm = normalize(features_raw)

    # Step 4 — Build padded sequence tensor for LSTM/GRU
    seq_tensor, actual_len = build_padded_sequence(features_norm)

    return potential_V, features_norm, seq_tensor, actual_len
