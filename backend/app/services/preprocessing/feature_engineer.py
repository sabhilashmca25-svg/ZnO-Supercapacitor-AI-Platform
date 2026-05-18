"""
feature_engineer.py — Generate the 10 model input features for a CV sweep.

WHAT THIS MODULE DOES:
  Given a scan rate and an optional potential window, this module:
  1. Generates the potential sweep array  (V_lower → V_upper → V_lower)
  2. Assigns sweep direction labels       (0.0 = anodic, 1.0 = cathodic)
  3. Computes all 10 engineered features

  The output DataFrame has the SAME column names and order as during training.
  DO NOT change column order — models depend on it.

ELECTROCHEMICAL CONVENTIONS USED:
  - V_lower = -0.65 V  (cathodic limit)
  - V_upper =  0.0  V  (anodic limit)
  - Anodic sweep:   V increases from V_lower → V_upper  (sweep_direction = 0.0)
  - Cathodic sweep: V decreases from V_upper → V_lower  (sweep_direction = 1.0)

  WHY sweep_direction = 0.0 for ANODIC (matches training data):
    The master_long_format.parquet stores sweep_direction = 0 for the anodic
    half-sweep (direction 0) and sweep_direction = 1 for the cathodic half-sweep
    (direction 1). The model was trained on these parquet values directly, so
    inference MUST use the same convention.

    Consequence for direction_x_potential = sweep_direction × potential:
      anodic   (dir=0.0): 0.0 × potential∈[-0.65, 0] →  0         (constant)
      cathodic (dir=1.0): 1.0 × potential∈[-0.65, 0] → [-0.65, 0] (varies)
    This gives direction_x_potential a range of [-0.65, 0], which matches
    the scaler stored in scalers.json → global_features → direction_x_potential.
"""
import numpy as np
import pandas as pd


# ── Electrochemical constants ─────────────────────────────────────────────────
V_LOWER: float = -0.65   # V — cathodic (lower) potential limit
V_UPPER: float =  0.0    # V — anodic  (upper) potential limit
N_POINTS: int  =  651    # total data points per full sweep (matches training)

# ── Feature column names — EXACT order used during training ──────────────────
FEATURE_NAMES = [
    "potential_V_norm",           # 1. Electrode potential (V)
    "scan_rate_mVs_norm",         # 2. Scan rate (mV/s)
    "log_scan_rate_norm",         # 3. Natural log of scan rate
    "sqrt_scan_rate_norm",        # 4. Square root of scan rate
    "potential_from_lower_norm",  # 5. Distance from lower potential limit (V)
    "potential_from_upper_norm",  # 6. Distance from upper potential limit (V)
    "sr_x_potential_norm",        # 7. Scan rate × potential (interaction feature)
    "direction_x_potential_norm", # 8. Sweep direction × potential (interaction feature)
    "sweep_direction",            # 9. 0.0 = anodic, 1.0 = cathodic (not normalized — matches parquet)
    "sweep_position",             # 10. Position in sweep [0, 1] (not normalized)
]


def generate_potential_sweep(
    v_lower: float = V_LOWER,
    v_upper: float = V_UPPER,
    n_points: int  = N_POINTS,
) -> np.ndarray:
    """
    Generate the potential array for one full CV cycle.

    The sweep goes: V_lower → V_upper (anodic) then V_upper → V_lower (cathodic).
    Total points = n_points (default 651, matching training data).

    Returns:
        np.ndarray of shape (n_points,) with potential values in Volts.
    """
    # Split n_points between forward and reverse half-sweeps
    # e.g. for n_points=651: n_forward=326, n_reverse=325
    n_forward = (n_points + 1) // 2       # 326 points: V_lower → V_upper
    n_reverse = n_points - n_forward       # 325 points: V_upper → V_lower

    V_forward = np.linspace(v_lower, v_upper, n_forward)
    # Exclude V_upper from reverse (it's the last point of V_forward)
    V_reverse = np.linspace(v_upper, v_lower, n_reverse + 1)[1:]

    return np.concatenate([V_forward, V_reverse])   # shape: (651,)


def build_raw_features(
    scan_rate_mVs: float,
    v_lower: float = V_LOWER,
    v_upper: float = V_UPPER,
    n_points: int  = N_POINTS,
) -> pd.DataFrame:
    """
    Build a DataFrame of RAW (un-normalised) features for one CV sweep.

    This is Step 1 of the preprocessing pipeline.
    Pass the output to normalizer.normalize() to get the final model input.

    Args:
        scan_rate_mVs: Scan rate in mV/s (e.g. 30.0)
        v_lower:       Lower potential limit in V (default -0.65)
        v_upper:       Upper potential limit in V (default 0.0)
        n_points:      Total number of data points (default 651)

    Returns:
        pd.DataFrame with shape (n_points, 10) and columns = FEATURE_NAMES
    """
    V   = generate_potential_sweep(v_lower, v_upper, n_points)
    n   = len(V)
    sr  = float(scan_rate_mVs)

    # Number of points in the anodic half-sweep
    n_forward = (n + 1) // 2   # 326 for n=651
    n_reverse = n - n_forward  # 325 for n=651

    # Pre-compute constant features
    log_sr  = np.log(sr)         # log(scan_rate) — natural log
    sqrt_sr = np.sqrt(sr)        # sqrt(scan_rate)

    # Build each row
    rows = []
    for i, v in enumerate(V):
        # ── Sweep direction ────────────────────────────────────────────────
        # 0.0 = anodic   (V increasing, index < n_forward)  ← matches parquet
        # 1.0 = cathodic (V decreasing, index >= n_forward) ← matches parquet
        # MUST match the parquet convention used during training:
        #   parquet sweep_direction=0 → anodic, sweep_direction=1 → cathodic
        sweep_dir = 0.0 if i < n_forward else 1.0

        # ── Sweep position ─────────────────────────────────────────────────
        # IMPORTANT: computed PER DIRECTION (0 → 1 within each half-sweep),
        # matching the parquet convention where each direction resets to 0.
        #   parquet anodic:   step_index / (n_anodic  - 1)  → [0, 1]
        #   parquet cathodic: step_index / (n_cathodic - 1) → [0, 1]
        if i < n_forward:
            sweep_pos = i / (n_forward - 1) if n_forward > 1 else 0.0
        else:
            j = i - n_forward  # index within cathodic half, 0-based
            sweep_pos = j / (n_reverse - 1) if n_reverse > 1 else 0.0

        # ── Potential distances ────────────────────────────────────────────
        from_lower = v - v_lower    # V + 0.65, range [0, 0.65]
        from_upper = v_upper - v    # 0.0 - V,  range [0, 0.65]

        # ── Interaction features ───────────────────────────────────────────
        sr_x_pot  = sr * v                   # scan_rate × potential
        dir_x_pot = sweep_dir * v            # direction × potential

        rows.append({
            "potential_V_norm":           v,
            "scan_rate_mVs_norm":         sr,
            "log_scan_rate_norm":         log_sr,
            "sqrt_scan_rate_norm":        sqrt_sr,
            "potential_from_lower_norm":  from_lower,
            "potential_from_upper_norm":  from_upper,
            "sr_x_potential_norm":        sr_x_pot,
            "direction_x_potential_norm": dir_x_pot,
            "sweep_direction":            sweep_dir,
            "sweep_position":             sweep_pos,
        })

    return pd.DataFrame(rows, columns=FEATURE_NAMES)
