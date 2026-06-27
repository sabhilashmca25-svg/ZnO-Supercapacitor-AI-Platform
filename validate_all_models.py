"""
validate_all_models.py - Stage 3: Comprehensive model artifact validation.
Loads every deployed model, runs inference on all 3 test partitions,
computes RMSE/MAE/MaxErr/R2 and compares against Colab ground-truth.
"""
import json, sys, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent
DATA    = BASE / "data" / "processed"
MODELS  = BASE / "models"
# Ground-truth comparison files (Colab training outputs).
# Set env var TRUTH_PATH to override, or place files in research/comparison_inputs/.
import os as _os
TRUTH   = Path(_os.environ.get("TRUTH_PATH", "")) if _os.environ.get("TRUTH_PATH") \
          else BASE / "research" / "comparison_inputs"

FEATURES = [
    "potential_V_norm", "scan_rate_mVs_norm", "log_scan_rate_norm",
    "sqrt_scan_rate_norm", "potential_from_lower_norm", "potential_from_upper_norm",
    "sr_x_potential_norm", "direction_x_potential_norm",
    "sweep_direction", "sweep_position",
]
TARGET = "current_normalized"
MAX_SEQ_LEN = 651
TOLERANCE = 1.5  # uA - differences within this are PASS

# ── Load splits ───────────────────────────────────────────────────────────────
print("Loading data partitions...")
val_df     = pd.read_parquet(DATA / "val_set.parquet")
test_sr_df = pd.read_parquet(DATA / "test_scanrate.parquet")
test_mat_df= pd.read_parquet(DATA / "test_material.parquet")
print(f"  val: {len(val_df):,}  test_sr: {len(test_sr_df):,}  test_mat: {len(test_mat_df):,}")

def compute_metrics(df, pred_norm, label):
    rng    = df["current_norm_max"] - df["current_norm_min"]
    pred_A = pred_norm * rng + df["current_norm_min"]
    rmse_uA   = np.sqrt(mean_squared_error(df["current_A"], pred_A)) * 1e6
    mae_uA    = mean_absolute_error(df["current_A"], pred_A) * 1e6
    maxerr_uA = float(np.abs(df["current_A"] - pred_A).max()) * 1e6
    r2        = r2_score(df[TARGET], pred_norm)
    return {"partition": label, "RMSE_uA": rmse_uA, "MAE_uA": mae_uA,
            "MaxErr_uA": maxerr_uA, "R2": r2}

def load_truth(fname):
    fpath = TRUTH / fname
    if not fpath.exists():
        return None
    with open(fpath) as f:
        d = json.load(f)
    # map short key -> truth row
    out = {}
    for r in d["metrics"]:
        p = r["partition"]
        if "val" in p.lower() and "SR=30" in p:
            out["val"] = r
        elif "test_SR" in p or "SR=50" in p:
            out["test_sr"] = r
        elif "test_MAT" in p or "NM4" in p:
            out["test_mat"] = r
    return out

def predict_tabular(model, df):
    return model.predict(df[FEATURES].to_numpy(dtype=np.float64))

def predict_sequence(model, df):
    """Group by (nm_id, scan_rate_mVs, half_sweep_id) — same key as training.

    Each triplet is one sequence of 649-651 points (fits in MAX_SEQ_LEN=651).
    Rows within each group are sorted by step_index (matching Colab training).
    Predictions are scattered back to each row's original DataFrame position.
    """
    pred_norm = np.zeros(len(df), dtype=np.float32)
    grp_keys  = ["nm_id", "scan_rate_mVs", "half_sweep_id"]
    for keys, grp in df.groupby(grp_keys, sort=True):
        grp_sorted = grp.sort_values("step_index")
        X_seq  = grp_sorted[FEATURES].to_numpy(dtype=np.float32)
        T      = len(X_seq)
        assert T <= MAX_SEQ_LEN, (
            f"Sequence {keys} has {T} rows > MAX_SEQ_LEN={MAX_SEQ_LEN}. "
            "Check grouping key or MAX_SEQ_LEN constant."
        )
        padded = np.zeros((1, MAX_SEQ_LEN, len(FEATURES)), dtype=np.float32)
        padded[0, :T, :] = X_seq
        out = model.predict(padded, verbose=0)  # (1, MAX_SEQ_LEN, 1)
        pred_norm[grp_sorted.index] = out[0, :T, 0]
    return pred_norm

results = {}

def validate_model(model_id, truth_file, predict_val, predict_sr, predict_mat):
    print(f"\n{'='*62}")
    print(f"  MODEL: {model_id.upper()}")
    print(f"{'='*62}")
    truth = load_truth(truth_file)
    results[model_id] = {}

    for key, df, pred_fn in [
        ("val",      val_df,      predict_val),
        ("test_sr",  test_sr_df,  predict_sr),
        ("test_mat", test_mat_df, predict_mat),
    ]:
        pred_norm = pred_fn(df)
        m = compute_metrics(df, pred_norm, key)
        t = truth.get(key)
        if t:
            diff = abs(m["RMSE_uA"] - t["RMSE_uA"])
            status = "PASS" if diff <= TOLERANCE else "FAIL"
            print(f"  {key:<12}  actual={m['RMSE_uA']:7.2f} uA  "
                  f"truth={t['RMSE_uA']:7.2f} uA  R2={m['R2']:.4f}  "
                  f"D={diff:.2f}  {status}")
        else:
            print(f"  {key:<12}  actual={m['RMSE_uA']:7.2f} uA  R2={m['R2']:.4f}  [no truth]")
        results[model_id][key] = {"actual": m, "truth": t}

# ── TABULAR MODELS ────────────────────────────────────────────────────────────
print("\nLoading RF...")
rf = joblib.load(MODELS / "rf" / "rf_baseline.joblib")
validate_model("rf", "rf_baseline_metrics.json",
    lambda df: predict_tabular(rf, df),
    lambda df: predict_tabular(rf, df),
    lambda df: predict_tabular(rf, df))

print("\nLoading LightGBM...")
lgb = joblib.load(MODELS / "lightgbm" / "lightgbm_model.joblib")
validate_model("lightgbm", "lightgbm_metrics.json",
    lambda df: predict_tabular(lgb, df),
    lambda df: predict_tabular(lgb, df),
    lambda df: predict_tabular(lgb, df))

print("\nLoading XGBoost...")
xgb = joblib.load(MODELS / "xgboost" / "xgboost_model.joblib")
validate_model("xgboost", "xgboost_metrics.json",
    lambda df: predict_tabular(xgb, df),
    lambda df: predict_tabular(xgb, df),
    lambda df: predict_tabular(xgb, df))

# ── DEEP LEARNING MODELS ──────────────────────────────────────────────────────
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

print("\nLoading ANN...")
ann = tf.keras.models.load_model(str(MODELS / "ann" / "ann_model.keras"))
validate_model("ann", "ann_metrics.json",
    lambda df: ann.predict(df[FEATURES].to_numpy(dtype=np.float32), verbose=0).flatten(),
    lambda df: ann.predict(df[FEATURES].to_numpy(dtype=np.float32), verbose=0).flatten(),
    lambda df: ann.predict(df[FEATURES].to_numpy(dtype=np.float32), verbose=0).flatten())

print("\nLoading LSTM...")
lstm = tf.keras.models.load_model(str(MODELS / "lstm" / "lstm_model.keras"))
validate_model("lstm", "lstm_metrics.json",
    lambda df: predict_sequence(lstm, df),
    lambda df: predict_sequence(lstm, df),
    lambda df: predict_sequence(lstm, df))

print("\nLoading GRU...")
gru = tf.keras.models.load_model(str(MODELS / "gru" / "gru_model.keras"))
validate_model("gru", "gru_metrics.json",
    lambda df: predict_sequence(gru, df),
    lambda df: predict_sequence(gru, df),
    lambda df: predict_sequence(gru, df))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*72)
print("FINAL VALIDATION SUMMARY")
print("="*72)
print(f"{'Model':<10} {'Partition':<12} {'Actual':>10} {'Truth':>10} {'Diff':>7} {'Status':<6}")
print("-"*72)

all_pass = True
for mid, partitions in results.items():
    for pkey, data in partitions.items():
        a = data["actual"]
        t = data["truth"]
        if t:
            diff = abs(a["RMSE_uA"] - t["RMSE_uA"])
            status = "PASS" if diff <= TOLERANCE else "WARN" if diff <= 3.0 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"{mid:<10} {pkey:<12} {a['RMSE_uA']:>9.2f}  {t['RMSE_uA']:>9.2f}  "
                  f"{diff:>6.2f}  {status}")

print()
if all_pass:
    print("ALL MODELS PASS (within tolerance).")
else:
    print("FAILURES DETECTED - see above.")
    sys.exit(1)
