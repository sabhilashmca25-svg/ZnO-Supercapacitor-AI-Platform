"""
build_xgboost_notebook.py
Generates notebooks/03_xgboost_model.ipynb
Run once: python src/build_xgboost_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb    = nbf.v4.new_notebook()
cells = []

def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
# Notebook 03 — XGBoost Model
## ZnO Supercapacitor CV Current Prediction · Gradient Boosting with Early Stopping

### Purpose
Train an **XGBoost gradient-boosted model** on the same scientifically correct
partitions established in Notebook 01, and compare against the Random Forest
baseline from Notebook 02.  This notebook:
- Loads pre-built artefacts from Notebook 01 — no preprocessing is repeated.
- Trains XGBRegressor with early stopping monitored on the validation set.
- Evaluates on the same three held-out partitions as the RF baseline.
- Produces 10 publication-ready verification figures saved to organised subdirectories.
- Delivers a head-to-head RF vs XGBoost comparison table.

### Three generalisation questions answered
| Partition | Question |
|-----------|----------|
| **Val** (NM1/2/3 × SR=30) | Can XGBoost interpolate to an unseen intermediate scan rate? |
| **Test-SR** (NM1/2/3 × SR=50) | Can XGBoost interpolate to SR=50 from {10,20,...,100}? |
| **Test-MAT** (NM4 × all SRs) | Can XGBoost generalise to a completely unseen ZnO formulation? |

### Why XGBoost may outperform RF on CV data
Gradient boosting builds trees **sequentially**, each correcting the residuals of
the prior ensemble.  For ZnO supercapacitor CV curves:
- **Residual correction**: scan-rate effects near electrode turning points benefit
  from iterative refinement that RF's parallel averaging smooths over.
- **L1/L2 regularisation** (`reg_alpha`, `reg_lambda`): explicit penalty on leaf
  weights prevents overfitting to training scan rates.
- **Gain importance**: XGBoost reports *gain* (total improvement in loss attributed
  to each feature split) which is more calibrated than RF's impurity-based MDI.
- **Early stopping**: automatically selects the optimal number of trees using the
  held-out validation set — no manual grid search on `n_estimators`.""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — UPLOAD + DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 0 — Upload Processed Files

Upload the **6 files** produced by Notebook 01 from your local machine.
A file-picker dialog will appear — select all 6 at once (Ctrl+click):

```
master_long_format.parquet
train_set.parquet
val_set.parquet
test_scanrate.parquet
test_material.parquet
scalers.json
```

All 6 are in the `data/processed/` folder of your PBL Project directory.""")

code("""\
from google.colab import files as _cf
from pathlib import Path

REQUIRED = [
    "master_long_format.parquet",
    "train_set.parquet",
    "val_set.parquet",
    "test_scanrate.parquet",
    "test_material.parquet",
    "scalers.json",
]

print("Select all 6 files in the dialog (Ctrl+click each):")
for f in REQUIRED:
    print(f"  {f}")
print()

_cf.upload()

PROCESSED_DIR = Path("/content")

BASE_DIR    = Path("/content/PBL_Project")
RESULTS_DIR = BASE_DIR / "results" / "xgboost"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
CSV_DIR     = RESULTS_DIR / "csv"
MODEL_DIR   = RESULTS_DIR / "model"

for d in [FIGURES_DIR, METRICS_DIR, CSV_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

missing = [f for f in REQUIRED if not (PROCESSED_DIR / f).exists()]
assert not missing, (
    "These files were not uploaded: " + str(missing)
    + "\\nRe-run this cell and select them in the dialog."
)

print("✓ All 6 files uploaded successfully")
print(f"  Source  : {PROCESSED_DIR}")
print(f"  Results : {RESULTS_DIR}")
print(f"    figures/ : {FIGURES_DIR}")
print(f"    metrics/ : {METRICS_DIR}")
print(f"    csv/     : {CSV_DIR}")
print(f"    model/   : {MODEL_DIR}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 1 — Load Preprocessing Artefacts

Loads the six outputs created by Notebook 01.  **Nothing is recomputed here** —
splits, normalisation, and feature engineering are fixed.""")

code("""\
import numpy  as np
import pandas as pd
import json
import warnings
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.6e}".format)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# ── Install / verify XGBoost ──────────────────────────────────────────────────
try:
    import xgboost as xgb
    print(f"XGBoost {xgb.__version__} already available")
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "xgboost", "-q"], check=True)
    import xgboost as xgb
    print(f"XGBoost installed: {xgb.__version__}")

# ── Load split DataFrames ─────────────────────────────────────────────────────
master_df   = pd.read_parquet(PROCESSED_DIR / "master_long_format.parquet")
train_df    = pd.read_parquet(PROCESSED_DIR / "train_set.parquet")
val_df      = pd.read_parquet(PROCESSED_DIR / "val_set.parquet")
test_sr_df  = pd.read_parquet(PROCESSED_DIR / "test_scanrate.parquet")
test_mat_df = pd.read_parquet(PROCESSED_DIR / "test_material.parquet")

# ── Load normalisation scalers ────────────────────────────────────────────────
with open(PROCESSED_DIR / "scalers.json") as fh:
    scalers = json.load(fh)

print()
print("✓ All artefacts loaded")
print(f"  master        : {master_df.shape}")
print(f"  train         : {train_df.shape}")
print(f"  val           : {val_df.shape}")
print(f"  test_scanrate : {test_sr_df.shape}")
print(f"  test_material : {test_mat_df.shape}")
print(f"  scaler groups : {len(scalers['current_per_group'])} (nm x scan_rate combinations)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SPLIT INTEGRITY VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 2 — Split Integrity Verification

Re-verify the split partitions before any modelling begins.
This is a guard against accidentally loading stale or mismatched artefacts.""")

code("""\
print("=== SPLIT INTEGRITY CHECK ===\\n")

# 1. Expected row counts
expected = {"train": 62400, "val": 7800, "test_scanrate": 7800, "test_material": 26000}
for name, exp_rows in expected.items():
    df = {"train": train_df, "val": val_df,
          "test_scanrate": test_sr_df, "test_material": test_mat_df}[name]
    assert len(df) == exp_rows, f"{name}: expected {exp_rows}, got {len(df)}"
    print(f"  [PASS] {name:15s}: {len(df):,} rows")

# 2. Correct materials per partition
assert set(train_df["nm_id"].unique())    == {"NM1", "NM2", "NM3"}
assert set(val_df["nm_id"].unique())      == {"NM1", "NM2", "NM3"}
assert set(test_sr_df["nm_id"].unique())  == {"NM1", "NM2", "NM3"}
assert set(test_mat_df["nm_id"].unique()) == {"NM4"}
print("  [PASS] Material assignments correct")

# 3. Correct scan rates per partition
assert set(train_df["scan_rate_mVs"].unique())    == {10, 20, 40, 60, 70, 80, 90, 100}
assert set(val_df["scan_rate_mVs"].unique())      == {30}
assert set(test_sr_df["scan_rate_mVs"].unique())  == {50}
print("  [PASS] Scan-rate assignments correct")

# 4. No feature NaN values
FEATURES = [
    "potential_V_norm", "scan_rate_mVs_norm", "log_scan_rate_norm",
    "sqrt_scan_rate_norm", "potential_from_lower_norm", "potential_from_upper_norm",
    "sr_x_potential_norm", "direction_x_potential_norm",
    "sweep_direction", "sweep_position",
]
TARGET = "current_normalized"

for name, df in [("train", train_df), ("val", val_df),
                  ("test_sr", test_sr_df), ("test_mat", test_mat_df)]:
    n_nan = df[FEATURES + [TARGET]].isna().sum().sum()
    assert n_nan == 0, f"NaN in {name}: {n_nan}"
print("  [PASS] Zero NaN in features and target across all partitions")

# 5. Target in [0, 1]
for name, df in [("train", train_df), ("val", val_df),
                  ("test_sr", test_sr_df), ("test_mat", test_mat_df)]:
    assert df[TARGET].between(-1e-9, 1 + 1e-9).all(), f"Target out of [0,1] in {name}"
print("  [PASS] current_normalized in [0, 1] for all partitions")

# 6. No half-sweep crosses split boundary
block_splits = (master_df.groupby(["nm_id", "scan_rate_mVs", "half_sweep_id"])["split"]
                          .nunique())
assert (block_splits == 1).all()
print("  [PASS] No half-sweep block crosses a split boundary")

print("\\n✓ ALL INTEGRITY CHECKS PASSED — safe to proceed with modelling")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE MATRIX CONSTRUCTION + CORRELATION PLOT
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 3 — Feature Matrix Construction

### Feature set rationale
Identical to the RF baseline — 10 normalised features.

| Feature | Why it is included |
|---------|-------------------|
| `potential_V_norm` | Primary independent variable in CV |
| `potential_from_lower_norm` | Distance from cathodic limit — controls reduction |
| `potential_from_upper_norm` | Distance from anodic limit — controls oxidation |
| `scan_rate_mVs_norm` | Linear scan-rate scaling (capacitive current) |
| `log_scan_rate_norm` | Log-linear relationships in electrochemistry |
| `sqrt_scan_rate_norm` | Randles-Sevcik: diffusion-limited current ~ sqrt(v) |
| `direction_x_potential_norm` | **Key interaction** — same V has different meaning per direction |
| `sr_x_potential_norm` | Scan-rate modulation of voltage-dependent response |
| `sweep_direction` | **Mandatory** — resolves the two-current many-to-one mapping |
| `sweep_position` | Normalised progress (0=start, 1=turning point) |

### Strictly excluded from model input
`current_A`, `current_norm_min`, `current_norm_max`, `split`, `nm_id`,
`half_sweep_id`, `cycle_id`, `step_index`""")

code("""\
FEATURES = [
    "potential_V_norm",
    "scan_rate_mVs_norm",
    "log_scan_rate_norm",
    "sqrt_scan_rate_norm",
    "potential_from_lower_norm",
    "potential_from_upper_norm",
    "sr_x_potential_norm",
    "direction_x_potential_norm",
    "sweep_direction",
    "sweep_position",
]
TARGET = "current_normalized"

# ── Numpy arrays (for metric computation and permutation importance) ───────────
X_train    = train_df[FEATURES].to_numpy(dtype=np.float64)
y_train    = train_df[TARGET].to_numpy(dtype=np.float64)

X_val      = val_df[FEATURES].to_numpy(dtype=np.float64)
y_val      = val_df[TARGET].to_numpy(dtype=np.float64)

X_test_sr  = test_sr_df[FEATURES].to_numpy(dtype=np.float64)
y_test_sr  = test_sr_df[TARGET].to_numpy(dtype=np.float64)

X_test_mat = test_mat_df[FEATURES].to_numpy(dtype=np.float64)
y_test_mat = test_mat_df[TARGET].to_numpy(dtype=np.float64)

# ── DataFrames for XGBoost fit (preserves named feature columns in booster) ───
Xdf_train = train_df[FEATURES]
Xdf_val   = val_df[FEATURES]

print("✓ Feature matrices constructed")
print(f"  X_train    : {X_train.shape}   y_train  : {y_train.shape}")
print(f"  X_val      : {X_val.shape}     y_val    : {y_val.shape}")
print(f"  X_test_sr  : {X_test_sr.shape}    y_test_sr : {y_test_sr.shape}")
print(f"  X_test_mat : {X_test_mat.shape}  y_test_mat: {y_test_mat.shape}")
print(f"\\n  {len(FEATURES)} input features  |  target: {TARGET}")""")

code("""\
# ── Figure 00: Feature correlation matrix (training data) ─────────────────────
feat_df = train_df[FEATURES].copy()
corr    = feat_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 7}, linewidths=0.4)
ax.set_title("Feature Correlation Matrix (training set)", pad=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "00_feature_correlation.png", bbox_inches="tight")
plt.show()
print("Saved figures/00_feature_correlation.png")
print()
print("Note: high correlation between potential-derived features is expected.")
print("XGBoost handles multicollinearity via regularisation and gain-based splits.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — XGBOOST TRAINING
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 4 — XGBoost Training with Early Stopping

### Hyperparameter choices
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 1000 | Upper bound; actual trees used selected by early stopping |
| `learning_rate` | 0.05 | Conservative step size; balances convergence speed vs overfitting |
| `max_depth` | 6 | Standard for tabular regression; deep enough for electrochemical interactions |
| `min_child_weight` | 3 | Minimum Hessian sum per leaf; regularises against noise rows |
| `subsample` | 0.8 | Row subsampling per tree — reduces variance without depth pruning |
| `colsample_bytree` | 0.8 | Feature subsampling per tree — decorrelates trees |
| `reg_alpha` | 0.05 | L1 regularisation — sparse feature weights |
| `reg_lambda` | 1.5 | L2 regularisation — shrinks leaf weights |
| `gamma` | 0.1 | Minimum loss reduction for a split — prevents trivial splits |
| `tree_method` | `"hist"` | Histogram-based algorithm — fastest for this dataset size |
| `eval_metric` | `"rmse"` | Monitors normalised RMSE on eval sets during training |
| `early_stopping_rounds` | 50 | Stop if val RMSE does not improve for 50 consecutive rounds |

**Early stopping**: XGBoost monitors the **second** eval set (val, SR=30) and halts
training when val RMSE has not improved for 50 consecutive boosting rounds.
`predict()` automatically uses the best iteration, not the final one.""")

code("""\
from xgboost import XGBRegressor

XGB_PARAMS = {
    "n_estimators"      : 1000,
    "learning_rate"     : 0.05,
    "max_depth"         : 6,
    "min_child_weight"  : 3,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "reg_alpha"         : 0.05,
    "reg_lambda"        : 1.5,
    "gamma"             : 0.1,
    "tree_method"       : "hist",
    "eval_metric"       : "rmse",
    "early_stopping_rounds": 50,
    "random_state"      : 42,
    "n_jobs"            : -1,
}

print("Training XGBoost ...")
print(f"  Parameters : {XGB_PARAMS}")
print(f"  Training on: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
print(f"  eval_set[0]: training RMSE (informational only)")
print(f"  eval_set[1]: val RMSE (early stopping monitor)")
print()

xgb_model = XGBRegressor(**XGB_PARAMS)

t0 = time.time()
xgb_model.fit(
    Xdf_train, y_train,
    eval_set=[(Xdf_train, y_train), (Xdf_val, y_val)],
    verbose=50,
)
elapsed = time.time() - t0

print(f"\\n✓ Training complete in {elapsed:.1f} s")
print(f"  Best iteration (0-indexed): {xgb_model.best_iteration}")
print(f"  Trees used for inference  : {xgb_model.best_iteration + 1}")
print(f"  Best val RMSE (normalised): {xgb_model.best_score:.6f}")
print()
print("  (Inference automatically uses best_iteration, not final round)")""")

code("""\
# ── Training curve: normalised RMSE per boosting round ────────────────────────
results = xgb_model.evals_result()
train_rmse_curve = results["validation_0"]["rmse"]
val_rmse_curve   = results["validation_1"]["rmse"]
rounds = range(len(train_rmse_curve))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rounds, train_rmse_curve, color="#2ca02c", lw=1.0, alpha=0.8, label="Train RMSE")
ax.plot(rounds, val_rmse_curve,   color="#1f77b4", lw=1.5, alpha=0.9, label="Val RMSE (SR=30)")
ax.axvline(xgb_model.best_iteration, color="#d62728", lw=1.5, ls="--",
           label=f"Best iteration = {xgb_model.best_iteration}")
ax.set_xlabel("Boosting round")
ax.set_ylabel("RMSE (normalised current)")
ax.set_title("XGBoost Training Curve — Normalised RMSE per Boosting Round", pad=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "xgb_training_curve.png", bbox_inches="tight")
plt.show()
print("Saved figures/xgb_training_curve.png")
print()
print(f"  Train converged to : {train_rmse_curve[-1]:.6f}")
print(f"  Best val RMSE      : {min(val_rmse_curve):.6f}  at round {xgb_model.best_iteration}")
print(f"  Final val RMSE     : {val_rmse_curve[-1]:.6f}  (may be worse due to over-boosting)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PREDICTIONS ON ALL PARTITIONS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 5 — Generate Predictions on All Partitions

Predictions are generated for every partition and attached back to the DataFrames
so they can be sorted and plotted in electrochemical (potential-sweep) order.

`xgb_model.predict()` automatically uses `best_iteration` trees for inference.

Inverse transform: `I_pred (A) = I_pred_norm * (I_max - I_min) + I_min`
where `I_min`, `I_max` are the per-group training-fitted scaler parameters
stored row-wise in each parquet file (`current_norm_min`, `current_norm_max`).""")

code("""\
# Work on copies so originals stay clean
train_df    = train_df.copy()
val_df      = val_df.copy()
test_sr_df  = test_sr_df.copy()
test_mat_df = test_mat_df.copy()

# ── Predict on all partitions (uses best_iteration automatically) ──────────────
train_df["pred_norm"]    = xgb_model.predict(X_train)
val_df["pred_norm"]      = xgb_model.predict(X_val)
test_sr_df["pred_norm"]  = xgb_model.predict(X_test_sr)
test_mat_df["pred_norm"] = xgb_model.predict(X_test_mat)

# ── Inverse transform: normalised -> Amperes ──────────────────────────────────
for df in [train_df, val_df, test_sr_df, test_mat_df]:
    rng          = df["current_norm_max"] - df["current_norm_min"]
    df["pred_A"] = df["pred_norm"] * rng + df["current_norm_min"]

print("✓ Predictions generated and inverse-transformed to Amperes")
print()
for label, df in [("train", train_df), ("val", val_df),
                   ("test_sr", test_sr_df), ("test_mat", test_mat_df)]:
    print(
        f"  {label:10s}: pred_A range "
        f"[{df['pred_A'].min():.4e}, {df['pred_A'].max():.4e}] A"
    )""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 6 — Evaluation Metrics

Metrics are computed in **both** spaces:
- **Normalised** (dimensionless): for direct model comparison across notebooks.
- **Physical (Amperes)**: the scientifically meaningful quantity.

R² is invariant under the linear inverse transform, so `R²_norm = R²_A`.

### RF baseline for comparison (from Notebook 02, honest evaluation)
| Partition | RF RMSE (μA) | RF R² |
|-----------|-------------|-------|
| Val (SR=30) | 26.59 | 0.9851 |
| Test-SR (SR=50) | 43.71 | 0.9777 |
| Test-MAT (NM4) | 33.65 | 0.9707 |""")

code("""\
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true_norm, y_pred_norm, actual_A, pred_A, label):
    rmse_norm = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))
    rmse_A    = np.sqrt(mean_squared_error(actual_A, pred_A))
    mae_A     = mean_absolute_error(actual_A, pred_A)
    r2        = r2_score(y_true_norm, y_pred_norm)
    max_err_A = np.abs(actual_A - pred_A).max()
    return {
        "partition" : label,
        "n_samples" : len(y_true_norm),
        "RMSE_norm" : rmse_norm,
        "RMSE_uA"   : rmse_A   * 1e6,
        "MAE_uA"    : mae_A    * 1e6,
        "MaxErr_uA" : max_err_A * 1e6,
        "R2"        : r2,
    }

metrics_rows = []

m = compute_metrics(
    y_train, train_df["pred_norm"].values,
    train_df["current_A"].values, train_df["pred_A"].values,
    "train (in-sample)")
metrics_rows.append(m)

m = compute_metrics(
    y_val, val_df["pred_norm"].values,
    val_df["current_A"].values, val_df["pred_A"].values,
    "val  (SR=30 interpolation)")
metrics_rows.append(m)

m = compute_metrics(
    y_test_sr, test_sr_df["pred_norm"].values,
    test_sr_df["current_A"].values, test_sr_df["pred_A"].values,
    "test_SR (SR=50 interpolation)")
metrics_rows.append(m)

m = compute_metrics(
    y_test_mat, test_mat_df["pred_norm"].values,
    test_mat_df["current_A"].values, test_mat_df["pred_A"].values,
    "test_MAT (NM4 extrapolation)")
metrics_rows.append(m)

metrics_df = pd.DataFrame(metrics_rows).set_index("partition")

print("=" * 70)
print("XGBOOST MODEL — EVALUATION SUMMARY")
print("=" * 70)
print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
print()
print("Units: RMSE / MAE / MaxErr in micro-Amperes (uA).  R2 is dimensionless.")
print()
print("Scientific interpretation:")
print("  - Train RMSE near 0: XGBoost overfits training rows (gradient boosting")
print("    is high-variance; regularisation only partially prevents memorisation).")
print("  - Val / Test-SR RMSE: scan-rate interpolation — XGBoost vs RF comparison")
print("    in Section 9.")
print("  - Test-MAT RMSE: material extrapolation — hardest generalisation axis.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7a — CV CURVE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7a — Predicted vs Actual CV Curves

The most physically meaningful test: does XGBoost reconstruct the correct
closed-loop voltammogram shape?  Rows are sorted by `(half_sweep_id, step_index)`
to recover the electrochemical trajectory from the tabular predictions.

Solid lines = actual current.  Dashed lines = XGBoost prediction.""")

code("""\
# ── Helper: plot one CV curve (actual vs predicted) ───────────────────────────
def plot_cv_curve(ax, group_df, title, model_label="XGBoost", show_legend=True):
    grp = group_df.sort_values(["half_sweep_id", "step_index"])
    colours   = {0: "#1f77b4", 1: "#d62728"}
    dir_labels = {0: "Cathodic (dir=0)", 1: "Anodic (dir=1)"}
    legend_done = set()
    for _, seg in grp.groupby("half_sweep_id"):
        d     = int(seg["sweep_direction"].iloc[0])
        label = dir_labels[d] if d not in legend_done else None
        ax.plot(seg["potential_V"], seg["current_A"] * 1e6,
                color=colours[d], lw=1.5, alpha=0.9, label=label)
        ax.plot(seg["potential_V"], seg["pred_A"] * 1e6,
                color=colours[d], lw=1.2, ls="--", alpha=0.7)
        legend_done.add(d)
    ax.set_xlabel("Potential (V vs SCE)", fontsize=8)
    ax.set_ylabel("Current (uA)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(fontsize=7, title=f"Solid=actual  Dashed={model_label}")


# ── Figure 01: Validation curves — NM1, NM2, NM3 at SR=30 ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nm in enumerate(["NM1", "NM2", "NM3"]):
    sub = val_df[(val_df["nm_id"] == nm) & (val_df["scan_rate_mVs"] == 30)]
    plot_cv_curve(axes[i], sub, f"VAL: {nm} @ 30 mV/s", show_legend=(i == 0))
fig.suptitle("Validation Set — Scan-Rate Interpolation (SR=30, seen materials)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_cv_curves_val.png", bbox_inches="tight")
plt.show()
print("Saved figures/01_cv_curves_val.png")""")

code("""\
# ── Figure 02: Test-SR curves — NM1, NM2, NM3 at SR=50 ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nm in enumerate(["NM1", "NM2", "NM3"]):
    sub = test_sr_df[(test_sr_df["nm_id"] == nm) & (test_sr_df["scan_rate_mVs"] == 50)]
    plot_cv_curve(axes[i], sub, f"TEST-SR: {nm} @ 50 mV/s", show_legend=(i == 0))
fig.suptitle("Test-SR Set — Scan-Rate Interpolation (SR=50, seen materials)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_cv_curves_test_sr.png", bbox_inches="tight")
plt.show()
print("Saved figures/02_cv_curves_test_sr.png")""")

code("""\
# ── Figure 03: Test-MAT curves — NM4 at 6 representative scan rates ──────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
sr_showcase = [10, 30, 50, 70, 90, 100]

for i, sr in enumerate(sr_showcase):
    sub = test_mat_df[(test_mat_df["nm_id"] == "NM4") & (test_mat_df["scan_rate_mVs"] == sr)]
    plot_cv_curve(axes[i], sub, f"TEST-MAT: NM4 @ {sr} mV/s", show_legend=(i == 0))

fig.suptitle("Test-MAT Set — Material Extrapolation (NM4, fully unseen formulation)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_cv_curves_test_mat.png", bbox_inches="tight")
plt.show()
print("Saved figures/03_cv_curves_test_mat.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7b — PARITY PLOTS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7b — Parity Plots (Predicted vs Actual)

A perfect model lies exactly on the diagonal `y = x`.
Colour encodes scan rate to reveal whether prediction quality varies with scan rate.""")

code("""\
# ── Figure 04: Parity plots ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_configs = [
    (val_df,      "Val (SR=30)"),
    (test_sr_df,  "Test-SR (SR=50)"),
    (test_mat_df, "Test-MAT (NM4)"),
]

for ax, (df, title) in zip(axes, plot_configs):
    sc = ax.scatter(
        df["current_A"] * 1e6,
        df["pred_A"]    * 1e6,
        c=df["scan_rate_mVs"], cmap="plasma",
        s=1.5, alpha=0.3, rasterized=True
    )
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k-", lw=1.2, label="y = x (perfect)")
    ax.set_xlabel("Actual current (uA)", fontsize=9)
    ax.set_ylabel("Predicted current (uA)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label="Scan rate (mV/s)", pad=0.01)

fig.suptitle("Parity Plots — XGBoost Model", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_parity_plots.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved figures/04_parity_plots.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7c — RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7c — Residual Analysis

**Figure 05**: Residual histograms (predicted − actual in μA).
A good model shows zero-centred, approximately Gaussian residuals.
Systematic bias or heavy tails indicate a structural limitation.

**Figure 06**: Residuals vs potential (coloured by sweep direction).
Uniform scatter around zero means no voltage-dependent bias.
Systematic patterns (e.g. larger residuals near turning points −0.65 V and 0 V)
indicate the model under-captures electrochemical reversibility.""")

code("""\
# ── Figure 05: Residual distributions ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

dist_configs = [
    (val_df,      "Val (SR=30)",     "#ff7f0e"),
    (test_sr_df,  "Test-SR (SR=50)", "#1f77b4"),
    (test_mat_df, "Test-MAT (NM4)",  "#d62728"),
]

for ax, (df, title, colour) in zip(axes, dist_configs):
    residuals = (df["pred_A"] - df["current_A"]) * 1e6
    ax.hist(residuals, bins=80, color=colour, alpha=0.75, edgecolor="none",
            density=True)
    ax.axvline(0,                  color="k",       lw=1.5, ls="-",
               label="zero bias")
    ax.axvline(residuals.mean(),   color="darkred",  lw=1.5, ls="--",
               label=f"mean={residuals.mean():.1f} uA")
    ax.axvline(residuals.median(), color="navy",     lw=1.2, ls=":",
               label=f"median={residuals.median():.1f} uA")
    ax.set_xlabel("Residual (uA): predicted - actual", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

fig.suptitle("Residual Distributions — XGBoost Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_residual_distributions.png", bbox_inches="tight")
plt.show()
print("Saved figures/05_residual_distributions.png")""")

code("""\
# ── Figure 06: Residuals vs potential (sweep direction coloured) ─────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (df, title, _) in zip(axes, dist_configs):
    residuals = (df["pred_A"] - df["current_A"]) * 1e6
    ax.scatter(df["potential_V"], residuals,
               c=df["sweep_direction"], cmap="coolwarm",
               s=1.0, alpha=0.25, rasterized=True)
    ax.axhline(0, color="k", lw=1.0, ls="--")
    ax.set_xlabel("Potential (V vs SCE)", fontsize=8)
    ax.set_ylabel("Residual (uA)", fontsize=8)
    ax.set_title(f"{title} — residual vs potential", fontsize=9)
    ax.grid(True, alpha=0.25)

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
           markersize=6, label="Cathodic (dir=0)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
           markersize=6, label="Anodic (dir=1)"),
]
axes[0].legend(handles=legend_elems, fontsize=7)

fig.suptitle("Residuals vs Potential — XGBoost Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_residual_vs_potential.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved figures/06_residual_vs_potential.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7d — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7d — Feature Importance

Two complementary metrics for XGBoost:

**Gain importance** (XGBoost-native) — total improvement in the loss function
attributed to splits on each feature, averaged across all trees.  More physically
interpretable than RF's MDI because it reflects how much each feature reduces
prediction error, not just how often it splits.

**Permutation importance** — computed on the **validation set** (not training).
Shuffles one feature at a time and measures RMSE increase.  Generalisation-aware
and model-agnostic, confirming gain-importance findings on held-out data.""")

code("""\
from sklearn.inspection import permutation_importance

# ── XGBoost gain importance (built-in, from booster) ─────────────────────────
booster = xgb_model.get_booster()
gain_raw = booster.get_score(importance_type="gain")
# get_score returns only features that appear in at least one split.
# Fill any unused features with 0 (rare but possible for binary/low-variance feat).
gain_imp = pd.Series(
    {f: gain_raw.get(f, 0.0) for f in FEATURES}
).sort_values(ascending=False)

# Normalise to sum-to-1 for comparability with MDI
gain_imp_norm = gain_imp / gain_imp.sum()

# ── Permutation importance on val set ────────────────────────────────────────
print("Computing permutation importance on val set (n_repeats=20) ...")
t0 = time.time()
perm = permutation_importance(
    xgb_model, X_val, y_val,
    n_repeats=20,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
print(f"  Done in {time.time() - t0:.1f} s")

perm_means = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
perm_stds  = pd.Series(perm.importances_std,  index=FEATURES)

# ── Figure 07: Feature importance (gain + permutation) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gain importance
axes[0].barh(gain_imp_norm.index[::-1], gain_imp_norm.values[::-1],
             color="#2ca02c", alpha=0.8)
axes[0].set_xlabel("Normalised gain importance", fontsize=9)
axes[0].set_title(
    "XGBoost Gain Importance\\n(total loss reduction per feature, normalised)",
    fontsize=9)
axes[0].grid(True, alpha=0.3, axis="x")

# Permutation importance
xerr = perm_stds.loc[perm_means.index][::-1].values
axes[1].barh(
    perm_means.index[::-1], perm_means.values[::-1],
    xerr=xerr, color="#1f77b4", alpha=0.8,
    error_kw={"elinewidth": 1.2, "capsize": 3}
)
axes[1].set_xlabel("Permutation importance (RMSE increase on val set)", fontsize=9)
axes[1].set_title(
    "Permutation Feature Importance\\n(validation set — generalisation-aware)",
    fontsize=9)
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_feature_importance.png", bbox_inches="tight")
plt.show()
print("Saved figures/07_feature_importance.png")

print()
print("Gain ranking (normalised):")
for feat, val in gain_imp_norm.items():
    print(f"  {feat:35s}: {val:.4f}")
print()
print("Permutation ranking (val set):")
for feat in perm_means.index:
    print(f"  {feat:35s}: {perm_means[feat]:.6f} +/- {perm_stds[feat]:.6f}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7e — PER-GROUP PERFORMANCE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7e — Per-Group Performance Breakdown

Break down RMSE by (nm_id, scan_rate) to reveal:
- Whether prediction quality is consistent across materials
- Which scan rates are hardest to interpolate/extrapolate
- How NM4 RMSE scales with scan rate (does current magnitude drive error?)""")

code("""\
def group_rmse(df, label):
    rows = []
    for (nm, sr), grp in df.groupby(["nm_id", "scan_rate_mVs"]):
        rmse_uA = np.sqrt(mean_squared_error(
            grp["current_A"], grp["pred_A"])) * 1e6
        r2 = r2_score(grp["current_normalized"], grp["pred_norm"])
        rows.append({"nm_id": nm, "scan_rate_mVs": sr,
                     "RMSE_uA": rmse_uA, "R2": r2, "partition": label})
    return pd.DataFrame(rows)

val_grp      = group_rmse(val_df,      "val")
test_sr_grp  = group_rmse(test_sr_df,  "test_sr")
test_mat_grp = group_rmse(test_mat_df, "test_mat")

# ── Figure 08: Per-group RMSE bars ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
bars = ax.bar(val_grp["nm_id"], val_grp["RMSE_uA"], color="#ff7f0e", alpha=0.85)
for i, row in val_grp.iterrows():
    ax.text(i, row["RMSE_uA"] + 0.05, f"R2={row['R2']:.3f}",
            ha="center", fontsize=8)
ax.set_xlabel("Material")
ax.set_ylabel("RMSE (uA)")
ax.set_title("Val (SR=30) — RMSE per material")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
bars = ax.bar(test_sr_grp["nm_id"], test_sr_grp["RMSE_uA"], color="#1f77b4", alpha=0.85)
for i, row in test_sr_grp.iterrows():
    ax.text(i, row["RMSE_uA"] + 0.05, f"R2={row['R2']:.3f}",
            ha="center", fontsize=8)
ax.set_xlabel("Material")
ax.set_ylabel("RMSE (uA)")
ax.set_title("Test-SR (SR=50) — RMSE per material")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[2]
ax.bar(test_mat_grp["scan_rate_mVs"].astype(int).astype(str),
       test_mat_grp["RMSE_uA"], color="#d62728", alpha=0.85)
ax.set_xlabel("Scan rate (mV/s)")
ax.set_ylabel("RMSE (uA)")
ax.set_title("Test-MAT (NM4) — RMSE per scan rate")
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Per-Group RMSE Breakdown — XGBoost Model", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_per_group_rmse.png", bbox_inches="tight")
plt.show()
print("Saved figures/08_per_group_rmse.png")""")

code("""\
# ── Figure 09: NM4 RMSE and R2 vs scan rate ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(test_mat_grp["scan_rate_mVs"], test_mat_grp["RMSE_uA"],
        "o-", color="#d62728", lw=1.8, ms=7)
ax.set_xlabel("Scan rate (mV/s)")
ax.set_ylabel("RMSE (uA)")
ax.set_title("NM4 — RMSE vs scan rate\\n(does error scale with scan rate?)")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(test_mat_grp["scan_rate_mVs"], test_mat_grp["R2"],
        "s-", color="#9467bd", lw=1.8, ms=7)
ax.axhline(0, color="k", lw=0.8, ls="--", label="R2=0 (mean prediction)")
ax.set_xlabel("Scan rate (mV/s)")
ax.set_ylabel("R2")
ax.set_title("NM4 — R2 vs scan rate\\n(does shape fidelity degrade?)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("NM4 Scan-Rate Trend — XGBoost Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "09_nm4_scan_rate_trend.png", bbox_inches="tight")
plt.show()
print("Saved figures/09_nm4_scan_rate_trend.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SAVE MODEL AND RESULTS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 8 — Save Model and Results")

code("""\
import joblib

# ── Save trained model ────────────────────────────────────────────────────────
model_path = MODEL_DIR / "xgboost_model.joblib"
joblib.dump(xgb_model, model_path)
print(f"Saved model : {model_path}  ({model_path.stat().st_size / 1e3:.0f} KB)")

# ── Save native XGBoost model (JSON format — portable, version-independent) ───
booster_path = MODEL_DIR / "xgboost_booster.json"
xgb_model.get_booster().save_model(str(booster_path))
print(f"Saved booster: {booster_path}")

# ── Save metrics as JSON ──────────────────────────────────────────────────────
metrics_out = {
    "model"         : "XGBRegressor",
    "params"        : XGB_PARAMS,
    "best_iteration": int(xgb_model.best_iteration),
    "best_val_rmse_norm": float(xgb_model.best_score),
    "metrics"       : metrics_df.reset_index().to_dict(orient="records"),
    "per_group"     : {
        "val"     : val_grp.to_dict(orient="records"),
        "test_sr" : test_sr_grp.to_dict(orient="records"),
        "test_mat": test_mat_grp.to_dict(orient="records"),
    },
    "feature_importance_gain": {
        f: float(v) for f, v in gain_imp.items()
    },
    "feature_importance_gain_norm": {
        f: float(v) for f, v in gain_imp_norm.items()
    },
    "feature_importance_permutation": {
        f: {"mean": float(perm_means[f]), "std": float(perm_stds[f])}
        for f in FEATURES
    },
}

metrics_path = METRICS_DIR / "xgboost_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(metrics_out, fh, indent=2)
print(f"Saved metrics: {metrics_path}")

# ── Save per-group RMSE tables ────────────────────────────────────────────────
for name, df_grp in [("val", val_grp), ("test_sr", test_sr_grp),
                      ("test_mat", test_mat_grp)]:
    path = CSV_DIR / f"per_group_rmse_{name}.csv"
    df_grp.to_csv(path, index=False)
    print(f"Saved {path.name}")

print()
print(f"✓ All results saved to:")
print(f"    figures/ — 10 mandatory figures + training curve")
print(f"    metrics/ — xgboost_metrics.json")
print(f"    csv/     — 3 per-group RMSE tables")
print(f"    model/   — xgboost_model.joblib + xgboost_booster.json")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — RF vs XGBOOST COMPARISON + SCIENTIFIC SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 9 — RF vs XGBoost Comparison and Scientific Summary

### Comparison methodology
Both models use:
- **Identical training data** (NM1/2/3 × SR∈{10,20,40,60,70,80,90,100}, 62,400 rows)
- **Identical feature set** (10 normalised features)
- **Identical evaluation partitions** (val SR=30, test-SR SR=50, test-MAT NM4)
- **Identical inverse transform** (per-group scaler parameters from Notebook 01)

Differences:
- RF: 300 trees, parallel bagging, unlimited depth, no regularisation
- XGBoost: sequential boosting, L1+L2 regularisation, early stopping on val set

### Why material extrapolation (NM4) is the hardest test
NM4 is a **completely unseen ZnO formulation** during training.  Both models must
transfer electrochemical patterns from NM1/2/3 to NM4 purely via the feature space.
A model that memorises NM1/2/3 patterns will fail here; one that captures
generalizable electrochemical physics (capacitance scaling, Randles-Sevcik) will succeed.

### Next steps in the pipeline
| Notebook | Model | What it adds over XGBoost |
|----------|-------|--------------------------|
| 04_ann_baseline | ANN (dense, 4 layers) | Non-linear interactions; replicates paper architecture |
| 05_lstm_baseline | LSTM | Explicit sequence modelling of the CV trajectory |""")

code("""\
# ── RF baseline values (Notebook 02, verified run) ────────────────────────────
RF_RESULTS = {
    "val  (SR=30 interpolation)"  : {"RMSE_uA": 26.59, "R2": 0.9851},
    "test_SR (SR=50 interpolation)": {"RMSE_uA": 43.71, "R2": 0.9777},
    "test_MAT (NM4 extrapolation)" : {"RMSE_uA": 33.65, "R2": 0.9707},
}

# ── Build comparison table ────────────────────────────────────────────────────
partitions = [
    "val  (SR=30 interpolation)",
    "test_SR (SR=50 interpolation)",
    "test_MAT (NM4 extrapolation)",
]

print("=" * 80)
print("RF vs XGBOOST — HEAD-TO-HEAD COMPARISON")
print("=" * 80)
print(f"  {'Partition':<32} {'RF RMSE':>10} {'XGB RMSE':>10} {'Diff':>8} "
      f"{'RF R2':>8} {'XGB R2':>8}")
print("  " + "-" * 76)

for p in partitions:
    rf_rmse = RF_RESULTS[p]["RMSE_uA"]
    rf_r2   = RF_RESULTS[p]["R2"]
    xgb_rmse = metrics_df.loc[p, "RMSE_uA"]
    xgb_r2   = metrics_df.loc[p, "R2"]
    diff     = xgb_rmse - rf_rmse
    arrow    = "▼" if diff < 0 else "▲"
    print(f"  {p:<32} {rf_rmse:>9.2f}  {xgb_rmse:>9.2f}  "
          f"{arrow}{abs(diff):>6.2f}  {rf_r2:>8.4f}  {xgb_r2:>8.4f}")

print()
print("  Units: RMSE in micro-Amperes (uA).  R2 dimensionless.")
print("  ▼ = XGBoost improves over RF   ▲ = XGBoost worse than RF")
print()

# ── Summary verdict ───────────────────────────────────────────────────────────
xgb_val_rmse     = metrics_df.loc["val  (SR=30 interpolation)",   "RMSE_uA"]
xgb_testsr_rmse  = metrics_df.loc["test_SR (SR=50 interpolation)", "RMSE_uA"]
xgb_testmat_rmse = metrics_df.loc["test_MAT (NM4 extrapolation)",  "RMSE_uA"]

improvements = [
    xgb_val_rmse     < RF_RESULTS["val  (SR=30 interpolation)"]["RMSE_uA"],
    xgb_testsr_rmse  < RF_RESULTS["test_SR (SR=50 interpolation)"]["RMSE_uA"],
    xgb_testmat_rmse < RF_RESULTS["test_MAT (NM4 extrapolation)"]["RMSE_uA"],
]
n_improved = sum(improvements)

print(f"  XGBoost improved on {n_improved}/3 partitions vs RF baseline.")
if n_improved == 3:
    print("  Conclusion: XGBoost outperforms RF on all three generalisation axes.")
elif n_improved >= 2:
    print("  Conclusion: XGBoost outperforms RF on most generalisation axes.")
else:
    print("  Conclusion: RF competitive or better — XGBoost hyperparameters may")
    print("  benefit from further tuning (lower learning_rate, wider grid search).")

print()
print("Notebook complete. Run Section 10 below to download all results.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — DOWNLOAD RESULTS TO LOCAL MACHINE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 10 — Download Results to Your Local Machine

Run this cell to zip the entire `results/xgboost/` folder and download it.
The zip will contain all figures, metrics JSON, CSV tables, and the saved model.

**Run this before closing the Colab session** — files are lost when the runtime disconnects.""")

code("""\
import shutil, os
from google.colab import files as _cf

zip_base = "/content/xgboost_results"
print(f"Creating zip of {RESULTS_DIR} ...")
shutil.make_archive(zip_base, "zip", str(RESULTS_DIR))

zip_path = zip_base + ".zip"
size_mb  = os.path.getsize(zip_path) / 1e6
print(f"Created : {zip_path}  ({size_mb:.1f} MB)")
print()

# List contents
for root, dirs, files in os.walk(str(RESULTS_DIR)):
    rel = os.path.relpath(root, str(RESULTS_DIR))
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        fsize = os.path.getsize(fpath) / 1e3
        print(f"  {os.path.join(rel, fname):<55} {fsize:>8.0f} KB")

print()
print("Downloading to your local machine ...")
_cf.download(zip_path)
print("Done — check your browser Downloads folder.")""")

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE & WRITE
# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    },
    "colab": {
        "provenance": []
    }
})

out_path = Path(__file__).resolve().parent.parent.parent / "research" / "notebooks" / "03_xgboost_model.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out_path))
print(f"Written : {out_path}")
print(f"Cells   : {len(nb.cells)}")
