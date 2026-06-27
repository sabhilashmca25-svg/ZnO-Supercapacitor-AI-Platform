"""
build_rf_notebook.py
Generates notebooks/02_rf_baseline.ipynb
Run once: python src/build_rf_notebook.py
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
# Notebook 02 — Random Forest Baseline
## ZnO Supercapacitor CV Current Prediction · Scientifically Correct Baseline

### Purpose
Establish a **scientifically valid Random Forest baseline** before proceeding to
XGBoost, ANN, and LSTM models.  This notebook:
- Loads pre-built artefacts from Notebook 01 — no preprocessing is repeated.
- Trains RF on the training partition only.
- Evaluates on three held-out partitions that test different generalisation axes.
- Reports all metrics in **physical Amperes** (primary) and normalised units.
- Produces publication-ready verification plots.

### Three generalisation questions answered
| Partition | Question |
|-----------|----------|
| **Val** (NM1/2/3 × SR=30) | Can RF interpolate to an unseen intermediate scan rate? |
| **Test-SR** (NM1/2/3 × SR=50) | Can RF interpolate to SR=50 from {10,20,...,100}? |
| **Test-MAT** (NM4 × all SRs) | Can RF generalise to a completely unseen ZnO formulation? |

### Scientific note on RF and CV data
Random Forest treats each measurement row **independently** — it has no concept of
sweep order.  The CV loop shape emerges purely from the feature space
(potential, direction, scan-rate interactions).  If the loop is well-reconstructed,
it confirms the feature set adequately encodes the electrochemical state at
each measurement point without requiring sequential modelling.""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — ENVIRONMENT SETUP
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

# Files land in /content/ after upload
PROCESSED_DIR = Path("/content")
RESULTS_DIR   = Path("/content/rf_baseline_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Verify all 6 arrived
missing = [f for f in REQUIRED if not (PROCESSED_DIR / f).exists()]
assert not missing, (
    "These files were not uploaded: " + str(missing)
    + "\\nRe-run this cell and select them in the dialog."
)

print("✓ All 6 files uploaded successfully")
print(f"  Source : {PROCESSED_DIR}")
print(f"  Results: {RESULTS_DIR}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 1 — Load Preprocessing Artefacts

Loads the six outputs created by Notebook 01.  **Nothing is recomputed here** —
splits, normalization, and feature engineering are fixed.""")

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

# ── Load split DataFrames ─────────────────────────────────────────────────────
master_df   = pd.read_parquet(PROCESSED_DIR / "master_long_format.parquet")
train_df    = pd.read_parquet(PROCESSED_DIR / "train_set.parquet")
val_df      = pd.read_parquet(PROCESSED_DIR / "val_set.parquet")
test_sr_df  = pd.read_parquet(PROCESSED_DIR / "test_scanrate.parquet")
test_mat_df = pd.read_parquet(PROCESSED_DIR / "test_material.parquet")

# ── Load normalisation scalers ────────────────────────────────────────────────
with open(PROCESSED_DIR / "scalers.json") as fh:
    scalers = json.load(fh)

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
assert set(train_df["nm_id"].unique())    == {"NM1","NM2","NM3"}
assert set(val_df["nm_id"].unique())      == {"NM1","NM2","NM3"}
assert set(test_sr_df["nm_id"].unique())  == {"NM1","NM2","NM3"}
assert set(test_mat_df["nm_id"].unique()) == {"NM4"}
print("  [PASS] Material assignments correct")

# 3. Correct scan rates per partition
assert set(train_df["scan_rate_mVs"].unique()) == {10,20,40,60,70,80,90,100}
assert set(val_df["scan_rate_mVs"].unique())   == {30}
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
    assert df[TARGET].between(-1e-9, 1+1e-9).all(), f"Target out of [0,1] in {name}"
print("  [PASS] current_normalized in [0, 1] for all partitions")

# 6. No half-sweep crosses split boundaries (block check)
block_splits = (master_df.groupby(["nm_id","scan_rate_mVs","half_sweep_id"])["split"]
                         .nunique())
assert (block_splits == 1).all()
print("  [PASS] No half-sweep block crosses a split boundary")

print("\\n✓ ALL INTEGRITY CHECKS PASSED — safe to proceed with modelling")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE MATRIX CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 3 — Feature Matrix Construction

### Feature set rationale
All 10 features are normalised to [0, 1] except `sweep_direction` ({0, 1}).

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
`half_sweep_id`, `cycle_id`, `step_index`
(first three are leakage targets; rest are structural labels, not electrochemical state)""")

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

# ── Build numpy arrays for sklearn ────────────────────────────────────────────
X_train    = train_df[FEATURES].to_numpy(dtype=np.float64)
y_train    = train_df[TARGET].to_numpy(dtype=np.float64)

X_val      = val_df[FEATURES].to_numpy(dtype=np.float64)
y_val      = val_df[TARGET].to_numpy(dtype=np.float64)

X_test_sr  = test_sr_df[FEATURES].to_numpy(dtype=np.float64)
y_test_sr  = test_sr_df[TARGET].to_numpy(dtype=np.float64)

X_test_mat = test_mat_df[FEATURES].to_numpy(dtype=np.float64)
y_test_mat = test_mat_df[TARGET].to_numpy(dtype=np.float64)

print("✓ Feature matrices constructed")
print(f"  X_train    : {X_train.shape}   y_train  : {y_train.shape}")
print(f"  X_val      : {X_val.shape}     y_val    : {y_val.shape}")
print(f"  X_test_sr  : {X_test_sr.shape}    y_test_sr: {y_test_sr.shape}")
print(f"  X_test_mat : {X_test_mat.shape}  y_test_mat: {y_test_mat.shape}")
print(f"\\n  {len(FEATURES)} input features  |  target: {TARGET}")
print(f"  Features: {FEATURES}")""")

code("""\
# ── Feature correlation matrix (training data) ────────────────────────────────
feat_df = train_df[FEATURES].copy()
corr    = feat_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 7}, linewidths=0.4)
ax.set_title("Feature Correlation Matrix (training set)", pad=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "00_feature_correlation.png", bbox_inches="tight")
plt.show()
print("Saved 00_feature_correlation.png")
print()
print("Note: high correlation between potential-derived features is expected.")
print("RF handles multicollinearity without issue (tree splits are discrete).")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RANDOM FOREST TRAINING
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 4 — Random Forest Baseline Training

### Hyperparameter choices
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 300 | Stable variance estimates; 100–200 often sufficient but 300 is reliable |
| `max_depth` | None | Unlimited depth; RF controls variance via bagging, not depth pruning |
| `min_samples_leaf` | 2 | Minimal regularisation; prevents single-sample leaves |
| `max_features` | `"sqrt"` | sqrt(10) ~ 3 features/split — standard RF setting |
| `n_jobs` | -1 | Parallel on all cores |
| `oob_score` | True | Out-of-bag score = free validation estimate during bagging |
| `random_state` | 42 | Reproducibility |

These are well-established defaults for tabular regression.
Further tuning (RandomizedSearchCV with GroupKFold) is left for a dedicated
optimisation notebook so this baseline remains clean and reproducible.""")

code("""\
from sklearn.ensemble import RandomForestRegressor

RF_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : None,
    "min_samples_leaf": 2,
    "max_features"    : "sqrt",
    "n_jobs"          : -1,
    "random_state"    : 42,
    "oob_score"       : True,
}

print("Training Random Forest ...")
print(f"  Parameters : {RF_PARAMS}")
print(f"  Training on: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
print()

t0 = time.time()
rf = RandomForestRegressor(**RF_PARAMS)
rf.fit(X_train, y_train)
elapsed = time.time() - t0

print(f"✓ Training complete in {elapsed:.1f} s")
print()
print(f"  OOB R^2 score (free estimate, no val data used): {rf.oob_score_:.6f}")
print()
print("  The OOB score is an unbiased estimate of generalisation performance")
print("  computed from the ~37% of training samples not used in each tree.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PREDICTIONS ON ALL PARTITIONS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 5 — Generate Predictions on All Partitions

Predictions are generated for every partition and attached back to the DataFrames
so they can be sorted and plotted in electrochemical (potential-sweep) order.

Inverse transform: `I_pred (A) = I_pred_norm * (I_max - I_min) + I_min`
where `I_min` and `I_max` are the per-group training-fitted scaler parameters
stored in each row of the parquet files (`current_norm_min`, `current_norm_max`).""")

code("""\
# ── Predict on all partitions ─────────────────────────────────────────────────
# Work on copies so originals stay clean.
train_df   = train_df.copy()
val_df     = val_df.copy()
test_sr_df = test_sr_df.copy()
test_mat_df= test_mat_df.copy()

train_df["pred_norm"]    = rf.predict(X_train)
val_df["pred_norm"]      = rf.predict(X_val)
test_sr_df["pred_norm"]  = rf.predict(X_test_sr)
test_mat_df["pred_norm"] = rf.predict(X_test_mat)

# ── Inverse transform: normalised prediction -> Amperes ───────────────────────
# current_norm_min / current_norm_max are the per-(nm_id, scan_rate) scaler
# boundaries stored row-wise during Notebook 01.
# Excluded from model features — used only here for inverse transform.
for df in [train_df, val_df, test_sr_df, test_mat_df]:
    rng           = df["current_norm_max"] - df["current_norm_min"]
    df["pred_A"]  = df["pred_norm"] * rng + df["current_norm_min"]

print("✓ Predictions generated and inverse-transformed to Amperes")
print()
for label, df in [("train", train_df), ("val", val_df),
                   ("test_sr", test_sr_df), ("test_mat", test_mat_df)]:
    neg_preds = (df["pred_A"] < df["current_norm_min"].min()).sum()
    over_preds = (df["pred_A"] > df["current_norm_max"].max()).sum()
    print(
        f"  {label:10s}: pred_A range "
        f"[{df['pred_A'].min():.4e}, {df['pred_A'].max():.4e}] A"
        f"  | out-of-scaler-range: {neg_preds + over_preds}"
    )""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 6 — Evaluation Metrics

Metrics are computed in **both** spaces:
- **Normalised** (dimensionless): for direct model comparison across notebooks.
- **Physical (Amperes)**: the scientifically meaningful quantity.

Note: R² is invariant under the linear inverse transform, so `R²_norm = R²_A`.

### Why physical-unit RMSE matters
The paper reported RMSE = 0.00017 A (RF) with row-wise splitting.
Our partitions test genuine generalisation, so higher RMSE is expected and
scientifically honest.""")

code("""\
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true_norm, y_pred_norm, actual_A, pred_A, label):
    rmse_norm = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))
    rmse_A    = np.sqrt(mean_squared_error(actual_A, pred_A))
    mae_A     = mean_absolute_error(actual_A, pred_A)
    r2        = r2_score(y_true_norm, y_pred_norm)
    max_err_A = np.abs(actual_A - pred_A).max()
    return {
        "partition"   : label,
        "n_samples"   : len(y_true_norm),
        "RMSE_norm"   : rmse_norm,
        "RMSE_uA"     : rmse_A   * 1e6,
        "MAE_uA"      : mae_A    * 1e6,
        "MaxErr_uA"   : max_err_A * 1e6,
        "R2"          : r2,
    }

metrics_rows = []

# Train (shows in-sample fit; RF will overfit training data — expected)
m = compute_metrics(
    y_train, train_df["pred_norm"].values,
    train_df["current_A"].values, train_df["pred_A"].values,
    "train (in-sample)")
metrics_rows.append(m)

# Validation: unseen scan rate SR=30
m = compute_metrics(
    y_val, val_df["pred_norm"].values,
    val_df["current_A"].values, val_df["pred_A"].values,
    "val  (SR=30 interpolation)")
metrics_rows.append(m)

# Test-SR: unseen scan rate SR=50
m = compute_metrics(
    y_test_sr, test_sr_df["pred_norm"].values,
    test_sr_df["current_A"].values, test_sr_df["pred_A"].values,
    "test_SR (SR=50 interpolation)")
metrics_rows.append(m)

# Test-MAT: unseen material NM4
m = compute_metrics(
    y_test_mat, test_mat_df["pred_norm"].values,
    test_mat_df["current_A"].values, test_mat_df["pred_A"].values,
    "test_MAT (NM4 extrapolation)")
metrics_rows.append(m)

metrics_df = pd.DataFrame(metrics_rows).set_index("partition")

print("=" * 70)
print("RANDOM FOREST BASELINE — EVALUATION SUMMARY")
print("=" * 70)
print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
print()
print("Units: RMSE / MAE / MaxErr in micro-Amperes (uA).  R2 is dimensionless.")
print()
print("Scientific interpretation:")
print("  - Train RMSE near 0 is expected (RF memorises training rows).")
print("  - Val / Test-SR RMSE shows scan-rate interpolation capability.")
print("  - Test-MAT RMSE shows material extrapolation capability.")
print("  - Paper baseline (inflated by row-wise split): RF RMSE = 170 uA.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7a — CV CURVE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7a — Predicted vs Actual CV Curves

The most physically meaningful test: does the model reconstruct the correct
closed-loop voltammogram shape?  Rows are sorted by `(half_sweep_id, step_index)`
to recover the electrochemical trajectory from the tabular predictions.""")

code("""\
# ── Helper: plot one CV curve (actual vs predicted) ───────────────────────────
def plot_cv_curve(ax, group_df, title, show_legend=True):
    grp = group_df.sort_values(["half_sweep_id", "step_index"])
    colours = {0: "#1f77b4", 1: "#d62728"}
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
        ax.legend(fontsize=7, title="Solid=actual  Dashed=RF")


# ── Validation curves: NM1, NM2, NM3 at SR=30 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nm in enumerate(["NM1", "NM2", "NM3"]):
    sub = val_df[(val_df["nm_id"]==nm) & (val_df["scan_rate_mVs"]==30)]
    plot_cv_curve(axes[i], sub, f"VAL: {nm} @ 30 mV/s", show_legend=(i==0))
fig.suptitle("Validation Set — Scan-Rate Interpolation (SR=30, seen materials)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "01_cv_curves_val.png", bbox_inches="tight")
plt.show()
print("Saved 01_cv_curves_val.png")""")

code("""\
# ── Test-SR curves: NM1, NM2, NM3 at SR=50 ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nm in enumerate(["NM1", "NM2", "NM3"]):
    sub = test_sr_df[(test_sr_df["nm_id"]==nm) & (test_sr_df["scan_rate_mVs"]==50)]
    plot_cv_curve(axes[i], sub, f"TEST-SR: {nm} @ 50 mV/s", show_legend=(i==0))
fig.suptitle("Test-SR Set — Scan-Rate Interpolation (SR=50, seen materials)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_cv_curves_test_sr.png", bbox_inches="tight")
plt.show()
print("Saved 02_cv_curves_test_sr.png")""")

code("""\
# ── Test-MAT curves: NM4 at 6 representative scan rates ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
sr_showcase = [10, 30, 50, 70, 90, 100]

for i, sr in enumerate(sr_showcase):
    sub = test_mat_df[(test_mat_df["nm_id"]=="NM4") & (test_mat_df["scan_rate_mVs"]==sr)]
    plot_cv_curve(axes[i], sub, f"TEST-MAT: NM4 @ {sr} mV/s", show_legend=(i==0))

fig.suptitle("Test-MAT Set — Material Extrapolation (NM4, fully unseen formulation)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_cv_curves_test_mat.png", bbox_inches="tight")
plt.show()
print("Saved 03_cv_curves_test_mat.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7b — PARITY PLOTS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7b — Parity Plots (Predicted vs Actual)

A parity plot (predicted on y-axis vs actual on x-axis) with a perfect model
lying exactly on the diagonal `y = x`.  Colour encodes scan rate to show
whether prediction quality varies with scan rate.""")

code("""\
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
    # Identity line
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

fig.suptitle("Parity Plots — Random Forest Baseline", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "04_parity_plots.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved 04_parity_plots.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7c — RESIDUAL DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7c — Residual Distributions

Residual = predicted – actual (in μA).  A good model shows a zero-centred,
approximately Gaussian distribution.  Systematic bias or heavy tails indicate
a structural limitation of the RF feature set.""")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

plot_configs = [
    (val_df,      "Val (SR=30)",      "#ff7f0e"),
    (test_sr_df,  "Test-SR (SR=50)",  "#1f77b4"),
    (test_mat_df, "Test-MAT (NM4)",   "#d62728"),
]

for ax, (df, title, colour) in zip(axes, plot_configs):
    residuals = (df["pred_A"] - df["current_A"]) * 1e6   # uA
    ax.hist(residuals, bins=80, color=colour, alpha=0.75, edgecolor="none",
            density=True)
    ax.axvline(0,                   color="k",       lw=1.5, ls="-",  label="zero bias")
    ax.axvline(residuals.mean(),    color="darkred",  lw=1.5, ls="--", label=f"mean={residuals.mean():.1f} uA")
    ax.axvline(residuals.median(),  color="navy",     lw=1.2, ls=":",  label=f"median={residuals.median():.1f} uA")
    ax.set_xlabel("Residual (uA): predicted - actual", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

fig.suptitle("Residual Distributions — Random Forest Baseline", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "05_residual_distributions.png", bbox_inches="tight")
plt.show()
print("Saved 05_residual_distributions.png")""")

code("""\
# ── Residuals vs potential (check for systematic bias across voltage range) ───
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (df, title, colour) in zip(axes, plot_configs):
    residuals = (df["pred_A"] - df["current_A"]) * 1e6
    ax.scatter(df["potential_V"], residuals,
               c=df["sweep_direction"], cmap="coolwarm",
               s=1.0, alpha=0.25, rasterized=True)
    ax.axhline(0, color="k", lw=1.0, ls="--")
    ax.set_xlabel("Potential (V vs SCE)", fontsize=8)
    ax.set_ylabel("Residual (uA)", fontsize=8)
    ax.set_title(f"{title} — residual vs potential", fontsize=9)
    ax.grid(True, alpha=0.25)

# Colourbar legend (blue=cathodic, red=anodic)
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#1f77b4",
           markersize=6, label="Cathodic (dir=0)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#d62728",
           markersize=6, label="Anodic (dir=1)"),
]
axes[0].legend(handles=legend_elems, fontsize=7)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "06_residual_vs_potential.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved 06_residual_vs_potential.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7d — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7d — Feature Importance

Two complementary importance metrics:

**MDI (Mean Decrease in Impurity)** — built into RF; fast but can inflate
importance of high-cardinality continuous features.

**Permutation Importance** — computed on the **validation set** (not training).
Shuffles one feature at a time and measures RMSE increase.  More reliable;
uses held-out data so it reflects generalisation, not in-sample fit.""")

code("""\
from sklearn.inspection import permutation_importance

# ── MDI importance (built-in RF, from training) ───────────────────────────────
mdi_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

# ── Permutation importance (on validation set — generalisation-aware) ─────────
print("Computing permutation importance on val set (n_repeats=20) ...")
t0 = time.time()
perm = permutation_importance(
    rf, X_val, y_val,
    n_repeats=20,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)
print(f"  Done in {time.time()-t0:.1f} s")

perm_means = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
perm_stds  = pd.Series(perm.importances_std,  index=FEATURES)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# MDI
axes[0].barh(mdi_imp.index[::-1], mdi_imp.values[::-1], color="#2ca02c", alpha=0.8)
axes[0].set_xlabel("MDI importance (mean decrease in impurity)", fontsize=9)
axes[0].set_title("MDI Feature Importance\\n(training data — may overrate continuous features)", fontsize=9)
axes[0].grid(True, alpha=0.3, axis="x")

# Permutation
xerr = perm_stds.loc[perm_means.index][::-1].values
axes[1].barh(
    perm_means.index[::-1], perm_means.values[::-1],
    xerr=xerr, color="#1f77b4", alpha=0.8,
    error_kw={"elinewidth": 1.2, "capsize": 3}
)
axes[1].set_xlabel("Permutation importance (RMSE increase on val set)", fontsize=9)
axes[1].set_title("Permutation Feature Importance\\n(validation set — generalisation-aware)", fontsize=9)
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "07_feature_importance.png", bbox_inches="tight")
plt.show()
print("Saved 07_feature_importance.png")

print()
print("MDI ranking:")
for feat, val in mdi_imp.items():
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
- Whether performance degrades at extreme scan rates
- Which NM4 scan rates are best/worst extrapolated
- Whether prediction quality is consistent across materials""")

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

val_grp     = group_rmse(val_df,      "val")
test_sr_grp = group_rmse(test_sr_df,  "test_sr")
test_mat_grp= group_rmse(test_mat_df, "test_mat")

# ── Validation: RMSE per material at SR=30 ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.bar(val_grp["nm_id"], val_grp["RMSE_uA"], color="#ff7f0e", alpha=0.85)
for i, row in val_grp.iterrows():
    ax.text(i, row["RMSE_uA"] + 0.2, f"R2={row['R2']:.3f}", ha="center", fontsize=8)
ax.set_xlabel("Material")
ax.set_ylabel("RMSE (uA)")
ax.set_title("Val (SR=30) — RMSE per material")
ax.grid(True, alpha=0.3, axis="y")

# Test-SR: RMSE per material at SR=50
ax = axes[1]
ax.bar(test_sr_grp["nm_id"], test_sr_grp["RMSE_uA"], color="#1f77b4", alpha=0.85)
for i, row in test_sr_grp.iterrows():
    ax.text(i, row["RMSE_uA"] + 0.2, f"R2={row['R2']:.3f}", ha="center", fontsize=8)
ax.set_xlabel("Material")
ax.set_ylabel("RMSE (uA)")
ax.set_title("Test-SR (SR=50) — RMSE per material")
ax.grid(True, alpha=0.3, axis="y")

# Test-MAT: RMSE per scan rate for NM4
ax = axes[2]
ax.bar(test_mat_grp["scan_rate_mVs"].astype(int).astype(str),
       test_mat_grp["RMSE_uA"], color="#d62728", alpha=0.85)
ax.set_xlabel("Scan rate (mV/s)")
ax.set_ylabel("RMSE (uA)")
ax.set_title("Test-MAT (NM4) — RMSE per scan rate")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "08_per_group_rmse.png", bbox_inches="tight")
plt.show()
print("Saved 08_per_group_rmse.png")""")

code("""\
# ── Test-MAT: RMSE trend vs scan rate ────────────────────────────────────────
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

plt.tight_layout()
plt.savefig(RESULTS_DIR / "09_nm4_scan_rate_trend.png", bbox_inches="tight")
plt.show()
print("Saved 09_nm4_scan_rate_trend.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SAVE MODEL AND RESULTS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 8 — Save Model and Results")

code("""\
import joblib

# ── Save trained model ────────────────────────────────────────────────────────
model_path = RESULTS_DIR / "rf_baseline_model.joblib"
joblib.dump(rf, model_path)
print(f"Saved model: {model_path}  ({model_path.stat().st_size/1e3:.0f} KB)")

# ── Save metrics as JSON ──────────────────────────────────────────────────────
metrics_out = {
    "model"     : "RandomForestRegressor",
    "params"    : RF_PARAMS,
    "oob_R2"    : float(rf.oob_score_),
    "metrics"   : metrics_df.reset_index().to_dict(orient="records"),
    "per_group" : {
        "val"     : val_grp.to_dict(orient="records"),
        "test_sr" : test_sr_grp.to_dict(orient="records"),
        "test_mat": test_mat_grp.to_dict(orient="records"),
    },
    "feature_importance_mdi": {
        f: float(v) for f, v in mdi_imp.items()
    },
    "feature_importance_permutation": {
        f: {"mean": float(perm_means[f]), "std": float(perm_stds[f])}
        for f in FEATURES
    },
}

metrics_path = RESULTS_DIR / "rf_baseline_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(metrics_out, fh, indent=2)
print(f"Saved metrics: {metrics_path}")

# ── Save per-group RMSE tables ────────────────────────────────────────────────
for name, df_grp in [("val", val_grp), ("test_sr", test_sr_grp),
                      ("test_mat", test_mat_grp)]:
    path = RESULTS_DIR / f"per_group_rmse_{name}.csv"
    df_grp.to_csv(path, index=False)
    print(f"Saved {path.name}")

print()
print(f"✓ All results saved to: {RESULTS_DIR}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SCIENTIFIC SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 9 — Scientific Summary and Next Steps

### What this baseline establishes
This notebook provides the first **scientifically valid** RF performance benchmark
for the ZnO supercapacitor CV prediction task.  Unlike the paper's RF (which
used row-wise splitting and reported inflated metrics), our evaluation partitions
test real generalisation.

### Key questions the results answer
1. **Scan-rate interpolation (Val / Test-SR):** Does the RF correctly infer CV
   behaviour at scan rates it never trained on?  If yes, the feature set
   (`sqrt_scan_rate_norm`, `log_scan_rate_norm`) successfully encodes
   the scan-rate physics.
2. **Material extrapolation (Test-MAT / NM4):** Can a model trained only on
   NM1–NM3 predict NM4 curves?  This is the hardest test — it probes whether
   the electrochemical feature space is universal across ZnO formulations.

### Limitations of RF for this problem
- RF sees rows **independently** — it cannot model capacitive hysteresis as a
  trajectory.  The CV loop shape emerges from features, not memory.
- RF cannot extrapolate beyond its training distribution (no scan rate > 100 mV/s).
- Prediction uncertainty is not calibrated (no confidence intervals from this model).

### Next steps in the pipeline
| Notebook | Model | What it adds over RF |
|----------|-------|----------------------|
| 03_xgboost_baseline | XGBoost | Gradient boosting; often outperforms RF on tabular data |
| 04_ann_baseline | ANN (dense) | Learns non-linear interactions; closer to paper architecture |
| 05_lstm_baseline | LSTM | Explicit sequence modelling of the CV trajectory |""")

code("""\
# ── Final printed summary ─────────────────────────────────────────────────────
print("=" * 70)
print("RANDOM FOREST BASELINE — FINAL SUMMARY")
print("=" * 70)
print()
print(f"  Model          : RandomForestRegressor")
print(f"  n_estimators   : {rf.n_estimators}")
print(f"  max_depth      : {rf.max_depth}")
print(f"  min_samples_lf : {rf.min_samples_leaf}")
print(f"  max_features   : {rf.max_features}")
print(f"  OOB R2         : {rf.oob_score_:.4f}")
print()
print(f"  {'Partition':<30} {'RMSE (uA)':>12} {'MAE (uA)':>10} {'R2':>8}")
print("  " + "-" * 62)
for _, row in metrics_df.iterrows():
    print(f"  {row.name:<30} {row['RMSE_uA']:>12.2f} {row['MAE_uA']:>10.2f} {row['R2']:>8.4f}")
print()
print(f"  Paper RF RMSE (row-wise split, inflated): 170 uA")
print(f"  Our RF RMSE (val, honest split)         : "
      + f"{metrics_df.loc['val  (SR=30 interpolation)','RMSE_uA']:.2f} uA")
print()

print()
print("Notebook complete. Run Section 10 below to download all results.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — DOWNLOAD RESULTS TO LOCAL MACHINE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 10 — Download Results to Your Local Machine

Run this cell to zip the entire `rf_baseline_results/` folder and download it.
The zip will contain all figures, metrics JSON, CSV tables, and the saved model.

**Run this before closing the Colab session** — files are lost when the runtime disconnects.""")

code("""\
import shutil, os
from google.colab import files as _cf

zip_base = "/content/rf_baseline_results_download"
print(f"Creating zip of {RESULTS_DIR} ...")
shutil.make_archive(zip_base, "zip", str(RESULTS_DIR))

zip_path = zip_base + ".zip"
size_mb  = os.path.getsize(zip_path) / 1e6
print(f"Created : {zip_path}  ({size_mb:.1f} MB)")
print()

for fname in sorted(os.listdir(str(RESULTS_DIR))):
    fpath = os.path.join(str(RESULTS_DIR), fname)
    fsize = os.path.getsize(fpath) / 1e3
    print(f"  {fname:<55} {fsize:>8.0f} KB")

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

out_path = Path(__file__).resolve().parent.parent.parent / "research" / "notebooks" / "02_rf_baseline.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out_path))
print(f"Written : {out_path}")
print(f"Cells   : {len(nb.cells)}")
