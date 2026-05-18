"""
build_lightgbm_notebook.py
Generates notebooks/07_lightgbm_model.ipynb
Run once: python src/build_lightgbm_notebook.py
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
# Notebook 07 — LightGBM Gradient Boosting Model
## ZnO Supercapacitor CV Trajectory Prediction · Leaf-Wise Boosting

### Purpose
Train a **LightGBM (Light Gradient Boosting Machine)** model on the same
scientifically correct partitions established in Notebook 01.  LightGBM is a
tabular model — identical flat-row representation to RF and XGBoost — allowing
a direct apples-to-apples comparison between three tree ensemble methods.

This notebook:
- Uses **leaf-wise (best-first) tree growth** instead of depth-wise (RF/XGBoost).
- Trains with **early stopping** monitored on the SR=30 validation partition.
- Reports **gain importance** (total information gain per feature) and
  **permutation importance** (RMSE degradation when feature is shuffled).
- Delivers a **six-model comparison table**
  (RF vs XGBoost vs ANN vs LSTM vs GRU vs LightGBM).
- Produces 12 publication-ready figures.

### Why LightGBM may outperform RF and XGBoost
| Property | Random Forest | XGBoost | LightGBM |
|---|---|---|---|
| Tree growth | Depth-wise | Depth-wise | **Leaf-wise (best-first)** |
| Split finding | Exact | Exact / approx | **Histogram-based** |
| Regularisation | None | L1 + L2 | L1 + L2 + `min_child_samples` |
| Training speed | Parallel trees | Sequential boost | **Fastest of the three** |
| Memory | High | Medium | **Lowest** |
| Typical accuracy | Baseline | Better than RF | **Competitive or best** |

Leaf-wise growth finds the leaf with highest loss reduction at each step —
fewer leaves needed to achieve the same training loss, reducing overfitting risk
on the engineered CV feature set.

### Model representation
```
Input:  (n_rows, 10)   — same as RF and XGBoost
Output: (n_rows,)      — predicted current_normalized per measurement row
```

Each measurement row is treated **independently** — identical to RF/XGBoost.
This is the scientifically fair comparison baseline for all tabular models.""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 0 — Environment Setup and File Upload

Set `EXECUTION_MODE` before running anything else.

| Mode | When to use |
|------|-------------|
| `"colab_upload"` | Google Colab — upload files from your PC via dialog |
| `"colab_drive"` | Google Colab — files already on Google Drive |
| `"local"` | Local machine (Jupyter / VS Code / terminal) |

**Required files** (from `data/processed/` in your PBL Project folder):
```
master_long_format.parquet    train_set.parquet
val_set.parquet               test_scanrate.parquet
test_material.parquet         scalers.json
```

### Runtime recommendation
**CPU runtime is sufficient** — LightGBM does not use GPU by default and
trains very fast on CPU with histogram-based splits.  No GPU needed.""")

code("""\
# ─────────────────────────────────────────────────────────────────────────────
# SET THIS BEFORE RUNNING ANY OTHER CELL
# ─────────────────────────────────────────────────────────────────────────────
#   "colab_upload"  — Google Colab: upload files via dialog
#   "colab_drive"   — Google Colab: files on Google Drive
#   "local"         — Local machine: files already on disk
#
EXECUTION_MODE = "colab_upload"       # <<< CHANGE THIS IF NEEDED

# For "colab_drive" only
DRIVE_DATA_PATH = "/content/drive/MyDrive/PBL_Project/data/processed"

# For "local" only — root of your PBL Project folder
LOCAL_PROJECT_PATH = r"D:\\mca\\2nd semester\\PBL Project"

# ── Required files ─────────────────────────────────────────────────────────────
REQUIRED = [
    "master_long_format.parquet",
    "train_set.parquet",
    "val_set.parquet",
    "test_scanrate.parquet",
    "test_material.parquet",
    "scalers.json",
]

from pathlib import Path
import shutil, os

if EXECUTION_MODE == "colab_upload":
    from google.colab import files as _cf
    BASE_DIR      = Path("/content/PBL_Project")
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Select all 6 files in the dialog (Ctrl+click each):")
    for f in REQUIRED:
        print(f"  {f}")
    print()
    _cf.upload()
    moved = 0
    for fname in REQUIRED:
        src = Path("/content") / fname
        dst = PROCESSED_DIR / fname
        if src.exists():
            shutil.move(str(src), str(dst))
            moved += 1
    print(f"  Moved {moved} file(s) into {PROCESSED_DIR}")

elif EXECUTION_MODE == "colab_drive":
    from google.colab import drive as _drive
    print("Mounting Google Drive ...")
    _drive.mount("/content/drive")
    drive_path    = Path(DRIVE_DATA_PATH)
    BASE_DIR      = Path("/content/PBL_Project")
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fname in REQUIRED:
        src = drive_path / fname
        dst = PROCESSED_DIR / fname
        if src.exists():
            shutil.copy(str(src), str(dst))
            copied += 1
        else:
            print(f"  [WARNING] Not found on Drive: {src}")
    print(f"  Copied {copied} file(s) from Drive into {PROCESSED_DIR}")

elif EXECUTION_MODE == "local":
    BASE_DIR      = Path(LOCAL_PROJECT_PATH)
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    print(f"Local mode — reading from: {PROCESSED_DIR}")

else:
    raise ValueError(
        f"Unknown EXECUTION_MODE: {EXECUTION_MODE!r}\\n"
        "Valid options: 'colab_upload', 'colab_drive', 'local'"
    )

# ── Create all results directories ────────────────────────────────────────────
RESULTS_DIR = BASE_DIR / "results" / "lightgbm"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
CSV_DIR     = RESULTS_DIR / "csv"
MODEL_DIR   = RESULTS_DIR / "model"

for d in [FIGURES_DIR, METRICS_DIR, CSV_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Verify all files present ───────────────────────────────────────────────────
missing = [f for f in REQUIRED if not (PROCESSED_DIR / f).exists()]
assert not missing, (
    f"Missing files in {PROCESSED_DIR}: {missing}\\n"
    "Re-run this cell and upload / provide all 6 files."
)

print()
print("All 6 files verified")
print(f"  Data     : {PROCESSED_DIR}")
print(f"  Results  : {RESULTS_DIR}")
print(f"    figures/ : {FIGURES_DIR}")
print(f"    metrics/ : {METRICS_DIR}")
print(f"    csv/     : {CSV_DIR}")
print(f"    model/   : {MODEL_DIR}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMPORTS + LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 1 — Imports and Load Artefacts

Loads all preprocessing outputs from Notebook 01.
**Nothing is recomputed** — splits, normalisation, and feature engineering are fixed.""")

code("""\
import numpy  as np
import pandas as pd
import json
import warnings
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.6e}".format)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# ── LightGBM ──────────────────────────────────────────────────────────────────
import lightgbm as lgb
print(f"LightGBM version : {lgb.__version__}")

# ── Load preprocessing artefacts ─────────────────────────────────────────────
master_df   = pd.read_parquet(PROCESSED_DIR / "master_long_format.parquet")
train_df    = pd.read_parquet(PROCESSED_DIR / "train_set.parquet")
val_df      = pd.read_parquet(PROCESSED_DIR / "val_set.parquet")
test_sr_df  = pd.read_parquet(PROCESSED_DIR / "test_scanrate.parquet")
test_mat_df = pd.read_parquet(PROCESSED_DIR / "test_material.parquet")

with open(PROCESSED_DIR / "scalers.json") as fh:
    scalers = json.load(fh)

print()
print("All artefacts loaded")
print(f"  master        : {master_df.shape}")
print(f"  train         : {train_df.shape}")
print(f"  val           : {val_df.shape}")
print(f"  test_scanrate : {test_sr_df.shape}")
print(f"  test_material : {test_mat_df.shape}")
print(f"  scaler groups : {len(scalers['current_per_group'])}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SPLIT INTEGRITY VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 2 — Split Integrity Verification

Identical PASS/FAIL checks as Notebooks 02–06.
Verifies the leakage-safe splits before any model training.""")

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

print("=== SPLIT INTEGRITY CHECK ===\\n")

expected = {"train": 62400, "val": 7800, "test_scanrate": 7800, "test_material": 26000}
for name, exp_rows in expected.items():
    df = {"train": train_df, "val": val_df,
          "test_scanrate": test_sr_df, "test_material": test_mat_df}[name]
    assert len(df) == exp_rows, f"{name}: expected {exp_rows}, got {len(df)}"
    print(f"  [PASS] {name:15s}: {len(df):,} rows")

assert set(train_df["nm_id"].unique())    == {"NM1","NM2","NM3"}
assert set(val_df["nm_id"].unique())      == {"NM1","NM2","NM3"}
assert set(test_sr_df["nm_id"].unique())  == {"NM1","NM2","NM3"}
assert set(test_mat_df["nm_id"].unique()) == {"NM4"}
print("  [PASS] Material assignments correct")

assert set(train_df["scan_rate_mVs"].unique())   == {10,20,40,60,70,80,90,100}
assert set(val_df["scan_rate_mVs"].unique())     == {30}
assert set(test_sr_df["scan_rate_mVs"].unique()) == {50}
print("  [PASS] Scan-rate assignments correct")

for name, df in [("train",train_df),("val",val_df),
                  ("test_sr",test_sr_df),("test_mat",test_mat_df)]:
    n_nan = df[FEATURES + [TARGET]].isna().sum().sum()
    assert n_nan == 0, f"NaN in {name}: {n_nan}"
print("  [PASS] Zero NaN in features and target")

for name, df in [("train",train_df),("val",val_df),
                  ("test_sr",test_sr_df),("test_mat",test_mat_df)]:
    assert df[TARGET].between(-1e-9, 1+1e-9).all(), f"Target out of [0,1] in {name}"
print("  [PASS] current_normalized in [0, 1]")

block_splits = (master_df.groupby(["nm_id","scan_rate_mVs","half_sweep_id"])["split"]
                          .nunique())
assert (block_splits == 1).all()
print("  [PASS] No half-sweep crosses a split boundary")

leakage_cols = ["current_A","current_norm_min","current_norm_max"]
for col in leakage_cols:
    assert col not in FEATURES, f"LEAKAGE: {col} in FEATURES!"
print("  [PASS] No leakage columns in FEATURES")

print("\\n All INTEGRITY CHECKS PASSED — safe to train LightGBM")

# ── Prepare feature matrices (flat tabular — same as RF/XGBoost) ─────────────
# LightGBM is a row-wise tabular model — NO sequence construction.
# Each measurement row is one sample, identical to RF/XGBoost representation.
X_train = train_df[FEATURES].values.astype(np.float32)
y_train = train_df[TARGET].values.astype(np.float32)

X_val   = val_df[FEATURES].values.astype(np.float32)
y_val   = val_df[TARGET].values.astype(np.float32)

X_test_sr  = test_sr_df[FEATURES].values.astype(np.float32)
y_test_sr  = test_sr_df[TARGET].values.astype(np.float32)

X_test_mat = test_mat_df[FEATURES].values.astype(np.float32)
y_test_mat = test_mat_df[TARGET].values.astype(np.float32)

print()
print("Feature matrix shapes (flat tabular — no sequences):")
print(f"  X_train    : {X_train.shape}   y_train    : {y_train.shape}")
print(f"  X_val      : {X_val.shape}    y_val      : {y_val.shape}")
print(f"  X_test_sr  : {X_test_sr.shape}    y_test_sr  : {y_test_sr.shape}")
print(f"  X_test_mat : {X_test_mat.shape}   y_test_mat : {y_test_mat.shape}")
print(f"  n_features : {len(FEATURES)}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 3 — Feature Correlation Matrix

Identical to RF and XGBoost notebooks — visualises linear correlations
between the 10 engineered features on the training set.""")

code("""\
# ── Figure 00: Feature correlation heatmap ────────────────────────────────────
corr = pd.DataFrame(X_train, columns=FEATURES).corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1,
    linewidths=0.5, cbar_kws={"shrink": 0.8},
    ax=ax, annot_kws={"size": 7},
)
ax.set_title(
    "Feature Correlation Matrix — Training Set (62,400 rows)\\n"
    "Lower triangle only (matrix is symmetric)",
    fontsize=11, pad=10,
)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "00_feature_correlation.png", bbox_inches="tight")
plt.show()
print("Saved figures/00_feature_correlation.png")
print()
print("Note: high correlation between scan-rate features is expected.")
print("Tree models handle correlated features better than linear models —")
print("they select the most discriminative split at each node regardless.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LIGHTGBM MODEL DEFINITION AND TRAINING
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 4 — LightGBM Model Definition and Training

### Parameter rationale
| Parameter | Value | Rationale |
|---|---|---|
| `boosting_type` | `'gbdt'` | Gradient Boosted Decision Trees — standard ensemble |
| `n_estimators` | 2000 | Upper bound; early stopping will terminate earlier |
| `learning_rate` | 0.03 | Slower than XGBoost (0.05) to allow finer convergence |
| `num_leaves` | 63 | Controls model complexity — leaf-wise allows deep trees cheaply |
| `max_depth` | -1 | Unlimited depth (leaf count is the real constraint) |
| `min_child_samples` | 20 | Minimum rows per leaf — prevents overfitting to small groups |
| `subsample` | 0.8 | 80% row sampling per tree — adds variance diversity |
| `colsample_bytree` | 0.8 | 80% feature sampling per tree — same as RF/XGBoost |
| `reg_alpha` | 0.05 | L1 regularisation — sparse feature usage |
| `reg_lambda` | 1.0 | L2 regularisation — weight shrinkage |

### Leaf-wise vs depth-wise growth
```
Depth-wise (RF, XGBoost):          Leaf-wise (LightGBM):
Level 0:   [Root]                  Step 1: [Root] → split best leaf
Level 1: [L] [R]                   Step 2: split leaf with max gain
Level 2: [L][R] [L][R]             Step 3: keep splitting max-gain leaf
→ balanced tree, more splits       → unbalanced tree, fewer splits for same loss
```
Leaf-wise finds the single leaf that most reduces loss at each step.
Result: fewer leaves needed → faster training, often lower val loss.

### Early stopping
Monitored on the SR=30 validation partition (same unseen scan rate
used for val evaluation in all other notebooks).  Stops when validation
RMSE has not improved for 50 consecutive rounds.""")

code("""\
# ── LightGBM hyperparameters ──────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective"       : "regression",   # standard MSE regression
    "metric"          : "rmse",         # evaluation metric for early stopping
    "boosting_type"   : "gbdt",         # gradient boosted decision trees
    "n_estimators"    : 2000,           # max trees (early stopping will cut earlier)
    "learning_rate"   : 0.03,           # slower lr -> finer convergence
    "num_leaves"      : 63,             # max leaves per tree (leaf-wise constraint)
    "max_depth"       : -1,             # no depth limit (leaves are the constraint)
    "min_child_samples": 20,            # min samples per leaf (prevents overfitting)
    "subsample"       : 0.8,            # 80% row sampling per tree
    "colsample_bytree": 0.8,            # 80% feature sampling per tree
    "reg_alpha"       : 0.05,           # L1 regularisation
    "reg_lambda"      : 1.0,            # L2 regularisation
    "random_state"    : 42,
    "n_jobs"          : -1,             # use all CPU cores
    "importance_type" : "gain",         # feature_importances_ returns gain
    "verbose"         : -1,             # suppress LightGBM internal verbosity
}

model = lgb.LGBMRegressor(**LGBM_PARAMS)

print("LightGBM model initialised")
print(f"  Params: {LGBM_PARAMS}")
print()

# ── Callbacks: early stopping + verbose evaluation log ───────────────────────
# early_stopping: stop if val RMSE has not improved for 50 rounds
# log_evaluation: print progress every 100 rounds
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100),
]

print("Training LightGBM ...")
print(f"  Training rows      : {X_train.shape[0]:,}")
print(f"  Validation rows    : {X_val.shape[0]:,}  (SR=30 — unseen scan rate)")
print(f"  Max trees          : {LGBM_PARAMS['n_estimators']}")
print(f"  Learning rate      : {LGBM_PARAMS['learning_rate']}")
print(f"  Num leaves         : {LGBM_PARAMS['num_leaves']}")
print(f"  Early stopping     : 50 rounds of no val RMSE improvement")
print()

t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=callbacks,
)
elapsed = time.time() - t0

# ── Access training history ───────────────────────────────────────────────────
# evals_result_ keys may vary by LightGBM version:
#   typical: {'training': {'rmse': [...]}, 'valid_0': {'rmse': [...]}}
evals      = model.evals_result_
all_keys   = list(evals.keys())
train_key  = all_keys[0]                          # 'training' or 'valid_0'
val_key    = all_keys[1]                          # 'valid_0'  or 'valid_1'
metric_key = list(evals[train_key].keys())[0]     # 'rmse'

train_rmse_hist = np.array(evals[train_key][metric_key])
val_rmse_hist   = np.array(evals[val_key][metric_key])
best_iter       = model.best_iteration_           # 1-indexed best round

print()
print(f"Training complete in {elapsed:.1f} s  ({elapsed/60:.1f} min)")
print(f"  Total trees built    : {len(train_rmse_hist)}")
print(f"  Best iteration       : {best_iter}  (1-indexed)")
print(f"  Best val RMSE (norm) : {val_rmse_hist[best_iter-1]:.6f}")
print(f"  Train RMSE at best   : {train_rmse_hist[best_iter-1]:.6f}")
print(f"  Overfitting gap      : {val_rmse_hist[best_iter-1] - train_rmse_hist[best_iter-1]:.6f}")""")

code("""\
# ── Figure 01: LightGBM training curve ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
round_axis = np.arange(1, len(train_rmse_hist) + 1)

ax = axes[0]
ax.plot(round_axis, train_rmse_hist, color="#2ca02c", lw=0.8, alpha=0.85,
        label="Train RMSE")
ax.plot(round_axis, val_rmse_hist,   color="#1f77b4", lw=1.5, alpha=0.9,
        label="Val RMSE (SR=30)")
ax.axvline(best_iter, color="#d62728", lw=1.8, ls="--",
           label=f"Best iteration = {best_iter}")
ax.set_xlabel("Boosting round (tree number)")
ax.set_ylabel("RMSE (normalised current)")
ax.set_title("LightGBM Boosting Curve — RMSE per Round", fontsize=10, pad=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Zoomed view around convergence
ax = axes[1]
zoom_start = max(0, best_iter - 200)
zoom_end   = min(len(train_rmse_hist), best_iter + 200)
ax.plot(round_axis[zoom_start:zoom_end], train_rmse_hist[zoom_start:zoom_end],
        color="#2ca02c", lw=0.8, alpha=0.85, label="Train RMSE")
ax.plot(round_axis[zoom_start:zoom_end], val_rmse_hist[zoom_start:zoom_end],
        color="#1f77b4", lw=1.5, alpha=0.9,  label="Val RMSE (SR=30)")
ax.axvline(best_iter, color="#d62728", lw=1.8, ls="--",
           label=f"Best iteration = {best_iter}")
ax.set_xlabel("Boosting round")
ax.set_ylabel("RMSE (normalised current)")
ax.set_title(f"Zoomed view (rounds {zoom_start+1}–{zoom_end})\\nConvergence region",
             fontsize=10, pad=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("LightGBM Training History — Leaf-Wise Gradient Boosting", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_lgbm_training_curve.png", bbox_inches="tight")
plt.show()
print("Saved figures/01_lgbm_training_curve.png")
print()
print("Scientific interpretation:")
print(f"  Best iteration = {best_iter}  (out of max {LGBM_PARAMS['n_estimators']})")
print(f"  Early stopping fired at round {len(train_rmse_hist)}, "
      f"reverting to round {best_iter}")
print("  — Leaf-wise growth: each round adds the leaf that most reduces loss.")
print("  — Monotone decrease in train RMSE: boosting always reduces training error.")
print("  — Val RMSE plateau before train RMSE: model begins memorising scan-rate")
print("    patterns not seen in the val set (SR=30).")
print("  — Gap between train and val RMSE reflects interpolation difficulty:")
print("    the 10 engineered features help, but SR=30 trajectory shape differs")
print("    from the 8 training scan rates in ways leaf splits cannot fully capture.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PREDICTIONS + INVERSE TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 5 — Predictions and Inverse Transform

`model.predict()` automatically uses the **best iteration** weights
(restored by early stopping).  Predictions are in normalised units [0,1]
and inverse-transformed to Amperes using the per-group scalers from Notebook 01.

### Inverse transform
`I_pred (A) = pred_norm × (I_max − I_min) + I_min`

where `I_min` and `I_max` are stored row-wise in each parquet file
(they are group-specific — different for each `nm_id × scan_rate` combination).""")

code("""\
# ── Generate predictions on all four partitions ────────────────────────────────
# model.predict() uses best_iteration automatically after early stopping.
print("Generating predictions ...")
t0 = time.time()

train_df["pred_norm"]    = model.predict(X_train)
val_df["pred_norm"]      = model.predict(X_val)
test_sr_df["pred_norm"]  = model.predict(X_test_sr)
test_mat_df["pred_norm"] = model.predict(X_test_mat)

print(f"  Done in {time.time()-t0:.2f} s")

# Verify no NaN predictions
for name, df in [("train",train_df),("val",val_df),
                  ("test_sr",test_sr_df),("test_mat",test_mat_df)]:
    n_nan = df["pred_norm"].isna().sum()
    assert n_nan == 0, f"NaN predictions in {name}: {n_nan}"
print("  [PASS] No NaN in predictions")

# ── Inverse transform: normalised -> Amperes ──────────────────────────────────
for df in [train_df, val_df, test_sr_df, test_mat_df]:
    rng_         = df["current_norm_max"] - df["current_norm_min"]
    df["pred_A"] = df["pred_norm"] * rng_ + df["current_norm_min"]

print()
print("Predictions inverse-transformed to Amperes")
for label, df in [("train",train_df),("val",val_df),
                   ("test_sr",test_sr_df),("test_mat",test_mat_df)]:
    print(f"  {label:10s}: pred_A range "
          f"[{df['pred_A'].min():.4e}, {df['pred_A'].max():.4e}] A")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 6 — Evaluation Metrics

Identical metric computation to all previous notebooks.
LightGBM predictions are row-level — directly comparable with RF, XGBoost,
ANN, LSTM, and GRU (after LSTM/GRU predictions are flattened).""")

code("""\
def compute_metrics(y_true_norm, y_pred_norm, actual_A, pred_A, label):
    rmse_norm = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))
    rmse_A    = np.sqrt(mean_squared_error(actual_A, pred_A))
    mae_A     = mean_absolute_error(actual_A, pred_A)
    r2        = r2_score(y_true_norm, y_pred_norm)
    max_err   = np.abs(actual_A - pred_A).max()
    return {
        "partition" : label,
        "n_samples" : len(y_true_norm),
        "RMSE_norm" : rmse_norm,
        "RMSE_uA"   : rmse_A   * 1e6,
        "MAE_uA"    : mae_A    * 1e6,
        "MaxErr_uA" : max_err  * 1e6,
        "R2"        : r2,
    }

metrics_rows = []
metrics_rows.append(compute_metrics(
    train_df["current_normalized"].values, train_df["pred_norm"].values,
    train_df["current_A"].values,          train_df["pred_A"].values,
    "train (in-sample)"))
metrics_rows.append(compute_metrics(
    val_df["current_normalized"].values,   val_df["pred_norm"].values,
    val_df["current_A"].values,            val_df["pred_A"].values,
    "val  (SR=30 interpolation)"))
metrics_rows.append(compute_metrics(
    test_sr_df["current_normalized"].values,  test_sr_df["pred_norm"].values,
    test_sr_df["current_A"].values,           test_sr_df["pred_A"].values,
    "test_SR (SR=50 interpolation)"))
metrics_rows.append(compute_metrics(
    test_mat_df["current_normalized"].values, test_mat_df["pred_norm"].values,
    test_mat_df["current_A"].values,          test_mat_df["pred_A"].values,
    "test_MAT (NM4 extrapolation)"))

metrics_df = pd.DataFrame(metrics_rows).set_index("partition")

print("=" * 70)
print("LIGHTGBM MODEL — EVALUATION SUMMARY")
print("=" * 70)
print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
print()
print("Units: RMSE / MAE / MaxErr in micro-Amperes (uA).  R2 dimensionless.")
print()
print("RF baseline for comparison:")
print("  val RMSE=26.59 uA  |  test_SR RMSE=43.71 uA  |  test_MAT RMSE=33.65 uA")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7a — CV CURVES
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7a — Predicted vs Actual CV Curves

The primary physical test: does LightGBM reconstruct the closed-loop
voltammogram shape with adequate fidelity?

Because LightGBM is a **row-wise model**, it cannot explicitly model
the electrochemical trajectory continuity.  Instead, it relies on the
engineered features (`sweep_direction`, `sweep_position`, `direction_x_potential_norm`)
to implicitly encode trajectory context at each independent measurement point.""")

code("""\
def plot_cv_curve(ax, group_df, title, model_label="LightGBM", show_legend=True):
    grp = group_df.sort_values(["half_sweep_id","step_index"])
    colours    = {0: "#1f77b4", 1: "#d62728"}
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

# ── Figure 02: Validation curves — SR=30 ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nm in enumerate(["NM1","NM2","NM3"]):
    sub = val_df[(val_df["nm_id"]==nm) & (val_df["scan_rate_mVs"]==30)]
    plot_cv_curve(axes[i], sub, f"VAL: {nm} @ 30 mV/s", show_legend=(i==0))
fig.suptitle("Validation Set — Scan-Rate Interpolation (SR=30, seen materials)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_cv_curves_val.png", bbox_inches="tight")
plt.show()
print("Saved figures/02_cv_curves_val.png")""")

code("""\
# ── Figure 03: Test-SR curves — SR=50 ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nm in enumerate(["NM1","NM2","NM3"]):
    sub = test_sr_df[(test_sr_df["nm_id"]==nm) & (test_sr_df["scan_rate_mVs"]==50)]
    plot_cv_curve(axes[i], sub, f"TEST-SR: {nm} @ 50 mV/s", show_legend=(i==0))
fig.suptitle("Test-SR Set — Scan-Rate Interpolation (SR=50, seen materials)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_cv_curves_test_sr.png", bbox_inches="tight")
plt.show()
print("Saved figures/03_cv_curves_test_sr.png")""")

code("""\
# ── Figure 04: Test-MAT curves — NM4 ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, sr in enumerate([10, 30, 50, 70, 90, 100]):
    sub = test_mat_df[(test_mat_df["nm_id"]=="NM4") & (test_mat_df["scan_rate_mVs"]==sr)]
    plot_cv_curve(axes[i], sub, f"TEST-MAT: NM4 @ {sr} mV/s", show_legend=(i==0))
fig.suptitle("Test-MAT Set — Material Extrapolation (NM4, fully unseen formulation)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_cv_curves_test_mat.png", bbox_inches="tight")
plt.show()
print("Saved figures/04_cv_curves_test_mat.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7b — PARITY PLOTS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 7b — Parity Plots (Predicted vs Actual)")

code("""\
# ── Figure 05: Parity plots ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_configs = [
    (val_df,      "Val (SR=30)"),
    (test_sr_df,  "Test-SR (SR=50)"),
    (test_mat_df, "Test-MAT (NM4)"),
]
for ax, (df, title) in zip(axes, plot_configs):
    sc = ax.scatter(df["current_A"]*1e6, df["pred_A"]*1e6,
                    c=df["scan_rate_mVs"], cmap="plasma",
                    s=1.5, alpha=0.3, rasterized=True)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k-", lw=1.2, label="y=x (perfect)")
    ax.set_xlabel("Actual current (uA)", fontsize=9)
    ax.set_ylabel("Predicted current (uA)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label="Scan rate (mV/s)", pad=0.01)
fig.suptitle("Parity Plots — LightGBM Model", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_parity_plots.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved figures/05_parity_plots.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7c — RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7c — Residual Analysis

Row-wise tree models (RF, XGBoost, LightGBM) may show **structured residuals**
near CV turning points (−0.65 V and 0.0 V) because these are
electrochemically complex regions where current reverses — a trajectory-level
phenomenon that independent row predictions cannot fully capture.

Compare Figure 07 across all notebooks: if LightGBM residuals at turning points
are similar to RF/XGB but larger than LSTM/GRU, it confirms that sequential
modelling adds genuine value at electrochemical transition regions.""")

code("""\
# ── Figure 06: Residual distributions ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
dist_configs = [
    (val_df,      "Val (SR=30)",     "#ff7f0e"),
    (test_sr_df,  "Test-SR (SR=50)", "#1f77b4"),
    (test_mat_df, "Test-MAT (NM4)",  "#d62728"),
]
for ax, (df, title, colour) in zip(axes, dist_configs):
    res = (df["pred_A"] - df["current_A"]) * 1e6
    ax.hist(res, bins=80, color=colour, alpha=0.75, edgecolor="none", density=True)
    ax.axvline(0,          color="k",       lw=1.5, ls="-",  label="zero bias")
    ax.axvline(res.mean(), color="darkred", lw=1.5, ls="--", label=f"mean={res.mean():.2f} uA")
    ax.axvline(res.median(),color="navy",   lw=1.2, ls=":",  label=f"med={res.median():.2f} uA")
    ax.set_xlabel("Residual (uA): predicted - actual", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
fig.suptitle("Residual Distributions — LightGBM Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_residual_distributions.png", bbox_inches="tight")
plt.show()
print("Saved figures/06_residual_distributions.png")""")

code("""\
# ── Figure 07: Residuals vs potential ─────────────────────────────────────────
from matplotlib.lines import Line2D
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (df, title, _) in zip(axes, dist_configs):
    res = (df["pred_A"] - df["current_A"]) * 1e6
    ax.scatter(df["potential_V"], res,
               c=df["sweep_direction"], cmap="coolwarm",
               s=1.0, alpha=0.25, rasterized=True)
    ax.axhline(0, color="k", lw=1.0, ls="--")
    ax.set_xlabel("Potential (V vs SCE)", fontsize=8)
    ax.set_ylabel("Residual (uA)", fontsize=8)
    ax.set_title(f"{title} — residual vs potential", fontsize=9)
    ax.grid(True, alpha=0.25)
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#1f77b4",
           markersize=6, label="Cathodic (dir=0)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#d62728",
           markersize=6, label="Anodic (dir=1)"),
]
axes[0].legend(handles=legend_elems, fontsize=7)
fig.suptitle("Residuals vs Potential — LightGBM Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_residual_vs_potential.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved figures/07_residual_vs_potential.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7d — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7d — Feature Importance: Gain and Permutation

### Two complementary importance measures

**Gain importance** (LightGBM native):
- Total information gain accumulated by all splits on a feature across all trees.
- Measures: how much a feature *actually reduces loss* in the training data.
- Limitation: biased toward high-cardinality or correlated features.

**Permutation importance** (model-agnostic):
- Shuffles one feature column at random and measures RMSE increase on val set.
- Measures: how much the model *relies* on the feature for generalisation.
- More reliable for correlated features — shuffling breaks both value and
  correlation structure.

### Expected result for CV data
`direction_x_potential_norm` encodes the interaction between sweep direction and
potential — the primary driver of CV hysteresis.  We expect it to rank highly
in both importance measures.  `potential_from_upper_norm` and
`potential_from_lower_norm` encode distance from turning points — also critical
for predicting the Faradaic and capacitive current near the reversal points.""")

code("""\
# ── Gain importance (LightGBM native) ────────────────────────────────────────
gain_arr = model.booster_.feature_importance(importance_type="gain")
gain_imp  = pd.Series(gain_arr, index=FEATURES).sort_values(ascending=False)

print("LightGBM Gain Importance ranking:")
for feat in gain_imp.index:
    bar = "#" * int(50 * gain_imp[feat] / gain_imp.max())
    print(f"  {feat:35s}: {gain_imp[feat]:12.1f}  {bar}")

# ── Figure 08: Gain importance ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(gain_imp.index[::-1], gain_imp.values[::-1],
        color="#e377c2", alpha=0.85)
ax.set_xlabel("Total gain across all splits and all trees", fontsize=9)
ax.set_title(
    "LightGBM Gain Feature Importance\\n"
    "Total information gain contributed by each feature", fontsize=9, pad=8)
ax.grid(True, alpha=0.3, axis="x")

# Annotate top feature
top_feat = gain_imp.index[0]
ax.text(gain_imp.iloc[0] * 0.02, len(FEATURES) - 1,
        f"Top: {top_feat}", va="center", fontsize=8, color="darkred")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_lgbm_gain_importance.png", bbox_inches="tight")
plt.show()
print("Saved figures/08_lgbm_gain_importance.png")""")

code("""\
# ── Permutation importance (model-agnostic, val set) ─────────────────────────
def lgbm_permutation_importance(model, X, y, feature_names,
                                 n_repeats=10, random_state=42):
    '''
    Permutation importance for LightGBM (row-wise).

    Shuffles feature j across ALL rows of the val set, measures RMSE increase.
    No padding concerns — all rows are real (no sequence padding).

    Parameters
    ----------
    model        : fitted LGBMRegressor
    X            : np.float32  (n_rows, n_features)  — val feature matrix
    y            : np.float32  (n_rows,)              — val target (normalised)
    feature_names: list of feature name strings
    n_repeats    : number of shuffles per feature
    random_state : seed for reproducibility
    '''
    rng           = np.random.default_rng(random_state)
    baseline_pred = model.predict(X)
    baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))

    imp_matrix = np.zeros((len(feature_names), n_repeats))
    for j in range(len(feature_names)):
        for r in range(n_repeats):
            X_perm    = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_rmse = np.sqrt(mean_squared_error(y, model.predict(X_perm)))
            imp_matrix[j, r] = perm_rmse - baseline_rmse

    means = pd.Series(imp_matrix.mean(axis=1), index=feature_names)
    stds  = pd.Series(imp_matrix.std(axis=1),  index=feature_names)
    return means.sort_values(ascending=False), stds


print(f"Computing permutation importance (n_repeats=10, {len(FEATURES)} features) ...")
print(f"  {10 * len(FEATURES)} model.predict() calls on X_val ({len(X_val):,} rows)")
t0 = time.time()
perm_means, perm_stds = lgbm_permutation_importance(
    model, X_val, y_val, FEATURES, n_repeats=10, random_state=42)
print(f"  Done in {time.time()-t0:.1f} s")
print()
print("LightGBM Permutation Importance ranking (val set):")
for feat in perm_means.index:
    print(f"  {feat:35s}: {perm_means[feat]:.6f} +/- {perm_stds[feat]:.6f}")""")

code("""\
# ── Figure 09: Permutation importance ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
xerr = perm_stds.loc[perm_means.index][::-1].values
ax.barh(perm_means.index[::-1], perm_means.values[::-1],
        xerr=xerr, color="#7f7f7f", alpha=0.85,
        error_kw={"elinewidth": 1.2, "capsize": 3})
ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("RMSE increase when feature is randomly shuffled (normalised units)", fontsize=9)
ax.set_title(
    "LightGBM Permutation Feature Importance (val set, n_repeats=10)\\n"
    "Positive = feature matters for generalisation to SR=30", fontsize=9, pad=8)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "09_lgbm_permutation_importance.png", bbox_inches="tight")
plt.show()
print("Saved figures/09_lgbm_permutation_importance.png")
print()
print("Scientific note:")
print("  Compare gain vs permutation importance rankings.")
print("  If a feature ranks high in gain but low in permutation:")
print("    -> model uses it heavily during training but it may be a correlated proxy.")
print("  If direction_x_potential_norm ranks top in permutation:")
print("    -> CV hysteresis encoding is genuinely important for generalisation.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7e — PER-GROUP RMSE
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 7e — Per-Group Performance Breakdown")

code("""\
def group_rmse(df, label):
    rows = []
    for (nm, sr), grp in df.groupby(["nm_id","scan_rate_mVs"]):
        rmse_uA = np.sqrt(mean_squared_error(grp["current_A"], grp["pred_A"])) * 1e6
        r2      = r2_score(grp["current_normalized"], grp["pred_norm"])
        rows.append({"nm_id": nm, "scan_rate_mVs": sr,
                     "RMSE_uA": rmse_uA, "R2": r2, "partition": label})
    return pd.DataFrame(rows)

val_grp      = group_rmse(val_df,      "val")
test_sr_grp  = group_rmse(test_sr_df,  "test_sr")
test_mat_grp = group_rmse(test_mat_df, "test_mat")

# ── Figure 10: Per-group RMSE bars ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.bar(val_grp["nm_id"], val_grp["RMSE_uA"], color="#ff7f0e", alpha=0.85)
for i, row in val_grp.iterrows():
    ax.text(i, row["RMSE_uA"]+0.05, f"R2={row['R2']:.3f}", ha="center", fontsize=8)
ax.set_xlabel("Material"); ax.set_ylabel("RMSE (uA)")
ax.set_title("Val (SR=30) — RMSE per material"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.bar(test_sr_grp["nm_id"], test_sr_grp["RMSE_uA"], color="#1f77b4", alpha=0.85)
for i, row in test_sr_grp.iterrows():
    ax.text(i, row["RMSE_uA"]+0.05, f"R2={row['R2']:.3f}", ha="center", fontsize=8)
ax.set_xlabel("Material"); ax.set_ylabel("RMSE (uA)")
ax.set_title("Test-SR (SR=50) — RMSE per material"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[2]
ax.bar(test_mat_grp["scan_rate_mVs"].astype(int).astype(str),
       test_mat_grp["RMSE_uA"], color="#d62728", alpha=0.85)
ax.set_xlabel("Scan rate (mV/s)"); ax.set_ylabel("RMSE (uA)")
ax.set_title("Test-MAT (NM4) — RMSE per scan rate"); ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Per-Group RMSE Breakdown — LightGBM Model", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "10_per_group_rmse.png", bbox_inches="tight")
plt.show()
print("Saved figures/10_per_group_rmse.png")""")

code("""\
# ── Figure 11: NM4 scan-rate trend ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(test_mat_grp["scan_rate_mVs"], test_mat_grp["RMSE_uA"],
        "o-", color="#d62728", lw=1.8, ms=7)
ax.set_xlabel("Scan rate (mV/s)"); ax.set_ylabel("RMSE (uA)")
ax.set_title("NM4 — RMSE vs scan rate\\n(does error scale with scan rate?)")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(test_mat_grp["scan_rate_mVs"], test_mat_grp["R2"],
        "s-", color="#e377c2", lw=1.8, ms=7)
ax.axhline(0, color="k", lw=0.8, ls="--", label="R2=0 (mean prediction)")
ax.set_xlabel("Scan rate (mV/s)"); ax.set_ylabel("R2")
ax.set_title("NM4 — R2 vs scan rate\\n(does shape fidelity degrade?)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("NM4 Scan-Rate Trend — LightGBM Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "11_nm4_scan_rate_trend.png", bbox_inches="tight")
plt.show()
print("Saved figures/11_nm4_scan_rate_trend.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 8 — Save All Outputs")

code("""\
# ── Trained LightGBM model (joblib) ──────────────────────────────────────────
model_path = MODEL_DIR / "lightgbm_model.joblib"
joblib.dump(model, model_path)
print(f"Saved model  : {model_path}")

# ── Booster also saved in LightGBM native format ─────────────────────────────
booster_path = MODEL_DIR / "lightgbm_booster.txt"
model.booster_.save_model(str(booster_path))
print(f"Saved booster: {booster_path}  (LightGBM text format, reloadable)")

# ── Comprehensive metrics JSON ────────────────────────────────────────────────
lgbm_cfg = {
    "params":       LGBM_PARAMS,
    "n_trees_built": int(len(train_rmse_hist)),
    "best_iteration": int(best_iter),
    "best_val_rmse_norm": float(val_rmse_hist[best_iter - 1]),
    "train_rmse_at_best": float(train_rmse_hist[best_iter - 1]),
}
metrics_out = {
    "model":        "LightGBM LGBMRegressor",
    "model_config": lgbm_cfg,
    "metrics":      metrics_df.reset_index().to_dict(orient="records"),
    "per_group":    {"val": val_grp.to_dict(orient="records"),
                     "test_sr": test_sr_grp.to_dict(orient="records"),
                     "test_mat": test_mat_grp.to_dict(orient="records")},
    "gain_importance": {f: float(v) for f, v in gain_imp.items()},
    "permutation_importance": {
        f: {"mean": float(perm_means.get(f, 0.0)),
            "std":  float(perm_stds.get(f,  0.0))}
        for f in FEATURES},
}
metrics_path = METRICS_DIR / "lightgbm_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(metrics_out, fh, indent=2)
print(f"Saved metrics: {metrics_path}")

for name, df_grp in [("val",val_grp),("test_sr",test_sr_grp),("test_mat",test_mat_grp)]:
    path = CSV_DIR / f"per_group_rmse_{name}.csv"
    df_grp.to_csv(path, index=False)
    print(f"Saved {path.name}")

print()
print("All outputs saved")
print(f"  model/   : lightgbm_model.joblib + lightgbm_booster.txt")
print(f"  metrics/ : lightgbm_metrics.json")
print(f"  csv/     : 3 per-group RMSE tables")
print(f"  figures/ : 12 figures (00 through 11)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SIX-MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 9 — Scientific Summary: RF vs XGBoost vs ANN vs LSTM vs GRU vs LightGBM

### Comparison methodology
All six models share:
- **Identical training data** (NM1/2/3 × SR∈{10,20,40,60,70,80,90,100}, 62,400 rows)
- **Identical feature set** (10 normalised features)
- **Identical evaluation partitions** (val SR=30, test-SR SR=50, test-MAT NM4)
- **Identical inverse transform** (per-group scaler from Notebook 01)

### Scientific discussion

**1. Why tree ensembles dominate engineered-feature datasets**
The 10 features in this pipeline are already highly informative:
`direction_x_potential_norm` encodes hysteresis, `sweep_position` encodes
trajectory progress, `potential_from_upper/lower_norm` encode turning-point proximity.
Tree ensembles partition this feature space with axis-aligned cuts — perfectly
suited to the discrete, combinatorial structure of (scan rate × direction × position).

**2. LightGBM vs RF — leaf-wise advantage**
RF averages 500+ fully-grown trees independently.  LightGBM builds trees
sequentially: each tree corrects the residuals of all previous trees.
Leaf-wise growth finds the single highest-gain leaf at each step —
more efficient use of model capacity for the same number of leaves.

**3. LightGBM vs XGBoost — histogram binning**
XGBoost uses exact split finding (or approximate).  LightGBM bins continuous
features into 255 histogram buckets before splitting — O(#bins) instead of
O(#rows).  Identical regularisation but faster convergence.

**4. Whether boosting improved peak current reconstruction**
Compare Figure 05 (parity plots) across RF, XGBoost, and LightGBM.  If
LightGBM parity points cluster more tightly around y=x at extreme currents
(near CV peaks), boosting is recovering fine-grained peak structure that
averaging (RF) misses.

**5. Sequence learning vs gradient boosting**
If LightGBM outperforms LSTM/GRU, it means the 10 engineered features already
encode the electrochemical trajectory context that the sequence models learn
dynamically.  This has a key publication implication: **feature engineering
may substitute for architecture complexity** in electrochemical CV prediction.

**6. Interpolation vs extrapolation**
- val (SR=30) + test_SR (SR=50): interpolation — scan rate seen nearby in training
- test_MAT (NM4): extrapolation — entirely new material formulation
All tree models share the same weakness on test_MAT: they cannot extrapolate
beyond the training materials' feature distribution.  Neural networks with
regularisation may handle this better if NM4 physics differs significantly.""")

code("""\
# ── RF baseline (Notebook 02, verified run) ───────────────────────────────────
RF_RESULTS = {
    "val  (SR=30 interpolation)"   : {"RMSE_uA": 26.59, "R2": 0.9851},
    "test_SR (SR=50 interpolation)": {"RMSE_uA": 43.71, "R2": 0.9777},
    "test_MAT (NM4 extrapolation)" : {"RMSE_uA": 33.65, "R2": 0.9707},
}

# ── Load XGBoost, ANN, LSTM, GRU results if available ────────────────────────
def load_model_results(results_dir, json_name):
    path = results_dir / json_name
    if not path.exists():
        return None, str(path)
    with open(path) as fh:
        data = json.load(fh)
    return {m["partition"]: {"RMSE_uA": m["RMSE_uA"], "R2": m["R2"]}
            for m in data["metrics"]}, None

XGB_RESULTS,  xgb_msg  = load_model_results(
    BASE_DIR / "results" / "xgboost" / "metrics", "xgboost_metrics.json")
ANN_RESULTS,  ann_msg  = load_model_results(
    BASE_DIR / "results" / "ann"     / "metrics", "ann_metrics.json")
LSTM_RESULTS, lstm_msg = load_model_results(
    BASE_DIR / "results" / "lstm"    / "metrics", "lstm_metrics.json")
GRU_RESULTS,  gru_msg  = load_model_results(
    BASE_DIR / "results" / "gru"     / "metrics", "gru_metrics.json")

for label, res, msg, nb_num in [
        ("XGBoost", XGB_RESULTS,  xgb_msg,  "03"),
        ("ANN",     ANN_RESULTS,  ann_msg,  "04"),
        ("LSTM",    LSTM_RESULTS, lstm_msg, "05"),
        ("GRU",     GRU_RESULTS,  gru_msg,  "06")]:
    if res:
        print(f"  {label} metrics loaded")
    else:
        print(f"  {label} metrics not found ({msg})")
        print(f"    Run Notebook {nb_num} first to populate this column.")

# ── Build six-model comparison table ──────────────────────────────────────────
PARTITIONS = [
    "val  (SR=30 interpolation)",
    "test_SR (SR=50 interpolation)",
    "test_MAT (NM4 extrapolation)",
]
PAD = 32

print()
print("=" * 135)
print("RF vs XGBoost vs ANN vs LSTM vs GRU vs LightGBM — RMSE (uA) COMPARISON")
print("=" * 135)
header = (f"  {'Partition':<{PAD}}  {'RF':>8}  {'XGBoost':>8}  "
          f"{'ANN':>8}  {'LSTM':>8}  {'GRU':>8}  {'LightGBM':>10}  {'Best':>9}")
print(header)
print("  " + "-" * 127)

for p in PARTITIONS:
    rf_r   = RF_RESULTS[p]["RMSE_uA"]
    xgb_r  = XGB_RESULTS[p]["RMSE_uA"]  if XGB_RESULTS  and p in XGB_RESULTS  else None
    ann_r  = ANN_RESULTS[p]["RMSE_uA"]  if ANN_RESULTS  and p in ANN_RESULTS  else None
    lstm_r = LSTM_RESULTS[p]["RMSE_uA"] if LSTM_RESULTS and p in LSTM_RESULTS else None
    gru_r  = GRU_RESULTS[p]["RMSE_uA"]  if GRU_RESULTS  and p in GRU_RESULTS  else None
    lgbm_r = metrics_df.loc[p, "RMSE_uA"]

    vals = {k: v for k, v in [
        ("RF", rf_r), ("XGBoost", xgb_r), ("ANN", ann_r),
        ("LSTM", lstm_r), ("GRU", gru_r), ("LightGBM", lgbm_r)]
        if v is not None}
    best_name = min(vals, key=vals.get)

    def fmt(v): return f"{v:8.2f}" if v is not None else "     ---"
    row = (f"  {p:<{PAD}}  {fmt(rf_r)}  {fmt(xgb_r)}  "
           f"{fmt(ann_r)}  {fmt(lstm_r)}  {fmt(gru_r)}  {fmt(lgbm_r):>10}  {best_name:>9}")
    print(row)

print()
print("  Units: RMSE in micro-Amperes (uA).  Lower is better.")
print()

# ── R2 table ──────────────────────────────────────────────────────────────────
print("RF vs XGBoost vs ANN vs LSTM vs GRU vs LightGBM — R2 COMPARISON")
print("=" * 135)
header2 = (f"  {'Partition':<{PAD}}  {'RF':>8}  {'XGBoost':>8}  "
           f"{'ANN':>8}  {'LSTM':>8}  {'GRU':>8}  {'LightGBM':>10}")
print(header2)
print("  " + "-" * 105)
for p in PARTITIONS:
    rf_r2   = RF_RESULTS[p]["R2"]
    xgb_r2  = XGB_RESULTS[p]["R2"]  if XGB_RESULTS  and p in XGB_RESULTS  else None
    ann_r2  = ANN_RESULTS[p]["R2"]  if ANN_RESULTS  and p in ANN_RESULTS  else None
    lstm_r2 = LSTM_RESULTS[p]["R2"] if LSTM_RESULTS and p in LSTM_RESULTS else None
    gru_r2  = GRU_RESULTS[p]["R2"]  if GRU_RESULTS  and p in GRU_RESULTS  else None
    lgbm_r2 = metrics_df.loc[p, "R2"]
    def fmtr(v): return f"{v:8.4f}" if v is not None else "     ---"
    print(f"  {p:<{PAD}}  {fmtr(rf_r2)}  {fmtr(xgb_r2)}  {fmtr(ann_r2)}  "
          f"{fmtr(lstm_r2)}  {fmtr(gru_r2)}  {fmtr(lgbm_r2):>10}")

print()
print("  R2: higher is better.  R2=1.0 = perfect.  R2=0.0 = predicts the mean.")
print()

# ── LightGBM-specific verdicts ────────────────────────────────────────────────
lgbm_vs_rf = sum(metrics_df.loc[p, "RMSE_uA"] < RF_RESULTS[p]["RMSE_uA"]
                 for p in PARTITIONS)
print(f"  LightGBM improved on {lgbm_vs_rf}/3 partitions vs RF baseline.")
if lgbm_vs_rf == 3:
    print("  => Leaf-wise boosting outperforms depth-wise averaging (RF) on all")
    print("     three evaluation partitions — gradient boosting wins this dataset.")
elif lgbm_vs_rf >= 1:
    print("  => Partial improvement: LightGBM wins on some but not all partitions.")
else:
    print("  => RF baseline is competitive — averaging 500 trees matches or exceeds")
    print("     leaf-wise boosting on these engineered electrochemical features.")

if XGB_RESULTS:
    lgbm_vs_xgb = sum(
        metrics_df.loc[p, "RMSE_uA"] < XGB_RESULTS[p]["RMSE_uA"]
        for p in PARTITIONS if p in XGB_RESULTS)
    print(f"  LightGBM improved on {lgbm_vs_xgb}/3 partitions vs XGBoost.")
    if lgbm_vs_xgb >= 2:
        print("  => Histogram-based leaf-wise splits outperform XGBoost's depth-wise.")
    else:
        print("  => XGBoost and LightGBM perform comparably — both boosting variants")
        print("     achieve similar accuracy on this feature-engineered CV dataset.")

print()
print("Computational efficiency summary:")
print("  LightGBM training time: see above  (typically fastest of the three tree models)")
print("  Inference time        : microseconds per row (same for all tree models)")
print("  Memory usage          : lowest of all models (histogram binning)")
print()
print("Pipeline complete.")
print("  Notebooks 02-07 cover: RF, XGBoost, ANN, LSTM, GRU, LightGBM")
print("  All use identical preprocessing, splits, features, and evaluation.")
print("  Model spectrum: RF < XGBoost ≈ LightGBM < ANN < GRU ≈ LSTM")
print("  (ordered by architectural complexity, not necessarily accuracy)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 10 — Download Results to Your Local Machine

Run this cell **before closing the Colab session**.
All files are lost when the runtime disconnects.

The cell zips the entire `results/lightgbm/` folder and triggers a browser
download on Colab.  On local execution it simply lists the saved files.""")

code("""\
import shutil, os

if EXECUTION_MODE in ("colab_upload", "colab_drive"):
    from google.colab import files as _cf

    zip_base = str(BASE_DIR / "lightgbm_results")
    print(f"Creating zip of {RESULTS_DIR} ...")
    shutil.make_archive(zip_base, "zip", str(RESULTS_DIR))

    zip_path = zip_base + ".zip"
    size_mb  = os.path.getsize(zip_path) / 1e6
    print(f"Created : {zip_path}  ({size_mb:.1f} MB)")
    print()
    print("Contents:")
    for root, dirs, files in os.walk(str(RESULTS_DIR)):
        rel = os.path.relpath(root, str(RESULTS_DIR))
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            fsize = os.path.getsize(fpath) / 1e3
            print(f"  {os.path.join(rel, fname):<55}  {fsize:>7.0f} KB")
    print()
    print("Initiating download to your local machine ...")
    _cf.download(zip_path)
    print("Done — check your browser Downloads folder.")

else:
    print(f"Local execution — results saved to: {RESULTS_DIR}")
    print()
    for root, dirs, files in os.walk(str(RESULTS_DIR)):
        rel = os.path.relpath(root, str(RESULTS_DIR))
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            fsize = os.path.getsize(fpath) / 1e3
            print(f"  {os.path.join(rel, fname):<55}  {fsize:>7.0f} KB")""")

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
    "language_info": {"name": "python", "version": "3.10.0"},
    "colab": {"provenance": []},
})

out_path = Path(r"D:/mca/2nd semester/PBL Project/notebooks/07_lightgbm_model.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out_path))
print(f"Written : {out_path}")
print(f"Cells   : {len(nb.cells)}")
