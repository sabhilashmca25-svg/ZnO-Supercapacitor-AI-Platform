"""
build_ann_notebook.py
Generates notebooks/04_ann_model.ipynb
Run once: python src/build_ann_notebook.py
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
# Notebook 04 — ANN Model (Dense Feedforward Neural Network)
## ZnO Supercapacitor CV Current Prediction · TensorFlow / Keras

### Purpose
Train a **Dense Feedforward Artificial Neural Network** on the same scientifically
correct partitions established in Notebook 01, and compare against the RF and
XGBoost baselines.  This notebook:
- Loads pre-built artefacts from Notebook 01 — **no preprocessing is repeated**.
- Builds a 4-layer dense ANN with He initialisation and Dropout regularisation.
- Trains with EarlyStopping and ReduceLROnPlateau callbacks.
- Evaluates on the same three held-out partitions as Notebooks 02 and 03.
- Produces 11 publication-ready verification figures.
- Delivers a head-to-head RF vs XGBoost vs ANN comparison table.

### Three generalisation questions answered
| Partition | Question |
|-----------|----------|
| **Val** (NM1/2/3 × SR=30) | Can the ANN interpolate to an unseen intermediate scan rate? |
| **Test-SR** (NM1/2/3 × SR=50) | Can the ANN interpolate to SR=50 from {10,20,...,100}? |
| **Test-MAT** (NM4 × all SRs) | Can the ANN generalise to a completely unseen ZnO formulation? |

### ANN architecture
```
Input(10)               ← 10 normalised electrochemical features
Dense(128, ReLU)        ← He initialisation
Dropout(0.15)
Dense(64,  ReLU)        ← He initialisation
Dropout(0.10)
Dense(32,  ReLU)        ← He initialisation
Dense(1,   Linear)      ← regression output (normalised current)
```

### Why ANN after tree-based models?
Tree models (RF, XGBoost) learn **axis-aligned** decision boundaries and cannot
model truly smooth response surfaces.  A dense ANN with ReLU activations can
approximate any smooth continuous function (universal approximation theorem).
- ANN learns **hierarchical feature compositions**: potential × scan-rate
  interactions emerge naturally from stacked layers.
- Dropout regularisation prevents overfitting to training scan rates.
- Gradient-based optimisation (Adam) is efficient for smooth loss landscapes.
- ANN is architecturally the predecessor to LSTM — it establishes the
  non-sequential performance ceiling before Notebook 05 adds temporal context.""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — ENVIRONMENT SETUP + FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 0 — Environment Setup and File Upload

### Step 1: Set your execution mode
Edit `EXECUTION_MODE` at the top of the cell below before running anything.

| Mode | When to use |
|------|-------------|
| `"colab_upload"` | Google Colab — upload files from your PC via dialog |
| `"colab_drive"` | Google Colab — files already on your Google Drive |
| `"local"` | Local machine (Jupyter / VS Code / terminal) |

### Step 2: Run the cell
All six required files will be placed into `PBL_Project/data/processed/`
and all results directories will be created automatically.

**Required files** (from `data/processed/` in your PBL Project folder):
```
master_long_format.parquet    train_set.parquet
val_set.parquet               test_scanrate.parquet
test_material.parquet         scalers.json
```

### Runtime recommendation (Google Colab)
Go to **Runtime → Change runtime type → T4 GPU** for faster ANN training.
CPU will also work but training will take longer.""")

code("""\
# ─────────────────────────────────────────────────────────────────────────────
# SET THIS BEFORE RUNNING ANY OTHER CELL
# ─────────────────────────────────────────────────────────────────────────────
#
#   "colab_upload"  — Google Colab: upload files manually via dialog
#   "colab_drive"   — Google Colab: files already on Google Drive
#   "local"         — Local machine: files already on disk
#
EXECUTION_MODE = "colab_upload"       # <<< CHANGE THIS IF NEEDED

# For "colab_drive" only — path to your data/processed folder on Google Drive
DRIVE_DATA_PATH = "/content/drive/MyDrive/PBL_Project/data/processed"

# For "local" only — absolute path to the root of your PBL Project folder
LOCAL_PROJECT_PATH = str(Path(__file__).resolve().parent.parent.parent)

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED FILES
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# MODE-SPECIFIC SETUP
# ─────────────────────────────────────────────────────────────────────────────

if EXECUTION_MODE == "colab_upload":
    # ── Google Colab: upload from local PC ───────────────────────────────────
    from google.colab import files as _cf
    BASE_DIR      = Path("/content/PBL_Project")
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("A file-picker dialog will open.")
    print("Select all 6 files at once (Ctrl+click each):")
    for f in REQUIRED:
        print(f"  {f}")
    print()
    _cf.upload()   # files land in /content/ after upload

    # Move each uploaded file from /content/ into the project data directory
    moved = 0
    for fname in REQUIRED:
        src = Path("/content") / fname
        dst = PROCESSED_DIR / fname
        if src.exists():
            shutil.move(str(src), str(dst))
            moved += 1
    print(f"  Moved {moved} file(s) into {PROCESSED_DIR}")

elif EXECUTION_MODE == "colab_drive":
    # ── Google Colab: copy from Google Drive ─────────────────────────────────
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
    # ── Local machine: files already on disk ─────────────────────────────────
    BASE_DIR      = Path(LOCAL_PROJECT_PATH)
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    print(f"Local mode — reading from: {PROCESSED_DIR}")

else:
    raise ValueError(
        f"Unknown EXECUTION_MODE: {EXECUTION_MODE!r}\\n"
        "Valid options: 'colab_upload', 'colab_drive', 'local'"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CREATE ALL RESULTS DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = BASE_DIR / "results" / "ann"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
CSV_DIR     = RESULTS_DIR / "csv"
MODEL_DIR   = RESULTS_DIR / "model"

for d in [FIGURES_DIR, METRICS_DIR, CSV_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# VERIFY ALL 6 FILES ARE PRESENT
# ─────────────────────────────────────────────────────────────────────────────
missing = [f for f in REQUIRED if not (PROCESSED_DIR / f).exists()]
assert not missing, (
    f"Missing files in {PROCESSED_DIR}: {missing}\\n"
    "Re-run this cell and upload / provide all 6 files."
)

print()
print("✓ All 6 files verified")
print(f"  Data     : {PROCESSED_DIR}")
print(f"  Results  : {RESULTS_DIR}")
print(f"    figures/ : {FIGURES_DIR}")
print(f"    metrics/ : {METRICS_DIR}")
print(f"    csv/     : {CSV_DIR}")
print(f"    model/   : {MODEL_DIR}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMPORTS + TENSORFLOW SETUP + LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 1 — Imports, TensorFlow Setup, and Load Artefacts

Loads all preprocessing outputs from Notebook 01.
**Nothing is recomputed here** — splits, normalisation, and feature engineering
are fixed and immutable.""")

code("""\
# ── Suppress TensorFlow info/warning messages (keep only errors) ──────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Standard scientific libraries ────────────────────────────────────────────
import numpy  as np
import pandas as pd
import json
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.6e}".format)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow version : {tf.__version__}")
print(f"Keras version      : {tf.keras.__version__}")
print()

# ── Seed everything for reproducibility ──────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Confirm GPU availability ──────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU available: {gpus[0].name}")
    print("  Training will use GPU acceleration.")
else:
    print("No GPU detected — training on CPU.")
    print("  For faster training on Colab: Runtime > Change runtime type > T4 GPU")

# ── Load all preprocessing artefacts ─────────────────────────────────────────
master_df   = pd.read_parquet(PROCESSED_DIR / "master_long_format.parquet")
train_df    = pd.read_parquet(PROCESSED_DIR / "train_set.parquet")
val_df      = pd.read_parquet(PROCESSED_DIR / "val_set.parquet")
test_sr_df  = pd.read_parquet(PROCESSED_DIR / "test_scanrate.parquet")
test_mat_df = pd.read_parquet(PROCESSED_DIR / "test_material.parquet")

with open(PROCESSED_DIR / "scalers.json") as fh:
    scalers = json.load(fh)

print()
print("✓ All artefacts loaded")
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

Identical verification to Notebooks 02 and 03.
Guards against loading stale or mismatched artefacts before any modelling.""")

code("""\
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=== SPLIT INTEGRITY CHECK ===\\n")

# Check 1 — Row counts
expected = {"train": 62400, "val": 7800, "test_scanrate": 7800, "test_material": 26000}
for name, exp_rows in expected.items():
    df = {"train": train_df, "val": val_df,
          "test_scanrate": test_sr_df, "test_material": test_mat_df}[name]
    assert len(df) == exp_rows, f"{name}: expected {exp_rows}, got {len(df)}"
    print(f"  [PASS] {name:15s}: {len(df):,} rows")

# Check 2 — Material assignments
assert set(train_df["nm_id"].unique())    == {"NM1", "NM2", "NM3"}
assert set(val_df["nm_id"].unique())      == {"NM1", "NM2", "NM3"}
assert set(test_sr_df["nm_id"].unique())  == {"NM1", "NM2", "NM3"}
assert set(test_mat_df["nm_id"].unique()) == {"NM4"}
print("  [PASS] Material assignments correct")

# Check 3 — Scan-rate assignments
assert set(train_df["scan_rate_mVs"].unique())   == {10, 20, 40, 60, 70, 80, 90, 100}
assert set(val_df["scan_rate_mVs"].unique())     == {30}
assert set(test_sr_df["scan_rate_mVs"].unique()) == {50}
print("  [PASS] Scan-rate assignments correct")

# Check 4 — Feature definitions (same as RF/XGB)
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

# Check 5 — No NaN values
for name, df in [("train", train_df), ("val", val_df),
                  ("test_sr", test_sr_df), ("test_mat", test_mat_df)]:
    n_nan = df[FEATURES + [TARGET]].isna().sum().sum()
    assert n_nan == 0, f"NaN in {name}: {n_nan}"
print("  [PASS] Zero NaN in features and target")

# Check 6 — Target in [0, 1]
for name, df in [("train", train_df), ("val", val_df),
                  ("test_sr", test_sr_df), ("test_mat", test_mat_df)]:
    assert df[TARGET].between(-1e-9, 1 + 1e-9).all(), f"Target out of [0,1] in {name}"
print("  [PASS] current_normalized in [0, 1]")

# Check 7 — No half-sweep crosses partition boundary
block_splits = (master_df.groupby(["nm_id", "scan_rate_mVs", "half_sweep_id"])["split"]
                          .nunique())
assert (block_splits == 1).all()
print("  [PASS] No half-sweep crosses a split boundary")

# Check 8 — No feature leakage
leakage_cols = ["current_A", "current_norm_min", "current_norm_max"]
for col in leakage_cols:
    assert col not in FEATURES, f"LEAKAGE: {col} is in FEATURES!"
print("  [PASS] No target-derived leakage columns in FEATURES")

print("\\n✓ ALL INTEGRITY CHECKS PASSED — safe to proceed with ANN modelling")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE MATRIX CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 3 — Feature Matrix Construction

### Feature set — identical to RF and XGBoost baselines
All 10 features are normalised to [0, 1] (except `sweep_direction` which is {0, 1}).

| Feature | Physical meaning |
|---------|-----------------|
| `potential_V_norm` | Primary independent variable in CV |
| `potential_from_lower_norm` | Distance from cathodic limit (−0.65 V) |
| `potential_from_upper_norm` | Distance from anodic limit (0.0 V) |
| `scan_rate_mVs_norm` | Linear scan-rate scaling |
| `log_scan_rate_norm` | Log-linear electrochemical relationships |
| `sqrt_scan_rate_norm` | Randles-Sevcik: diffusion current ~ √v |
| `direction_x_potential_norm` | **Key** — same V, different current per direction |
| `sr_x_potential_norm` | Scan-rate modulation of voltage response |
| `sweep_direction` | **Mandatory** — resolves many-to-one V→I mapping |
| `sweep_position` | Normalised progress 0→1 within each half-sweep |

### Strictly excluded
`current_A`, `current_norm_min`, `current_norm_max` (leakage targets),
`nm_id`, `half_sweep_id`, `cycle_id`, `step_index`, `split` (structural labels)""")

code("""\
# ── Build numpy arrays for TensorFlow (float32 is preferred by Keras) ─────────
# float32 is used for model inputs; float64 for metric computation
X_train    = train_df[FEATURES].to_numpy(dtype=np.float32)
y_train    = train_df[TARGET].to_numpy(dtype=np.float32)

X_val      = val_df[FEATURES].to_numpy(dtype=np.float32)
y_val      = val_df[TARGET].to_numpy(dtype=np.float32)

X_test_sr  = test_sr_df[FEATURES].to_numpy(dtype=np.float32)
y_test_sr  = test_sr_df[TARGET].to_numpy(dtype=np.float32)

X_test_mat = test_mat_df[FEATURES].to_numpy(dtype=np.float32)
y_test_mat = test_mat_df[TARGET].to_numpy(dtype=np.float32)

# float64 copies for sklearn metric functions (require double precision)
X_val_64      = X_val.astype(np.float64)
y_val_64      = y_val.astype(np.float64)
X_test_sr_64  = X_test_sr.astype(np.float64)
y_test_sr_64  = y_test_sr.astype(np.float64)
X_test_mat_64 = X_test_mat.astype(np.float64)
y_test_mat_64 = y_test_mat.astype(np.float64)

print("✓ Feature matrices constructed")
print(f"  X_train    : {X_train.shape}  dtype={X_train.dtype}")
print(f"  X_val      : {X_val.shape}  dtype={X_val.dtype}")
print(f"  X_test_sr  : {X_test_sr.shape}  dtype={X_test_sr.dtype}")
print(f"  X_test_mat : {X_test_mat.shape}  dtype={X_test_mat.dtype}")
print(f"\\n  {len(FEATURES)} features  |  target: {TARGET}")""")

code("""\
# ── Figure 00: Feature correlation matrix (training set) ─────────────────────
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
print("Note: ANN handles multicollinearity differently from trees.")
print("Correlated inputs can still be useful because the network learns to")
print("weight and combine them in ways that reduce prediction error.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ANN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 4 — ANN Architecture

### Layer-by-layer design rationale
| Layer | Units | Activation | Purpose |
|-------|-------|-----------|---------|
| Dense 1 | 128 | ReLU | Wide first layer captures diverse feature combinations |
| Dropout 1 | — | — | 15% dropout; prevents neurons from co-adapting |
| Dense 2 | 64 | ReLU | Compresses representation — learns scan-rate × potential interactions |
| Dropout 2 | — | — | 10% dropout; smaller because gradients are more stable here |
| Dense 3 | 32 | ReLU | Final nonlinear compression before regression head |
| Dense 4 | 1 | Linear | Regression output — unbounded prediction in [0,1] |

### He initialisation
ReLU kills half its inputs (negative → 0), so weights must start larger than
for sigmoid/tanh.  He initialisation: `W ~ N(0, 2/fan_in)` prevents
vanishing gradients from the start.

### Compilation
- **Loss**: MSE — penalises large errors quadratically; gradients scale with error.
- **Metric**: MAE — easier to interpret than MSE during training monitoring.
- **Optimiser**: Adam (lr=1e-3) — adaptive per-parameter learning rates.

### Callbacks
| Callback | Setting | Purpose |
|----------|---------|---------|
| EarlyStopping | patience=25, restore_best_weights=True | Stops when val_loss stagnates; uses best model |
| ReduceLROnPlateau | factor=0.5, patience=10 | Halves lr when val_loss plateaus; helps convergence |""")

code("""\
# ── Build the ANN model ───────────────────────────────────────────────────────
model = Sequential([
    # Input: 10 normalised electrochemical features
    Input(shape=(10,), name="input"),

    # Layer 1: 128 neurons, ReLU activation, He initialisation
    # Wide first layer: gives the network capacity to learn many feature combinations
    Dense(128, activation="relu", kernel_initializer="he_normal", name="dense_1"),

    # Dropout 1: randomly deactivates 15% of neurons during each training step
    # This forces the network to learn redundant, robust representations
    Dropout(0.15, name="dropout_1"),

    # Layer 2: 64 neurons — compresses the 128-dim representation
    # Learns interaction terms: potential × direction, scan_rate × potential, etc.
    Dense(64, activation="relu", kernel_initializer="he_normal", name="dense_2"),

    # Dropout 2: 10% — smaller because gradients at this depth need stability
    Dropout(0.10, name="dropout_2"),

    # Layer 3: 32 neurons — final non-linear compression before output
    Dense(32, activation="relu", kernel_initializer="he_normal", name="dense_3"),

    # Output: 1 neuron, linear activation — regression head
    # Predicts normalised current; no activation limits the range
    Dense(1, activation="linear", name="output"),
], name="ZnO_CV_ANN")

# ── Compile with Adam, MSE loss, MAE metric ───────────────────────────────────
model.compile(
    # Adam: adaptive learning rate; combines momentum with RMSProp scaling
    optimizer=Adam(learning_rate=1e-3),
    # MSE (Mean Squared Error): standard regression loss
    # Gradient proportional to error — large mistakes penalised quadratically
    loss="mse",
    # Track MAE during training (more interpretable than sqrt(MSE) per epoch)
    metrics=["mae"],
)

# ── Print architecture summary ────────────────────────────────────────────────
print("ANN Architecture Summary")
print("=" * 60)
model.summary()
print()

# Count parameters
total_params     = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print()
print("Architecture: Input(10) → Dense(128,ReLU) → Dropout(0.15)")
print("           → Dense(64,ReLU)  → Dropout(0.10)")
print("           → Dense(32,ReLU)  → Dense(1,Linear)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRAINING WITH CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 5 — Training with Early Stopping and LR Reduction

### Training configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `epochs` | 300 | Upper bound; actual epochs determined by EarlyStopping |
| `batch_size` | 512 | Large enough to use GPU efficiently; small enough for stable gradients |
| `validation_data` | Val (SR=30) | Unseen scan rate — ensures early stopping reflects true generalisation |
| EarlyStopping patience | 25 | Allows plateau before stopping; `restore_best_weights=True` |
| ReduceLROnPlateau patience | 10 | Halves lr after 10 stagnant epochs; `factor=0.5` |

### What to look for in the training output
- **Train loss ≫ Val loss early on**: underfitting — model needs more capacity or epochs.
- **Val loss stops improving while train loss drops**: overfitting — Dropout is helping.
- **EarlyStopping trigger**: confirms the model converged; `best_epoch` is the optimum.
- **LR reductions** in verbose output: each halving helps escape local minima.""")

code("""\
# ── Define callbacks ──────────────────────────────────────────────────────────

# EarlyStopping: monitors val_loss; stops when no improvement for 25 epochs
# restore_best_weights=True ensures predictions use the best checkpoint
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True,
    verbose=1,
)

# ReduceLROnPlateau: halves learning rate when val_loss stagnates for 10 epochs
# min_lr prevents the learning rate from becoming uselessly small
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1,
)

# ── Train the ANN ──────────────────────────────────────────────────────────────
print("Training ANN ...")
print(f"  Training samples   : {X_train.shape[0]:,}")
print(f"  Validation samples : {X_val.shape[0]:,}  (SR=30 — unseen scan rate)")
print(f"  Batch size         : 512")
print(f"  Max epochs         : 300  (EarlyStopping will cut this short)")
print(f"  EarlyStopping      : patience=25, restore_best_weights=True")
print(f"  ReduceLROnPlateau  : patience=10, factor=0.5")
print()

t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=512,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)
elapsed = time.time() - t0

epochs_ran = len(history.history["loss"])
best_epoch = int(np.argmin(history.history["val_loss"]))

print()
print(f"✓ Training complete in {elapsed:.1f} s  ({elapsed/60:.1f} min)")
print(f"  Epochs run       : {epochs_ran}  (max was 300)")
print(f"  Best epoch       : {best_epoch + 1}  (1-indexed, 0-indexed={best_epoch})")
print(f"  Best val MSE     : {min(history.history['val_loss']):.6f}")
print(f"  Best val RMSE    : {np.sqrt(min(history.history['val_loss'])):.6f}  (normalised)")
print(f"  Best val MAE     : {min(history.history['val_mae']):.6f}  (normalised)")
print()
print("  model.predict() will automatically use weights from best_epoch")
print("  because EarlyStopping was set with restore_best_weights=True")""")

code("""\
# ── Figure 01: ANN training curves (RMSE + MAE) ───────────────────────────────
hist       = history.history
n_epochs   = len(hist["loss"])
epoch_axis = range(1, n_epochs + 1)

# Convert MSE -> RMSE for more interpretable axis
train_rmse = np.sqrt(np.array(hist["loss"]))
val_rmse   = np.sqrt(np.array(hist["val_loss"]))
train_mae  = np.array(hist["mae"])
val_mae    = np.array(hist["val_mae"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: RMSE curve
ax = axes[0]
ax.plot(epoch_axis, train_rmse, color="#2ca02c", lw=1.0, alpha=0.85,
        label="Train RMSE")
ax.plot(epoch_axis, val_rmse,   color="#1f77b4", lw=1.5, alpha=0.9,
        label="Val RMSE (SR=30)")
ax.axvline(best_epoch + 1, color="#d62728", lw=1.8, ls="--",
           label=f"Best epoch = {best_epoch + 1}")
ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("RMSE (normalised current)", fontsize=9)
ax.set_title("ANN Training — RMSE per Epoch", fontsize=10, pad=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right panel: MAE curve
ax = axes[1]
ax.plot(epoch_axis, train_mae, color="#2ca02c", lw=1.0, alpha=0.85,
        label="Train MAE")
ax.plot(epoch_axis, val_mae,   color="#1f77b4", lw=1.5, alpha=0.9,
        label="Val MAE (SR=30)")
ax.axvline(best_epoch + 1, color="#d62728", lw=1.8, ls="--",
           label=f"Best epoch = {best_epoch + 1}")
ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("MAE (normalised current)", fontsize=9)
ax.set_title("ANN Training — MAE per Epoch", fontsize=10, pad=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("ANN Training History — Convergence and Early Stopping", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_ann_training_curve.png", bbox_inches="tight")
plt.show()
print("Saved figures/01_ann_training_curve.png")
print()
print("Scientific interpretation:")
print(f"  — Epochs run = {n_epochs}  (EarlyStopping cut short from 300 max)")
print(f"  — Best epoch = {best_epoch + 1}")
print("  — If train RMSE >> val RMSE: underfitting (need more capacity)")
print("  — If train RMSE << val RMSE: overfitting (Dropout is limiting it)")
print("  — Plateaus followed by LR reductions indicate ReduceLROnPlateau fired")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PREDICTIONS ON ALL PARTITIONS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 6 — Generate Predictions on All Partitions

`model.predict()` uses the weights restored from the best epoch.

Inverse transform: `I_pred (A) = I_pred_norm × (I_max − I_min) + I_min`
where `I_min`, `I_max` are the per-group scaler parameters stored row-wise
in each parquet file (`current_norm_min`, `current_norm_max`).""")

code("""\
# Work on copies to preserve original DataFrames
train_df    = train_df.copy()
val_df      = val_df.copy()
test_sr_df  = test_sr_df.copy()
test_mat_df = test_mat_df.copy()

# ── ANN predictions (batch_size=512 for memory efficiency) ───────────────────
# verbose=0 suppresses the Keras progress bar during inference
train_df["pred_norm"]    = model.predict(X_train,    batch_size=512, verbose=0).ravel()
val_df["pred_norm"]      = model.predict(X_val,      batch_size=512, verbose=0).ravel()
test_sr_df["pred_norm"]  = model.predict(X_test_sr,  batch_size=512, verbose=0).ravel()
test_mat_df["pred_norm"] = model.predict(X_test_mat, batch_size=512, verbose=0).ravel()

# ── Inverse transform: normalised current -> physical Amperes ─────────────────
# current_norm_min and current_norm_max are the per-(nm_id, scan_rate) scaler
# boundaries stored as columns in each parquet row (from Notebook 01)
for df in [train_df, val_df, test_sr_df, test_mat_df]:
    rng          = df["current_norm_max"] - df["current_norm_min"]
    df["pred_A"] = df["pred_norm"] * rng + df["current_norm_min"]

print("✓ Predictions generated and inverse-transformed to Amperes")
print()
for label, df in [("train",    train_df),
                   ("val",      val_df),
                   ("test_sr",  test_sr_df),
                   ("test_mat", test_mat_df)]:
    print(f"  {label:10s}: pred_A range "
          f"[{df['pred_A'].min():.4e}, {df['pred_A'].max():.4e}] A")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7 — Evaluation Metrics

All metrics are computed in **both** spaces:
- **Normalised** (dimensionless): direct comparison across notebooks.
- **Physical (Amperes / μA)**: scientifically meaningful quantity.

R² is invariant under the linear inverse transform: `R²_norm = R²_A`.

### RF and XGBoost baselines for reference
| Partition | RF RMSE (μA) | RF R² | XGB RMSE (μA) | XGB R² |
|-----------|-------------|-------|--------------|--------|
| Val (SR=30) | 26.59 | 0.9851 | — | — |
| Test-SR (SR=50) | 43.71 | 0.9777 | — | — |
| Test-MAT (NM4) | 33.65 | 0.9707 | — | — |

*(XGBoost values will be filled in Section 10 if Notebook 03 output is available)*""")

code("""\
def compute_metrics(y_true_norm, y_pred_norm, actual_A, pred_A, label):
    '''Compute regression metrics in both normalised and physical units.'''
    rmse_norm  = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))
    rmse_A     = np.sqrt(mean_squared_error(actual_A, pred_A))
    mae_A      = mean_absolute_error(actual_A, pred_A)
    r2         = r2_score(y_true_norm, y_pred_norm)
    max_err_A  = np.abs(actual_A - pred_A).max()
    return {
        "partition" : label,
        "n_samples" : len(y_true_norm),
        "RMSE_norm" : rmse_norm,
        "RMSE_uA"   : rmse_A    * 1e6,
        "MAE_uA"    : mae_A     * 1e6,
        "MaxErr_uA" : max_err_A * 1e6,
        "R2"        : r2,
    }

metrics_rows = []

# Train — shows in-sample fit; ANN may memorise training data
m = compute_metrics(
    y_train.astype(np.float64),  train_df["pred_norm"].values,
    train_df["current_A"].values, train_df["pred_A"].values,
    "train (in-sample)")
metrics_rows.append(m)

# Val — SR=30: unseen scan rate interpolation
m = compute_metrics(
    y_val_64, val_df["pred_norm"].values,
    val_df["current_A"].values, val_df["pred_A"].values,
    "val  (SR=30 interpolation)")
metrics_rows.append(m)

# Test-SR — SR=50: unseen scan rate interpolation
m = compute_metrics(
    y_test_sr_64, test_sr_df["pred_norm"].values,
    test_sr_df["current_A"].values, test_sr_df["pred_A"].values,
    "test_SR (SR=50 interpolation)")
metrics_rows.append(m)

# Test-MAT — NM4: unseen material extrapolation (hardest test)
m = compute_metrics(
    y_test_mat_64, test_mat_df["pred_norm"].values,
    test_mat_df["current_A"].values, test_mat_df["pred_A"].values,
    "test_MAT (NM4 extrapolation)")
metrics_rows.append(m)

metrics_df = pd.DataFrame(metrics_rows).set_index("partition")

print("=" * 70)
print("ANN MODEL — EVALUATION SUMMARY")
print("=" * 70)
print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
print()
print("Units: RMSE / MAE / MaxErr in micro-Amperes (uA).  R2 dimensionless.")
print()
print("Interpretation:")
print("  train in-sample  : Low RMSE expected — ANN can memorise rows.")
print("  val SR=30        : Scan-rate interpolation capability.")
print("  test_SR SR=50    : Scan-rate interpolation (harder gap).")
print("  test_MAT NM4     : Material extrapolation — hardest axis.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8a — CV CURVE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8a — Predicted vs Actual CV Curves

The most physically meaningful test: does the ANN reconstruct the correct
closed-loop voltammogram shape?

Solid lines = actual measured current.  Dashed lines = ANN prediction.
Rows are sorted by `(half_sweep_id, step_index)` to recover the correct
electrochemical trajectory from the tabular predictions.""")

code("""\
# ── Helper: plot one CV curve ─────────────────────────────────────────────────
def plot_cv_curve(ax, group_df, title, model_label="ANN", show_legend=True):
    grp = group_df.sort_values(["half_sweep_id", "step_index"])
    colours    = {0: "#1f77b4", 1: "#d62728"}
    dir_labels = {0: "Cathodic (dir=0)", 1: "Anodic (dir=1)"}
    legend_done = set()
    for _, seg in grp.groupby("half_sweep_id"):
        d     = int(seg["sweep_direction"].iloc[0])
        label = dir_labels[d] if d not in legend_done else None
        # Solid = actual measured current
        ax.plot(seg["potential_V"], seg["current_A"] * 1e6,
                color=colours[d], lw=1.5, alpha=0.9, label=label)
        # Dashed = ANN prediction
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


# ── Figure 02: Validation curves — NM1, NM2, NM3 at SR=30 ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nm in enumerate(["NM1", "NM2", "NM3"]):
    sub = val_df[(val_df["nm_id"] == nm) & (val_df["scan_rate_mVs"] == 30)]
    plot_cv_curve(axes[i], sub, f"VAL: {nm} @ 30 mV/s", show_legend=(i == 0))
fig.suptitle("Validation Set — Scan-Rate Interpolation (SR=30, seen materials)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_cv_curves_val.png", bbox_inches="tight")
plt.show()
print("Saved figures/02_cv_curves_val.png")""")

code("""\
# ── Figure 03: Test-SR curves — NM1, NM2, NM3 at SR=50 ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, nm in enumerate(["NM1", "NM2", "NM3"]):
    sub = test_sr_df[(test_sr_df["nm_id"] == nm) & (test_sr_df["scan_rate_mVs"] == 50)]
    plot_cv_curve(axes[i], sub, f"TEST-SR: {nm} @ 50 mV/s", show_legend=(i == 0))
fig.suptitle("Test-SR Set — Scan-Rate Interpolation (SR=50, seen materials)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_cv_curves_test_sr.png", bbox_inches="tight")
plt.show()
print("Saved figures/03_cv_curves_test_sr.png")""")

code("""\
# ── Figure 04: Test-MAT curves — NM4 at 6 representative scan rates ──────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
sr_showcase = [10, 30, 50, 70, 90, 100]

for i, sr in enumerate(sr_showcase):
    sub = test_mat_df[(test_mat_df["nm_id"] == "NM4") & (test_mat_df["scan_rate_mVs"] == sr)]
    plot_cv_curve(axes[i], sub, f"TEST-MAT: NM4 @ {sr} mV/s", show_legend=(i == 0))

fig.suptitle("Test-MAT Set — Material Extrapolation (NM4, fully unseen formulation)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_cv_curves_test_mat.png", bbox_inches="tight")
plt.show()
print("Saved figures/04_cv_curves_test_mat.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8b — PARITY PLOTS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8b — Parity Plots (Predicted vs Actual)

A perfect model lies on the diagonal `y = x`.
Colour encodes scan rate to reveal whether prediction quality changes with scan rate.

**Interpretation**: systematic deviations from the diagonal at specific scan rates
indicate the ANN has not fully generalised to that kinetic regime.""")

code("""\
# ── Figure 05: Parity plots ───────────────────────────────────────────────────
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

fig.suptitle("Parity Plots — ANN Model", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_parity_plots.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved figures/05_parity_plots.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8c — RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8c — Residual Analysis

**Figure 06**: Residual histograms (predicted − actual in μA).
Ideal: zero-centred, symmetric, approximately Gaussian.
Heavy tails → ANN struggles with peak currents at turning points.
Systematic offset → structural bias (e.g. scan-rate-dependent bias).

**Figure 07**: Residuals vs potential (coloured by sweep direction).
Uniform scatter around zero → no voltage-dependent error.
Fan-shaped pattern → heteroscedastic error (larger near peaks).
Asymmetry between cathodic/anodic → ANN not perfectly resolving direction.

The ANN, unlike trees, can learn smooth non-linear response surfaces,
so structured residuals seen in RF may be reduced here.""")

code("""\
# ── Figure 06: Residual distributions ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

dist_configs = [
    (val_df,      "Val (SR=30)",     "#ff7f0e"),
    (test_sr_df,  "Test-SR (SR=50)", "#1f77b4"),
    (test_mat_df, "Test-MAT (NM4)",  "#d62728"),
]

for ax, (df, title, colour) in zip(axes, dist_configs):
    residuals = (df["pred_A"] - df["current_A"]) * 1e6   # uA
    ax.hist(residuals, bins=80, color=colour, alpha=0.75,
            edgecolor="none", density=True)
    ax.axvline(0,                  color="k",      lw=1.5, ls="-",
               label="zero bias")
    ax.axvline(residuals.mean(),   color="darkred", lw=1.5, ls="--",
               label=f"mean={residuals.mean():.2f} uA")
    ax.axvline(residuals.median(), color="navy",    lw=1.2, ls=":",
               label=f"med={residuals.median():.2f} uA")
    ax.set_xlabel("Residual (uA): predicted - actual", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

fig.suptitle("Residual Distributions — ANN Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_residual_distributions.png", bbox_inches="tight")
plt.show()
print("Saved figures/06_residual_distributions.png")""")

code("""\
# ── Figure 07: Residuals vs potential (sweep direction coloured) ─────────────
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

fig.suptitle("Residuals vs Potential — ANN Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_residual_vs_potential.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved figures/07_residual_vs_potential.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8d — PERMUTATION FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8d — Permutation Feature Importance

ANN has **no native feature importance** (unlike RF's MDI or XGBoost's gain).
We therefore use **permutation importance** on the validation set:
- Shuffle one feature column at a time across all validation rows.
- Measure the increase in RMSE caused by that shuffle.
- A feature that is important will cause a large RMSE increase when shuffled.
- A feature that is redundant or irrelevant will cause little change.

This is computed on the **validation set** (SR=30 — unseen scan rate),
so it reflects what features are important for *generalisation*, not memorisation.

### Expected findings
- `direction_x_potential_norm` should rank high — it encodes the direction-voltage
  interaction that resolves the two-current CV hysteresis.
- `sweep_direction` alone should also be important — it's the only binary feature
  that distinguishes cathodic from anodic half-sweeps.
- Scan-rate features (`sqrt_scan_rate_norm`, `log_scan_rate_norm`) should rank
  higher for ANN than for RF because the ANN can learn smooth sr→current mappings.

### Comparison with RF and XGBoost
The same permutation importance is run in Notebooks 02 and 03.
Differences in ranking reveal which features are more or less exploitable by each
model family (tree vs gradient boosting vs neural network).""")

code("""\
def ann_permutation_importance(model, X, y, feature_names, n_repeats=20,
                                random_state=42, batch_size=512):
    '''
    Compute permutation importance for a Keras model.

    For each feature: shuffle its values n_repeats times, predict, and measure
    RMSE increase vs baseline.  A positive increase means the feature matters.

    Parameters
    ----------
    model        : trained Keras model with .predict()
    X            : numpy array, shape (n_samples, n_features)
    y            : numpy array, shape (n_samples,)
    feature_names: list of feature name strings
    n_repeats    : number of shuffle repeats per feature
    random_state : seed for reproducibility
    batch_size   : batch size for model.predict()

    Returns
    -------
    means : pd.Series  — mean RMSE increase per feature, sorted descending
    stds  : pd.Series  — std of RMSE increase per feature
    '''
    rng = np.random.default_rng(random_state)

    # Baseline RMSE on unshuffled data
    baseline_pred = model.predict(X, batch_size=batch_size, verbose=0).ravel()
    baseline_rmse = np.sqrt(mean_squared_error(y.astype(np.float64),
                                               baseline_pred.astype(np.float64)))

    n_features = len(feature_names)
    imp_matrix = np.zeros((n_features, n_repeats))   # [feature, repeat]

    for j in range(n_features):
        for r in range(n_repeats):
            X_perm = X.copy()
            # Shuffle column j in-place across all rows
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_pred = model.predict(X_perm, batch_size=batch_size,
                                      verbose=0).ravel()
            perm_rmse = np.sqrt(mean_squared_error(
                y.astype(np.float64), perm_pred.astype(np.float64)))
            imp_matrix[j, r] = perm_rmse - baseline_rmse  # > 0 means feature matters

    means = pd.Series(imp_matrix.mean(axis=1), index=feature_names)
    stds  = pd.Series(imp_matrix.std(axis=1),  index=feature_names)
    return means.sort_values(ascending=False), stds


print("Computing permutation importance on val set (n_repeats=20) ...")
print(f"  This runs 20 × {len(FEATURES)} = {20 * len(FEATURES)} model.predict() calls on X_val.")
t0 = time.time()

perm_means, perm_stds = ann_permutation_importance(
    model, X_val, y_val, FEATURES, n_repeats=20, random_state=42
)

print(f"  Done in {time.time() - t0:.1f} s")
print()
print("ANN Permutation Importance ranking (val set):")
for feat in perm_means.index:
    print(f"  {feat:35s}: {perm_means[feat]:.6f} +/- {perm_stds[feat]:.6f}")""")

code("""\
# ── Figure 08: ANN permutation feature importance ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

xerr = perm_stds.loc[perm_means.index][::-1].values
ax.barh(
    perm_means.index[::-1], perm_means.values[::-1],
    xerr=xerr, color="#9467bd", alpha=0.8,
    error_kw={"elinewidth": 1.2, "capsize": 3}
)
ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Permutation importance — RMSE increase when feature is shuffled", fontsize=9)
ax.set_title(
    "ANN Permutation Feature Importance (val set, n_repeats=20)\\n"
    "Higher = more important for generalisation to SR=30",
    fontsize=9, pad=8
)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_ann_feature_importance.png", bbox_inches="tight")
plt.show()
print("Saved figures/08_ann_feature_importance.png")
print()
print("Scientific note:")
print("  direction_x_potential_norm is typically the top feature across all models.")
print("  It encodes the key interaction: same voltage, different current per sweep direction.")
print("  High importance confirms the feature engineering correctly captured CV hysteresis.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8e — PER-GROUP PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8e — Per-Group Performance Breakdown

Break down RMSE by (nm_id, scan_rate) to reveal:
- Whether prediction quality is consistent across materials (NM1, NM2, NM3).
- Which scan rates are hardest to interpolate.
- How NM4 RMSE scales with scan rate (does error grow with current magnitude?).

At high scan rates, capacitive currents are larger, so absolute RMSE may be
higher even when relative accuracy is the same.""")

code("""\
def group_rmse(df, label):
    '''Compute RMSE and R2 per (nm_id, scan_rate_mVs) group.'''
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

# ── Figure 09: Per-group RMSE bars ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.bar(val_grp["nm_id"], val_grp["RMSE_uA"], color="#ff7f0e", alpha=0.85)
for i, row in val_grp.iterrows():
    ax.text(i, row["RMSE_uA"] + 0.05, f"R2={row['R2']:.3f}",
            ha="center", fontsize=8)
ax.set_xlabel("Material")
ax.set_ylabel("RMSE (uA)")
ax.set_title("Val (SR=30) — RMSE per material")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.bar(test_sr_grp["nm_id"], test_sr_grp["RMSE_uA"], color="#1f77b4", alpha=0.85)
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

fig.suptitle("Per-Group RMSE Breakdown — ANN Model", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "09_per_group_rmse.png", bbox_inches="tight")
plt.show()
print("Saved figures/09_per_group_rmse.png")""")

code("""\
# ── Figure 10: NM4 RMSE and R2 vs scan rate ───────────────────────────────────
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

fig.suptitle("NM4 Scan-Rate Trend — ANN Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "10_nm4_scan_rate_trend.png", bbox_inches="tight")
plt.show()
print("Saved figures/10_nm4_scan_rate_trend.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 9 — Save All Outputs")

code("""\
# ── Save trained ANN model (.keras format — TF2 native) ───────────────────────
model_path = MODEL_DIR / "ann_model.keras"
model.save(str(model_path))
print(f"Saved model  : {model_path}")

# ── Save training history as JSON ─────────────────────────────────────────────
history_out = {
    k: [float(v) for v in vs]
    for k, vs in history.history.items()
}
history_out["epochs_ran"]  = epochs_ran
history_out["best_epoch"]  = best_epoch
history_out["best_epoch_1indexed"] = best_epoch + 1

history_path = MODEL_DIR / "training_history.json"
with open(history_path, "w") as fh:
    json.dump(history_out, fh, indent=2)
print(f"Saved history: {history_path}")

# ── Save comprehensive metrics JSON ───────────────────────────────────────────
ann_arch = {
    "layers": [
        {"name": "Dense 1", "units": 128, "activation": "relu", "init": "he_normal"},
        {"name": "Dropout 1", "rate": 0.15},
        {"name": "Dense 2", "units": 64, "activation": "relu", "init": "he_normal"},
        {"name": "Dropout 2", "rate": 0.10},
        {"name": "Dense 3", "units": 32, "activation": "relu", "init": "he_normal"},
        {"name": "Output", "units": 1, "activation": "linear"},
    ],
    "total_params": int(model.count_params()),
    "optimizer": "Adam",
    "learning_rate_initial": 1e-3,
    "loss": "mse",
    "metrics": ["mae"],
}

training_cfg = {
    "epochs_max": 300,
    "epochs_ran": epochs_ran,
    "best_epoch": best_epoch,
    "batch_size": 512,
    "early_stopping_patience": 25,
    "reduce_lr_patience": 10,
    "reduce_lr_factor": 0.5,
    "best_val_mse": float(min(history.history["val_loss"])),
    "best_val_rmse_norm": float(np.sqrt(min(history.history["val_loss"]))),
}

metrics_out = {
    "model"      : "Dense ANN (TensorFlow/Keras)",
    "architecture": ann_arch,
    "training"   : training_cfg,
    "metrics"    : metrics_df.reset_index().to_dict(orient="records"),
    "per_group"  : {
        "val"     : val_grp.to_dict(orient="records"),
        "test_sr" : test_sr_grp.to_dict(orient="records"),
        "test_mat": test_mat_grp.to_dict(orient="records"),
    },
    "feature_importance_permutation": {
        f: {"mean": float(perm_means.get(f, 0.0)), "std": float(perm_stds.get(f, 0.0))}
        for f in FEATURES
    },
}

metrics_path = METRICS_DIR / "ann_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(metrics_out, fh, indent=2)
print(f"Saved metrics: {metrics_path}")

# ── Save per-group RMSE CSVs ──────────────────────────────────────────────────
for name, df_grp in [("val", val_grp), ("test_sr", test_sr_grp),
                      ("test_mat", test_mat_grp)]:
    path = CSV_DIR / f"per_group_rmse_{name}.csv"
    df_grp.to_csv(path, index=False)
    print(f"Saved {path.name}")

print()
print("✓ All outputs saved")
print(f"  model/   : ann_model.keras + training_history.json")
print(f"  metrics/ : ann_metrics.json")
print(f"  csv/     : 3 per-group RMSE tables")
print(f"  figures/ : 11 figures (00 through 10)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — SCIENTIFIC SUMMARY + COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 10 — Scientific Summary and RF vs XGBoost vs ANN Comparison

### Comparison methodology
All three models use:
- **Identical training data** (NM1/2/3 × SR∈{10,20,40,60,70,80,90,100}, 62,400 rows)
- **Identical feature set** (10 normalised features)
- **Identical evaluation partitions** (val SR=30, test-SR SR=50, test-MAT NM4)
- **Identical inverse transform** (per-group scaler parameters from Notebook 01)

### ANN-specific discussion
- **Peak fidelity**: ANN can learn smooth continuous response surfaces, so
  prediction at the turning points (−0.65 V and 0.0 V) may be improved vs trees.
- **Direction × potential interaction**: the `direction_x_potential_norm` feature
  captures the key many-to-one V→I hysteresis; high permutation importance confirms
  the ANN is using it effectively.
- **Structured residuals**: unlike RF/XGBoost which show staircase artifacts at
  decision boundaries, ANN residuals should be smoother and more Gaussian.
- **Extrapolation (NM4)**: ANN cannot extrapolate beyond its training distribution
  in the same way a physical model can — if NM4 electrochemistry is qualitatively
  different from NM1/2/3, ANN will struggle here too.""")

code("""\
# ── RF baseline (Notebook 02, verified run) ───────────────────────────────────
RF_RESULTS = {
    "val  (SR=30 interpolation)"   : {"RMSE_uA": 26.59, "R2": 0.9851},
    "test_SR (SR=50 interpolation)": {"RMSE_uA": 43.71, "R2": 0.9777},
    "test_MAT (NM4 extrapolation)" : {"RMSE_uA": 33.65, "R2": 0.9707},
}

# ── XGBoost baseline — load from Notebook 03 output if available ──────────────
XGB_RESULTS = {}
xgb_json = BASE_DIR / "results" / "xgboost" / "metrics" / "xgboost_metrics.json"
if xgb_json.exists():
    with open(xgb_json) as fh:
        _xgb_data = json.load(fh)
    for m in _xgb_data["metrics"]:
        XGB_RESULTS[m["partition"]] = {
            "RMSE_uA": m["RMSE_uA"], "R2": m["R2"]
        }
    print(f"XGBoost metrics loaded from {xgb_json}")
else:
    print(f"XGBoost metrics not found at: {xgb_json}")
    print("  Run Notebook 03 first to populate the XGBoost comparison.")
    print("  XGBoost column will show  '---'  below.")

# ── Build comparison table ────────────────────────────────────────────────────
partitions = [
    "val  (SR=30 interpolation)",
    "test_SR (SR=50 interpolation)",
    "test_MAT (NM4 extrapolation)",
]

print()
print("=" * 95)
print("RF vs XGBoost vs ANN — HEAD-TO-HEAD COMPARISON")
print("=" * 95)
print(f"  {'Partition':<32}  {'RF RMSE':>9}  {'XGB RMSE':>9}  {'ANN RMSE':>9}  "
      f"{'RF R2':>7}  {'XGB R2':>7}  {'ANN R2':>7}")
print("  " + "-" * 89)

for p in partitions:
    rf_rmse  = RF_RESULTS[p]["RMSE_uA"]
    rf_r2    = RF_RESULTS[p]["R2"]

    xgb_rmse = XGB_RESULTS[p]["RMSE_uA"] if p in XGB_RESULTS else None
    xgb_r2   = XGB_RESULTS[p]["R2"]     if p in XGB_RESULTS else None

    ann_rmse = metrics_df.loc[p, "RMSE_uA"]
    ann_r2   = metrics_df.loc[p, "R2"]

    xgb_rmse_str = f"{xgb_rmse:9.2f}" if xgb_rmse is not None else "      ---"
    xgb_r2_str   = f"{xgb_r2:7.4f}"  if xgb_r2   is not None else "    ---"

    # Arrows vs RF baseline
    ann_arrow = "▼" if ann_rmse < rf_rmse else "▲"

    print(f"  {p:<32}  {rf_rmse:9.2f}  {xgb_rmse_str}  "
          f"{ann_arrow}{ann_rmse:8.2f}  {rf_r2:7.4f}  {xgb_r2_str}  {ann_r2:7.4f}")

print()
print("  Units: RMSE in micro-Amperes (uA).  R2 dimensionless.")
print("  ▼ = ANN improves over RF baseline   ▲ = ANN worse than RF")
print()

# Summary verdict
improvements = sum(
    metrics_df.loc[p, "RMSE_uA"] < RF_RESULTS[p]["RMSE_uA"]
    for p in partitions
)
print(f"  ANN improved on {improvements}/3 partitions vs RF baseline.")

# ANN vs XGBoost (if available)
if XGB_RESULTS:
    xgb_improvements = sum(
        metrics_df.loc[p, "RMSE_uA"] < XGB_RESULTS[p]["RMSE_uA"]
        for p in partitions if p in XGB_RESULTS
    )
    print(f"  ANN improved on {xgb_improvements}/3 partitions vs XGBoost baseline.")

print()
print("Final best model so far  :", end=" ")
best_val = min(
    ("RF",      RF_RESULTS["val  (SR=30 interpolation)"]["RMSE_uA"]),
    ("ANN",     metrics_df.loc["val  (SR=30 interpolation)", "RMSE_uA"]),
    *([("XGBoost", XGB_RESULTS["val  (SR=30 interpolation)"]["RMSE_uA"])]
      if "val  (SR=30 interpolation)" in XGB_RESULTS else []),
    key=lambda x: x[1]
)
print(f"{best_val[0]} (val RMSE = {best_val[1]:.2f} uA)")
print()
print("Next: Notebook 05 — LSTM (adds explicit sequence modelling of CV trajectories)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — DOWNLOAD RESULTS TO LOCAL MACHINE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 11 — Download Results to Your Local Machine

Run this cell **before closing the Colab session** — all files are lost when
the runtime disconnects.

The cell automatically detects your execution mode:
- **Colab (upload / drive)**: creates a zip of the entire `results/ann/` folder
  and triggers a browser download.
- **Local**: lists the saved files (already on disk — no download needed).""")

code("""\
import shutil, os

if EXECUTION_MODE in ("colab_upload", "colab_drive"):
    from google.colab import files as _cf

    zip_base = str(BASE_DIR / "ann_results")
    print(f"Creating zip archive of {RESULTS_DIR} ...")
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
    print()
    print("Unzip into your PBL Project directory to restore the full structure:")
    print(f"  Extract to: .../PBL_Project/results/ann/")

else:
    # Local execution — files are already saved to disk
    print(f"Local execution — results already saved to: {RESULTS_DIR}")
    print()
    print("Saved files:")
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
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    },
    "colab": {
        "provenance": []
    }
})

out_path = Path(__file__).resolve().parent.parent.parent / "research" / "notebooks" / "04_ann_model.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out_path))
print(f"Written : {out_path}")
print(f"Cells   : {len(nb.cells)}")
