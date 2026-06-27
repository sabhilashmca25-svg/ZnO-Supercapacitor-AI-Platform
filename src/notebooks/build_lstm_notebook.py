"""
build_lstm_notebook.py
Generates notebooks/05_lstm_model.ipynb
Run once: python src/build_lstm_notebook.py
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
# Notebook 05 — LSTM Sequence Model
## ZnO Supercapacitor CV Trajectory Prediction · Temporal Deep Learning

### Purpose
Train an **LSTM (Long Short-Term Memory) sequence model** on the same scientifically
correct partitions established in Notebook 01.  Unlike previous notebooks, the LSTM
receives each **half-sweep as one contiguous sequence** rather than as independent
rows — correctly representing the electrochemical trajectory as a time series.

This notebook:
- Constructs 3D sequence arrays `(samples, timesteps, features)` from the flat parquet files.
- Trains a stacked LSTM with TimeDistributed output head — predicting at every timestep.
- Evaluates on the same three held-out partitions as RF, XGBoost, and ANN.
- Delivers a **four-model comparison table** (RF vs XGBoost vs ANN vs LSTM).
- Produces 11 publication-ready figures.

### Why LSTM is the scientifically correct architecture for CV data
| Model | Data representation | CV physics captured |
|-------|-------------------|---------------------|
| RF | Independent rows | None — pure feature-space regression |
| XGBoost | Independent rows | None — sequential boosting of row errors |
| ANN | Independent rows | None — smooth non-linear row-to-row mapping |
| **LSTM** | **Sequences of ~650 timesteps** | **Electrochemical history dependence** |

A cyclic voltammetry sweep is not a set of independent measurements.
At each potential step, the current depends on:
- The **history of applied potentials** (charging of the double layer).
- **Faradaic processes** that began at earlier timesteps.
- The **direction** of the sweep and how far along it has progressed.

LSTM's hidden state `h_t` accumulates this history, enabling it to model
trajectory-level patterns that are invisible to row-wise models.

### Architecture overview
```
Input(timesteps, 10)
↓
Masking(mask_value=0.0)          ← ignores padded timesteps
↓
LSTM(64, return_sequences=True)  ← learns temporal dynamics
↓
Dropout(0.20)
↓
LSTM(32, return_sequences=True)  ← refines representation at each timestep
↓
Dropout(0.10)
↓
TimeDistributed(Dense(16, ReLU)) ← per-timestep nonlinear projection
↓
TimeDistributed(Dense(1))        ← per-timestep current prediction
```

### Three generalisation questions answered
| Partition | Question |
|-----------|----------|
| **Val** (NM1/2/3 × SR=30) | Can LSTM interpolate trajectories at an unseen scan rate? |
| **Test-SR** (NM1/2/3 × SR=50) | Can LSTM interpolate SR=50 trajectories? |
| **Test-MAT** (NM4 × all SRs) | Can LSTM generalise trajectories to an unseen ZnO formulation? |""")

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
**T4 GPU** — Runtime → Change runtime type → T4 GPU → Save.
LSTM training benefits from GPU acceleration more than ANN due to sequential
matrix ops across 651 timesteps per sequence.""")

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
LOCAL_PROJECT_PATH = str(Path(__file__).resolve().parent.parent.parent)

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
RESULTS_DIR = BASE_DIR / "results" / "lstm"
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
**Nothing is recomputed** — splits, normalisation, and feature engineering are fixed.""")

code("""\
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy  as np
import pandas as pd
import json
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.6e}".format)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import (LSTM, Dense, Dropout,
                                          TimeDistributed, Masking, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow : {tf.__version__}")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU        : {gpus[0].name}  (training will use GPU acceleration)")
else:
    print("GPU        : not detected — training on CPU")
    print("  For faster training: Runtime > Change runtime type > T4 GPU")

# ── Load preprocessing artefacts ─────────────────────────────────────────────
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

Identical PASS/FAIL checks as Notebooks 02–04.
Verifies the leakage-safe splits before any sequence construction.""")

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

print("\\n✓ ALL INTEGRITY CHECKS PASSED — safe to construct sequences")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SEQUENCE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 3 — Sequence Construction: Flat Rows → 3D LSTM Input

### The fundamental difference from RF / XGBoost / ANN

Previous notebooks fed each measurement row independently:
```
RF/XGB/ANN input:  (n_rows, 10)   — e.g. (62400, 10)
```

LSTM requires sequences — each **half-sweep** becomes one sample:
```
LSTM input:  (n_sequences, timesteps, 10)   — e.g. (96, 651, 10)
```

### Grouping key
Each unique `(nm_id, scan_rate_mVs, half_sweep_id)` triplet defines one sequence.

### Sequence ordering
Within each sequence, rows are sorted by `step_index` (ascending) to preserve
the electrochemical sweep direction: cathodic (decreasing V) followed by anodic
(increasing V) within each half-sweep.

### Variable-length handling
Half-sweeps are 649–651 timesteps (inclusive turning-point edges cause ±1 variation).
All sequences are **padded to `max_seq_len`** with zeros.
A `Masking(mask_value=0.0)` layer in the model ignores padded timesteps during
training — they contribute zero gradient.

**Why padding is safe**: at any real measurement timestep, at least one feature
(e.g. `potential_from_upper_norm`) is always non-zero, so a real timestep is
never masked.

### Output shapes
```
X : (n_seqs, max_seq_len, 10)   float32 — input to LSTM
Y : (n_seqs, max_seq_len, 1)    float32 — target at every timestep
```
The LSTM predicts current at **every timestep** within the sequence
(time-distributed regression), not just at the final step.""")

code("""\
# ── Compute global max and min sequence length from master ────────────────────
seq_lens    = master_df.groupby(["nm_id","scan_rate_mVs","half_sweep_id"]).size()
max_seq_len = int(seq_lens.max())
min_seq_len = int(seq_lens.min())

print(f"Sequence length statistics across all half-sweeps:")
print(f"  min = {min_seq_len}  max = {max_seq_len}  "
      f"variation = {max_seq_len - min_seq_len} timestep(s)")
print(f"  All sequences padded to : {max_seq_len} timesteps")
print(f"  Padding amount per seq  : 0–{max_seq_len - min_seq_len} zeros")
print()

# ── Sequence construction function ────────────────────────────────────────────
def build_sequences(df, features, target, max_len):
    '''
    Convert a flat DataFrame into 3D LSTM input arrays.

    Groups by (nm_id, scan_rate_mVs, half_sweep_id) and sorts each group
    by step_index.  Pads shorter sequences with zeros to max_len.

    Parameters
    ----------
    df       : DataFrame with all required columns
    features : list of feature column names
    target   : string, target column name
    max_len  : int, all sequences are padded/kept to this length

    Returns
    -------
    X       : np.float32  (n_seqs, max_len, n_features)
    Y       : np.float32  (n_seqs, max_len, 1)
    info_df : DataFrame   (nm_id, scan_rate_mVs, half_sweep_id, actual_len)
    row_idx : np.ndarray  original DataFrame index, length = sum(actual_len)
    '''
    seqs_X, seqs_y, info_list, all_row_idx = [], [], [], []

    grp_keys = ["nm_id", "scan_rate_mVs", "half_sweep_id"]
    for (nm, sr, hs), grp in df.groupby(grp_keys, sort=True):
        g           = grp.sort_values("step_index")
        x           = g[features].to_numpy(dtype=np.float32)
        y           = g[target].to_numpy(dtype=np.float32)
        actual_len  = len(x)

        # Pad to max_len with zeros (Masking layer will ignore them)
        if actual_len < max_len:
            pad = max_len - actual_len
            x   = np.vstack([x, np.zeros((pad, len(features)), dtype=np.float32)])
            y   = np.concatenate([y, np.zeros(pad, dtype=np.float32)])

        seqs_X.append(x)
        seqs_y.append(y[:, None])   # (max_len, 1) for TimeDistributed target
        info_list.append({"nm_id": nm, "scan_rate_mVs": sr,
                           "half_sweep_id": hs, "actual_len": actual_len})
        all_row_idx.extend(g.index.tolist())

    X       = np.stack(seqs_X)                 # (n_seqs, max_len, n_features)
    Y       = np.stack(seqs_y)                 # (n_seqs, max_len, 1)
    info_df = pd.DataFrame(info_list)
    row_idx = np.array(all_row_idx, dtype=np.int64)
    return X, Y, info_df, row_idx


# ── Flatten-prediction helper ─────────────────────────────────────────────────
def flatten_predictions(pred_3d, info_df):
    '''
    Remove padding from LSTM output and concatenate across sequences.

    Parameters
    ----------
    pred_3d : np.ndarray  (n_seqs, max_len, 1)  — raw model.predict() output
    info_df : DataFrame   with column 'actual_len' per sequence

    Returns
    -------
    flat : np.ndarray  (total_timesteps,)  — one prediction per original row
    '''
    parts = [pred_3d[i, :row["actual_len"], 0]
             for i, row in info_df.iterrows()]
    return np.concatenate(parts)


# ── Attach predictions back to original DataFrame ─────────────────────────────
def attach_predictions(df, pred_3d, info_df, row_idx):
    '''
    Map flattened LSTM predictions onto the original DataFrame rows.

    Uses row_idx (original df.index values in sequence order) to align
    predictions correctly regardless of the DataFrame's current row order.
    '''
    flat = flatten_predictions(pred_3d, info_df)
    pred_series = pd.Series(flat, index=row_idx)
    out = df.copy()
    out["pred_norm"] = pred_series.reindex(df.index).values
    return out


print("✓ Sequence functions defined: build_sequences / flatten_predictions / attach_predictions")""")

code("""\
# ── Build 3D arrays for all four partitions ────────────────────────────────────
print("Building sequence arrays ...")
t0 = time.time()

X_train,    Y_train,    info_train,    idx_train    = build_sequences(
    train_df,    FEATURES, TARGET, max_seq_len)
X_val,      Y_val,      info_val,      idx_val      = build_sequences(
    val_df,      FEATURES, TARGET, max_seq_len)
X_test_sr,  Y_test_sr,  info_test_sr,  idx_test_sr  = build_sequences(
    test_sr_df,  FEATURES, TARGET, max_seq_len)
X_test_mat, Y_test_mat, info_test_mat, idx_test_mat = build_sequences(
    test_mat_df, FEATURES, TARGET, max_seq_len)

print(f"  Done in {time.time()-t0:.1f} s")
print()
print("Sequence array shapes:")
print(f"  X_train    : {X_train.shape}    Y_train    : {Y_train.shape}")
print(f"  X_val      : {X_val.shape}     Y_val      : {Y_val.shape}")
print(f"  X_test_sr  : {X_test_sr.shape}     Y_test_sr  : {Y_test_sr.shape}")
print(f"  X_test_mat : {X_test_mat.shape}    Y_test_mat : {Y_test_mat.shape}")
print()
print(f"  Interpretation: (n_sequences, timesteps, n_features)")
print(f"  LSTM sees {X_train.shape[0]} complete CV half-sweep trajectories during training")
print()

# Verify no leakage: no sequence group appears in more than one partition
train_keys   = set(zip(info_train["nm_id"],   info_train["scan_rate_mVs"],   info_train["half_sweep_id"]))
val_keys     = set(zip(info_val["nm_id"],     info_val["scan_rate_mVs"],     info_val["half_sweep_id"]))
test_sr_keys = set(zip(info_test_sr["nm_id"],  info_test_sr["scan_rate_mVs"],  info_test_sr["half_sweep_id"]))
test_mat_keys= set(zip(info_test_mat["nm_id"], info_test_mat["scan_rate_mVs"], info_test_mat["half_sweep_id"]))

assert not (train_keys & val_keys),     "LEAKAGE: train and val share sequences!"
assert not (train_keys & test_sr_keys), "LEAKAGE: train and test_sr share sequences!"
assert not (train_keys & test_mat_keys),"LEAKAGE: train and test_mat share sequences!"
print("  [PASS] No sequence appears in more than one partition")
print()
print(f"  [PASS] Sequence counts:")
print(f"    train    : {len(info_train):3d} sequences  "
      f"({len(info_train['nm_id'].unique())} materials x "
      f"{len(info_train['scan_rate_mVs'].unique())} scan rates x 4 half-sweeps)")
print(f"    val      : {len(info_val):3d} sequences  "
      f"({len(info_val['nm_id'].unique())} materials x "
      f"{len(info_val['scan_rate_mVs'].unique())} scan rate x 4 half-sweeps)")
print(f"    test_sr  : {len(info_test_sr):3d} sequences")
print(f"    test_mat : {len(info_test_mat):3d} sequences")""")

code("""\
# ── Figure 00: Sequence construction visualisation ────────────────────────────
fig = plt.figure(figsize=(15, 4.5))
gs  = fig.add_gridspec(1, 3, wspace=0.35)

# Panel 1: sequence counts per partition
ax1 = fig.add_subplot(gs[0])
parts   = ["train", "val", "test_SR", "test_MAT"]
counts  = [len(info_train), len(info_val), len(info_test_sr), len(info_test_mat)]
colours = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]
bars = ax1.bar(parts, counts, color=colours, alpha=0.85)
for bar, cnt in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, cnt + 0.3,
             str(cnt), ha="center", va="bottom", fontsize=9)
ax1.set_ylabel("Number of sequences")
ax1.set_title("Sequence counts per partition\\n(each = one CV half-sweep)", fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")

# Panel 2: sequence length histogram (all half-sweeps)
ax2 = fig.add_subplot(gs[1])
all_lens = seq_lens.values
ax2.hist(all_lens, bins=range(min_seq_len-1, max_seq_len+2),
         color="#9467bd", alpha=0.8, edgecolor="white")
ax2.axvline(max_seq_len, color="#d62728", lw=1.5, ls="--",
            label=f"Padded to: {max_seq_len}")
ax2.set_xlabel("Sequence length (timesteps)")
ax2.set_ylabel("Count")
ax2.set_title(f"Sequence length distribution\\n(range: {min_seq_len}–{max_seq_len} timesteps)", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3: example trajectory (one half-sweep from training set)
ax3 = fig.add_subplot(gs[2])
ex_row = info_train.iloc[0]
ex_grp = train_df[
    (train_df["nm_id"]          == ex_row["nm_id"]) &
    (train_df["scan_rate_mVs"]  == ex_row["scan_rate_mVs"]) &
    (train_df["half_sweep_id"]  == ex_row["half_sweep_id"])
].sort_values("step_index")

colours_dir = {0: "#1f77b4", 1: "#d62728"}
for d, grp in ex_grp.groupby("sweep_direction"):
    ax3.plot(grp["step_index"], grp["potential_V_norm"],
             color=colours_dir[d], lw=1.2, alpha=0.8,
             label=f"dir={d} ({'Cathodic' if d==0 else 'Anodic'})")
ax3_r = ax3.twinx()
ax3_r.plot(ex_grp["step_index"], ex_grp["current_normalized"],
           color="#8c564b", lw=1.0, alpha=0.6, ls=":", label="current_norm")
ax3.set_xlabel("step_index (timestep)")
ax3.set_ylabel("potential_V_norm", color="#1f77b4")
ax3_r.set_ylabel("current_normalized", color="#8c564b")
ax3.set_title(
    f"Example sequence:\\n{ex_row['nm_id']} SR={ex_row['scan_rate_mVs']} "
    f"HS={ex_row['half_sweep_id']}  (len={ex_row['actual_len']})", fontsize=9)
ax3.legend(fontsize=7, loc="upper left")
ax3.grid(True, alpha=0.2)

fig.suptitle("Sequence Construction — CV Half-Sweeps as LSTM Input", fontsize=11, y=1.02)
plt.savefig(FIGURES_DIR / "00_sequence_construction.png", bbox_inches="tight")
plt.show()
print("Saved figures/00_sequence_construction.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LSTM ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 4 — LSTM Architecture

### Layer design rationale
| Layer | Config | Purpose |
|-------|--------|---------|
| `Masking` | mask_value=0.0 | Ignore padded timesteps — no gradient from zero-padding |
| `LSTM(64, return_sequences=True)` | — | First layer: captures sweep-level dynamics across all timesteps |
| `Dropout(0.20)` | — | Regularise: prevents the LSTM from memorising scan-rate-specific patterns |
| `LSTM(32, return_sequences=True)` | — | Second layer: refines the trajectory representation at each timestep |
| `Dropout(0.10)` | — | Lighter regularisation on the compressed representation |
| `TimeDistributed(Dense(16, ReLU))` | — | Per-timestep nonlinear projection of LSTM state |
| `TimeDistributed(Dense(1))` | — | Per-timestep current prediction (regression at every step) |

### Why `return_sequences=True`?
With `return_sequences=True`, the LSTM outputs a hidden state vector `h_t` at
**every** timestep, not just the final one.  This allows:
- The second LSTM to process the full sequence.
- `TimeDistributed(Dense)` to produce a prediction at every timestep.
- Loss computed across all 651 measurement points — full trajectory supervision.

### Why `Masking`?
Sequences are padded with zeros.  Without Masking, the LSTM would treat padded
zeros as real measurements, introducing noise into the final hidden states.
Masking marks each zero-padded timestep as invalid — the LSTM copies its
previous state `h_{t-1}` instead of updating, and the loss ignores those steps.

### TimeDistributed output
The final output shape is `(batch, timesteps, 1)` — one predicted current
value at every timestep.  After inference, these are flattened back to the
row-level and inverse-transformed to Amperes for evaluation.""")

code("""\
# ── Build the stacked LSTM model ──────────────────────────────────────────────
model = Sequential([
    # Input: sequences of max_seq_len timesteps, each with 10 features
    Input(shape=(max_seq_len, len(FEATURES)), name="input"),

    # Masking: timesteps where ALL features == 0.0 are skipped
    # (Only zero-padded timesteps meet this condition — real data is never all-zero)
    Masking(mask_value=0.0, name="masking"),

    # First LSTM: 64 units, return full sequence (one hidden state per timestep)
    # Learns temporal sweep dynamics: how current evolves across the trajectory
    LSTM(64, return_sequences=True, name="lstm_1"),

    # Dropout 1: randomly zeros 20% of LSTM outputs during training
    Dropout(0.20, name="dropout_1"),

    # Second LSTM: 32 units, also return full sequence
    # Refines the 64-dim representation to a more compressed 32-dim trajectory
    LSTM(32, return_sequences=True, name="lstm_2"),

    # Dropout 2: lighter 10% dropout at this deeper representation
    Dropout(0.10, name="dropout_2"),

    # TimeDistributed(Dense(16, ReLU)):
    # Applies the same Dense(16) independently at every timestep
    # Adds non-linear capacity to the per-timestep prediction head
    TimeDistributed(Dense(16, activation="relu"), name="td_dense"),

    # TimeDistributed(Dense(1)):
    # Predicts one normalised current value at every timestep
    # Output shape: (batch, max_seq_len, 1)
    TimeDistributed(Dense(1), name="td_output"),
], name="ZnO_CV_LSTM")

# ── Compile with Adam, MSE loss, MAE metric ───────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)

print("LSTM Architecture Summary")
print("=" * 60)
model.summary()
print()
print(f"Input  shape : (batch, {max_seq_len}, {len(FEATURES)})")
print(f"Output shape : (batch, {max_seq_len}, 1)  — prediction at every timestep")
print(f"Total params : {model.count_params():,}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 5 — Training with Early Stopping and LR Reduction

### Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `epochs` | 150 | Upper bound; EarlyStopping will terminate earlier |
| `batch_size` | 16 | 16 complete half-sweep sequences per gradient step |
| `validation_data` | Val sequences (SR=30) | Unseen scan rate — genuine generalisation signal |
| EarlyStopping `patience` | 20 | 20 epochs of no val improvement before stopping |
| ReduceLROnPlateau `patience` | 8 | Halves lr after 8 stagnant val-loss epochs |

### LSTM training scale
- Training sequences: **`X_train.shape[0]`** half-sweeps × 651 timesteps each.
- Each epoch processes all training sequences in batches of 16.
- Loss is averaged across all `batch × timesteps × 1` predictions per batch.
- This means each gradient step uses far more information than RF/XGB/ANN batches.

### What to watch
- **Train loss dropping faster than val**: LSTM memorising temporal order of
  training scan rates — Dropout is the main defence.
- **LR reductions in verbose output**: `ReduceLROnPlateau` firing means the
  val loss has plateaued; smaller steps help navigate the loss landscape.
- **Early stopping trigger**: The message `Restoring model weights from
  the end of the best epoch` confirms `restore_best_weights=True` worked.""")

code("""\
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1,
)

print("Training LSTM ...")
print(f"  Training sequences   : {X_train.shape[0]}")
print(f"  Validation sequences : {X_val.shape[0]}  (SR=30 half-sweeps)")
print(f"  Sequence length      : {max_seq_len} timesteps")
print(f"  Input shape          : {X_train.shape}")
print(f"  Target shape         : {Y_train.shape}")
print(f"  Batch size           : 16 sequences")
print(f"  Batches per epoch    : {X_train.shape[0] // 16} (approx)")
print(f"  Max epochs           : 150")
print()

t0 = time.time()
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=150,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)
elapsed = time.time() - t0

epochs_ran = len(history.history["loss"])
best_epoch = int(np.argmin(history.history["val_loss"]))

print()
print(f"✓ Training complete in {elapsed:.1f} s  ({elapsed/60:.1f} min)")
print(f"  Epochs run           : {epochs_ran}  (max was 150)")
print(f"  Best epoch           : {best_epoch + 1}  (0-indexed: {best_epoch})")
print(f"  Best val MSE         : {min(history.history['val_loss']):.6f}")
print(f"  Best val RMSE (norm) : {np.sqrt(min(history.history['val_loss'])):.6f}")""")

code("""\
# ── Figure 01: LSTM training curves ───────────────────────────────────────────
hist       = history.history
n_epochs   = len(hist["loss"])
epoch_axis = range(1, n_epochs + 1)

train_rmse = np.sqrt(np.array(hist["loss"]))
val_rmse   = np.sqrt(np.array(hist["val_loss"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(epoch_axis, train_rmse, color="#2ca02c", lw=1.0, alpha=0.85, label="Train RMSE")
ax.plot(epoch_axis, val_rmse,   color="#1f77b4", lw=1.5, alpha=0.9,  label="Val RMSE (SR=30 seqs)")
ax.axvline(best_epoch + 1, color="#d62728", lw=1.8, ls="--",
           label=f"Best epoch = {best_epoch + 1}")
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE (normalised current)")
ax.set_title("LSTM Training — RMSE per Epoch", fontsize=10, pad=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(epoch_axis, hist["mae"],     color="#2ca02c", lw=1.0, alpha=0.85, label="Train MAE")
ax.plot(epoch_axis, hist["val_mae"], color="#1f77b4", lw=1.5, alpha=0.9,  label="Val MAE (SR=30 seqs)")
ax.axvline(best_epoch + 1, color="#d62728", lw=1.8, ls="--",
           label=f"Best epoch = {best_epoch + 1}")
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE (normalised current)")
ax.set_title("LSTM Training — MAE per Epoch", fontsize=10, pad=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("LSTM Training History — Temporal Sequence Learning", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_lstm_training_curve.png", bbox_inches="tight")
plt.show()
print("Saved figures/01_lstm_training_curve.png")
print()
print("Scientific interpretation:")
print(f"  Epochs run = {n_epochs}  (EarlyStopping from 150 max)")
print(f"  Best epoch = {best_epoch + 1}")
print("  — Faster val convergence than ANN suggests LSTM exploits temporal structure.")
print("  — Large train/val gap = LSTM memorising training scan-rate trajectories.")
print("  — LR reductions visible as slope changes after plateau periods.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PREDICTIONS + INVERSE TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 6 — Predictions, Sequence Flattening, and Inverse Transform

### Flattening procedure (LSTM-specific)
`model.predict()` returns shape `(n_seqs, max_seq_len, 1)`.

Each sequence in this output contains `actual_len` real predictions followed by
`max_seq_len - actual_len` predictions for zero-padded timesteps.
We discard the padded predictions and keep only the `actual_len` real ones.

The `flatten_predictions()` function reads `actual_len` from `info_df` for each
sequence and concatenates real predictions in the same order as the original rows.
`attach_predictions()` then uses the saved `row_idx` array to reindex predictions
back to each DataFrame's original index — guaranteeing correct row alignment.

### Inverse transform
Identical to all previous notebooks:
`I_pred (A) = pred_norm × (I_max − I_min) + I_min`
where `I_min`, `I_max` are stored row-wise in each parquet file.""")

code("""\
# ── Predict on all partitions (model uses best_epoch weights) ─────────────────
print("Generating predictions ...")
t0 = time.time()

pred_train_3d    = model.predict(X_train,    batch_size=16, verbose=0)
pred_val_3d      = model.predict(X_val,      batch_size=16, verbose=0)
pred_test_sr_3d  = model.predict(X_test_sr,  batch_size=16, verbose=0)
pred_test_mat_3d = model.predict(X_test_mat, batch_size=16, verbose=0)

print(f"  Prediction shapes: {pred_train_3d.shape}  "
      f"{pred_val_3d.shape}  {pred_test_sr_3d.shape}  {pred_test_mat_3d.shape}")
print(f"  Done in {time.time() - t0:.1f} s")

# ── Flatten predictions and attach to DataFrames ──────────────────────────────
# attach_predictions() removes padding, reindexes to original df rows
train_df    = attach_predictions(train_df,    pred_train_3d,    info_train,    idx_train)
val_df      = attach_predictions(val_df,      pred_val_3d,      info_val,      idx_val)
test_sr_df  = attach_predictions(test_sr_df,  pred_test_sr_3d,  info_test_sr,  idx_test_sr)
test_mat_df = attach_predictions(test_mat_df, pred_test_mat_3d, info_test_mat, idx_test_mat)

# Verify no NaN from the reindex operation
for name, df in [("train",train_df),("val",val_df),("test_sr",test_sr_df),("test_mat",test_mat_df)]:
    n_nan = df["pred_norm"].isna().sum()
    assert n_nan == 0, f"NaN predictions in {name}: {n_nan}"

print("  [PASS] No NaN in flattened predictions")

# ── Inverse transform: normalised -> Amperes ──────────────────────────────────
for df in [train_df, val_df, test_sr_df, test_mat_df]:
    rng          = df["current_norm_max"] - df["current_norm_min"]
    df["pred_A"] = df["pred_norm"] * rng + df["current_norm_min"]

print()
print("✓ Predictions inverse-transformed to Amperes")
for label, df in [("train",train_df),("val",val_df),("test_sr",test_sr_df),("test_mat",test_mat_df)]:
    print(f"  {label:10s}: pred_A range "
          f"[{df['pred_A'].min():.4e}, {df['pred_A'].max():.4e}] A")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7 — Evaluation Metrics

Identical metric computation to all previous notebooks.
After flattening, LSTM predictions are row-level — directly comparable with RF/XGB/ANN.""")

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

y_train_flat  = train_df["current_normalized"].values
y_val_flat    = val_df["current_normalized"].values
y_testsr_flat = test_sr_df["current_normalized"].values
y_testm_flat  = test_mat_df["current_normalized"].values

metrics_rows = []
metrics_rows.append(compute_metrics(
    y_train_flat, train_df["pred_norm"].values,
    train_df["current_A"].values, train_df["pred_A"].values,
    "train (in-sample)"))
metrics_rows.append(compute_metrics(
    y_val_flat, val_df["pred_norm"].values,
    val_df["current_A"].values, val_df["pred_A"].values,
    "val  (SR=30 interpolation)"))
metrics_rows.append(compute_metrics(
    y_testsr_flat, test_sr_df["pred_norm"].values,
    test_sr_df["current_A"].values, test_sr_df["pred_A"].values,
    "test_SR (SR=50 interpolation)"))
metrics_rows.append(compute_metrics(
    y_testm_flat, test_mat_df["pred_norm"].values,
    test_mat_df["current_A"].values, test_mat_df["pred_A"].values,
    "test_MAT (NM4 extrapolation)"))

metrics_df = pd.DataFrame(metrics_rows).set_index("partition")

print("=" * 70)
print("LSTM MODEL — EVALUATION SUMMARY")
print("=" * 70)
print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
print()
print("Units: RMSE / MAE / MaxErr in micro-Amperes (uA).  R2 dimensionless.")
print()
print("RF baseline for comparison:")
print("  val RMSE=26.59 uA  |  test_SR RMSE=43.71 uA  |  test_MAT RMSE=33.65 uA")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8a — CV CURVES
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8a — Predicted vs Actual CV Curves

The primary physical test: does the LSTM reconstruct the closed-loop
voltammogram shape with trajectory-level fidelity?

Because LSTM processes each half-sweep as a sequence, it can model
**temporal dependencies within the sweep** — e.g. the gradual charging of the
double-layer as potential changes, or the history-dependent Faradaic response
near the turning points.""")

code("""\
def plot_cv_curve(ax, group_df, title, model_label="LSTM", show_legend=True):
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
# SECTION 8b — PARITY PLOTS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 8b — Parity Plots (Predicted vs Actual)")

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
fig.suptitle("Parity Plots — LSTM Model", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_parity_plots.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved figures/05_parity_plots.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8c — RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8c — Residual Analysis

Because LSTM models the trajectory as a whole, residuals near **turning points**
(−0.65 V and 0.0 V) should be smaller than for row-wise models.  The LSTM hidden
state accumulates context about how far along the sweep the electrode has progressed,
improving predictions at electrochemically complex transition regions.""")

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
fig.suptitle("Residual Distributions — LSTM Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_residual_distributions.png", bbox_inches="tight")
plt.show()
print("Saved figures/06_residual_distributions.png")""")

code("""\
# ── Figure 07: Residuals vs potential ────────────────────────────────────────
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
fig.suptitle("Residuals vs Potential — LSTM Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_residual_vs_potential.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved figures/07_residual_vs_potential.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8d — PERMUTATION FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8d — Permutation Feature Importance (Sequence-Level)

For LSTM, permutation importance shuffles a feature across **all actual
(non-padded) timesteps and all sequences** simultaneously.  This destroys the
temporal structure of that feature across the entire validation set — measuring
how much the LSTM relies on that feature's time-series signal.

> **Why non-padded only?**  Sequences are zero-padded to a common length and
> the `Masking(mask_value=0.0)` layer suppresses padded timesteps.  If we
> shuffle the entire time axis including padding, non-zero values land in
> previously-zero positions and the Masking layer un-masks them — corrupting
> the LSTM's sequence-length assumption and triggering a CuDNN Assert error.
> Restricting the shuffle to actual timesteps keeps the padding intact.

Compared to row-wise models:
- **Row-wise models (RF/XGB/ANN)**: shuffling a column breaks only the
  feature-label pairing for that row.
- **LSTM**: shuffling a column breaks both the feature-label pairing AND
  the temporal dependencies the LSTM learned within sequences for that feature.

This means LSTM permutation importance captures a richer notion of importance:
a feature is important if either its **value** or its **temporal pattern** matters.""")

code("""\
def lstm_permutation_importance(model, X_seqs, y_flat, info_df,
                                 feature_names, n_repeats=10, random_state=42,
                                 batch_size=16):
    '''
    Permutation importance for a sequence LSTM model.

    Shuffles feature j across actual (non-padded) timesteps only, then
    measures RMSE increase relative to the baseline.  Positive = feature matters.

    Padding positions remain all-zeros so the Masking layer mask is never
    corrupted — avoids the CuDNN LSTM InvalidArgumentError that occurs when
    a non-zero value is placed into a previously all-zero padded timestep.

    Parameters
    ----------
    model        : trained Keras LSTM (output shape: batch x timesteps x 1)
    X_seqs       : np.float32  (n_seqs, max_seq_len, n_features)
    y_flat       : np.float64  (total_actual_timesteps,)  — flat actual targets
    info_df      : DataFrame with 'actual_len' per sequence (index 0..n_seqs-1)
    feature_names: list of feature name strings
    n_repeats    : number of shuffles per feature
    random_state : seed for reproducibility
    batch_size   : passed to model.predict()
    '''
    rng = np.random.default_rng(random_state)
    actual_lens = info_df["actual_len"].astype(int).to_numpy()  # (n_seqs,)

    # Baseline RMSE on unshuffled validation sequences
    pred_base  = model.predict(X_seqs, batch_size=batch_size, verbose=0)
    pred_flat  = flatten_predictions(pred_base, info_df).astype(np.float64)
    baseline   = np.sqrt(mean_squared_error(y_flat, pred_flat))

    n_features = len(feature_names)
    imp_matrix = np.zeros((n_features, n_repeats))

    for j in range(n_features):
        # Collect actual (non-padded) values for feature j across all sequences.
        # Padded positions (index >= actual_len) are all-zero and are excluded.
        real_vals = np.concatenate([
            X_seqs[i, :actual_lens[i], j] for i in range(len(actual_lens))
        ])  # shape: (total_actual_timesteps,)

        for r in range(n_repeats):
            X_perm = X_seqs.copy()
            shuffled = rng.permutation(real_vals)

            # Write shuffled values back into non-padded positions only.
            # Padded positions remain 0 → Masking layer mask is unchanged.
            offset = 0
            for i in range(len(actual_lens)):
                alen = actual_lens[i]
                X_perm[i, :alen, j] = shuffled[offset:offset + alen]
                offset += alen

            pred_p      = model.predict(X_perm, batch_size=batch_size, verbose=0)
            pred_flat_p = flatten_predictions(pred_p, info_df).astype(np.float64)
            imp_matrix[j, r] = np.sqrt(mean_squared_error(y_flat, pred_flat_p)) - baseline

    means = pd.Series(imp_matrix.mean(axis=1), index=feature_names)
    stds  = pd.Series(imp_matrix.std(axis=1),  index=feature_names)
    return means.sort_values(ascending=False), stds


# Build flat y_val for importance computation
y_val_flat_64 = val_df.loc[val_df.index, "current_normalized"].values.astype(np.float64)
# Reorder to match sequence order (row_idx order)
y_val_seq_order = pd.Series(y_val_flat_64, index=val_df.index).reindex(idx_val).values

print(f"Computing LSTM permutation importance (n_repeats=10) ...")
print(f"  {10 * len(FEATURES)} model.predict() calls on X_val ({X_val.shape[0]} sequences each)")
t0 = time.time()
perm_means, perm_stds = lstm_permutation_importance(
    model, X_val, y_val_seq_order, info_val, FEATURES,
    n_repeats=10, random_state=42,
)
print(f"  Done in {time.time()-t0:.1f} s")
print()
print("LSTM Permutation Importance ranking (val sequences):")
for feat in perm_means.index:
    print(f"  {feat:35s}: {perm_means[feat]:.6f} +/- {perm_stds[feat]:.6f}")""")

code("""\
# ── Figure 08: LSTM permutation feature importance ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
xerr = perm_stds.loc[perm_means.index][::-1].values
ax.barh(perm_means.index[::-1], perm_means.values[::-1],
        xerr=xerr, color="#17becf", alpha=0.85,
        error_kw={"elinewidth": 1.2, "capsize": 3})
ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Permutation importance — RMSE increase when feature is shuffled", fontsize=9)
ax.set_title(
    "LSTM Permutation Feature Importance (val sequences, n_repeats=10)\\n"
    "Shuffling destroys both feature value AND temporal pattern", fontsize=9, pad=8)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_lstm_feature_importance.png", bbox_inches="tight")
plt.show()
print("Saved figures/08_lstm_feature_importance.png")
print()
print("Scientific note:")
print("  LSTM importance reflects both the feature VALUE and its TEMPORAL PATTERN.")
print("  direction_x_potential_norm captures the CV hysteresis interaction.")
print("  sweep_position is likely more important for LSTM than row-wise models")
print("  because the LSTM uses it to track progression within the trajectory.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8e — PER-GROUP RMSE
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 8e — Per-Group Performance Breakdown")

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

# ── Figure 09: Per-group RMSE bars ────────────────────────────────────────────
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

fig.suptitle("Per-Group RMSE Breakdown — LSTM Model", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "09_per_group_rmse.png", bbox_inches="tight")
plt.show()
print("Saved figures/09_per_group_rmse.png")""")

code("""\
# ── Figure 10: NM4 scan-rate trend ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(test_mat_grp["scan_rate_mVs"], test_mat_grp["RMSE_uA"],
        "o-", color="#d62728", lw=1.8, ms=7)
ax.set_xlabel("Scan rate (mV/s)"); ax.set_ylabel("RMSE (uA)")
ax.set_title("NM4 — RMSE vs scan rate\\n(does error scale with scan rate?)")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(test_mat_grp["scan_rate_mVs"], test_mat_grp["R2"],
        "s-", color="#17becf", lw=1.8, ms=7)
ax.axhline(0, color="k", lw=0.8, ls="--", label="R2=0 (mean prediction)")
ax.set_xlabel("Scan rate (mV/s)"); ax.set_ylabel("R2")
ax.set_title("NM4 — R2 vs scan rate\\n(does shape fidelity degrade?)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("NM4 Scan-Rate Trend — LSTM Model", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "10_nm4_scan_rate_trend.png", bbox_inches="tight")
plt.show()
print("Saved figures/10_nm4_scan_rate_trend.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 9 — Save All Outputs")

code("""\
# ── Trained LSTM model (.keras format) ───────────────────────────────────────
model_path = MODEL_DIR / "lstm_model.keras"
model.save(str(model_path))
print(f"Saved model  : {model_path}")

# ── Training history JSON ─────────────────────────────────────────────────────
history_out = {k: [float(v) for v in vs] for k, vs in history.history.items()}
history_out["epochs_ran"]        = epochs_ran
history_out["best_epoch"]        = best_epoch
history_out["best_epoch_1indexed"]= best_epoch + 1
with open(MODEL_DIR / "training_history.json", "w") as fh:
    json.dump(history_out, fh, indent=2)
print(f"Saved history: {MODEL_DIR / 'training_history.json'}")

# ── Comprehensive metrics JSON ────────────────────────────────────────────────
lstm_cfg = {
    "architecture": [
        {"layer": "Masking",                "mask_value": 0.0},
        {"layer": "LSTM",    "units": 64,   "return_sequences": True},
        {"layer": "Dropout", "rate": 0.20},
        {"layer": "LSTM",    "units": 32,   "return_sequences": True},
        {"layer": "Dropout", "rate": 0.10},
        {"layer": "TimeDistributed(Dense)", "units": 16, "activation": "relu"},
        {"layer": "TimeDistributed(Dense)", "units": 1},
    ],
    "total_params":    int(model.count_params()),
    "optimizer":       "Adam",
    "learning_rate":   1e-3,
    "loss":            "mse",
    "max_seq_len":     max_seq_len,
    "min_seq_len":     min_seq_len,
}
training_cfg = {
    "epochs_max":  150, "epochs_ran": epochs_ran, "best_epoch": best_epoch,
    "batch_size":  16,
    "early_stopping_patience": 20, "reduce_lr_patience": 8, "reduce_lr_factor": 0.5,
    "best_val_mse_norm":  float(min(history.history["val_loss"])),
    "best_val_rmse_norm": float(np.sqrt(min(history.history["val_loss"]))),
    "n_train_seqs": int(X_train.shape[0]),
    "n_val_seqs":   int(X_val.shape[0]),
}
metrics_out = {
    "model":       "Stacked LSTM (TensorFlow/Keras)",
    "model_config": lstm_cfg,
    "training":    training_cfg,
    "metrics":     metrics_df.reset_index().to_dict(orient="records"),
    "per_group":   {"val": val_grp.to_dict(orient="records"),
                    "test_sr": test_sr_grp.to_dict(orient="records"),
                    "test_mat": test_mat_grp.to_dict(orient="records")},
    "feature_importance_permutation": {
        f: {"mean": float(perm_means.get(f, 0.0)), "std": float(perm_stds.get(f, 0.0))}
        for f in FEATURES},
}
metrics_path = METRICS_DIR / "lstm_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(metrics_out, fh, indent=2)
print(f"Saved metrics: {metrics_path}")

for name, df_grp in [("val",val_grp),("test_sr",test_sr_grp),("test_mat",test_mat_grp)]:
    path = CSV_DIR / f"per_group_rmse_{name}.csv"
    df_grp.to_csv(path, index=False)
    print(f"Saved {path.name}")

print()
print("✓ All outputs saved")
print(f"  model/   : lstm_model.keras + training_history.json")
print(f"  metrics/ : lstm_metrics.json")
print(f"  csv/     : 3 per-group RMSE tables")
print(f"  figures/ : 11 figures (00 through 10)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FOUR-MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 10 — Scientific Summary: RF vs XGBoost vs ANN vs LSTM

### Comparison methodology
All four models share:
- **Identical training data** (NM1/2/3 × SR∈{10,20,40,60,70,80,90,100}, 62,400 rows)
- **Identical feature set** (10 normalised features)
- **Identical evaluation partitions** (val SR=30, test-SR SR=50, test-MAT NM4)
- **Identical inverse transform** (per-group scaler parameters from Notebook 01)

The only difference is the **model representation**:
- RF/XGB/ANN treat each measurement row independently.
- LSTM treats each half-sweep as a 651-timestep sequence.

### Key scientific claims to verify from the comparison table
1. **Hysteresis learning**: LSTM val/test RMSE < ANN implies the temporal context
   adds information beyond the hand-crafted `direction_x_potential_norm` feature.
2. **Trajectory fidelity**: LSTM residuals near turning points should be smaller
   (compare Figure 07 across notebooks).
3. **Scan-rate interpolation (val/test_SR)**: LSTM learning sweep dynamics should
   help at unseen scan rates if scan-rate-dependent trajectory shape is generalizable.
4. **Material extrapolation (test_MAT)**: If NM4 dynamics differ qualitatively from
   NM1/2/3, LSTM may struggle — the hidden state generalises only if the dynamics
   are structurally similar across formulations.""")

code("""\
# ── RF baseline (Notebook 02, verified run) ───────────────────────────────────
RF_RESULTS = {
    "val  (SR=30 interpolation)"   : {"RMSE_uA": 26.59, "R2": 0.9851},
    "test_SR (SR=50 interpolation)": {"RMSE_uA": 43.71, "R2": 0.9777},
    "test_MAT (NM4 extrapolation)" : {"RMSE_uA": 33.65, "R2": 0.9707},
}

# ── Load XGBoost and ANN results if available ─────────────────────────────────
def load_model_results(results_dir, json_name):
    path = results_dir / json_name
    if not path.exists():
        return None, str(path)
    with open(path) as fh:
        data = json.load(fh)
    return {m["partition"]: {"RMSE_uA": m["RMSE_uA"], "R2": m["R2"]}
            for m in data["metrics"]}, None

XGB_RESULTS, xgb_msg = load_model_results(
    BASE_DIR / "results" / "xgboost" / "metrics", "xgboost_metrics.json")
ANN_RESULTS, ann_msg = load_model_results(
    BASE_DIR / "results" / "ann"     / "metrics", "ann_metrics.json")

for label, res, msg in [("XGBoost", XGB_RESULTS, xgb_msg),
                         ("ANN",     ANN_RESULTS, ann_msg)]:
    if res:
        print(f"✓ {label} metrics loaded")
    else:
        print(f"  {label} metrics not found ({msg})")
        print(f"    Run Notebook {'03' if label == 'XGBoost' else '04'} first.")

# ── Build four-model comparison table ─────────────────────────────────────────
PARTITIONS = [
    "val  (SR=30 interpolation)",
    "test_SR (SR=50 interpolation)",
    "test_MAT (NM4 extrapolation)",
]
PAD = 32

print()
print("=" * 105)
print("RF vs XGBoost vs ANN vs LSTM — RMSE (uA) COMPARISON")
print("=" * 105)
header = (f"  {'Partition':<{PAD}}  {'RF':>9}  {'XGBoost':>9}  "
          f"{'ANN':>9}  {'LSTM':>9}  {'Best':>8}")
print(header)
print("  " + "-" * 99)

for p in PARTITIONS:
    rf_r   = RF_RESULTS[p]["RMSE_uA"]
    xgb_r  = XGB_RESULTS[p]["RMSE_uA"] if XGB_RESULTS and p in XGB_RESULTS else None
    ann_r  = ANN_RESULTS[p]["RMSE_uA"] if ANN_RESULTS and p in ANN_RESULTS else None
    lstm_r = metrics_df.loc[p, "RMSE_uA"]

    vals  = {k: v for k, v in [("RF", rf_r), ("XGBoost", xgb_r),
                                 ("ANN", ann_r), ("LSTM", lstm_r)] if v is not None}
    best_name = min(vals, key=vals.get)

    def fmt(v): return f"{v:9.2f}" if v is not None else "      ---"
    row = (f"  {p:<{PAD}}  {fmt(rf_r)}  {fmt(xgb_r)}  "
           f"{fmt(ann_r)}  {fmt(lstm_r)}  {best_name:>8}")
    print(row)

print()
print("  Units: RMSE in micro-Amperes (uA).  Lower is better.")
print()

# ── R2 table ─────────────────────────────────────────────────────────────────
print("RF vs XGBoost vs ANN vs LSTM — R2 COMPARISON")
print("=" * 105)
header2 = (f"  {'Partition':<{PAD}}  {'RF':>8}  {'XGBoost':>8}  "
           f"{'ANN':>8}  {'LSTM':>8}")
print(header2)
print("  " + "-" * 85)
for p in PARTITIONS:
    rf_r2   = RF_RESULTS[p]["R2"]
    xgb_r2  = XGB_RESULTS[p]["R2"] if XGB_RESULTS and p in XGB_RESULTS else None
    ann_r2  = ANN_RESULTS[p]["R2"] if ANN_RESULTS and p in ANN_RESULTS else None
    lstm_r2 = metrics_df.loc[p, "R2"]
    def fmtr(v): return f"{v:8.4f}" if v is not None else "     ---"
    print(f"  {p:<{PAD}}  {fmtr(rf_r2)}  {fmtr(xgb_r2)}  {fmtr(ann_r2)}  {fmtr(lstm_r2)}")

print()
print("  R2: higher is better.  R2 = 1.0 = perfect.  R2 = 0.0 = predicts the mean.")
print()

# ── LSTM-specific verdict ─────────────────────────────────────────────────────
lstm_vs_rf = sum(
    metrics_df.loc[p, "RMSE_uA"] < RF_RESULTS[p]["RMSE_uA"]
    for p in PARTITIONS)
print(f"  LSTM improved on {lstm_vs_rf}/3 partitions vs RF baseline.")

if ANN_RESULTS:
    lstm_vs_ann = sum(
        metrics_df.loc[p, "RMSE_uA"] < ANN_RESULTS[p]["RMSE_uA"]
        for p in PARTITIONS if p in ANN_RESULTS)
    print(f"  LSTM improved on {lstm_vs_ann}/3 partitions vs ANN baseline.")
    if lstm_vs_ann >= 2:
        print("  => Temporal modelling adds measurable value over row-wise ANN.")
    else:
        print("  => Feature engineering (direction_x_potential_norm) may already")
        print("     encode most of the trajectory information accessible to LSTM.")

print()
print("Pipeline complete.")
print("  Notebooks 02–05 establish four scientifically honest baselines.")
print("  All use the same preprocessing, splits, features, and evaluation.")
print("  Model complexity order: RF < XGBoost < ANN < LSTM (sequence-aware)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 11 — Download Results to Your Local Machine

Run this cell **before closing the Colab session**.
All files are lost when the runtime disconnects.

The cell zips the entire `results/lstm/` folder and triggers a browser download
on Colab.  On local execution it simply lists the saved files.""")

code("""\
import shutil, os

if EXECUTION_MODE in ("colab_upload", "colab_drive"):
    from google.colab import files as _cf

    zip_base = str(BASE_DIR / "lstm_results")
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

out_path = Path(__file__).resolve().parent.parent.parent / "research" / "notebooks" / "05_lstm_model.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out_path))
print(f"Written : {out_path}")
print(f"Cells   : {len(nb.cells)}")
