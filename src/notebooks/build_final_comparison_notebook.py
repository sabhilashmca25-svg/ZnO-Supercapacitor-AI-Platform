"""
build_final_comparison_notebook.py
===================================
Generates  notebooks/08_final_comparison.ipynb

Publication-quality final comparative analysis of all 6 trained models:
Random Forest · XGBoost · LightGBM · ANN · LSTM · GRU

Run once from the project root:
    python src/build_final_comparison_notebook.py
"""
import nbformat as nbf
from pathlib import Path

ROOT   = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)
NB_PATH = NB_DIR / "08_final_comparison.ipynb"

nb    = nbf.v4.new_notebook()
cells = []

def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))


# ══════════════════════════════════════════════════════════════════════════════
# CELL 01 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
md("""\
# Notebook 08 — Final Model Comparison
## ZnO Supercapacitor CV Trajectory Prediction: Comprehensive Multi-Model Analysis

**Project**: AI-Based Mobile Application for Predicting Supercapacitor Performance Using Machine Learning
**Dataset**: ZnO-based supercapacitor cyclic voltammetry (CV) curves — NM1 / NM2 / NM3 / NM4
**Task**: Predict the full current–voltage trajectory from scan-rate and electrode-potential features
**Models compared**: Random Forest · XGBoost · LightGBM · ANN · LSTM · GRU

---

> **What this notebook does**
> Loads the six pre-computed `*_metrics.json` result files from all prior notebooks
> and produces **twelve publication-quality comparison figures**, a ranked leaderboard,
> a **twelve-point scientific analysis**, and **six export files**.
> No raw parquet data, no model weights, and no GPU are required.

### Three evaluation partitions
| Partition | Split | Purpose |
|-----------|-------|---------|
| **Val** | NM1/2/3 × SR=30 mV/s | Scan-rate interpolation (held out during training) |
| **Test-SR** | NM1/2/3 × SR=50 mV/s | Scan-rate interpolation (unseen rate, seen materials) |
| **Test-MAT** | NM4 × all 10 scan rates | Material extrapolation (completely unseen ZnO morphology) |""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 02 — HOW TO RUN
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## How to Run This Notebook

### Option A — Google Colab (recommended)
1. Upload the **six `*_metrics.json` files** using the upload cell below
   *(only ~6 KB each — not the large parquet files or model weights)*
2. **Runtime → Run all**

### Option B — Google Colab + Drive
Set `DRIVE_MODE = True` and `DRIVE_RESULTS_PATH` in the config cell, then **Run all**.

### Option C — Local Python (Jupyter / VS Code)
Run locally — the notebook auto-detects paths under `results/`.

---
### Exactly which files do you need to upload? (6 total)

| File | Location in project |
|------|---------------------|
| `rf_baseline_metrics.json` | `results/rf_baseline/` |
| `xgboost_metrics.json` | `results/xgboost/metrics/` |
| `lightgbm_metrics.json` | `results/lightgbm/metrics/` |
| `ann_metrics.json` | `results/ann/metrics/` |
| `lstm_metrics.json` | `results/lstm/metrics/` |
| `gru_metrics.json` | `results/gru/metrics/` |

> **Tip**: In your file manager, navigate into each folder in turn and copy the JSON file
> to a single staging folder. Then upload all 6 at once in the dialog below.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 03 — IMPORTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════
code("""\
import json, os, sys, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "figure.dpi":      150,
    "savefig.dpi":     200,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Colab detection ──────────────────────────────────────────────────────────
try:
    from google.colab import files as _cf
    IN_COLAB = True
except ImportError:
    _cf      = None
    IN_COLAB = False

# ── User configuration — edit only these two lines if needed ─────────────────
DRIVE_MODE         = False
DRIVE_RESULTS_PATH = "/content/drive/MyDrive/PBL_Project/results"

# ── Model metadata (fixed constants) ─────────────────────────────────────────
MODEL_ORDER = ["RF", "XGBoost", "LightGBM", "ANN", "LSTM", "GRU"]

COLORS = {
    "RF":       "#2196F3",   # blue
    "XGBoost":  "#FF5722",   # deep-orange
    "LightGBM": "#4CAF50",   # green
    "ANN":      "#9C27B0",   # purple
    "LSTM":     "#FF9800",   # amber
    "GRU":      "#00BCD4",   # cyan
}

# Approximate wall-clock training times (minutes) — not stored in JSON files
TRAIN_TIME_MIN = {
    "RF": 4, "XGBoost": 6, "LightGBM": 2, "ANN": 10, "LSTM": 25, "GRU": 20,
}

# Serialised model file sizes (MB)
MODEL_SIZE_MB = {
    "RF": 615.0, "XGBoost": 0.322, "LightGBM": 1.6,
    "ANN": 0.172, "LSTM": 0.424, "GRU": 0.335,
}

# Approximate learnable parameters / number of trees (best iteration)
COMPLEXITY = {
    "RF": 300, "XGBoost": 193, "LightGBM": 278,
    "ANN": 11_777, "LSTM": 32_161, "GRU": 24_545,
}
COMPLEXITY_LABEL = {
    "RF":       "300 trees",
    "XGBoost":  "193 trees",
    "LightGBM": "278 trees",
    "ANN":      "11.8k params",
    "LSTM":     "32.2k params",
    "GRU":      "24.5k params",
}

print("Imports OK — Colab:", IN_COLAB)
print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M"))""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 04 — OUTPUT DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════
code("""\
if IN_COLAB:
    OUT_ROOT = Path("/content/final_comparison")
else:
    # Local: save alongside other results
    _local_project = Path(__file__).parent.parent if "__file__" in dir() else Path(".")
    OUT_ROOT = _local_project / "results" / "final_comparison"

FIG_DIR = OUT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Output root :", OUT_ROOT.resolve())
print("Figures dir :", FIG_DIR.resolve())""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 05 — SECTION 1 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 1 — Upload / Locate the Six Metrics JSON Files

Run the cell below.
- **Colab**: a file-picker dialog will open — select all 6 `*_metrics.json` files.
- **Local**: the notebook auto-discovers them from `results/`.

Only the six tiny JSON files are needed — no parquet data, no model weights.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 06 — UPLOAD / DISCOVER JSONs
# ══════════════════════════════════════════════════════════════════════════════
code("""\
EXPECTED_NAMES = [
    "rf_baseline_metrics.json",
    "xgboost_metrics.json",
    "lightgbm_metrics.json",
    "ann_metrics.json",
    "lstm_metrics.json",
    "gru_metrics.json",
]

JSON_PATHS = {}   # fname  →  Path

# ── helper: auto-discover local paths ────────────────────────────────────────
def _find_local_jsons():
    candidates = {
        "rf_baseline_metrics.json": Path("results/rf_baseline/rf_baseline_metrics.json"),
        "xgboost_metrics.json":     Path("results/xgboost/metrics/xgboost_metrics.json"),
        "lightgbm_metrics.json":    Path("results/lightgbm/metrics/lightgbm_metrics.json"),
        "ann_metrics.json":         Path("results/ann/metrics/ann_metrics.json"),
        "lstm_metrics.json":        Path("results/lstm/metrics/lstm_metrics.json"),
        "gru_metrics.json":         Path("results/gru/metrics/gru_metrics.json"),
    }
    return {k: v for k, v in candidates.items() if v.exists()}

# ── branch on execution environment ──────────────────────────────────────────
if IN_COLAB and DRIVE_MODE:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    base = Path(DRIVE_RESULTS_PATH)
    JSON_PATHS = {
        "rf_baseline_metrics.json": base / "rf_baseline" / "rf_baseline_metrics.json",
        "xgboost_metrics.json":     base / "xgboost"    / "metrics" / "xgboost_metrics.json",
        "lightgbm_metrics.json":    base / "lightgbm"   / "metrics" / "lightgbm_metrics.json",
        "ann_metrics.json":         base / "ann"         / "metrics" / "ann_metrics.json",
        "lstm_metrics.json":        base / "lstm"        / "metrics" / "lstm_metrics.json",
        "gru_metrics.json":         base / "gru"         / "metrics" / "gru_metrics.json",
    }
    print("Drive mode — reading from:", DRIVE_RESULTS_PATH)

elif IN_COLAB and not DRIVE_MODE:
    print("Upload all 6  *_metrics.json  files (select all in one dialog)...")
    uploaded = _cf.upload()
    for fname, data in uploaded.items():
        p = Path("/content") / fname
        p.write_bytes(data)
        JSON_PATHS[fname] = p
    missing = [n for n in EXPECTED_NAMES if n not in JSON_PATHS]
    if missing:
        print("WARNING — not uploaded:", missing)
        print("Re-run this cell and upload the missing files.")
    else:
        print("All 6 JSON files received.")

else:
    JSON_PATHS = _find_local_jsons()
    if len(JSON_PATHS) == 6:
        print("Auto-located all 6 JSON files locally.")
    else:
        print(f"Found {len(JSON_PATHS)}/6 locally. Check results/ folder.")

print()
for name in EXPECTED_NAMES:
    p = JSON_PATHS.get(name)
    ok = p is not None and Path(p).exists()
    print(f"  {'OK' if ok else 'MISSING':<7} {name}")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 07 — LOAD ALL 6 JSONs
# ══════════════════════════════════════════════════════════════════════════════
code("""\
def _load_json(name):
    p = JSON_PATHS.get(name)
    if p is None or not Path(p).exists():
        raise FileNotFoundError(
            f"Cannot find {name}.  Re-run the upload cell."
        )
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)

raw = {
    "RF":       _load_json("rf_baseline_metrics.json"),
    "XGBoost":  _load_json("xgboost_metrics.json"),
    "LightGBM": _load_json("lightgbm_metrics.json"),
    "ANN":      _load_json("ann_metrics.json"),
    "LSTM":     _load_json("lstm_metrics.json"),
    "GRU":      _load_json("gru_metrics.json"),
}

print("All 6 metrics JSONs loaded successfully.")
for model, d in raw.items():
    print(f"  {model:<10}  model={d.get('model', 'N/A')!r}")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 08 — PARSE METRICS INTO DataFrames
# ══════════════════════════════════════════════════════════════════════════════
code("""\
# Exact partition label strings used in all JSON files
PART_MAP = {
    "train":    "train (in-sample)",
    "val":      "val  (SR=30 interpolation)",    # NOTE: two spaces before (SR
    "test_SR":  "test_SR (SR=50 interpolation)",
    "test_MAT": "test_MAT (NM4 extrapolation)",
}

def get_metric(metrics_list, partition_label, metric, default=np.nan):
    for row in metrics_list:
        if row["partition"] == partition_label:
            return row.get(metric, default)
    return default

# ── Build long-format summary DataFrame ──────────────────────────────────────
rows = []
for model in MODEL_ORDER:
    ml = raw[model]["metrics"]
    for part_key, part_label in PART_MAP.items():
        rows.append({
            "Model":     model,
            "Partition": part_key,
            "RMSE_uA":   get_metric(ml, part_label, "RMSE_uA"),
            "MAE_uA":    get_metric(ml, part_label, "MAE_uA"),
            "MaxErr_uA": get_metric(ml, part_label, "MaxErr_uA"),
            "R2":        get_metric(ml, part_label, "R2"),
            "RMSE_norm": get_metric(ml, part_label, "RMSE_norm"),
        })

df_all = pd.DataFrame(rows)

# ── Pivot tables: rows=model, cols=partition ──────────────────────────────────
def pivot(metric):
    return df_all.pivot(index="Model", columns="Partition",
                        values=metric).reindex(MODEL_ORDER)

pv_rmse = pivot("RMSE_uA")
pv_mae  = pivot("MAE_uA")
pv_maxe = pivot("MaxErr_uA")
pv_r2   = pivot("R2")

# ── NM4 per-scan-rate data ────────────────────────────────────────────────────
pg_rows = []
for model in MODEL_ORDER:
    for rec in raw[model].get("per_group", {}).get("test_mat", []):
        pg_rows.append({
            "Model":         model,
            "scan_rate_mVs": rec["scan_rate_mVs"],
            "RMSE_uA":       rec["RMSE_uA"],
            "R2":            rec["R2"],
        })
df_pg = pd.DataFrame(pg_rows)

print("Long-format table:", df_all.shape, "rows")
print("NM4 per-scan-rate:", df_pg.shape, "rows")
print()
print("RMSE (µA) pivot table:")
print(pv_rmse[["val", "test_SR", "test_MAT"]].round(2).to_string())""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 09 — OVERVIEW TABLE HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 2 — Metrics Overview

Full RMSE and R² summary table for all six models across three evaluation partitions.
This is the primary reference table for all figures that follow.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — PRINT OVERVIEW TABLE
# ══════════════════════════════════════════════════════════════════════════════
code("""\
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 110)

divider = "=" * 88

print(divider)
print("  RMSE (µA)  —  lower is better")
print(divider)
tmp = pv_rmse[["val", "test_SR", "test_MAT"]].copy()
tmp.columns = ["Val (SR=30)", "Test-SR (SR=50)", "Test-MAT (NM4)"]
print(tmp.round(2).to_string())

print()
print(divider)
print("  MAE (µA)  —  lower is better")
print(divider)
tmp2 = pv_mae[["val", "test_SR", "test_MAT"]].copy()
tmp2.columns = ["Val (SR=30)", "Test-SR (SR=50)", "Test-MAT (NM4)"]
print(tmp2.round(2).to_string())

print()
print(divider)
print("  R²  —  higher is better")
print(divider)
tmp3 = pv_r2[["val", "test_SR", "test_MAT"]].copy()
tmp3.columns = ["Val (SR=30)", "Test-SR (SR=50)", "Test-MAT (NM4)"]
print(tmp3.round(4).to_string())""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 — FIGURE 01 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 3 — Figure 01: RMSE Comparison Across All Partitions

Grouped bar chart showing RMSE (µA) for each model across all three held-out
evaluation partitions.  Lower bars indicate better predictions.
Numbers above each bar show the exact RMSE value in µA.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 12 — FIGURE 01: RMSE GROUPED BAR
# ══════════════════════════════════════════════════════════════════════════════
code("""\
parts  = ["val", "test_SR", "test_MAT"]
labels = ["Val  (SR=30 interp.)", "Test-SR  (SR=50 interp.)", "Test-MAT  (NM4 extrap.)"]
n      = len(MODEL_ORDER)
w      = 0.13
x      = np.arange(len(parts))
offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

fig, ax = plt.subplots(figsize=(13, 6))
for i, model in enumerate(MODEL_ORDER):
    vals = [pv_rmse.loc[model, p] for p in parts]
    bars = ax.bar(x + offsets[i], vals, w, label=model,
                  color=COLORS[model], alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("RMSE (µA)", fontsize=12)
ax.set_title(
    "Figure 01 — RMSE Comparison: All Models × All Evaluation Partitions",
    fontsize=13, fontweight="bold", pad=12,
)
ax.legend(title="Model", loc="upper left", fontsize=9, title_fontsize=9)
ax.set_ylim(0, pv_rmse[parts].max().max() * 1.28)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig01_rmse_comparison.png")
plt.show()
print("Figure 01 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 13 — FIGURE 02 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 4 — Figure 02: R² Comparison Across All Partitions

R² (coefficient of determination) measures how well the model explains the variance
of the measured current signal.  An R² > 0.95 is considered excellent for CV trajectory
prediction.  Y-axis is zoomed to [0.92, 1.00] for visual clarity.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 14 — FIGURE 02: R² GROUPED BAR
# ══════════════════════════════════════════════════════════════════════════════
code("""\
fig, ax = plt.subplots(figsize=(13, 6))
for i, model in enumerate(MODEL_ORDER):
    vals = [pv_r2.loc[model, p] for p in parts]
    bars = ax.bar(x + offsets[i], vals, w, label=model,
                  color=COLORS[model], alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0004,
                f"{v:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("R²", fontsize=12)
ax.set_title(
    "Figure 02 — R² Comparison: All Models × All Evaluation Partitions",
    fontsize=13, fontweight="bold", pad=12,
)
ax.legend(title="Model", loc="lower left", fontsize=9, title_fontsize=9)
ax.set_ylim(0.92, 1.003)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig02_r2_comparison.png")
plt.show()
print("Figure 02 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 15 — FIGURE 03 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 5 — Figure 03: MAE Comparison

Mean Absolute Error is less sensitive to outliers than RMSE and directly represents
the average point-by-point current prediction error in µA.
Comparing MAE to RMSE (Figures 01 vs 03) indicates the skewness of the error distribution.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 16 — FIGURE 03: MAE GROUPED BAR
# ══════════════════════════════════════════════════════════════════════════════
code("""\
fig, ax = plt.subplots(figsize=(13, 6))
for i, model in enumerate(MODEL_ORDER):
    vals = [pv_mae.loc[model, p] for p in parts]
    bars = ax.bar(x + offsets[i], vals, w, label=model,
                  color=COLORS[model], alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("MAE (µA)", fontsize=12)
ax.set_title(
    "Figure 03 — MAE Comparison: All Models × All Evaluation Partitions",
    fontsize=13, fontweight="bold", pad=12,
)
ax.legend(title="Model", loc="upper left", fontsize=9, title_fontsize=9)
ax.set_ylim(0, pv_mae[parts].max().max() * 1.28)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig03_mae_comparison.png")
plt.show()
print("Figure 03 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 17 — FIGURE 04 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 6 — Figure 04: Maximum Error Comparison

The maximum absolute error (worst-case single-point prediction error) is critical
for automated CV integration workflows, where large outlier errors can produce
significant artefacts in computed capacitance or charge values.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 18 — FIGURE 04: MaxErr GROUPED BAR
# ══════════════════════════════════════════════════════════════════════════════
code("""\
fig, ax = plt.subplots(figsize=(13, 6))
for i, model in enumerate(MODEL_ORDER):
    vals = [pv_maxe.loc[model, p] for p in parts]
    bars = ax.bar(x + offsets[i], vals, w, label=model,
                  color=COLORS[model], alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Max Absolute Error (µA)", fontsize=12)
ax.set_title(
    "Figure 04 — Maximum Error Comparison: All Models × All Evaluation Partitions",
    fontsize=13, fontweight="bold", pad=12,
)
ax.legend(title="Model", loc="upper left", fontsize=9, title_fontsize=9)
ax.set_ylim(0, pv_maxe[parts].max().max() * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig04_maxerr_comparison.png")
plt.show()
print("Figure 04 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 19 — FIGURE 05 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 7 — Figure 05: Training Efficiency — Time vs Validation RMSE

Scatter plot mapping approximate training time against validation RMSE.
Models in the **lower-left quadrant** offer the best accuracy-for-compute ratio.
Dotted cross-hairs mark the group mean on each axis.

> **Note**: Training times are approximate wall-clock estimates; they were not
> instrumented in the training scripts. GPU availability and batch sizes affect
> actual times significantly.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 20 — FIGURE 05: TIME vs VAL RMSE SCATTER
# ══════════════════════════════════════════════════════════════════════════════
code("""\
fig, ax = plt.subplots(figsize=(9, 6))
for model in MODEL_ORDER:
    t = TRAIN_TIME_MIN[model]
    r = pv_rmse.loc[model, "val"]
    ax.scatter(t, r, color=COLORS[model], s=220, zorder=5,
               edgecolors="white", linewidth=1.5)
    ax.annotate(model, (t, r), textcoords="offset points", xytext=(8, 4),
                fontsize=10, color=COLORS[model], fontweight="bold")

mean_t = np.mean(list(TRAIN_TIME_MIN.values()))
mean_r = pv_rmse["val"].mean()
ax.axvline(mean_t, color="gray", linestyle=":", alpha=0.55, linewidth=1.2)
ax.axhline(mean_r, color="gray", linestyle=":", alpha=0.55, linewidth=1.2)
ax.text(0.02, 0.97, "faster  &  accurate",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, color="gray", style="italic")

ax.set_xlabel("Approximate Training Time (minutes)", fontsize=12)
ax.set_ylabel("Validation RMSE (µA)", fontsize=12)
ax.set_title(
    "Figure 05 — Training Efficiency: Time vs Validation RMSE",
    fontsize=13, fontweight="bold", pad=12,
)
ax.grid(linestyle="--", alpha=0.35)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig05_training_time_vs_rmse.png")
plt.show()
print("Figure 05 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 21 — FIGURE 06 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 8 — Figure 06: Deployment Footprint — Model Size vs Extrapolation RMSE

Practical deployment consideration: serialised model file size on disk (log scale)
vs extrapolation accuracy on the held-out material NM4.
**Bubble area** is proportional to log(model size).
RF's 615 MB serialisation is orders of magnitude larger than all other models.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 22 — FIGURE 06: SIZE vs RMSE SCATTER
# ══════════════════════════════════════════════════════════════════════════════
code("""\
sizes_mb = [MODEL_SIZE_MB[m] for m in MODEL_ORDER]
rmse_mat = [pv_rmse.loc[m, "test_MAT"] for m in MODEL_ORDER]
log_max  = np.log1p(max(sizes_mb))
bubbles  = [max(80, 1400 * np.log1p(s) / log_max) for s in sizes_mb]

fig, ax = plt.subplots(figsize=(9, 6))
for i, model in enumerate(MODEL_ORDER):
    ax.scatter(sizes_mb[i], rmse_mat[i], color=COLORS[model],
               s=bubbles[i], zorder=5, edgecolors="white", linewidth=1.5, alpha=0.85)
    ax.annotate(
        f"{model}\\n({sizes_mb[i]:.1f} MB)",
        (sizes_mb[i], rmse_mat[i]),
        textcoords="offset points", xytext=(9, 3),
        fontsize=9, color=COLORS[model], fontweight="bold",
    )

ax.set_xscale("log")
ax.set_xlabel("Model Size on Disk (MB, log scale)", fontsize=12)
ax.set_ylabel("Test-MAT RMSE (µA)", fontsize=12)
ax.set_title(
    "Figure 06 — Deployment Footprint: Model Size vs Extrapolation Accuracy",
    fontsize=13, fontweight="bold", pad=12,
)
ax.grid(linestyle="--", alpha=0.35, which="both")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig06_model_size_vs_rmse.png")
plt.show()
print("Figure 06 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 23 — FIGURE 07 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 9 — Figure 07: Multi-Metric Radar Chart

Spider/radar chart normalising six performance dimensions to [0, 1].
**Larger filled area = better overall profile.**

| Radar axis | Raw metric | Direction |
|-----------|-----------|-----------|
| Val R² | R² on validation set | Higher is better |
| Val RMSE (inv.) | RMSE on validation set | Lower raw = higher score |
| Test-SR RMSE (inv.) | RMSE on scan-rate test | Lower raw = higher score |
| Test-MAT RMSE (inv.) | RMSE on material test | Lower raw = higher score |
| Test-MAT R² | R² on material test | Higher is better |
| Training Efficiency | 1/training_time | Faster = higher score |""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 24 — FIGURE 07: RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════
code("""\
import math

RADAR_LABELS = [
    "Val R²",
    "Val RMSE\\n(inv.)",
    "Test-SR RMSE\\n(inv.)",
    "Test-MAT RMSE\\n(inv.)",
    "Test-MAT R²",
    "Training\\nEfficiency",
]
N_AXES = len(RADAR_LABELS)

# Collect raw values per model: [val_r2, val_rmse, tsr_rmse, tmat_rmse, tmat_r2, train_min]
def _raw_radar(model):
    return [
        pv_r2.loc[model,   "val"],
        pv_rmse.loc[model, "val"],
        pv_rmse.loc[model, "test_SR"],
        pv_rmse.loc[model, "test_MAT"],
        pv_r2.loc[model,   "test_MAT"],
        TRAIN_TIME_MIN[model],
    ]

raw_mat  = np.array([_raw_radar(m) for m in MODEL_ORDER])   # (6, 6)
ax_min   = raw_mat.min(axis=0)
ax_max   = raw_mat.max(axis=0)
ax_rng   = np.where(ax_max - ax_min == 0, 1e-9, ax_max - ax_min)

# Axes 0 and 4 (R²): higher raw = higher score
# Axes 1,2,3,5 (RMSE, time): lower raw = higher score
INVERT = np.array([False, True, True, True, False, True])

def _normalize_radar(model):
    raw   = np.array(_raw_radar(model), dtype=float)
    score = (raw - ax_min) / ax_rng
    score[INVERT] = 1.0 - score[INVERT]
    return np.clip(score, 0.0, 1.0)

norm_scores = {m: _normalize_radar(m) for m in MODEL_ORDER}

angles = np.linspace(0, 2 * math.pi, N_AXES, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(math.pi / 2)
ax.set_theta_direction(-1)

for model in MODEL_ORDER:
    vals = norm_scores[model].tolist() + [norm_scores[model][0]]
    ax.plot(angles, vals, "-o", color=COLORS[model], linewidth=2.2,
            markersize=5.5, label=model)
    ax.fill(angles, vals, color=COLORS[model], alpha=0.10)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_LABELS, size=9.5)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=8, color="gray")
ax.set_title(
    "Figure 07 — Multi-Metric Radar Chart\\n(normalized 0-1; outer ring = best)",
    fontsize=13, fontweight="bold", pad=22,
)
ax.legend(loc="lower right", bbox_to_anchor=(1.40, -0.08), fontsize=9)
ax.grid(alpha=0.30)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig07_radar_chart.png")
plt.show()
print("Figure 07 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 25 — FIGURE 08 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 10 — Figure 08: Performance Heatmap

Colour-coded heatmap of RMSE (µA) across all six models and four evaluation partitions
(including the in-sample training partition).
Darker colour intensity = higher error = worse performance.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 26 — FIGURE 08: HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
code("""\
all_parts  = ["train", "val", "test_SR", "test_MAT"]
all_labels = ["Train\\n(in-sample)", "Val\\n(SR=30)", "Test-SR\\n(SR=50)", "Test-MAT\\n(NM4)"]
matrix = pv_rmse[all_parts].values.astype(float)

try:
    import seaborn as sns
    _has_sns = True
except ImportError:
    _has_sns = False

fig, ax = plt.subplots(figsize=(9, 5.5))

if _has_sns:
    sns.heatmap(
        pd.DataFrame(matrix, index=MODEL_ORDER, columns=all_labels),
        annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "RMSE (µA)", "shrink": 0.80},
        ax=ax,
    )
else:
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="RMSE (µA)", shrink=0.80)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    pct60 = np.percentile(matrix, 60)
    for r in range(len(MODEL_ORDER)):
        for c in range(len(all_parts)):
            clr = "white" if matrix[r, c] > pct60 else "black"
            ax.text(c, r, f"{matrix[r,c]:.1f}", ha="center", va="center",
                    fontsize=10, color=clr)

ax.set_title(
    "Figure 08 — RMSE (µA) Heatmap: Models × Partitions",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("")
ax.set_ylabel("")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig08_performance_heatmap.png")
plt.show()
print("Figure 08 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 27 — FIGURE 09 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 11 — Figure 09: NM4 Material Extrapolation by Scan Rate

Line plots showing RMSE and R² on the NM4 held-out material across all 10 scan rates
(10, 20, 30 … 100 mV/s).  This reveals how each model's extrapolation accuracy evolves
with scan rate — a key question for the mobile prediction application.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 28 — FIGURE 09: NM4 SCAN RATE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for model in MODEL_ORDER:
    sub = df_pg[df_pg["Model"] == model].sort_values("scan_rate_mVs")
    ax.plot(sub["scan_rate_mVs"], sub["RMSE_uA"], "-o",
            color=COLORS[model], label=model, linewidth=1.9, markersize=5.5)
ax.set_xlabel("Scan Rate (mV/s)", fontsize=11)
ax.set_ylabel("RMSE (µA)", fontsize=11)
ax.set_title("NM4 RMSE vs Scan Rate", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5)
ax.grid(linestyle="--", alpha=0.35)
ax.set_xticks(range(10, 110, 10))

ax = axes[1]
for model in MODEL_ORDER:
    sub = df_pg[df_pg["Model"] == model].sort_values("scan_rate_mVs")
    ax.plot(sub["scan_rate_mVs"], sub["R2"], "-o",
            color=COLORS[model], label=model, linewidth=1.9, markersize=5.5)
ax.set_xlabel("Scan Rate (mV/s)", fontsize=11)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("NM4 R² vs Scan Rate", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5)
ax.grid(linestyle="--", alpha=0.35)
ax.set_xticks(range(10, 110, 10))

fig.suptitle(
    "Figure 09 — NM4 Material Extrapolation Performance by Scan Rate",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig09_nm4_extrapolation_by_scan_rate.png")
plt.show()
print("Figure 09 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 29 — FIGURE 10 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 12 — Figure 10: Generalisation Gap Analysis

Key question: does performance degrade progressively as we move from training data
(seen samples) → validation (unseen scan rate) → test-SR (further unseen SR) → test-MAT
(completely unseen material)?

A flat or slowly rising line indicates **robust generalisation**.
A steep rise indicates **domain shift sensitivity**.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 30 — FIGURE 10: GENERALISATION GAP
# ══════════════════════════════════════════════════════════════════════════════
code("""\
part_order  = ["train", "val", "test_SR", "test_MAT"]
part_ticks  = ["Train\\n(in-sample)", "Val\\n(SR=30)", "Test-SR\\n(SR=50)", "Test-MAT\\n(NM4)"]
xp = np.arange(len(part_order))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for model in MODEL_ORDER:
    vals = [pv_rmse.loc[model, p] for p in part_order]
    ax.plot(xp, vals, "-o", color=COLORS[model], label=model,
            linewidth=2.0, markersize=6.5)
ax.set_xticks(xp)
ax.set_xticklabels(part_ticks, fontsize=10)
ax.set_ylabel("RMSE (µA)", fontsize=11)
ax.set_title("RMSE Across All Partitions", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(linestyle="--", alpha=0.35)

ax = axes[1]
for model in MODEL_ORDER:
    vals = [pv_r2.loc[model, p] for p in part_order]
    ax.plot(xp, vals, "-o", color=COLORS[model], label=model,
            linewidth=2.0, markersize=6.5)
ax.set_xticks(xp)
ax.set_xticklabels(part_ticks, fontsize=10)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("R² Across All Partitions", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower left")
ax.grid(linestyle="--", alpha=0.35)

fig.suptitle(
    "Figure 10 — Generalisation Gap: Train → Val → Test-SR → Test-MAT",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig10_generalisation_gap.png")
plt.show()
print("Figure 10 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 31 — FIGURE 11 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 13 — Figure 11: Error Spread on Test-MAT (RMSE / MAE / Max Error)

Three-panel chart for the most critical partition (Test-MAT, NM4 extrapolation),
comparing RMSE, MAE, and maximum absolute error side-by-side.

- **MAE/RMSE ratio close to 1.0** → errors are uniformly distributed
- **MAE/RMSE ratio << 1.0** → a few large spikes dominate the RMSE
- **Max Error** reveals worst-case outlier behaviour, critical for integration workflows""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 32 — FIGURE 11: ERROR SPREAD SUBPLOTS
# ══════════════════════════════════════════════════════════════════════════════
code("""\
clrs = [COLORS[m] for m in MODEL_ORDER]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric, title, ylabel in zip(
        axes,
        ["RMSE_uA", "MAE_uA", "MaxErr_uA"],
        ["RMSE", "MAE", "Max Abs Error"],
        ["RMSE (µA)", "MAE (µA)", "Max Error (µA)"],
):
    vals = [
        float(df_all.query("Model == @m and Partition == 'test_MAT'")[metric].values[0])
        for m in MODEL_ORDER
    ]
    bars = ax.bar(MODEL_ORDER, vals, color=clrs, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ceiling = max(vals) * 1.25
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ceiling * 0.012,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, ceiling)
    ax.set_xticklabels(MODEL_ORDER, rotation=28, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

fig.suptitle(
    "Figure 11 — Error Spread on Test-MAT (NM4 Material Extrapolation)",
    fontsize=13, fontweight="bold", y=1.02,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig11_error_spread_test_mat.png")
plt.show()
print("Figure 11 saved.")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 33 — FIGURE 12 HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 14 — Figure 12: Accuracy vs Model Complexity

Scatter plot mapping model complexity (x-axis, log scale) against Test-MAT RMSE
(y-axis).  Complexity is measured as:

- **Tree models**: number of trees at best iteration (100s scale)
- **Neural networks**: total trainable parameters (10,000s scale)

**Bubble area** is proportional to log(model disk size in MB), so RF's enormous bubble
immediately communicates its deployment impracticality.

The **lower-left region** of the chart represents the ideal trade-off:
low complexity, low extrapolation error.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 34 — FIGURE 12: ACCURACY vs COMPLEXITY
# ══════════════════════════════════════════════════════════════════════════════
code("""\
log_max_size = np.log1p(max(MODEL_SIZE_MB.values()))

fig, ax = plt.subplots(figsize=(9.5, 6))
for model in MODEL_ORDER:
    cx  = COMPLEXITY[model]
    ry  = pv_rmse.loc[model, "test_MAT"]
    bsz = max(80, 1400 * np.log1p(MODEL_SIZE_MB[model]) / log_max_size)
    ax.scatter(cx, ry, color=COLORS[model], s=bsz, zorder=5,
               edgecolors="white", linewidth=1.5, alpha=0.88)
    ax.annotate(
        f"{model}\\n({COMPLEXITY_LABEL[model]})",
        (cx, ry), textcoords="offset points", xytext=(9, 4),
        fontsize=8.5, color=COLORS[model], fontweight="bold",
    )

ax.set_xscale("log")
ax.set_xlabel("Model Complexity  (trees for ensemble · params for neural nets, log scale)",
              fontsize=11)
ax.set_ylabel("Test-MAT RMSE (µA)", fontsize=11)
ax.set_title(
    "Figure 12 — Accuracy vs Complexity Tradeoff\\n"
    "(bubble size proportional to log model disk size)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.grid(linestyle="--", alpha=0.35, which="both")

# Ideal-region annotation
ax.text(0.03, 0.06, "ideal: simple + accurate",
        transform=ax.transAxes, fontsize=9, color="gray", style="italic")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig12_accuracy_vs_complexity.png")
plt.show()
print("Figure 12 saved.")
print()
print(f"All 12 figures saved to: {FIG_DIR}")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 35 — SCIENTIFIC ANALYSIS HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 15 — Twelve-Point Scientific Analysis

Key findings from the rigorous comparative evaluation of six machine-learning
approaches on the ZnO supercapacitor CV trajectory prediction task.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 36 — SCIENTIFIC ANALYSIS (12 FINDINGS — MARKDOWN)
# ══════════════════════════════════════════════════════════════════════════════
md("""\
### Finding 01 — Random Forest Leads All Models on Validation Accuracy
RF achieves the lowest validation RMSE (26.59 µA, R² = 0.9851) and the lowest
test-SR RMSE (43.71 µA), confirming that ensemble tree averaging with 300 trees excels
when the test distribution closely mirrors training scan-rate conditions.
RF's OOB R² = 0.963 indicates minimal overfitting despite unconstrained tree depth.

---

### Finding 02 — GRU Is the Strongest Model for Material Extrapolation (R²)
On the hardest partition — predicting CV curves for the completely unseen NM4 material —
GRU achieves the **highest R² = 0.9751** (vs RF's 0.9707).
On RMSE: RF is marginally better (33.65 µA vs GRU's 34.52 µA, a 2.6% gap).
The divergence between RMSE and R² rankings indicates that GRU produces
**proportionally accurate predictions relative to signal variance**, while RF achieves
slightly lower absolute errors.  For a material-agnostic deployment scenario, the GRU
provides the strongest variance-normalised generalisation.

---

### Finding 03 — LightGBM Is the Best Tabular Model for Extrapolation
Among the three tabular models (RF, XGBoost, LightGBM), LightGBM generalises best
to NM4 after RF: test-MAT RMSE = 35.18 µA vs XGBoost's 36.64 µA.
LightGBM's leaf-wise growth with L1/L2 regularisation (`reg_alpha=0.05, reg_lambda=1.0`)
outperforms XGBoost's depth-wise strategy for cross-material extrapolation.

---

### Finding 04 — Dense ANN Underperforms Across Every Partition
The dense ANN (11,777 parameters, point-wise training) achieves the worst scores on
every partition: val RMSE = 49.93 µA, test-MAT RMSE = 46.16 µA, test-MAT R² = 0.961.
Dense ANNs are ill-suited for CV trajectory prediction — they lack both the temporal
memory of RNNs and the non-parametric flexibility of tree ensembles.
They treat each voltage measurement as an independent sample, ignoring the
time-ordered electrochemical dynamics.

---

### Finding 05 — GRU Generalises Better Than LSTM with Fewer Parameters
Despite having ~24% fewer parameters (24,545 vs LSTM's 32,161), GRU outperforms LSTM:

| Metric | GRU | LSTM | GRU advantage |
|--------|-----|------|---------------|
| Test-MAT RMSE | 34.52 µA | 38.40 µA | **10.1% lower error** |
| Test-MAT R² | 0.9751 | 0.9674 | +0.0077 |
| Test-SR RMSE | 48.69 µA | 52.40 µA | 7.1% lower error |

GRU's simpler two-gate gating mechanism (reset + update) reduces the risk of
overfitting on the limited sequence dataset (96 training sequences).

---

### Finding 06 — LightGBM and XGBoost Have the Tightest Worst-Case Errors on Validation
On the validation set, LightGBM achieves **MaxErr = 99.3 µA** (only model below 100 µA)
and XGBoost achieves 104.9 µA.
In contrast, LSTM's MaxErr on test-MAT reaches **782.4 µA** and GRU reaches 445.9 µA,
indicating occasional large outlier spikes. These spikes likely occur at CV curve
extremities (current reversal points) where temporal dynamics are most abrupt.

---

### Finding 07 — Scan-Rate Dependence: Higher Error at Intermediate Scan Rates (NM4)
All models produce higher RMSE at intermediate scan rates (30–50 mV/s) on NM4,
with errors rising from 10 mV/s to a peak near 40–50 mV/s, then partially recovering.
This is physically meaningful: at intermediate scan rates, resistive (IR drop) and
capacitive (double-layer) contributions overlap, producing sharper CV curve features
that are intrinsically harder for all model architectures to predict.

---

### Finding 08 — Generalisation Profile Reveals Regime-Dependent Sensitivity
All models show larger test-SR RMSE than test-MAT RMSE — counter-intuitive since
test-SR is within the training material set (NM1/2/3) but at a more distant scan rate (50 vs 30 mV/s),
while test-MAT involves a completely unseen material.
This reveals that **scan-rate interpolation gap** is a more difficult challenge than
**cross-material transfer** for this feature representation.
Notably, RNNs (LSTM, GRU) show a smaller test-SR penalty than tree models,
suggesting temporal sequence modelling provides smoother scan-rate interpolation.

---

### Finding 09 — RF File Size (615 MB) Precludes Mobile Deployment
The serialised RF model (joblib, 615 MB) is:
- **~360× larger** than LightGBM (1.7 MB)
- **~1,416× larger** than LSTM (0.434 MB)
- **~1,792× larger** than GRU (0.343 MB)

For the project's stated mobile-app deployment target, RF is impractical.
LightGBM (1.7 MB) or GRU (0.343 MB, TFLite-convertible to ~86 KB) are the viable options.

---

### Finding 10 — LightGBM Offers the Optimal Speed-Accuracy-Size Balance
LightGBM trains in approximately 2 minutes (fastest of all six models),
achieves val RMSE = 26.86 µA (within 1% of RF's 26.59 µA), and has a 1.7 MB model file.
For production deployment in the mobile supercapacitor characterisation application,
**LightGBM is the optimal single-model choice** when temporal sequence modelling
is not a priority.

---

### Finding 11 — GRU Is Recommended for Material-Agnostic Real-Time Prediction
When the application must generalise to ZnO morphologies not represented in training
(new synthesis batches, different dopant concentrations), GRU provides the strongest
variance-normalised extrapolation (test-MAT R² = 0.9751).
Its 343 KB model file converts to approximately 86 KB with TensorFlow Lite int8 quantisation,
making real-time on-device inference feasible on mid-range smartphones.

---

### Finding 12 — Overall Rankings by Use Case
**By validation RMSE** (interpolation, seen materials):
RF (26.59) > LightGBM (26.86) > XGBoost (28.83) > LSTM (36.51) > GRU (36.84) > ANN (49.93)

**By test-MAT R²** (material extrapolation, unseen ZnO morphology):
GRU (0.9751) > RF (0.9707) > LSTM (0.9674) > LightGBM (0.9678) > XGBoost (0.9668) > ANN (0.9606)

**By deployment practicality** (accuracy + file size + inference speed):
LightGBM > GRU > XGBoost > ANN > LSTM >> RF

**Conclusion**: The optimal model choice is task-dependent.
RF for laboratory interpolation · LightGBM for mobile app deployment · GRU for new-material generalisation.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 37 — LEADERBOARD HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 16 — Final Leaderboard

Ranked comparison table using a **composite score** = mean rank across three RMSE partitions
(val, test-SR, test-MAT).  Rank 1 within each partition = lowest (best) RMSE.
A composite score of 1.0 would indicate a model ranked 1st on all three partitions simultaneously.""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 38 — LEADERBOARD COMPUTATION + PRINT
# ══════════════════════════════════════════════════════════════════════════════
code("""\
lb_parts = ["val", "test_SR", "test_MAT"]

lb = pd.DataFrame(index=MODEL_ORDER)
for p in lb_parts:
    lb[f"RMSE_{p}"]  = pv_rmse[p].values
    lb[f"rank_{p}"]  = pv_rmse[p].rank().values.astype(int)

lb["composite_rank"] = lb[[f"rank_{p}" for p in lb_parts]].mean(axis=1)
lb = lb.sort_values("composite_rank")
lb.insert(0, "Overall", range(1, len(lb) + 1))

for p in lb_parts:
    lb[f"R2_{p}"] = pv_r2[p].reindex(lb.index).values

lb["size_MB"]    = [MODEL_SIZE_MB[m] for m in lb.index]
lb["train_min"]  = [TRAIN_TIME_MIN[m] for m in lb.index]

# Pretty print
sep = "=" * 108
hdr = (f"{'Rank':<5} {'Model':<10} "
       f"{'Val RMSE':>9} {'TestSR RMSE':>12} {'TestMAT RMSE':>13} "
       f"{'Val R2':>8} {'TestMAT R2':>11} "
       f"{'Comp.Rank':>10} {'Size(MB)':>9}")
print(sep)
print(hdr)
print("-" * 108)
for idx, row in lb.iterrows():
    print(
        f"{int(row['Overall']):<5} {idx:<10} "
        f"{row['RMSE_val']:>9.2f} "
        f"{row['RMSE_test_SR']:>12.2f} "
        f"{row['RMSE_test_MAT']:>13.2f} "
        f"{row['R2_val']:>8.4f} "
        f"{row['R2_test_MAT']:>11.4f} "
        f"{row['composite_rank']:>10.2f} "
        f"{row['size_MB']:>9.3f}"
    )
print(sep)
print("Units: RMSE in µA  |  R² dimensionless  |  Composite = mean RMSE rank (1=best)")

df_leaderboard = lb.reset_index().rename(columns={"index": "Model"})""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 39 — EXPORT HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 17 — Export All Results

Saving six output files to `results/final_comparison/`:

| # | File | Contents |
|---|------|---------|
| 1 | `comparison_metrics.csv` | Full long-format table (all models × partitions × metrics) |
| 2 | `final_leaderboard.csv` | Ranked leaderboard with composite score |
| 3 | `nm4_per_scan_rate.csv` | NM4 RMSE and R² per scan rate per model |
| 4 | `final_summary.json` | Structured JSON with key findings and all metric values |
| 5 | `SUMMARY.md` | Publication-quality text summary with recommendations |
| 6 | `figures/` | All 12 PNG comparison figures (already saved above) |""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 40 — EXPORT CODE
# ══════════════════════════════════════════════════════════════════════════════
code("""\
# ── 1. comparison_metrics.csv ─────────────────────────────────────────────
df_all.to_csv(OUT_ROOT / "comparison_metrics.csv", index=False)
print("  comparison_metrics.csv saved")

# ── 2. final_leaderboard.csv ──────────────────────────────────────────────
df_leaderboard.to_csv(OUT_ROOT / "final_leaderboard.csv", index=False)
print("  final_leaderboard.csv saved")

# ── 3. nm4_per_scan_rate.csv ──────────────────────────────────────────────
df_pg.to_csv(OUT_ROOT / "nm4_per_scan_rate.csv", index=False)
print("  nm4_per_scan_rate.csv saved")

# ── 4. final_summary.json ─────────────────────────────────────────────────
summary_dict = {
    "generated":   datetime.now().isoformat(),
    "project":     "ZnO Supercapacitor CV Trajectory Prediction",
    "models":      MODEL_ORDER,
    "key_findings": {
        "best_by_val_rmse":     "RF",
        "best_by_testmat_r2":   "GRU",
        "best_for_deployment":  "LightGBM",
        "worst_model":          "ANN",
    },
    "metrics": {
        model: {
            part: {
                "RMSE_uA":   float(pv_rmse.loc[model, part]),
                "MAE_uA":    float(pv_mae.loc[model,  part]),
                "MaxErr_uA": float(pv_maxe.loc[model, part]),
                "R2":        float(pv_r2.loc[model,   part]),
            }
            for part in ["val", "test_SR", "test_MAT"]
        }
        for model in MODEL_ORDER
    },
    "approx_training_time_minutes": TRAIN_TIME_MIN,
    "model_size_mb":                MODEL_SIZE_MB,
    "complexity_params_or_trees":   COMPLEXITY,
}
with open(OUT_ROOT / "final_summary.json", "w", encoding="utf-8") as fj:
    json.dump(summary_dict, fj, indent=2)
print("  final_summary.json saved")

# ── 5. SUMMARY.md ─────────────────────────────────────────────────────────
def _r(m, p): return f"{pv_rmse.loc[m, p]:.2f}"
def _q(m, p): return f"{pv_r2.loc[m,   p]:.4f}"

lines = [
    "# ZnO CV Trajectory Prediction — Final Model Comparison Summary",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "## Experiment Overview",
    "Six machine-learning models (RF, XGBoost, LightGBM, ANN, LSTM, GRU) were trained",
    "to predict the full cyclic voltammetry (CV) current-voltage trajectory of ZnO-based",
    "supercapacitors from scan-rate and electrode-potential features.",
    "",
    "## Key Metrics (RMSE in uA)",
    "",
    "| Model     | Val (SR=30) | Test-SR (SR=50) | Test-MAT (NM4) | Val R2   | TestMAT R2 |",
    "|-----------|-------------|-----------------|----------------|----------|------------|",
    f"| RF        | {_r('RF','val'):<11} | {_r('RF','test_SR'):<15} | {_r('RF','test_MAT'):<14} | {_q('RF','val')} | {_q('RF','test_MAT')} |",
    f"| XGBoost   | {_r('XGBoost','val'):<11} | {_r('XGBoost','test_SR'):<15} | {_r('XGBoost','test_MAT'):<14} | {_q('XGBoost','val')} | {_q('XGBoost','test_MAT')} |",
    f"| LightGBM  | {_r('LightGBM','val'):<11} | {_r('LightGBM','test_SR'):<15} | {_r('LightGBM','test_MAT'):<14} | {_q('LightGBM','val')} | {_q('LightGBM','test_MAT')} |",
    f"| ANN       | {_r('ANN','val'):<11} | {_r('ANN','test_SR'):<15} | {_r('ANN','test_MAT'):<14} | {_q('ANN','val')} | {_q('ANN','test_MAT')} |",
    f"| LSTM      | {_r('LSTM','val'):<11} | {_r('LSTM','test_SR'):<15} | {_r('LSTM','test_MAT'):<14} | {_q('LSTM','val')} | {_q('LSTM','test_MAT')} |",
    f"| GRU       | {_r('GRU','val'):<11} | {_r('GRU','test_SR'):<15} | {_r('GRU','test_MAT'):<14} | {_q('GRU','val')} | {_q('GRU','test_MAT')} |",
    "",
    "## Recommendations",
    "",
    f"1. Best for interpolation (seen scan rates, seen materials): RF",
    f"   - Validation RMSE = {_r('RF','val')} uA, R2 = {_q('RF','val')}",
    "",
    f"2. Best for material extrapolation (by R2): GRU",
    f"   - Test-MAT R2 = {_q('GRU','test_MAT')}, RMSE = {_r('GRU','test_MAT')} uA",
    "",
    f"3. Best for mobile deployment (accuracy + compact size): LightGBM",
    f"   - Validation RMSE = {_r('LightGBM','val')} uA, model size = 1.7 MB",
    f"   - (vs RF at 615 MB, trains in ~2 min vs RF's ~4 min)",
    "",
    "## Scientific Conclusion",
    "Random Forest delivers the best interpolation performance (validation RMSE) and",
    "marginally the best extrapolation RMSE, while GRU uniquely achieves the highest",
    "variance-normalised extrapolation accuracy (test-MAT R2 = 0.9751).",
    "LightGBM provides the optimal accuracy-footprint trade-off for mobile deployment.",
    "Dense ANN is not recommended for this spatiotemporal regression task.",
]
summary_md = "\\n".join(lines)
with open(OUT_ROOT / "SUMMARY.md", "w", encoding="utf-8") as fs:
    fs.write(summary_md)
print("  SUMMARY.md saved")

print()
print(f"All outputs saved to : {OUT_ROOT.resolve()}")
n_figs = len(list(FIG_DIR.glob("*.png")))
print(f"Figures              : {FIG_DIR}  ({n_figs} PNG files)")""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 41 — DOWNLOAD HEADER
# ══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Section 18 — Download Results Archive

Downloads a ZIP file containing all figures and export files.
*(Colab only — on local Python the files are already on disk in `results/final_comparison/`)*""")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 42 — DOWNLOAD CODE
# ══════════════════════════════════════════════════════════════════════════════
code("""\
if IN_COLAB:
    import shutil
    zip_stem = "/content/08_final_comparison_results"
    shutil.make_archive(zip_stem, "zip", OUT_ROOT)
    _cf.download(zip_stem + ".zip")
    print("Download triggered: 08_final_comparison_results.zip")
    print("Check your browser Downloads folder.")
else:
    print("Local mode — all outputs already at:")
    print(f"  {OUT_ROOT.resolve()}")
    print()
    print("Contents:")
    for f in sorted(OUT_ROOT.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            print(f"  {str(f.relative_to(OUT_ROOT)):<55}  {sz:>8,} bytes")""")


# ══════════════════════════════════════════════════════════════════════════════
# WRITE NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
    },
    "colab": {"provenance": []},
}

with open(NB_PATH, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)

print(f"Notebook written : {NB_PATH}")
print(f"Total cells      : {len(cells)}")
print()
print("Cell breakdown:")
for i, cell in enumerate(cells):
    kind = "MD  " if cell.cell_type == "markdown" else "CODE"
    preview = cell.source[:60].replace("\n", " ").strip()
    print(f"  [{i+1:02d}] {kind}  {preview}")
