"""
build_notebook.py
Programmatically generates notebooks/01_correct_preprocessing.ipynb
Colab-compatible version.
Run once: python src/build_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
md("""# Notebook 01 — Scientifically Correct CV Data Preprocessing
## ZnO Supercapacitor Cyclic Voltammetry · ML Pipeline Foundation

### Scientific Constraints (enforced throughout every downstream step)
| # | Constraint |
|---|---|
| 1 | CV data is sequential electrochemical behaviour — **not** i.i.d. tabular data |
| 2 | **Half-sweeps (~650 rows)** are the minimum indivisible unit for any split |
| 3 | **Random row-wise splitting is forbidden** — it causes autocorrelation leakage |
| 4 | **Sweep direction is mandatory** — the same voltage produces two distinct currents |
| 5 | Normalization scalers are fitted on **training data only** |
| 6 | Evaluation must test **real generalization**: unseen scan rates + unseen materials |
| 7 | Reported metrics must also include **physical-unit errors** (Amperes), not only normalised scores |

### Notebook Sections
0. Colab file setup
1. Project configuration & imports
2. Robust Excel parsing
3. Long-format (tidy) assembly
4. Sweep-direction detection & structural features
5. Engineered electrochemical features
6. Verification visualisations
7. Scientifically correct train / validation / test split
8. Leakage-free normalisation
9. Save all artefacts
10. Final integrity check""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — COLAB FILE SETUP
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 0 — Colab File Setup

Choose **one** of the two options below by setting `UPLOAD_METHOD`:

| Option | `UPLOAD_METHOD` | When to use |
|--------|----------------|-------------|
| Direct upload | `"upload"` | Quick one-off run; file stays only for this session |
| Google Drive | `"drive"` | Recommended — outputs (parquet, figures) persist across sessions |

**If using Drive:** edit `DRIVE_EXCEL_PATH` and `DRIVE_PROJECT_ROOT` to match
where you stored the file and where you want outputs saved in your Drive.""")

code("""\
from pathlib import Path

# ── Choose upload method ─────────────────────────────────────────────────────
UPLOAD_METHOD = "upload"   # "upload"  or  "drive"

if UPLOAD_METHOD == "drive":
    from google.colab import drive
    drive.mount("/content/drive")

    # Edit these two paths to match your Google Drive layout:
    DRIVE_EXCEL_PATH   = "/content/drive/MyDrive/NM CV data sets.xlsx"
    DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/ZnO_PBL_Project"

    RAW_DATA_PATH = Path(DRIVE_EXCEL_PATH)
    PROJECT_ROOT  = Path(DRIVE_PROJECT_ROOT)
    print("Google Drive mounted.")

else:
    from google.colab import files as _colab_files
    print("Select  'NM CV data sets.xlsx'  in the upload dialog below …")
    _up    = _colab_files.upload()
    _fname = list(_up.keys())[0]
    RAW_DATA_PATH = Path(f"/content/{_fname}")
    PROJECT_ROOT  = Path("/content/ZnO_PBL_Project")

assert RAW_DATA_PATH.exists(), (
    f"File not found: {RAW_DATA_PATH}\\n"
    "If using Drive, check that DRIVE_EXCEL_PATH matches the actual file location."
)
print(f"\\n  Data file : {RAW_DATA_PATH}")
print(f"  Size      : {RAW_DATA_PATH.stat().st_size / 1e6:.1f} MB")
print(f"  Project   : {PROJECT_ROOT}")
print("\\n✓ File setup complete — proceed to the next cell.")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PROJECT CONFIGURATION & IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 1 — Project Configuration & Imports")

code("""\
# RAW_DATA_PATH and PROJECT_ROOT are defined in Section 0 above.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.6e}".format)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# ── Output directories ────────────────────────────────────────────────────────
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
SEQUENCES_DIR  = PROJECT_ROOT / "data" / "sequences"
FIGURES_DIR    = PROJECT_ROOT / "figures"

for d in [PROCESSED_DIR, SEQUENCES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Electrochemical constants ─────────────────────────────────────────────────
SHEET_NAMES          = ["NM1", "NM2", "NM3", "NM4"]
SCAN_RATES           = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # mV/s
V_LOWER              = -0.65   # V vs SCE — cathodic limit
V_UPPER              =  0.00   # V vs SCE — anodic limit
STEPS_PER_HALF_SWEEP = 650     # nominal: 2600 / 4; boundary sweeps may be ±1
V_STEP               = 0.001   # V

# ── Split configuration ───────────────────────────────────────────────────────
# Splits along MATERIAL and SCAN-RATE axes — never along the row axis.
TRAIN_NM         = ["NM1", "NM2", "NM3"]
TRAIN_SR         = [10, 20, 40, 60, 70, 80, 90, 100]

VAL_NM           = ["NM1", "NM2", "NM3"]
VAL_SR           = [30]

TEST_MATERIAL_NM = ["NM4"]
TEST_SR_NM       = ["NM1", "NM2", "NM3"]
TEST_SR_SR       = [50]

EXPECTED_TOTAL_ROWS = len(SHEET_NAMES) * len(SCAN_RATES) * 4 * STEPS_PER_HALF_SWEEP

print("✓ Configuration loaded")
print(f"  Data file    : {RAW_DATA_PATH}")
print(f"  Processed dir: {PROCESSED_DIR}")
print(f"  Expected rows: {EXPECTED_TOTAL_ROWS:,}  "
      f"({len(SHEET_NAMES)} mats x {len(SCAN_RATES)} SRs x 4 half-sweeps x {STEPS_PER_HALF_SWEEP} steps)")
print(f"  Train SRs    : {TRAIN_SR}")
print(f"  Val SR       : {VAL_SR}")
print(f"  Test SR      : {TEST_SR_SR}  |  Test material: {TEST_MATERIAL_NM}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EXCEL PARSING
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 2 — Robust Excel Parsing

### Why a custom parser is needed
The four sheets (NM1–NM4) have **inconsistent column offsets**:

| Sheet | Potential column (index) | Current columns |
|-------|--------------------------|-----------------|
| NM1   | 1                        | 2 – 11          |
| NM2   | 2                        | 3 – 12          |
| NM3   | 2                        | 3 – 12          |
| NM4   | 3                        | 4 – 13          |

The robust strategy: **locate the potential column by its known starting value (−0.65 V)**,
then take the 10 numeric columns that immediately follow it as the current data columns,
mapping them in order to scan rates 10, 20, …, 100 mV/s.""")

code("""\
def parse_single_sheet(xl_path: Path, sheet_name: str) -> pd.DataFrame:
    \"\"\"
    Parse one sheet of the CV Excel workbook into a clean wide-format DataFrame.

    Returns columns: [row_in_sheet, potential_V, sr_10, sr_20, ..., sr_100]
    where sr_XX is current (A) at scan rate XX mV/s.

    Strategy
    --------
    1. Read with header=None to expose raw layout.
    2. Detect the potential column: first column whose row-3 value is approx V_LOWER.
    3. Detect current columns: the 10 numeric columns immediately after potential.
    4. Drop rows where potential is NaN (blank separator rows in NM1).
    5. Validate: exactly 2600 non-NaN potential rows expected.
    \"\"\"
    raw = pd.read_excel(xl_path, sheet_name=sheet_name, header=None)

    # ── Step 1: locate potential column ──────────────────────────────────────
    pot_col = None
    for c in range(raw.shape[1]):
        val = raw.iloc[3, c]
        if pd.notna(val):
            try:
                fval = float(val)
                if abs(fval - V_LOWER) < 0.005:
                    pot_col = c
                    break
            except (ValueError, TypeError):
                pass
    assert pot_col is not None, f"[{sheet_name}] Could not locate potential column"

    # ── Step 2: locate the 10 current columns after potential ────────────────
    curr_cols = []
    for c in range(pot_col + 1, raw.shape[1]):
        col_data = pd.to_numeric(raw.iloc[3:, c], errors="coerce")
        if col_data.notna().mean() >= 0.90:
            curr_cols.append(c)
        if len(curr_cols) == len(SCAN_RATES):
            break
    assert len(curr_cols) == len(SCAN_RATES), (
        f"[{sheet_name}] Expected {len(SCAN_RATES)} current columns, found {len(curr_cols)}")

    # ── Step 3: extract data rows ─────────────────────────────────────────────
    data      = raw.iloc[3:].copy().reset_index(drop=True)
    potential = pd.to_numeric(data.iloc[:, pot_col], errors="coerce")

    records = {"potential_V": potential.values}
    for i, sr in enumerate(SCAN_RATES):
        records[f"sr_{sr}"] = pd.to_numeric(data.iloc[:, curr_cols[i]], errors="coerce").values

    df_wide = pd.DataFrame(records)

    # ── Step 4: drop rows with NaN potential ─────────────────────────────────
    df_wide = df_wide.dropna(subset=["potential_V"]).reset_index(drop=True)

    # ── Step 5: validate row count ────────────────────────────────────────────
    expected_rows = 4 * STEPS_PER_HALF_SWEEP
    assert len(df_wide) == expected_rows, (
        f"[{sheet_name}] Expected {expected_rows} rows, got {len(df_wide)}")

    df_wide.insert(0, "row_in_sheet", df_wide.index)
    print(f"  [{sheet_name}] parsed  — potential col: {pot_col}, "
          f"current cols: {curr_cols[0]}-{curr_cols[-1]}, "
          f"rows: {len(df_wide)}")
    return df_wide


print("Parsing all sheets ...")
wide_sheets = {nm: parse_single_sheet(RAW_DATA_PATH, nm) for nm in SHEET_NAMES}
print("\\n✓ All sheets parsed successfully")
for nm, df in wide_sheets.items():
    print(f"  {nm}: shape={df.shape}  |  "
          f"V range=[{df['potential_V'].min():.3f}, {df['potential_V'].max():.3f}] V  |  "
          f"I(sr10) range=[{df['sr_10'].min():.3e}, {df['sr_10'].max():.3e}] A")""")

code("""\
# ── Verification: display first 5 rows of NM1 and NM2 ───────────────────────
print("=== NM1 first 5 rows ===")
print(wide_sheets["NM1"].head())
print("\\n=== NM2 first 5 rows ===")
print(wide_sheets["NM2"].head())""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LONG-FORMAT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 3 — Long-Format (Tidy) Assembly

Each row in the wide format corresponds to one potential step with 10 current readings
(one per scan rate). We melt this into **one row per (material, scan-rate, potential step)**
— the tidy long format required for all downstream ML pipelines.

```
Wide:  [potential_V, sr_10, sr_20, ..., sr_100]  ->  2600 rows x 11 cols per sheet
Long:  [nm_id, scan_rate_mVs, potential_V, current_A]  ->  104 000 rows x 4 cols
```""")

code("""\
def wide_to_long(nm_id: str, df_wide: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Melt one wide-format sheet into tidy long format.\"\"\"
    sr_cols = {f"sr_{sr}": float(sr) for sr in SCAN_RATES}
    frames  = []
    for col_name, sr_val in sr_cols.items():
        tmp = pd.DataFrame({
            "nm_id"        : nm_id,
            "scan_rate_mVs": sr_val,
            "row_in_sheet" : df_wide["row_in_sheet"].values,
            "potential_V"  : df_wide["potential_V"].values,
            "current_A"    : df_wide[col_name].values,
        })
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


print("Assembling long-format master DataFrame ...")
long_frames = [wide_to_long(nm, wide_sheets[nm]) for nm in SHEET_NAMES]
master      = pd.concat(long_frames, ignore_index=True)

print(f"\\n✓ Master long-format assembled")
print(f"  Shape       : {master.shape}")
print(f"  Expected    : ({EXPECTED_TOTAL_ROWS:,}, 5)")
assert len(master) == EXPECTED_TOTAL_ROWS, "Row count mismatch!"
print(f"  Columns     : {list(master.columns)}")
print(f"  nm_id values: {master['nm_id'].unique().tolist()}")
print(f"  Scan rates  : {sorted(master['scan_rate_mVs'].unique().tolist())}")
print(f"  NaN current : {master['current_A'].isna().sum()}")
master.head(8)""")

code("""\
# ── Verification: row counts per group ───────────────────────────────────────
counts = master.groupby(["nm_id", "scan_rate_mVs"]).size().unstack()
print("Rows per (nm_id x scan_rate) — all must equal 2600:")
print(counts)
assert (counts == 4 * STEPS_PER_HALF_SWEEP).all().all(), "Unexpected group sizes!"
print("\\n✓ All groups have exactly 2600 rows")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SWEEP DIRECTION & STRUCTURAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 4 — Sweep Direction Detection & Structural Features

### The electrochemical problem
A CV sweep is a **closed loop**: the electrode visits every voltage twice per cycle —
once sweeping up (cathodic, reduction) and once sweeping down (anodic, oxidation).
At V = −0.3 V, the current on the upward sweep differs from the current on the
downward sweep due to capacitive hysteresis and kinetic asymmetry.

If we model `current = f(potential)` with no direction information, we have a
**many-to-one mapping**: the model cannot distinguish which branch it is on and
learns a blurred average — physically meaningless.

### Features generated in this section
| Feature | Type | Description |
|---|---|---|
| `dV` | float | Potential step: `+0.001` (forward) or `-0.001` (reverse) |
| `sweep_direction` | int 0/1 | 0 = cathodic (V increasing), 1 = anodic (V decreasing) |
| `half_sweep_id` | int 0-3 | Which of the 4 contiguous half-sweeps |
| `cycle_id` | int 0/1 | `half_sweep_id // 2` — full CV cycle index |
| `step_index` | int | Row index within the current half-sweep |
| `sweep_position` | float [0,1] | Normalised progress through the current half-sweep |""")

code("""\
def add_structural_features(group_df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Detect sweep direction and derive structural features for one
    (nm_id, scan_rate) group.  Operates entirely within the group
    — no cross-group state.

    Algorithm
    ---------
    1. Compute dV = diff(potential_V).  Forward: +0.001, Reverse: -0.001.
    2. Derive sweep_direction from sign of dV.
    3. Detect direction-change boundaries -> half_sweep_id (0,1,2,3).
    4. Derive cycle_id = half_sweep_id // 2.
    5. Compute step_index = row count within each half_sweep_id.
    6. Compute sweep_position = step_index / per-half-sweep max step.
    \"\"\"
    df = group_df.copy().reset_index(drop=True)
    n  = len(df)

    # ── 1. dV ─────────────────────────────────────────────────────────────────
    pot    = df["potential_V"].values.astype(float)
    dv     = np.empty(n)
    dv[1:] = pot[1:] - pot[:-1]
    dv[0]  = dv[1]              # boundary: inherit direction from first real step

    df["dV"] = np.round(dv, 6)

    # ── 2. sweep_direction ────────────────────────────────────────────────────
    direction  = np.where(dv > 0, 0, np.where(dv < 0, 1, np.nan)).astype(float)
    dir_series = pd.Series(direction).ffill().bfill()
    df["sweep_direction"] = dir_series.astype(int)

    # ── 3. half_sweep_id ──────────────────────────────────────────────────────
    dir_change       = (df["sweep_direction"].diff().abs() > 0).astype(int)
    df["half_sweep_id"] = dir_change.cumsum()

    # ── 4. cycle_id ───────────────────────────────────────────────────────────
    df["cycle_id"] = df["half_sweep_id"] // 2

    # ── 5. step_index ─────────────────────────────────────────────────────────
    df["step_index"] = df.groupby("half_sweep_id").cumcount()

    # ── 6. sweep_position ─────────────────────────────────────────────────────
    # Use per-half-sweep max step_index as denominator — boundary half-sweeps
    # may have 649 or 651 rows rather than exactly 650.
    hs_max = df.groupby("half_sweep_id")["step_index"].transform("max")
    df["sweep_position"] = df["step_index"] / hs_max.clip(lower=1)

    return df


print("Adding structural features ...")
# Explicit loop avoids pandas 2.x groupby().apply() column-dropping behaviour.
_frames = []
for (_nm, _sr), _grp in master.groupby(["nm_id", "scan_rate_mVs"], sort=False):
    _frames.append(add_structural_features(_grp))
master = pd.concat(_frames, ignore_index=True)

print(f"\\n✓ Structural features added.  Shape: {master.shape}")
print(f"  Columns: {list(master.columns)}")""")

code("""\
# ── Verification: structural integrity ────────────────────────────────────────
print("=== Structural feature verification ===\\n")

# 1. Every group must have exactly 4 half-sweeps
hs_check = (master.groupby(["nm_id", "scan_rate_mVs"])["half_sweep_id"].nunique())
assert (hs_check == 4).all(), "Not all groups have 4 half-sweeps!"
print("[PASS] All groups have exactly 4 half-sweeps")

# 2. Total rows per group must be exactly 2600
# (boundary half-sweeps vary by +-1 row; total is always 2600)
step_counts  = (master.groupby(["nm_id", "scan_rate_mVs", "half_sweep_id"]).size())
group_totals = step_counts.groupby(level=["nm_id", "scan_rate_mVs"]).sum()
assert (group_totals == 4 * STEPS_PER_HALF_SWEEP).all(), "Group total rows mismatch!"
print(f"[PASS] All groups total exactly {4*STEPS_PER_HALF_SWEEP} rows across 4 half-sweeps")
print(f"[INFO] Half-sweep sizes range: {step_counts.min()}-{step_counts.max()} rows")

# 3. Direction alternates correctly: 0,1,0,1
dir_per_hs = (master.groupby(["nm_id", "scan_rate_mVs", "half_sweep_id"])
                    ["sweep_direction"].first().unstack())
expected_first_dir = {0: 0, 1: 1, 2: 0, 3: 1}
for hs_id, exp_dir in expected_first_dir.items():
    assert (dir_per_hs[hs_id] == exp_dir).all(), (
        f"Direction pattern wrong for half_sweep_id={hs_id}")
print("[PASS] Direction alternates 0->1->0->1 correctly in every group")

# 4. dV values
dv_vals = master["dV"].round(4).unique()
print(f"[INFO] Unique dV values: {sorted(dv_vals)}")

# 5. sweep_position range
print(f"[INFO] sweep_position range: [{master['sweep_position'].min():.3f}, "
      f"{master['sweep_position'].max():.3f}]")

# 6. Sample rows
sample = master.query("nm_id=='NM1' and scan_rate_mVs==20").head(10)
print("\\nSample rows (NM1, SR=20):")
print(sample[["potential_V","dV","sweep_direction","half_sweep_id",
              "cycle_id","step_index","sweep_position","current_A"]])""")

code("""\
# ── Visualisation: CV loop with direction colour-coding (NM1, SR=50) ─────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

sample_nm, sample_sr = "NM1", 50.0
sub = master.query(f"nm_id=='{sample_nm}' and scan_rate_mVs=={sample_sr}")

colours = {0: "#1f77b4", 1: "#d62728"}
labels  = {0: "Cathodic (direction=0, V up)", 1: "Anodic (direction=1, V down)"}

ax = axes[0]
for hs_id in [0, 1, 2, 3]:
    seg = sub[sub["half_sweep_id"] == hs_id]
    d   = seg["sweep_direction"].iloc[0]
    lbl = labels[d] if hs_id <= 1 else None
    ax.plot(seg["potential_V"], seg["current_A"],
            color=colours[d], lw=1.4, label=lbl)
ax.set_xlabel("Potential (V vs SCE)")
ax.set_ylabel("Current (A)")
ax.set_title(f"CV Loop - {sample_nm}, {sample_sr} mV/s\\n(colour = sweep direction)")
ax.legend(fontsize=8)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axvline(V_LOWER, color="grey", lw=0.5, ls=":")
ax.axvline(V_UPPER, color="grey", lw=0.5, ls=":")
ax.grid(True, alpha=0.3)

ax = axes[1]
for hs_id in [0, 1]:
    seg = sub[sub["half_sweep_id"] == hs_id]
    d   = seg["sweep_direction"].iloc[0]
    ax.plot(seg["sweep_position"], seg["current_A"],
            color=colours[d], lw=1.4, label=labels[d])
ax.set_xlabel("Sweep position (0 = start, 1 = turning point)")
ax.set_ylabel("Current (A)")
ax.set_title(f"Sweep Position vs Current - Cycle 0\\n{sample_nm}, {sample_sr} mV/s")
ax.legend(fontsize=8)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_sweep_direction_verification.png", bbox_inches="tight")
plt.show()
print("Saved 01_sweep_direction_verification.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ENGINEERED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 5 — Engineered Electrochemical Features

These features encode known physical relationships so models can exploit them
directly rather than learning them from scratch.

| Feature | Formula | Physical basis |
|---|---|---|
| `potential_from_lower` | V - V_lower | Distance from cathodic limit |
| `potential_from_upper` | V_upper - V | Distance from anodic limit |
| `log_scan_rate` | ln(v) | Many electrochemical relationships are log-linear in scan rate |
| `sqrt_scan_rate` | sqrt(v) | Randles-Sevcik: diffusion-limited current proportional to v^0.5 |
| `direction_x_potential` | direction x V | Interaction: same voltage has asymmetric meaning per sweep direction |
| `sr_x_potential` | v x V | Interaction: scan rate modulates the voltage-dependent current response |""")

code("""\
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["potential_from_lower"]  = df["potential_V"] - V_LOWER
    df["potential_from_upper"]  = V_UPPER - df["potential_V"]
    df["log_scan_rate"]         = np.log(df["scan_rate_mVs"])
    df["sqrt_scan_rate"]        = np.sqrt(df["scan_rate_mVs"])
    df["direction_x_potential"] = df["sweep_direction"] * df["potential_V"]
    df["sr_x_potential"]        = df["scan_rate_mVs"]   * df["potential_V"]
    return df


master = add_engineered_features(master)

print("✓ Engineered features added")
print(f"  Shape: {master.shape}")

eng_features = ["potential_from_lower", "potential_from_upper",
                "log_scan_rate", "sqrt_scan_rate",
                "direction_x_potential", "sr_x_potential"]
print("\\nFeature ranges:")
for f in eng_features:
    print(f"  {f:30s}: [{master[f].min():.4f}, {master[f].max():.4f}]")""")

code("""\
print("=== MASTER DATAFRAME SUMMARY ===\\n")
print(master.dtypes)
print()
print(master.describe().T[["mean", "std", "min", "max"]])
print(f"\\nTotal rows : {len(master):,}")
print(f"Total cols : {len(master.columns)}")
print(f"Memory     : {master.memory_usage(deep=True).sum() / 1e6:.1f} MB")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — VERIFICATION VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 6 — Verification Visualisations

Five verification plots:
1. **CV loops at all scan rates for NM1** — confirm loop shape and scan-rate scaling
2. **All 4 materials at 50 mV/s** — confirm inter-material variation
3. **Potential & current vs row index** — verify sequential structure is preserved
4. **Current distributions** — confirm raw data scale before normalisation
5. **Scan-rate scaling of peak current** — confirm capacitive/diffusion physics""")

code("""\
# ── Plot 1: CV loops across scan rates for NM1 ───────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
axes = axes.flatten()
sample_nm   = "NM1"
colours_sr  = plt.cm.plasma(np.linspace(0.1, 0.9, len(SCAN_RATES)))

for i, sr in enumerate(SCAN_RATES):
    ax  = axes[i]
    sub = master.query(f"nm_id=='{sample_nm}' and scan_rate_mVs=={sr}")
    for hs_id in [0, 1, 2, 3]:
        seg = sub[sub["half_sweep_id"] == hs_id]
        ax.plot(seg["potential_V"], seg["current_A"],
                color=colours_sr[i], lw=1.2, alpha=0.8)
    ax.set_title(f"{sr} mV/s", fontsize=9)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("V (V vs SCE)", fontsize=7)
    if i % 5 == 0:
        ax.set_ylabel("Current (A)", fontsize=7)

fig.suptitle(f"CV Loops - {sample_nm} - All Scan Rates (2 complete cycles)", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_cv_loops_all_scanrates.png", bbox_inches="tight")
plt.show()
print("Saved 02_cv_loops_all_scanrates.png")""")

code("""\
# ── Plot 2: All 4 materials at scan rate 50 mV/s ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes       = axes.flatten()
mat_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
sr_plot     = 50.0

for i, nm in enumerate(SHEET_NAMES):
    ax  = axes[i]
    sub = master.query(f"nm_id=='{nm}' and scan_rate_mVs=={sr_plot}")
    for hs_id in [0, 1, 2, 3]:
        seg = sub[sub["half_sweep_id"] == hs_id]
        ax.plot(seg["potential_V"], seg["current_A"],
                color=mat_colours[i], lw=1.3, alpha=0.9)
    ax.set_title(f"{nm} @ {sr_plot} mV/s", fontsize=10)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Potential (V vs SCE)")
    ax.set_ylabel("Current (A)")
    ax.grid(True, alpha=0.3)

fig.suptitle("Inter-material CV Comparison - All 4 NM Formulations @ 50 mV/s", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_material_comparison.png", bbox_inches="tight")
plt.show()
print("Saved 03_material_comparison.png")""")

code("""\
# ── Plot 3: Sequential structure — potential & current vs row index ───────────
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
sub = master.query("nm_id=='NM1' and scan_rate_mVs==30.0")

axes[0].plot(sub["step_index"].reset_index(drop=True),
             sub["potential_V"].reset_index(drop=True),
             color="#1f77b4", lw=0.8)
axes[0].set_ylabel("Potential (V vs SCE)")
axes[0].set_title("NM1, 30 mV/s - Sequential structure (2 full CV cycles = 4 half-sweeps)")
for x in [650, 1300, 1950]:
    axes[0].axvline(x, color="grey", ls=":", lw=1)
for x, lbl in zip([325, 975, 1625, 2275],
                   ["HS0\\n(cat.)", "HS1\\n(ano.)", "HS2\\n(cat.)", "HS3\\n(ano.)"]):
    axes[0].text(x, -0.3, lbl, ha="center", fontsize=8, color="grey")

axes[1].plot(sub["step_index"].reset_index(drop=True),
             sub["current_A"].reset_index(drop=True),
             color="#d62728", lw=0.8)
axes[1].set_xlabel("Row index (step_index within full group)")
axes[1].set_ylabel("Current (A)")
for x in [650, 1300, 1950]:
    axes[1].axvline(x, color="grey", ls=":", lw=1)

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_sequential_structure.png", bbox_inches="tight")
plt.show()
print("Saved 04_sequential_structure.png")""")

code("""\
# ── Plot 4: Current distributions per material ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes = axes.flatten()

for i, nm in enumerate(SHEET_NAMES):
    ax     = axes[i]
    sub_nm = master[master["nm_id"] == nm]
    for j, sr in enumerate([20, 50, 80, 100]):
        vals = sub_nm[sub_nm["scan_rate_mVs"] == sr]["current_A"].values * 1e4
        ax.hist(vals, bins=60, alpha=0.6, density=True,
                label=f"{sr} mV/s", color=plt.cm.viridis(j / 3))
    ax.set_title(f"{nm} - Current distributions (x10^-4 A)")
    ax.set_xlabel("Current (x10^-4 A)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Current Distributions - Raw (before normalisation)", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_current_distributions.png", bbox_inches="tight")
plt.show()
print("Saved 05_current_distributions.png")""")

code("""\
# ── Plot 5: Scan-rate scaling of peak absolute current ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for i, nm in enumerate(SHEET_NAMES):
    i_peaks = []
    for sr in SCAN_RATES:
        i_max = (master.query(f"nm_id=='{nm}' and scan_rate_mVs=={sr}")
                       ["current_A"].abs().max())
        i_peaks.append(i_max * 1e4)
    axes[0].plot(SCAN_RATES, i_peaks, "o-", label=nm, lw=1.5)
    axes[1].loglog(SCAN_RATES, i_peaks, "o-", label=nm, lw=1.5)

for ax, title in zip(axes, ["Linear scale",
                              "Log-log scale (slope = power law exponent)"]):
    ax.set_xlabel("Scan rate (mV/s)")
    ax.set_ylabel("Peak |Current| (x10^-4 A)")
    ax.set_title(f"Peak Current vs Scan Rate\\n{title}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_scanrate_scaling.png", bbox_inches="tight")
plt.show()
print("Saved 06_scanrate_scaling.png")
print("  (Log-log slope ~1.0 -> purely capacitive; ~0.5 -> diffusion-limited)")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 7 — Scientifically Correct Train / Validation / Test Split

### Why row-wise splitting is forbidden
Adjacent rows in a CV sweep are separated by only 1 mV in potential. Their
currents are nearly identical (smooth continuous function). A random 80/20 row
split puts V = −0.300 V in training and V = −0.299 V in test — trivial
interpolation, not generalisation. The reported R² ≈ 0.999 in the paper reflects
autocorrelation, not model capability.

### The atomic unit: a half-sweep block
`(nm_id, scan_rate, half_sweep_id)` — ~650 rows, always kept together.
Total blocks: 4 materials × 10 scan rates × 4 half-sweeps = **160 blocks**.

### Split assignment
| Partition | Criterion | Blocks | Rows | % |
|---|---|---|---|---|
| **Train** | NM1/2/3 × SR in {10,20,40,60,70,80,90,100} | 96 | 62 400 | 60% |
| **Val** | NM1/2/3 × SR = 30 mV/s | 12 | 7 800 | 7.5% |
| **Test-SR** | NM1/2/3 × SR = 50 mV/s | 12 | 7 800 | 7.5% |
| **Test-MAT** | NM4 × all SRs | 40 | 26 000 | 25% |""")

code("""\
def assign_split(row) -> str:
    nm = row["nm_id"]
    sr = row["scan_rate_mVs"]
    if nm in TEST_MATERIAL_NM:
        return "test_material"
    if nm in TEST_SR_NM and sr in TEST_SR_SR:
        return "test_scanrate"
    if nm in VAL_NM and sr in VAL_SR:
        return "val"
    if nm in TRAIN_NM and sr in TRAIN_SR:
        return "train"
    return "unassigned"


master["split"] = master.apply(assign_split, axis=1)

unassigned = (master["split"] == "unassigned").sum()
assert unassigned == 0, f"{unassigned} rows unassigned — check split logic!"
print("✓ All rows assigned to a split partition")

split_stats = master.groupby("split").agg(
    rows       = ("current_A",     "count"),
    materials  = ("nm_id",         "nunique"),
    scan_rates = ("scan_rate_mVs", "nunique"),
).assign(pct=lambda x: (x["rows"] / len(master) * 100).round(1))
print("\\nSplit distribution:")
print(split_stats)

block_splits = (master.groupby(["nm_id", "scan_rate_mVs", "half_sweep_id"])["split"]
                      .nunique())
assert (block_splits == 1).all(), "A half-sweep block spans multiple splits — leakage!"
print("\\n[PASS] Every half-sweep block belongs to exactly one split partition")
print("[PASS] Zero autocorrelation leakage across split boundaries")""")

code("""\
# ── Visualisation: split map ──────────────────────────────────────────────────
split_colour = {
    "train"        : "#2ca02c",
    "val"          : "#ff7f0e",
    "test_scanrate": "#1f77b4",
    "test_material": "#d62728",
}

fig, ax = plt.subplots(figsize=(13, 4))

block_df = (master.groupby(["nm_id", "scan_rate_mVs", "half_sweep_id"])["split"]
                  .first().reset_index())
block_df["nm_idx"] = block_df["nm_id"].map({nm: i for i, nm in enumerate(SHEET_NAMES)})
block_df["sr_idx"] = block_df["scan_rate_mVs"].map(
    {sr: i for i, sr in enumerate(SCAN_RATES)})
block_df["hs_idx"] = block_df["half_sweep_id"]

for _, row in block_df.iterrows():
    x = row["sr_idx"] * 4 + row["hs_idx"] + row["nm_idx"] * 0.07
    y = row["nm_idx"]
    ax.scatter(x, y, c=split_colour[row["split"]], s=40, marker="s",
               alpha=0.85, edgecolors="none")

from matplotlib.patches import Patch
legend_elements = [Patch(fc=c, label=s) for s, c in split_colour.items()]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
ax.set_yticks(range(len(SHEET_NAMES)))
ax.set_yticklabels(SHEET_NAMES)
ax.set_xlabel("(scan_rate x 4 + half_sweep_id) position")
ax.set_ylabel("Material")
ax.set_title("Split Map: every square = one half-sweep block (~650 rows)")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_split_map.png", bbox_inches="tight")
plt.show()
print("Saved 07_split_map.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 8 — Leakage-Free Normalisation

### Two-tier normalisation strategy

**Tier A — Global features** (potential, scan rate and its transforms):
Fit min-max scalers on the **training set only**, apply to all partitions.
Global is correct because the voltage window and scan-rate range are fixed
experimental conditions.

**Tier B — Target current, per (nm_id, scan_rate) group**:
Fit separate min-max scalers for each group on **training rows only**.
Removes scale differences between materials and scan rates so the model
learns *curve shape*, not magnitude.

All scaler parameters are stored in `scalers.json` for deployment
inverse-transformation.

### Critical rule
Never re-fit scalers on validation or test data — that leaks test statistics
into preprocessing. For groups with no training rows (NM4, SR=30, SR=50),
the group's own statistics are used for normalisation only; the model never
trains on these rows.""")

code("""\
train_mask = master["split"] == "train"
train_df   = master[train_mask].copy()

# ═══════════════════════════════════════════════════════════════════════════
# TIER A — Global feature scalers (fit on training rows only)
# ═══════════════════════════════════════════════════════════════════════════
GLOBAL_FEATURES = [
    "potential_V",
    "scan_rate_mVs",
    "log_scan_rate",
    "sqrt_scan_rate",
    "potential_from_lower",
    "potential_from_upper",
    "sr_x_potential",
    "direction_x_potential",
]

global_scalers = {}
for feat in GLOBAL_FEATURES:
    f_min   = float(train_df[feat].min())
    f_max   = float(train_df[feat].max())
    f_range = f_max - f_min
    global_scalers[feat] = {"min": f_min, "max": f_max}
    norm_col = feat + "_norm"
    master[norm_col] = (master[feat] - f_min) / f_range if f_range > 0 else 0.0

# sweep_position and sweep_direction are already in [0,1] / {0,1} — no scaling needed
# dV sign is captured by sweep_direction — not used as a model input

print("✓ Tier-A global feature scalers fitted on training data")
for feat in GLOBAL_FEATURES:
    s = global_scalers[feat]
    print(f"  {feat:30s}: [{s['min']:.4f}, {s['max']:.4f}]")""")

code("""\
# ═══════════════════════════════════════════════════════════════════════════
# TIER B — Per-(nm_id, scan_rate) current scalers (fit on training rows only)
# ═══════════════════════════════════════════════════════════════════════════
current_scalers = {}

master["current_normalized"] = np.nan
master["current_norm_min"]   = np.nan
master["current_norm_max"]   = np.nan

for (nm_id, sr), group_idx in master.groupby(["nm_id", "scan_rate_mVs"]).groups.items():
    key = f"{nm_id}_{int(sr)}"

    train_group_mask = ((master["split"] == "train") &
                        (master["nm_id"] == nm_id) &
                        (master["scan_rate_mVs"] == sr))
    train_group = master.loc[train_group_mask, "current_A"]

    if len(train_group) == 0:
        # NM4 and test/val scan rates have no training rows.
        # Use group statistics for normalisation only (not for training).
        group_currents = master.loc[group_idx, "current_A"]
        c_min = float(group_currents.min())
        c_max = float(group_currents.max())
    else:
        c_min = float(train_group.min())
        c_max = float(train_group.max())

    c_range = c_max - c_min
    current_scalers[key] = {"min": c_min, "max": c_max, "range": c_range}

    if c_range > 0:
        master.loc[group_idx, "current_normalized"] = (
            (master.loc[group_idx, "current_A"] - c_min) / c_range)
    else:
        master.loc[group_idx, "current_normalized"] = 0.0

    master.loc[group_idx, "current_norm_min"] = c_min
    master.loc[group_idx, "current_norm_max"] = c_max

print(f"✓ Tier-B current scalers fitted for {len(current_scalers)} groups")
print(f"  NaN current_normalized : {master['current_normalized'].isna().sum()}")
print(f"  Min current_normalized : {master['current_normalized'].min():.6f}")
print(f"  Max current_normalized : {master['current_normalized'].max():.6f}")

norm_min = master["current_normalized"].min()
norm_max = master["current_normalized"].max()
assert abs(norm_min) < 1e-9,       f"Normalised current min not 0: {norm_min}"
assert abs(norm_max - 1.0) < 1e-9, f"Normalised current max not 1: {norm_max}"
print("[PASS] current_normalized in [0.0, 1.0]")""")

code("""\
# ── Verification: inverse-transform round-trip accuracy ──────────────────────
print("=== Inverse-transform round-trip verification ===\\n")
sample_groups = [("NM1", 20.0), ("NM2", 50.0), ("NM3", 100.0), ("NM4", 10.0)]
max_error = 0.0

for nm_id, sr in sample_groups:
    key = f"{nm_id}_{int(sr)}"
    s   = current_scalers[key]
    sub = master.query(f"nm_id=='{nm_id}' and scan_rate_mVs=={sr}").head(200)

    reconstructed = sub["current_normalized"] * s["range"] + s["min"]
    error = (reconstructed - sub["current_A"]).abs().max()
    max_error = max(max_error, error)
    print(f"  {nm_id}, SR={sr:4.0f}: max reconstruction error = {error:.2e} A")

assert max_error < 1e-12, f"Round-trip error too large: {max_error}"
print(f"\\n[PASS] Max round-trip reconstruction error: {max_error:.2e} A (< 1e-12)")""")

code("""\
# ── Normalisation verification plot ──────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 6))

for col_i, (nm_id, sr) in enumerate([("NM1",20),("NM1",50),("NM2",20),("NM2",80),
                                       ("NM3",30),("NM3",100),("NM4",50),("NM4",90)]):
    ax  = axes[col_i // 4][col_i % 4]
    sub = master.query(f"nm_id=='{nm_id}' and scan_rate_mVs=={sr}")
    for hs_id in [0, 1]:
        seg = sub[sub["half_sweep_id"] == hs_id]
        d   = seg["sweep_direction"].iloc[0]
        ax.plot(seg["potential_V_norm"], seg["current_normalized"],
                color="#1f77b4" if d == 0 else "#d62728", lw=1.0, alpha=0.85)
    ax.set_title(f"{nm_id} SR={sr}\\nsplit={sub['split'].iloc[0]}", fontsize=8)
    ax.set_xlabel("V_norm", fontsize=7)
    ax.set_ylabel("I_norm", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

plt.suptitle("Normalised CV Curves (all axes in [0,1])", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_normalised_curves.png", bbox_inches="tight")
plt.show()
print("Saved 08_normalised_curves.png")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 9 — Save All Artefacts

### Files written
| File | Format | Contents |
|---|---|---|
| `master_long_format.parquet` | Parquet | Full 104 000-row dataset with all features |
| `train_set.parquet` | Parquet | Training rows only |
| `val_set.parquet` | Parquet | Validation rows only |
| `test_scanrate.parquet` | Parquet | Test-SR rows (unseen scan rate in seen materials) |
| `test_material.parquet` | Parquet | Test-MAT rows (unseen material NM4) |
| `scalers.json` | JSON | All normalisation parameters for deployment |

**Why Parquet:** preserves float64 precision, ~8x more compact than CSV,
and loads ~6x faster. At 10^-4 – 10^-6 A magnitude, CSV round-trip
would introduce rounding errors.""")

code("""\
splits_to_save = {
    "master_long_format": master,
    "train_set"         : master[master["split"] == "train"],
    "val_set"           : master[master["split"] == "val"],
    "test_scanrate"     : master[master["split"] == "test_scanrate"],
    "test_material"     : master[master["split"] == "test_material"],
}

for name, df in splits_to_save.items():
    path    = PROCESSED_DIR / f"{name}.parquet"
    df.reset_index(drop=True).to_parquet(path, index=False)
    size_kb = path.stat().st_size / 1024
    print(f"  Saved {name:25s}: {len(df):8,} rows  |  {size_kb:7.1f} KB")

scalers_bundle = {
    "version"          : "1.0",
    "electrochemical"  : {
        "V_lower_V"          : V_LOWER,
        "V_upper_V"          : V_UPPER,
        "scan_rate_mVs_range": [min(SCAN_RATES), max(SCAN_RATES)],
        "steps_per_half_sweep": STEPS_PER_HALF_SWEEP,
    },
    "global_features"  : global_scalers,
    "current_per_group": current_scalers,
}

scalers_path = PROCESSED_DIR / "scalers.json"
with open(scalers_path, "w") as f:
    json.dump(scalers_bundle, f, indent=2)
print(f"\\n  Saved scalers.json  ({scalers_path.stat().st_size:,} bytes)")
print(f"\\n✓ All artefacts saved to: {PROCESSED_DIR}")""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FINAL INTEGRITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## Section 10 — Final Integrity Check

A comprehensive set of assertions that verify the entire pipeline end-to-end.
**All checks must pass before proceeding to ML model training.**""")

code("""\
print("=" * 60)
print("FINAL PREPROCESSING INTEGRITY CHECK")
print("=" * 60)
passed = 0
failed = 0

def check(condition: bool, label: str, detail: str = ""):
    global passed, failed
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status} {label}" + (f" -- {detail}" if detail else ""))
    if condition:
        passed += 1
    else:
        failed += 1

# 1. Total row count
check(len(master) == EXPECTED_TOTAL_ROWS,
      "Total row count",
      f"{len(master):,} == {EXPECTED_TOTAL_ROWS:,}")

# 2. All 4 splits present
check(set(master["split"].unique()) == {"train","val","test_scanrate","test_material"},
      "All 4 split labels present")

# 3. No half-sweep block spans two splits
block_split_count = (master.groupby(["nm_id","scan_rate_mVs","half_sweep_id"])["split"]
                           .nunique())
check((block_split_count == 1).all(),
      "No half-sweep block crosses split boundary (no leakage)")

# 4. NM4 entirely in test_material
nm4_splits = master[master["nm_id"] == "NM4"]["split"].unique()
check(set(nm4_splits) == {"test_material"},
      "NM4 entirely in test_material", str(nm4_splits))

# 5. SR=30 entirely in val for NM1/2/3
sr30 = master.query("nm_id != 'NM4' and scan_rate_mVs == 30")["split"].unique()
check(set(sr30) == {"val"},
      "SR=30 entirely in val", str(sr30))

# 6. SR=50 entirely in test_scanrate for NM1/2/3
sr50 = master.query("nm_id != 'NM4' and scan_rate_mVs == 50")["split"].unique()
check(set(sr50) == {"test_scanrate"},
      "SR=50 entirely in test_scanrate", str(sr50))

# 7. current_normalized in [0, 1]
check(master["current_normalized"].between(-1e-9, 1 + 1e-9).all(),
      "current_normalized in [0, 1] for all rows")

# 8. No NaN in current_normalized
check(master["current_normalized"].isna().sum() == 0,
      "No NaN in current_normalized")

# 9. Direction alternates 0->1->0->1 across half-sweeps in every group
dir_ok = True
for (nm, sr), g in master.groupby(["nm_id","scan_rate_mVs"]):
    hs_dirs = g.groupby("half_sweep_id")["sweep_direction"].first().values
    if not np.array_equal(hs_dirs, [0, 1, 0, 1]):
        dir_ok = False
        break
check(dir_ok, "Sweep direction alternates 0->1->0->1 in every group")

# 10. potential_V_norm in [0, 1]
check(master["potential_V_norm"].between(-1e-9, 1 + 1e-9).all(),
      "potential_V_norm in [0, 1]")

# 11. sweep_position in [0, 1]
check(master["sweep_position"].between(-1e-9, 1 + 1e-9).all(),
      "sweep_position in [0, 1]")

# 12. Parquet files reload correctly
for name in ["master_long_format","train_set","val_set","test_scanrate","test_material"]:
    df_reload = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    check(len(df_reload) == len(splits_to_save[name]),
          f"Reload check: {name}", f"{len(df_reload):,} rows")

# 13. scalers.json contains all 40 group scalers
with open(PROCESSED_DIR / "scalers.json") as f:
    sc = json.load(f)
expected_keys = {f"{nm}_{sr}" for nm in SHEET_NAMES for sr in SCAN_RATES}
actual_keys   = set(sc["current_per_group"].keys())
check(expected_keys == actual_keys,
      "scalers.json contains all 40 (nm x sr) current scalers")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print(f"{'='*60}")
print(f"  Result: {passed} passed, {failed} failed")
if failed == 0:
    print("  ALL CHECKS PASSED -- Dataset is ready for ML training")
else:
    print("  FAILURES DETECTED -- Review and fix before proceeding")
print(f"{'='*60}")""")

code("""\
# ── Final dataset overview ─────────────────────────────────────────────────────
print("\\n=== PREPROCESSED DATASET SUMMARY ===\\n")

summary = master.groupby("split").agg(
    rows        = ("current_A",     "count"),
    materials   = ("nm_id",         lambda x: list(x.unique())),
    scan_rates  = ("scan_rate_mVs", lambda x: sorted(x.unique().tolist())),
    half_sweeps = ("half_sweep_id", "nunique"),
    I_mean_uA   = ("current_A",     lambda x: x.mean() * 1e6),
    I_std_uA    = ("current_A",     lambda x: x.std()  * 1e6),
).rename(columns={"I_mean_uA": "mean I (uA)", "I_std_uA": "std I (uA)"})
print(summary.to_string())

print(
    "\\nColumns in master dataset:"
    "\\n  Identity  : nm_id, scan_rate_mVs, cycle_id, half_sweep_id, step_index"
    "\\n  Raw       : potential_V, current_A, dV"
    "\\n  Direction : sweep_direction, sweep_position"
    "\\n  Engineered: potential_from_lower, potential_from_upper,"
    "\\n              log_scan_rate, sqrt_scan_rate,"
    "\\n              direction_x_potential, sr_x_potential"
    "\\n  Normalised: potential_V_norm, scan_rate_mVs_norm, log_scan_rate_norm,"
    "\\n              sqrt_scan_rate_norm, potential_from_lower_norm,"
    "\\n              potential_from_upper_norm, sr_x_potential_norm,"
    "\\n              direction_x_potential_norm, current_normalized"
    "\\n  Split     : split"
    "\\n  Scaler ref: current_norm_min, current_norm_max"
    "\\n"
    "\\nReady for:  ANN  |  Random Forest  |  XGBoost  |  LSTM/sequence models"
)

if UPLOAD_METHOD == "upload":
    print("\\n--- NOTE ---")
    print("You used direct upload. Outputs are saved to /content/ZnO_PBL_Project/")
    print("Download the parquet files and scalers.json before closing this session")
    print("or they will be lost. Use the Files panel on the left, or run:")
    print("  from google.colab import files")
    print("  files.download('/content/ZnO_PBL_Project/data/processed/master_long_format.parquet')")""")

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

out_path = Path(__file__).resolve().parent.parent.parent / "research" / "notebooks" / "01_correct_preprocessing.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out_path))
print(f"Written : {out_path}")
print(f"Cells   : {len(nb.cells)}")
