# ZnO Supercapacitor AI Platform

> **AI-Based Research Platform for Predicting Supercapacitor Performance
> Using Machine Learning**
>
> MCA 2nd Semester PBL Project — Full-Stack Research Platform

---

> [!IMPORTANT]
> ## Before You Clone — Install These Three Things First
>
> You need **all three** installed on your machine before cloning.
> Missing any one of them will prevent the app from running.
>
> ### 1. Python 3.11 or newer
> - Download from **https://python.org/downloads**
> - During installation, **tick the "Add Python to PATH" checkbox** — without this, nothing works
> - Python 3.10 and older are **NOT supported** (TensorFlow 2.18 requires Python 3.11+)
>
> ### 2. Node.js LTS
> - Download from **https://nodejs.org** — choose the **LTS** version
> - Use default installation options
>
> ### 3. Git LFS (Large File Storage) — easy to miss!
> - Download from **https://git-lfs.com** and install it
> - Then open a terminal and run: `git lfs install`
> - **Why:** The Random Forest model is 201 MB — too large for regular Git.
>   It is stored via Git LFS. If you clone without Git LFS installed,
>   you get a 133-byte placeholder file instead of the real model,
>   and RF predictions will silently fail.
>
> ---
>
> **After installing all three, clone and run:**
> ```bash
> git lfs install
> git clone https://github.com/sabhilashmca25-svg/ZnO-Supercapacitor-AI-Platform.git
> cd ZnO-Supercapacitor-AI-Platform
> ```
> Then double-click **`START_APP.bat`** (Windows).
> The first launch downloads ~500 MB of packages automatically — keep internet on.

---

## What This Platform Does

- **Predict** full CV current–voltage trajectories for ZnO supercapacitors using 6 ML models
- **Compare** model predictions side-by-side with publication-quality Plotly charts
- **Explore** benchmark leaderboards, RMSE/R² metrics, and training histories
- **Validate** predictions against real experimental data with residual analysis
- **Run offline** — Progressive Web App with service worker caching

---

## Model Performance Summary

| Model | Val RMSE | TestMAT R² | Size | Notes |
|-------|----------|------------|------|-------|
| Random Forest | 26.59 µA | 0.9707 | 201 MB | Via Git LFS |
| LightGBM | 26.86 µA | 0.9678 | 1.7 MB | ✅ Recommended |
| XGBoost | 28.83 µA | 0.9668 | 0.3 MB | ✅ |
| GRU | 36.84 µA | 0.9751 | 0.3 MB | ✅ Best extrapolation |
| LSTM | 36.51 µA | 0.9674 | 0.4 MB | ✅ |
| ANN | 49.93 µA | 0.9606 | 0.2 MB | ✅ |

---

## System Requirements

| Tool | Minimum | Why it matters |
|------|---------|----------------|
| **Python 3.11+** | 3.11 (3.13 supported) | TensorFlow 2.18 does not support Python 3.10 or older — install will fail |
| **Node.js LTS** | 18 LTS | Runs the React frontend dev server |
| **Git LFS** | Any | RF model is 201 MB; without LFS you get a broken placeholder file |
| RAM | 8 GB | 16 GB recommended — RF model occupies ~4 GB when loaded |
| OS | Windows 10+ / macOS 12+ / Ubuntu 20.04+ | `START_APP.bat` is Windows only; use `start_app.sh` on Mac/Linux |

> [!CAUTION]
> **Python 3.10 and below will not work.** `pip install -r requirements.txt` will fail
> because TensorFlow 2.18 only supports Python 3.11, 3.12, and 3.13.

---

## Quick Start (Windows — Recommended)

> [!WARNING]
> Complete **all steps below in order**. Skipping step 1 or 2 means the app will not start correctly.

**Step 1 — Install prerequisites (one-time, do this before anything else)**

| What | Where | Notes |
|------|-------|-------|
| Python 3.11+ | https://python.org/downloads | Tick **"Add Python to PATH"** during install |
| Node.js LTS | https://nodejs.org | Use default options |
| Git LFS | https://git-lfs.com | Then run `git lfs install` in a terminal |

**Step 2 — Clone the repository**

```bash
git lfs install
git clone https://github.com/sabhilashmca25-svg/ZnO-Supercapacitor-AI-Platform.git
cd ZnO-Supercapacitor-AI-Platform
```

> `git lfs install` must be run **before** `git clone`, not after.
> Cloning without it downloads a 133-byte pointer instead of the 201 MB RF model.

**Step 3 — Launch**

Double-click **`START_APP.bat`**

On first launch it will automatically:
- Create a Python virtual environment
- Install all Python packages (TensorFlow ~350 MB — takes 10–30 min on slow connections)
- Install all Node.js packages (~2 min)
- Start the backend and frontend
- Open the app in your browser at `http://localhost:5173`

**To stop:** double-click `STOP_APP.bat`

---

## Quick Start (macOS / Linux)

```bash
# Install Git LFS (one-time)
brew install git-lfs          # macOS
sudo apt install git-lfs      # Ubuntu/Debian

# Clone (LFS downloads RF model automatically)
git lfs install
git clone <repository-url>
cd ZnO_Supercapacitor_AI_Platform

# Launch
bash start_app.sh
```

**To stop:** press `Ctrl+C` in the terminal, or run `bash stop_app.sh`

---

## Manual Setup (Backend)

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # macOS / Linux

pip install -r requirements.txt

# Optional: copy .env.example to .env to customise
cp .env.example .env

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health
- First startup loads all ML models — allow ~30–60 s (RF takes ~45 s)

---

## Manual Setup (Frontend)

```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

> Vite proxies `/api/*` → `http://127.0.0.1:8000` automatically — no `.env` needed.

---

## Git LFS Details

The Random Forest model (`models/rf/rf_baseline.joblib`, 201 MB) is stored with Git LFS
because it exceeds GitHub's 100 MB per-file limit. The remaining 5 models are committed
directly to git as they are all under 2 MB.

**Cloning with LFS (standard):**
```bash
git lfs install   # once per machine
git clone <url>   # LFS downloads RF model automatically
```

**Cloning without LFS (LFS not installed):**
```bash
git clone <url>   # RF model is a 134-byte LFS pointer, NOT the real file
```
In this case, RF predictions will fail with a "model file not found" error.
All other 5 models work normally. To get the RF model later: `git lfs pull`

**Without running RF:**
Edit `backend/.env` and set:
```
ENABLED_MODELS=["lightgbm","gru","xgboost","ann","lstm"]
```

---

## Model Files

| Model | File | Size | Storage |
|-------|------|------|---------|
| Random Forest | `models/rf/rf_baseline.joblib` | 201 MB | Git LFS |
| LightGBM | `models/lightgbm/lightgbm_model.joblib` | 1.7 MB | Git |
| XGBoost | `models/xgboost/xgboost_model.joblib` | 0.27 MB | Git |
| GRU | `models/gru/gru_model.keras` | 0.34 MB | Git |
| LSTM | `models/lstm/lstm_model.keras` | 0.43 MB | Git |
| ANN | `models/ann/ann_model.keras` | 0.18 MB | Git |

---

## Project Structure

```
ZnO_Supercapacitor_AI_Platform/
├── START_APP.bat           One-click launcher (Windows)
├── STOP_APP.bat            One-click stopper (Windows)
├── start_app.sh            Shell launcher (macOS / Linux)
├── stop_app.sh             Shell stopper (macOS / Linux)
├── launcher.py             Python launcher (starts backend + frontend)
├── stop_app.py             Python stopper (reads .launcher.lock)
│
├── backend/
│   ├── app/
│   │   ├── main.py         FastAPI application factory
│   │   ├── api/v1/         Route definitions (9 endpoint modules)
│   │   ├── core/           Config, exceptions, model registry
│   │   ├── services/       Prediction pipeline, preprocessing
│   │   └── schemas/        Pydantic request/response models
│   ├── requirements.txt    Python dependencies
│   └── .env.example        Environment configuration template
│
├── frontend/
│   ├── src/
│   │   ├── pages/          9 route pages (Dashboard, Prediction, etc.)
│   │   ├── components/     Reusable UI components
│   │   ├── hooks/          React hooks (health, prediction, PWA)
│   │   ├── store/          Redux state (prediction, UI)
│   │   ├── api/            Axios client + endpoint functions
│   │   └── types/          TypeScript type definitions
│   ├── public/             Static assets + PWA manifest
│   └── vite.config.ts      Vite build configuration
│
├── models/
│   ├── rf/                 Random Forest (201 MB via Git LFS)
│   ├── lightgbm/           LightGBM (1.7 MB)
│   ├── xgboost/            XGBoost (0.27 MB)
│   ├── gru/                GRU (0.34 MB)
│   ├── lstm/               LSTM (0.43 MB)
│   ├── ann/                ANN (0.18 MB)
│   └── shared/             Scalers, feature config, model registry
│
├── data/
│   └── processed/          master_long_format.parquet (experimental CV data)
│
└── research/
    ├── metrics/            Per-model JSON metric files
    ├── exports/            Benchmark summary JSON
    ├── per_group_csv/      Per-group RMSE CSV files
    ├── training_history/   Epoch-by-epoch training logs (JSON)
    └── figures/            Publication figures (PNG)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness probe |
| GET | `/api/v1/health/ready` | Readiness probe (hot models loaded?) |
| GET | `/api/v1/health/system` | Uptime, cache stats, model sizes |
| POST | `/api/v1/predict` | Single-model CV curve prediction |
| POST | `/api/v1/compare` | Multi-model comparison |
| GET | `/api/v1/models` | Registered model list |
| GET | `/api/v1/metrics` | All model evaluation metrics |
| GET | `/api/v1/metrics/{model_id}` | Single model metrics |
| GET | `/api/v1/benchmarks/leaderboard` | Ranked leaderboard |
| GET | `/api/v1/benchmarks/comparison` | Cross-partition comparison table |
| GET | `/api/v1/benchmarks/summary` | Benchmark summary + key findings |
| GET | `/api/v1/training-history` | All training curves |
| GET | `/api/v1/training-history/{model_id}` | Single model training history |
| GET | `/api/v1/experimental/{material}/{scan_rate}` | Experimental CV curve |
| POST | `/api/v1/validation/compare` | Predicted vs experimental overlay |
| GET | `/api/v1/validation/per-group/{model_id}` | Per-group RMSE heatmap data |

---

## Troubleshooting

**RF model missing after clone**
```
git lfs pull
```
Or check that `git lfs install` was run before cloning.

**`python` not recognised on Windows**
Install from https://python.org/downloads — check "Add Python to PATH".

**`node` not recognised**
Install LTS from https://nodejs.org — use default options.

**Backend port already in use**
The launcher auto-selects a free port. If port 8000 is busy, the backend
starts on the next available port automatically.

**Frontend shows "Backend unreachable"**
Wait 30–60 s for ML models to finish loading. The status badge turns green
automatically once the backend is ready.

**TensorFlow import error on first run**
Run `pip install -r backend/requirements.txt` to install or update TF.

**DLL load error on Windows after pip install**
Run `START_APP.bat` again — it automatically unblocks `.pyd`/`.dll` files
that Windows marks as downloaded-from-internet.

---

## Research Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_correct_preprocessing.ipynb` | Leakage-safe 4-partition split |
| 02 | `02_rf_baseline.ipynb` | Random Forest baseline |
| 03 | `03_xgboost_model.ipynb` | XGBoost gradient boosting |
| 04 | `04_ann_model.ipynb` | Dense neural network |
| 05 | `05_lstm_model.ipynb` | Stacked LSTM sequence model |
| 06 | `06_gru_model.ipynb` | Stacked GRU sequence model |
| 07 | `07_lightgbm_model.ipynb` | LightGBM leaf-wise boosting |
| 08 | `08_final_comparison.ipynb` | 6-model final comparison |

All notebooks live in `research/notebooks/`.

---

## License

Academic use. All rights reserved.
