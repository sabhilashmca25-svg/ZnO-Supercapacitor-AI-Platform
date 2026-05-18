# ZnO Supercapacitor AI Platform — Backend Guide

> **Who is this for?**  
> Complete beginners who want to run the FastAPI backend, test the endpoints,
> and understand what every piece does. No prior FastAPI or ML experience needed.

---

## Table of Contents

1. [What does this backend do?](#1-what-does-this-backend-do)
2. [Folder structure](#2-folder-structure)
3. [One-time setup](#3-one-time-setup)
4. [Running the server](#4-running-the-server)
5. [Testing with Swagger UI](#5-testing-with-swagger-ui)
6. [All endpoints explained](#6-all-endpoints-explained)
7. [Example requests & responses](#7-example-requests--responses)
8. [Environment variables (.env)](#8-environment-variables-env)
9. [Understanding the ML pipeline](#9-understanding-the-ml-pipeline)
10. [Error messages & what they mean](#10-error-messages--what-they-mean)
11. [Connecting the React frontend](#11-connecting-the-react-frontend)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What does this backend do?

This is a **REST API** (a server that answers questions over HTTP) built with
[FastAPI](https://fastapi.tiangolo.com/).

Given:
- A **ZnO nanomaterial** (`NM1`, `NM2`, `NM3`, or `NM4`)
- A **scan rate** (how fast the voltage is swept, in mV/s — values 10, 20 … 100)
- A **model choice** (`rf`, `lightgbm`, `gru`, etc.)

The backend returns a **full CV (Cyclic Voltammetry) curve** — 651 voltage
values and 651 predicted current values — ready to be plotted.

It also serves **pre-computed evaluation metrics** so you can compare model
performance without re-training anything.

---

## 2. Folder structure

```
ZnO_Supercapacitor_AI_Platform/
│
├── backend/                          ← everything in here is this guide
│   ├── .env                          ← your local configuration (edit this)
│   ├── requirements.txt              ← Python packages to install
│   └── app/
│       ├── main.py                   ← entry point — creates the FastAPI app
│       ├── core/
│       │   ├── config.py             ← reads .env, defines all file paths
│       │   ├── model_registry.py     ← loads ML models at startup
│       │   └── exceptions.py        ← custom error types + JSON error handlers
│       ├── api/v1/
│       │   ├── router.py             ← registers all URL routes
│       │   └── endpoints/
│       │       ├── health.py         ← GET  /api/v1/health
│       │       ├── predict.py        ← POST /api/v1/predict
│       │       ├── compare.py        ← POST /api/v1/compare
│       │       ├── models.py         ← GET  /api/v1/models
│       │       ├── metrics.py        ← GET  /api/v1/metrics
│       │       └── benchmarks.py     ← GET  /api/v1/benchmarks/*
│       ├── schemas/
│       │   ├── prediction.py         ← request/response types for /predict
│       │   ├── comparison.py         ← request/response types for /compare
│       │   └── common.py             ← shared types (HealthResponse, etc.)
│       └── services/
│           ├── predictor.py          ← orchestrates the full ML pipeline
│           ├── preprocess.py         ← builds features, normalises them
│           ├── ml/                   ← one file per model type
│           ├── preprocessing/
│           │   ├── feature_engineer.py  ← generates CV sweep + 10 features
│           │   ├── normalizer.py        ← min-max normalisation
│           │   └── sequence_builder.py  ← packs features into 3D tensor (LSTM/GRU)
│           └── postprocessing/
│               └── denormalizer.py   ← converts model output back to µA
│
├── models/                           ← trained model files (large — don't commit)
│   ├── rf/rf_baseline.joblib
│   ├── lightgbm/lightgbm_model.joblib
│   ├── gru/gru_model.keras
│   └── shared/
│       ├── model_registry.json       ← maps model IDs to their file paths
│       └── scalers/scalers.json      ← normalisation parameters (min/max per feature)
│
└── research/
    ├── metrics/                      ← per-model evaluation metrics (JSON)
    └── exports/                      ← leaderboard + comparison tables (CSV/JSON)
```

---

## 3. One-time setup

### Step 1 — Install Python 3.11+

Download from https://www.python.org/downloads/  
During installation, tick **"Add Python to PATH"**.

Verify in a terminal:
```bash
python --version
# Should print: Python 3.11.x or 3.12.x
```

### Step 2 — Create a virtual environment

A virtual environment is a sandboxed Python installation just for this project.
Open a terminal in the `ZnO_Supercapacitor_AI_Platform/` folder, then:

```bash
# Windows (Command Prompt or PowerShell)
python -m venv backend\venv

# macOS / Linux
python3 -m venv backend/venv
```

### Step 3 — Activate the virtual environment

```bash
# Windows Command Prompt
backend\venv\Scripts\activate.bat

# Windows PowerShell
backend\venv\Scripts\Activate.ps1

# macOS / Linux
source backend/venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt. This means the
virtual environment is active. **Always activate it before running the server.**

### Step 4 — Install dependencies

```bash
pip install -r backend/requirements.txt
```

This downloads ~20 packages (FastAPI, TensorFlow, scikit-learn, etc.).
It may take 2–5 minutes the first time.

### Step 5 — Check the .env file

Open `backend/.env` and verify the contents:

```env
APP_ENV=development
LOG_LEVEL=INFO
API_PORT=8000
ENABLED_MODELS=rf,lightgbm,gru
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

**`ENABLED_MODELS`** — which models to load at startup. Loading RF takes ~10 seconds
because it's 615 MB. Start with just `rf,lightgbm,gru` unless you need the others.

---

## 4. Running the server

Make sure your virtual environment is activated (Step 3 above), then:

```bash
# Always run from the backend/ directory
cd backend
uvicorn app.main:app --reload --port 8000
```

**What `--reload` does:** automatically restarts the server when you edit a `.py`
file. Useful during development. Remove it in production.

**Expected startup output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
08:00:01 | INFO     | app.main — ============================================================
08:00:01 | INFO     | app.main — ZnO Supercapacitor AI Platform — starting up
08:00:01 | INFO     | app.main — Environment : development
08:00:01 | INFO     | app.main — Models to load: ['rf', 'lightgbm', 'gru']
08:00:01 | INFO     | app.core.model_registry —   [rf] Loaded via joblib (615.0 MB)
08:00:11 | INFO     | app.core.model_registry —   [lightgbm] Loaded via joblib (1.7 MB)
08:00:11 | INFO     | app.core.model_registry —   [gru] Loaded via Keras
08:00:15 | INFO     | app.main — Models loaded: ['rf', 'lightgbm', 'gru']
08:00:15 | INFO     | app.main — Swagger docs  → http://localhost:8000/docs
08:00:15 | INFO     | app.main — ============================================================
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The server is ready when you see **"Application startup complete."**

---

## 5. Testing with Swagger UI

Open your browser and go to: **http://localhost:8000/docs**

You'll see an interactive page listing every endpoint. Here's how to test one:

### Test the /predict endpoint

1. Click on **POST /api/v1/predict**
2. Click **"Try it out"** (top right of that section)
3. Replace the example request body with:
   ```json
   {
     "model_name": "rf",
     "material_id": "NM1",
     "scan_rate_mVs": 20
   }
   ```
4. Click **"Execute"**
5. Scroll down to see the response — you'll get 651 voltage values and 651
   predicted current values

### Test the /health endpoint

1. Click **GET /api/v1/health**
2. Click **"Try it out"** → **"Execute"**
3. You should see `"status": "ok"` and a list of loaded models

---

## 6. All endpoints explained

| Method | URL | What it does |
|--------|-----|--------------|
| GET | `/` | API info + links to docs |
| GET | `/api/v1/health` | Server status + which models are loaded |
| POST | `/api/v1/predict` | Predict a full CV curve with ONE model |
| POST | `/api/v1/compare` | Run MULTIPLE models on the same input, get overlay data |
| GET | `/api/v1/models` | List all loaded models + their metadata |
| GET | `/api/v1/models/{model_name}` | Metadata for a single model |
| GET | `/api/v1/metrics` | Evaluation metrics for ALL models |
| GET | `/api/v1/metrics/{model_name}` | Evaluation metrics for ONE model |
| GET | `/api/v1/benchmarks/leaderboard` | Ranked comparison of all 6 models |
| GET | `/api/v1/benchmarks/comparison` | Full per-partition metrics table |
| GET | `/api/v1/benchmarks/summary` | Key findings + model size/speed info |

---

## 7. Example requests & responses

### POST /api/v1/predict

**Request:**
```json
{
  "model_name": "lightgbm",
  "material_id": "NM2",
  "scan_rate_mVs": 50
}
```

**Response (abbreviated):**
```json
{
  "model": "lightgbm",
  "material_id": "NM2",
  "scan_rate_mVs": 50.0,
  "n_points": 651,
  "potential_V": [-0.65, -0.648, -0.646, "...(651 values total)..."],
  "predicted_current_uA": [-45.2, -44.8, -43.1, "...(651 values total)..."],
  "statistics": {
    "peak_anodic_uA": 87.3,
    "peak_cathodic_uA": -52.1,
    "current_range_uA": 139.4,
    "integral_area": 29.6
  }
}
```

**How to plot this in Python:**
```python
import requests
import matplotlib.pyplot as plt

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"model_name": "rf", "material_id": "NM1", "scan_rate_mVs": 20}
)
data = response.json()

plt.plot(data["potential_V"], data["predicted_current_uA"])
plt.xlabel("Potential (V)")
plt.ylabel("Current (µA)")
plt.title(f"{data['model']} — {data['material_id']} @ {data['scan_rate_mVs']} mV/s")
plt.show()
```

---

### POST /api/v1/compare

**Request:**
```json
{
  "models": ["rf", "lightgbm", "gru"],
  "material_id": "NM1",
  "scan_rate_mVs": 20
}
```

**Response (abbreviated):**
```json
{
  "material_id": "NM1",
  "scan_rate_mVs": 20.0,
  "n_points": 651,
  "potential_V": [-0.65, -0.648, "..."],
  "predictions": {
    "rf": {
      "predicted_current_uA": [-45.2, -44.8, "..."],
      "peak_anodic_uA": 87.3,
      "peak_cathodic_uA": -52.1
    },
    "lightgbm": {
      "predicted_current_uA": [-44.9, -44.5, "..."],
      "peak_anodic_uA": 86.7,
      "peak_cathodic_uA": -51.8
    }
  },
  "models_failed": []
}
```

**Key:** all models share the same `potential_V` x-axis. Plot each as a
separate trace using `predictions["rf"]["predicted_current_uA"]` etc.

---

### GET /api/v1/benchmarks/leaderboard

**Response (first entry):**
```json
[
  {
    "rank": 1,
    "model_id": "rf",
    "model_display": "RF",
    "composite_rank": 1.0,
    "rmse_val_uA": 26.59,
    "rmse_test_sr_uA": 43.71,
    "rmse_test_mat_uA": 33.65,
    "r2_val": 0.9851,
    "r2_test_sr": 0.9777,
    "r2_test_mat": 0.9707,
    "size_MB": 615.0,
    "train_min": 4
  },
  ...
]
```

---

## 8. Environment variables (.env)

The file `backend/.env` controls the server behaviour without changing code.

| Variable | Default | What it does |
|----------|---------|--------------|
| `APP_ENV` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | Verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `API_PORT` | `8000` | Port to listen on (used by uvicorn command) |
| `ENABLED_MODELS` | `rf,lightgbm,gru` | Comma-separated list of models to load at startup |
| `ALLOWED_ORIGINS` | `http://localhost:5173,...` | CORS whitelist for your frontend |

**Example: load all 6 models**
```env
ENABLED_MODELS=rf,lightgbm,gru,xgboost,ann,lstm
```
> Note: RF is 615 MB and takes ~10 seconds to load. ANN/GRU/LSTM are fast.

**Example: production settings**
```env
APP_ENV=production
LOG_LEVEL=WARNING
ENABLED_MODELS=lightgbm,gru
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

---

## 9. Understanding the ML pipeline

When you call `POST /api/v1/predict`, here's what happens step by step:

```
Request: {model_name: "rf", material_id: "NM1", scan_rate_mVs: 20}
  │
  ├── 1. VALIDATE input (model name, material, scan rate range)
  │
  ├── 2. GENERATE potential sweep
  │       651 points: -0.65V → 0V → -0.65V
  │       Forward (anodic):  -0.65 → 0V  (326 points)
  │       Reverse (cathodic): 0V → -0.65V (325 points)
  │
  ├── 3. COMPUTE 10 electrochemical features per point
  │       potential_V         — the voltage at this point
  │       scan_rate_mVs       — same value repeated (20 mV/s)
  │       log_scan_rate       — log(20) = 2.996
  │       sqrt_scan_rate      — √20 = 4.472
  │       potential_from_lower — V - (-0.65) = V + 0.65
  │       potential_from_upper — 0 - V = -V
  │       sr_x_potential      — 20 × V
  │       direction_x_potential — 1×V (anodic) or 0×V (cathodic)
  │       sweep_direction     — 1.0 (anodic) or 0.0 (cathodic), NOT normalised
  │       sweep_position      — i / 650, from 0.0 to 1.0, NOT normalised
  │
  ├── 4. NORMALISE 8 of the 10 features using min-max scaling
  │       Each feature is scaled to [0, 1] using the stored min/max values
  │       from models/shared/scalers/scalers.json
  │       (sweep_direction and sweep_position are already in [0,1] — not rescaled)
  │
  ├── 5. RUN the model
  │       RF/LightGBM/XGBoost/ANN: predict(features_2D_array)  → (651,)
  │       GRU/LSTM:                predict(features_3D_tensor) → (651,)
  │
  ├── 6. DENORMALISE predictions back to real current values
  │       Using the per-group scaler for "NM1_20" from scalers.json
  │       The stored scalers are in Amperes → multiply by 1,000,000 → µA
  │
  └── 7. COMPUTE statistics (peak currents, range, integral area)
          Return PredictionResponse with all arrays and stats
```

---

## 10. Error messages & what they mean

### 404 — model_not_found
```json
{"error": "model_not_found", "message": "Model 'xyz' is not recognised.", "valid_models": [...]}
```
**Cause:** You typed the model name wrong (e.g. `"random_forest"` instead of `"rf"`).  
**Fix:** Use one of: `rf`, `lightgbm`, `gru`, `xgboost`, `ann`, `lstm`

---

### 503 — model_not_loaded
```json
{"error": "model_not_loaded", "message": "Model 'xgboost' exists but is not loaded.", "hint": "Add this model to ENABLED_MODELS in your .env file and restart."}
```
**Cause:** The model is valid but wasn't in `ENABLED_MODELS` when the server started.  
**Fix:** Add it to `.env`: `ENABLED_MODELS=rf,lightgbm,gru,xgboost` and restart uvicorn.

---

### 422 — validation_error
```json
{"error": "validation_error", "message": "Request body has invalid or missing fields.", "details": [...]}
```
**Cause:** Missing field, wrong type, or value out of range.  
**Fix:** Check the Swagger docs for the exact required fields. Common mistakes:
- `scan_rate_mVs` must be a number, not a string
- `material_id` must be exactly `"NM1"`, `"NM2"`, `"NM3"`, or `"NM4"`

---

### 500 — internal_server_error
```json
{"error": "internal_server_error", "message": "An unexpected error occurred. Check the backend logs."}
```
**Cause:** A bug in the server code or a missing/corrupt model file.  
**Fix:** Look at the terminal where uvicorn is running — the full traceback will
be printed there. Common causes:
- Model file is missing from `models/` directory
- `scalers.json` is missing or corrupt
- Incompatible package versions

---

## 11. Connecting the React frontend

The API is already configured for CORS (Cross-Origin Resource Sharing), so your
React app (running on `localhost:5173` or `localhost:3000`) can call it directly.

**Example using fetch:**
```javascript
const response = await fetch("http://localhost:8000/api/v1/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model_name: "lightgbm",
    material_id: "NM1",
    scan_rate_mVs: 20,
  }),
});
const data = await response.json();
// data.potential_V     → x-axis array
// data.predicted_current_uA → y-axis array
```

**Example using axios:**
```javascript
import axios from "axios";

const { data } = await axios.post("http://localhost:8000/api/v1/predict", {
  model_name: "rf",
  material_id: "NM2",
  scan_rate_mVs: 50,
});
```

For a **Plotly** multi-trace overlay chart using `/api/v1/compare`:
```javascript
const { data } = await axios.post("http://localhost:8000/api/v1/compare", {
  models: ["rf", "lightgbm", "gru"],
  material_id: "NM1",
  scan_rate_mVs: 20,
});

const traces = Object.entries(data.predictions).map(([model_id, pred]) => ({
  x: data.potential_V,
  y: pred.predicted_current_uA,
  name: model_id,
  type: "scatter",
  mode: "lines",
}));
```

---

## 12. Troubleshooting

### "uvicorn: command not found"
The virtual environment is not activated. Run:
```bash
# Windows
backend\venv\Scripts\activate.bat

# macOS/Linux
source backend/venv/bin/activate
```
Then try again.

---

### "ModuleNotFoundError: No module named 'app'"
You must run uvicorn from inside the `backend/` directory:
```bash
cd backend        # ← this is required
uvicorn app.main:app --reload --port 8000
```

---

### "No models loaded — health check shows models_loaded: []"
The model files are missing from the `models/` directory. Verify that these
files exist:
- `models/rf/rf_baseline.joblib`
- `models/lightgbm/lightgbm_model.joblib`
- `models/gru/gru_model.keras`

Also check that `models/shared/model_registry.json` exists.

---

### "scalers_ok: false" in health response
The file `models/shared/scalers/scalers.json` is missing. This file is required
for prediction to work. Check that it exists at that path.

---

### Loading RF is very slow (615 MB)
That is normal. RF requires 615 MB because it stores 300 full decision trees.
While testing, you can set `ENABLED_MODELS=lightgbm,gru` — both are under 2 MB
and load in under a second.

---

### TensorFlow prints lots of warnings on startup
This is normal. TensorFlow prints messages about CUDA, AVX, etc. These are
informational, not errors. Add this to suppress them:
```bash
# Windows
set TF_CPP_MIN_LOG_LEVEL=2

# macOS/Linux
export TF_CPP_MIN_LOG_LEVEL=2
```

---

### Port 8000 already in use
Another process is using port 8000. Either stop it or use a different port:
```bash
uvicorn app.main:app --reload --port 8001
```

---

*Last updated: 2026-05-16 | ZnO Supercapacitor AI Platform v1.0.0*
