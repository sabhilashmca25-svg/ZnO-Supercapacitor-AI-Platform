# ZnO Supercapacitor AI Platform

> **AI-Based Mobile Application for Predicting Supercapacitor Performance
> Using Machine Learning**
>
> MCA 2nd Semester PBL Project — Full-Stack Research Platform

---

## Live Platform
| Service | URL |
|---------|-----|
| Frontend (PWA) | _https://zno-supercapacitor-ai.vercel.app_ |
| Backend API    | _https://zno-supercapacitor-api.onrender.com_ |
| API Docs (Swagger) | _https://zno-supercapacitor-api.onrender.com/docs_ |

---

## What This Platform Does

Users can:
- **Predict** full CV current–voltage trajectories for ZnO supercapacitors
- **Compare** predictions from 6 ML models side-by-side
- **Explore** publication-quality benchmarking figures
- **Analyse** interpolation vs extrapolation performance
- **Study** model architectures, parameters, and trade-offs

---

## Model Performance Summary

| Model | Val RMSE | TestMAT R² | Size | Deploy |
|-------|----------|-----------|------|--------|
| Random Forest | 26.59 µA | 0.9707 | 615 MB | Lab only |
| XGBoost | 28.83 µA | 0.9668 | 0.3 MB | ✅ |
| LightGBM | 26.86 µA | 0.9678 | 1.7 MB | ✅ Recommended |
| ANN | 49.93 µA | 0.9606 | 0.2 MB | ✅ |
| LSTM | 36.51 µA | 0.9674 | 0.4 MB | ✅ |
| GRU | 36.84 µA | 0.9751 | 0.3 MB | ✅ Best extrapolation |

---

## Project Structure

```
ZnO_Supercapacitor_AI_Platform/
├── backend/        FastAPI Python API (6 model endpoints)
├── frontend/       React + MUI + Plotly PWA (8 pages)
├── models/         Trained model files + metadata
├── research/       Metrics, exports, training histories (ML phase)
├── data/           Processed parquet datasets + scalers
├── src/            Notebook builder scripts
└── docs/           Technical documentation
```

---

## System Requirements

| Tool | Minimum version |
|------|----------------|
| Python | 3.11 |
| Node.js | 18 |
| Git | Any recent version |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 |

> TensorFlow (required for GRU / LSTM / ANN) supports Python 3.11–3.13.

---

## Quick Start

### Backend
```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

# (Optional) copy .env.example to .env to customise settings
# cp .env.example .env

uvicorn app.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
# Health API: http://localhost:8000/api/v1/health
```

> First startup loads all ML models — allow ~30–60 s before the API badge turns green.

> **No RF model?** The backend starts fine without `models/rf/rf_baseline.joblib`. All 5 other models (LightGBM, XGBoost, GRU, LSTM, ANN) load and serve predictions normally. RF predictions return an error until the file is placed manually.

### Frontend
```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

> The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000` automatically — no `.env` needed for local development.

### Windows one-click
Double-click **`START_APP.bat`** at the project root — it launches both servers and opens the browser automatically.

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

## Model Files

| Model | File | Size | Committed? |
|---|---|---|---|
| Random Forest | `models/rf/rf_baseline.joblib` | ~201 MB | ❌ — exceeds GitHub 100 MB limit |
| LightGBM | `models/lightgbm/lightgbm_model.joblib` | 1.7 MB | ✓ |
| XGBoost | `models/xgboost/xgboost_model.joblib` | 0.27 MB | ✓ |
| GRU | `models/gru/gru_model.keras` | 0.34 MB | ✓ |
| LSTM | `models/lstm/lstm_model.keras` | 0.43 MB | ✓ |
| ANN | `models/ann/ann_model.keras` | 0.18 MB | ✓ |

> Place `rf_baseline.joblib` manually in `models/rf/` after cloning to enable Random Forest predictions. All 5 other models work without it.

---

## License
Academic use. All rights reserved.
