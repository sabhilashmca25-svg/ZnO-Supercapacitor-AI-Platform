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
├── research/       All notebooks, figures, metrics (ML phase)
├── data/           Processed parquet datasets + scalers
├── src/            Notebook builder scripts
├── deployment/     Render + Vercel + Docker configs
├── reports/        Paper, presentation, project report
└── docs/           Technical documentation
```

---

## Quick Start

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env          # edit MODEL_DIR if needed
uvicorn app.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
# create .env.development with: VITE_API_URL=http://localhost:8000
npm run dev
# App: http://localhost:5173
```

### Full Stack (Docker)
```bash
docker-compose -f deployment/docker/docker-compose.yml up --build
```

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

## Deployment

See `docs/deployment/` for step-by-step guides.
- **Backend → Render.com**: uses `deployment/render/render.yaml`
- **Frontend → Vercel**: uses `deployment/vercel/vercel.json`
- **Self-hosted → Docker**: uses `deployment/docker/docker-compose.yml`

---

## License
Academic use. All rights reserved.
