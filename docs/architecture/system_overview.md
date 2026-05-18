# System Architecture Overview

## Components
1. **Frontend PWA** (React + MUI + Plotly) — deployed on Vercel
2. **Backend API** (FastAPI + Python) — deployed on Render
3. **ML Models** (6 trained models) — loaded at backend startup
4. **Research Layer** (Notebooks + Figures) — static artefacts

## Data Flow
```
User Input (scan rate, voltage range)
  → Frontend (React form)
  → POST /api/v1/predict/{model}
  → Feature Engineering (feature_engineer.py)
  → Normalisation (normalizer.py)
  → Model Inference (predictor_factory.py)
  → Denormalisation (denormalizer.py)
  → JSON Response
  → Plotly CV curve render
```
