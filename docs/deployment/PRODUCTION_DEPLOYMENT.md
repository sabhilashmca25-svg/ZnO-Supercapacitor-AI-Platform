# ZnO Supercapacitor AI Platform — Production Deployment Guide

## Architecture

```
Browser → Vercel (React SPA + PWA)
              ↓ API calls
         Render (FastAPI + ML models)
```

---

## 1. Backend — Deploy to Render

### Prerequisites
- Render account: https://render.com
- GitHub repo with this codebase

### Steps

1. **Connect repo in Render dashboard**
   - New → Web Service → Connect your GitHub repo

2. **Configure the service**
   - Root Directory: `backend`
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health Check Path: `/api/v1/health`

3. **Set environment variables** (Render dashboard → Environment):
   ```
   APP_ENV=production
   LOG_LEVEL=INFO
   DOCS_ENABLED=true
   ENABLE_WARMUP=true
   PREDICTION_CACHE_SIZE=128
   ENABLED_MODELS=["lightgbm","gru","xgboost","ann","lstm"]
   ALLOWED_ORIGINS=["https://your-app.vercel.app"]
   ```
   > **Note on RF (615 MB → 211 MB compressed):**
   > Render free tier has 512 MB RAM. RF requires ~640 MB peak RAM.
   > Add `"rf"` to ENABLED_MODELS only on a paid plan (≥2 GB RAM).

4. **Deploy** — Render auto-deploys on push to main.

5. **Note your backend URL** — e.g. `https://zno-supercapacitor-api.onrender.com`

---

## 2. Frontend — Deploy to Vercel

### Prerequisites
- Vercel account: https://vercel.com
- Backend URL from step above

### Steps

1. **Import project in Vercel dashboard**
   - New Project → Import your GitHub repo

2. **Configure build settings**
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`

3. **Set environment variables** (Vercel dashboard → Settings → Environment Variables):
   ```
   VITE_API_URL = https://zno-supercapacitor-api.onrender.com
   ```

4. **Deploy** — Vercel auto-deploys on push to main.

5. **Update backend CORS** — Add your Vercel domain to `ALLOWED_ORIGINS` on Render.

---

## 3. Verify Deployment

After both services are live, check:

```bash
# Backend health
curl https://your-api.onrender.com/api/v1/health

# Backend readiness (all hot models loaded?)
curl https://your-api.onrender.com/api/v1/health/ready

# Test prediction
curl -X POST https://your-api.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name":"lightgbm","material_id":"NM1","scan_rate_mVs":30}'

# Frontend
open https://your-app.vercel.app
```

---

## 4. Model RAM Requirements

| Model     | File Size (compressed) | Peak RAM | Tier |
|-----------|----------------------|----------|------|
| RF        | 211 MB               | ~640 MB  | Hot  |
| LightGBM  | 1.7 MB               | ~50 MB   | Hot  |
| GRU       | 0.34 MB              | ~150 MB  | Hot  |
| XGBoost   | 0.27 MB              | ~30 MB   | Lazy |
| LSTM      | 0.43 MB              | ~150 MB  | Lazy |
| ANN       | 0.18 MB              | ~50 MB   | Lazy |

**Free Render tier (512 MB RAM):** Use `ENABLED_MODELS=["lightgbm","gru","xgboost","ann","lstm"]`

**Render Starter plan (2 GB RAM):** Can add `"rf"` to ENABLED_MODELS

---

## 5. Performance Expectations

### Cold start (Render free tier — spins down after inactivity)
- First request after sleep: 30-60s (model loading + warm-up)
- Subsequent requests: < 500ms (LightGBM/XGBoost/GRU)

### Warm start (always-on plans)
- LightGBM prediction: ~50ms
- GRU prediction: ~100ms
- XGBoost prediction: ~80ms
- RF prediction: ~2-4s (not recommended for production API)

### Frontend (Vercel CDN)
- Initial load: ~190 KB gzip (app shell — no Plotly)
- Plotly loaded on first chart page: ~1.4 MB gzip (cached after first visit)
- Subsequent navigations: instant (code-split chunks)

---

## 6. Environment Variables Summary

### Backend (.env)
| Variable | Dev default | Production |
|----------|-------------|------------|
| `APP_ENV` | development | production |
| `DOCS_ENABLED` | true | true |
| `ENABLE_WARMUP` | true | true |
| `PREDICTION_CACHE_SIZE` | 128 | 128 |
| `ENABLED_MODELS` | all 6 | exclude rf |
| `ALLOWED_ORIGINS` | localhost | your Vercel URL |

### Frontend (.env.production)
| Variable | Value |
|----------|-------|
| `VITE_API_URL` | `https://your-api.onrender.com` |
