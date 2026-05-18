# ZnO Supercapacitor AI Platform — Live Deployment Guide

**Target:** GitHub → Render (backend) + Vercel (frontend)

---

## STEP 1 — Push to GitHub

### 1.1 Create a new GitHub repository
1. Go to https://github.com/new
2. Name: `ZnO-Supercapacitor-AI-Platform` (or any name you prefer)
3. Visibility: Public or Private
4. **Do NOT** initialise with README (repo already has one)
5. Click **Create repository**

### 1.2 Connect and push

```bash
cd "D:\mca\2nd semester\ZnO_Supercapacitor_AI_Platform"

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/ZnO-Supercapacitor-AI-Platform.git

git push -u origin main
```

> **Expected output:** Files pushed successfully. GitHub will show ~293 files including
> model files (LightGBM 1.7 MB, XGBoost 0.3 MB, ANN/LSTM/GRU < 0.5 MB each).
> RF model is NOT pushed (gitignored — not needed for Render free tier).

---

## STEP 2 — Deploy Backend to Render

### 2.1 Create a new Web Service on Render

1. Go to https://dashboard.render.com → **New → Web Service**
2. Connect your GitHub repo
3. Configure:

| Setting | Value |
|---------|-------|
| Name | `zno-supercapacitor-api` |
| Root Directory | `backend` |
| Environment | `Python 3` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free (or Starter for better performance) |
| Health Check Path | `/api/v1/health` |

### 2.2 Set environment variables (Render Dashboard → Environment tab)

Add each variable individually:

| Key | Value |
|-----|-------|
| `APP_ENV` | `production` |
| `LOG_LEVEL` | `INFO` |
| `DOCS_ENABLED` | `true` |
| `ENABLE_WARMUP` | `true` |
| `WARMUP_MATERIAL` | `NM1` |
| `WARMUP_SCAN_RATE` | `30` |
| `PREDICTION_CACHE_SIZE` | `128` |
| `ENABLED_MODELS` | `["lightgbm","gru","xgboost","ann","lstm"]` |
| `ALLOWED_ORIGINS` | `["https://YOUR-APP.vercel.app"]` |

> **Note:** Set `ALLOWED_ORIGINS` after you have your Vercel URL (Step 3).
> You can temporarily use `["*"]` during initial testing, then lock it down.

### 2.3 Deploy

Click **Create Web Service**. Render will:
1. Clone the repo
2. Run `pip install -r requirements.txt` (~2–4 minutes)
3. Start uvicorn
4. Load ML models (LightGBM, GRU: ~10–20 seconds)
5. Run warm-up inference
6. Pass the health check at `/api/v1/health`

### 2.4 Note your backend URL

It will look like: `https://zno-supercapacitor-api.onrender.com`

### 2.5 Verify backend is live

```bash
# Liveness
curl https://zno-supercapacitor-api.onrender.com/api/v1/health

# Readiness (all hot models loaded)
curl https://zno-supercapacitor-api.onrender.com/api/v1/health/ready

# Test prediction
curl -X POST https://zno-supercapacitor-api.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name":"lightgbm","material_id":"NM1","scan_rate_mVs":30}'
```

Expected `/health/ready` response:
```json
{"status": "ready", "loaded": ["gru", "lightgbm"], "missing": []}
```

---

## STEP 3 — Deploy Frontend to Vercel

### 3.1 Import project on Vercel

1. Go to https://vercel.com/new
2. Import from GitHub → select your repo
3. Configure:

| Setting | Value |
|---------|-------|
| Framework Preset | Vite |
| Root Directory | `frontend` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Install Command | `npm ci` |

### 3.2 Set environment variable

In the Vercel import wizard (or Settings → Environment Variables after deploy):

| Key | Value |
|-----|-------|
| `VITE_API_URL` | `https://zno-supercapacitor-api.onrender.com` |

> Replace the URL with your actual Render backend URL from Step 2.4.

### 3.3 Deploy

Click **Deploy**. Vercel will:
1. Run `npm ci`
2. Run `npm run build` (~50 seconds)
3. Deploy to their CDN
4. Give you a URL like `https://zno-supercapacitor-ai-platform.vercel.app`

### 3.4 Update backend CORS

Go back to Render → Environment → update `ALLOWED_ORIGINS`:
```
["https://YOUR-ACTUAL-APP.vercel.app"]
```

Render redeploys automatically when you save env vars.

---

## STEP 4 — Verify End-to-End

Open your Vercel URL and test each page:

| Page | What to verify |
|------|---------------|
| Landing | Loads instantly (no Plotly at this point) |
| Dashboard | Spark chart appears, API connected |
| Prediction Studio | Run a LightGBM prediction → CV curve renders |
| Model Comparison | Compare 2–3 models → chart updates |
| Validation Analysis | Run validation → experimental overlay works |
| Research Analytics | Training Curves tab loads |
| Benchmark Results | Leaderboard table renders |
| Model Encyclopedia | All 6 model cards visible |

---

## STEP 5 — Common Deployment Failures & Fixes

### Backend

| Problem | Fix |
|---------|-----|
| Build fails: `No module named X` | Check `requirements.txt` is in `backend/` root |
| `ModuleNotFoundError: app.core.config` | Render runs from `backend/` — uvicorn path is `app.main:app` ✓ |
| `ENABLED_MODELS` parse error | Must use JSON array: `["lightgbm","gru"]` not `lightgbm,gru` |
| Health check fails | Check Render logs — model load error or wrong PORT handling |
| 503 on `/predict` | Model not loaded — check `ENABLED_MODELS` includes the requested model |
| CORS error in browser | `ALLOWED_ORIGINS` doesn't include your Vercel domain |
| Out of memory (OOM) | Remove `"rf"` from `ENABLED_MODELS` (needs 640 MB RAM) |

### Frontend

| Problem | Fix |
|---------|-----|
| Build fails: TypeScript error | Run `npx tsc --noEmit` locally to see errors |
| API calls return 404 | `VITE_API_URL` not set — check Vercel env vars |
| CORS error | Backend `ALLOWED_ORIGINS` missing Vercel domain |
| Blank page after deploy | Check browser console — likely `VITE_API_URL` missing |
| Plotly not loading | Check network tab — `vendor-plotly-xxx.js` chunk should load on first chart page |
| PWA not installable | Must be served over HTTPS (Vercel provides this automatically) |

---

## Environment Variables Checklist

### Backend (Render)
- [ ] `APP_ENV` = `production`
- [ ] `DOCS_ENABLED` = `true`
- [ ] `ENABLE_WARMUP` = `true`
- [ ] `PREDICTION_CACHE_SIZE` = `128`
- [ ] `ENABLED_MODELS` = `["lightgbm","gru","xgboost","ann","lstm"]`
- [ ] `ALLOWED_ORIGINS` = `["https://your-app.vercel.app"]`

### Frontend (Vercel)
- [ ] `VITE_API_URL` = `https://your-api.onrender.com`

---

## RAM Budget for Render Free Tier (512 MB)

| Model | Loaded RAM | Status |
|-------|-----------|--------|
| LightGBM | ~50 MB | Loaded at startup |
| GRU | ~150 MB | Loaded at startup |
| XGBoost | ~30 MB | Loaded on first request |
| LSTM | ~150 MB | Loaded on first request |
| ANN | ~50 MB | Loaded on first request |
| RF | ~640 MB | **EXCLUDED** (too large) |
| **Total (all 5)** | **~430 MB** | Within 512 MB limit |

> If all 5 non-RF models are requested in the same session, peak RAM may reach
> ~430 MB. This fits within Render's free 512 MB limit with ~80 MB headroom.

---

## Render Free Tier Notes

- **Cold start:** Render free services spin down after 15 minutes of inactivity.
  First request after a sleep will take 30–60 seconds (model reload + warm-up).
- **Bandwidth:** 100 GB/month on free tier — sufficient for a research platform.
- **Persistent disk:** Not available on free tier. All files come from git at each deploy.
- **Upgrade path:** Render Starter ($7/month) keeps the service always on and can enable RF.
