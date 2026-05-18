# Deploying Backend to Render.com

1. Push code to GitHub
2. Connect repo to Render
3. Use `deployment/render/render.yaml` as blueprint
4. Add environment variables in Render dashboard:
   - `MODEL_STORAGE=local` (or `cloud` if models are on GCS)
   - `ALLOWED_ORIGINS=https://your-frontend.vercel.app`
5. Upload large model files separately (RF is 615 MB — use cloud storage)
