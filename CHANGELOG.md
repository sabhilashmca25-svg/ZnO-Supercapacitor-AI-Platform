# Changelog

All notable changes to ZnO Supercapacitor AI Platform.

## [1.0.0] — 2026-05-16

### Added — ML Experimentation Phase (Complete)
- Notebook 01: Leakage-safe preprocessing with 4-partition split
- Notebook 02: Random Forest baseline (Val RMSE 26.59 µA, R² 0.985)
- Notebook 03: XGBoost model (Val RMSE 28.83 µA, from final comparison evaluation)
- Notebook 04: Dense ANN model (Val RMSE 49.93 µA)
- Notebook 05: Stacked LSTM with sequence packing (Val RMSE 36.51 µA)
- Notebook 06: Stacked GRU — best R² on NM4 extrapolation (0.9751)
- Notebook 07: LightGBM — best speed-accuracy-size balance
- Notebook 08: Final 6-model comparison with 12 publication figures

### Added — System Integration Phase (In Progress)
- Professional project structure: backend, frontend, models, research
- FastAPI backend scaffold with all 6 model endpoints
- React PWA frontend scaffold with 8 pages
- Docker + Render + Vercel deployment configs
