# ZnO CV Trajectory Prediction — Final Model Comparison Summary
Generated: 2026-05-16 07:45

## Experiment Overview
Six machine-learning models (RF, XGBoost, LightGBM, ANN, LSTM, GRU) were trained
to predict the full cyclic voltammetry (CV) current-voltage trajectory of ZnO-based
supercapacitors from scan-rate and electrode-potential features.

## Key Metrics (RMSE in uA)

| Model     | Val (SR=30) | Test-SR (SR=50) | Test-MAT (NM4) | Val R2   | TestMAT R2 |
|-----------|-------------|-----------------|----------------|----------|------------|
| RF        | 26.59       | 43.71           | 33.65          | 0.9851 | 0.9707 |
| XGBoost   | 28.83       | 45.40           | 36.64          | 0.9800 | 0.9668 |
| LightGBM  | 26.86       | 44.69           | 35.18          | 0.9831 | 0.9678 |
| ANN       | 49.93       | 63.61           | 46.16          | 0.9475 | 0.9606 |
| LSTM      | 36.51       | 52.40           | 38.40          | 0.9717 | 0.9674 |
| GRU       | 36.84       | 48.69           | 34.52          | 0.9723 | 0.9751 |

## Recommendations

1. Best for interpolation (seen scan rates, seen materials): RF
   - Validation RMSE = 26.59 uA, R2 = 0.9851

2. Best for material extrapolation (by R2): GRU
   - Test-MAT R2 = 0.9751, RMSE = 34.52 uA

3. Best for mobile deployment (accuracy + compact size): LightGBM
   - Validation RMSE = 26.86 uA, model size = 1.7 MB
   - (vs RF at 615 MB, trains in ~2 min vs RF's ~4 min)

## Scientific Conclusion
Random Forest delivers the best interpolation performance (validation RMSE) and
marginally the best extrapolation RMSE, while GRU uniquely achieves the highest
variance-normalised extrapolation accuracy (test-MAT R2 = 0.9751).
LightGBM provides the optimal accuracy-footprint trade-off for mobile deployment.
Dense ANN is not recommended for this spatiotemporal regression task.