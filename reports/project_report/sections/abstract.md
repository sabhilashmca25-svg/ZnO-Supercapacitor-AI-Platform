# Abstract

This project presents an AI-powered platform for predicting cyclic voltammetry (CV)
trajectories of ZnO-based supercapacitors using six machine learning models: Random Forest,
XGBoost, LightGBM, Dense ANN, Stacked LSTM, and Stacked GRU.

A leakage-safe 4-partition split (train / val / test-SR / test-MAT) ensures genuine
generalisation testing across unseen scan rates and unseen material formulations.

Key result: Random Forest achieves the best interpolation accuracy (Val RMSE 26.59 µA,
R² 0.985), while GRU achieves the highest material-extrapolation R² (0.9751).
LightGBM provides the optimal deployment balance (1.7 MB, Val RMSE 26.86 µA).
