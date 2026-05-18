# Live Demo Script

## Slide 1 — Introduction (30 sec)
Introduce the problem: predicting CV curves without running physical experiments.

## Slide 2 — Dataset (1 min)
Show NM1–NM4 materials, 10 scan rates, the 4-partition split diagram.

## Slide 3 — Live Demo (3 min)
1. Open the platform at https://zno-supercapacitor-ai.vercel.app
2. Go to Prediction Studio → select LightGBM → scan rate 30 mV/s → Predict
3. Go to Model Comparison → select all 6 → show overlay plot
4. Go to Benchmarks → show leaderboard and radar chart

## Slide 4 — Key Findings (1 min)
- RF: best interpolation (26.59 µA)
- GRU: best new-material R² (0.9751)
- LightGBM: best for mobile (1.7 MB)
