import type { ModelConfig } from "../types";

// ── Model IDs ─────────────────────────────────────────────────────────────
export const MODEL_IDS = {
  RF: "rf",
  XGBOOST: "xgboost",
  LIGHTGBM: "lightgbm",
  ANN: "ann",
  LSTM: "lstm",
  GRU: "gru",
} as const;

export type ModelId = (typeof MODEL_IDS)[keyof typeof MODEL_IDS];

// ── Leaderboard order ─────────────────────────────────────────────────────
export const MODEL_ORDER: ModelId[] = ["rf", "lightgbm", "xgboost", "gru", "lstm", "ann"];

// ── Color palette for CV curve overlays ──────────────────────────────────
export const MODEL_COLORS: Record<string, string> = {
  rf: "#3b82f6",        // blue
  lightgbm: "#22d3ee",  // cyan
  gru: "#a78bfa",       // violet
  xgboost: "#fb923c",   // orange
  lstm: "#f472b6",      // pink
  ann: "#facc15",       // yellow
};

// ── Complete model metadata ───────────────────────────────────────────────
export const MODELS: ModelConfig[] = [
  {
    id: "rf",
    display: "Random Forest",
    shortName: "RF",
    color: MODEL_COLORS.rf,
    type: "tree",
    description:
      "Ensemble of decision trees trained on CV feature space. Achieves lowest validation RMSE (26.59 µA) at the cost of a 615 MB model file. Ideal when prediction accuracy is the primary concern.",
    architecture: "300 estimators, max_depth=None (unlimited), max_features=sqrt, min_samples_leaf=2",
    strengths: [
      "Best validation RMSE (26.59 µA)",
      "High interpolation accuracy",
      "No hyperparameter sensitivity",
      "Deterministic predictions",
    ],
    weaknesses: [
      "615 MB on-disk footprint",
      "Slow load time (~4 s on CPU)",
      "Not suitable for edge deployment",
    ],
    deployScore: 52,
    size: "615 MB",
  },
  {
    id: "lightgbm",
    display: "LightGBM",
    shortName: "LGB",
    color: MODEL_COLORS.lightgbm,
    type: "boost",
    description:
      "Gradient-boosted decision trees with leaf-wise growth strategy. Best accuracy-to-size ratio — 1.7 MB with RMSE 26.86 µA. Recommended for production deployment.",
    architecture: "Leaf-wise GBDT, 2000 max estimators, lr=0.03, num_leaves=63, 278 trees at best_iteration",
    strengths: [
      "Only 1.7 MB — ~360× smaller than RF",
      "RMSE within 1% of RF",
      "Fast inference (~5 ms per curve)",
      "Best deployment candidate",
    ],
    weaknesses: [
      "Slightly lower testMAT R² than GRU",
      "Tree-based: limited sequence modelling",
    ],
    deployScore: 97,
    size: "1.7 MB",
  },
  {
    id: "xgboost",
    display: "XGBoost",
    shortName: "XGB",
    color: MODEL_COLORS.xgboost,
    type: "boost",
    description:
      "Regularised gradient boosting with second-order gradients. Competitive accuracy (RMSE 28.83 µA) with a 268 KB model — smaller than LightGBM but slightly lower accuracy.",
    architecture: "XGBoost 2.x, 1000 max estimators, lr=0.05, max_depth=6, colsample=0.8, 110 trees at best_iteration",
    strengths: [
      "Excellent regularisation",
      "Only 268 KB model size",
      "Robust to overfitting",
      "Fast inference",
    ],
    weaknesses: [
      "Slightly lower accuracy than LightGBM",
      "Slower training than LightGBM",
    ],
    deployScore: 88,
    size: "268 KB",
  },
  {
    id: "gru",
    display: "Stacked GRU",
    shortName: "GRU",
    color: MODEL_COLORS.gru,
    type: "deep",
    description:
      "Two-layer Gated Recurrent Unit network. Best testMAT R² (0.9751) — highest extrapolation ability across unseen materials. Ideal when generalisation to new ZnO composites is needed.",
    architecture: "GRU(64) → Dropout(0.2) → GRU(32) → Dropout(0.1) → TimeDistributed Dense(16) → TimeDistributed Dense(1), 651-step padded sequences",
    strengths: [
      "Best testMAT R² (0.9751)",
      "Highest extrapolation ability",
      "Captures voltage sweep sequence structure",
      "Only 343 KB weights",
    ],
    weaknesses: [
      "Requires TensorFlow runtime",
      "Higher validation RMSE than tree models",
      "~4 s cold-start load time",
    ],
    deployScore: 74,
    size: "343 KB",
  },
  {
    id: "lstm",
    display: "Stacked LSTM",
    shortName: "LSTM",
    color: MODEL_COLORS.lstm,
    type: "deep",
    description:
      "Long Short-Term Memory network with cell and hidden state. Competitive extrapolation performance but slightly higher RMSE than GRU for this dataset.",
    architecture: "LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.1) → TimeDistributed Dense(16) → TimeDistributed Dense(1)",
    strengths: [
      "Long-range temporal dependencies",
      "Good extrapolation (R²=0.9674)",
      "434 KB model size",
    ],
    weaknesses: [
      "Higher parameter count than GRU",
      "Slightly lower R² than GRU on testMAT",
    ],
    deployScore: 70,
    size: "434 KB",
  },
  {
    id: "ann",
    display: "Dense ANN",
    shortName: "ANN",
    color: MODEL_COLORS.ann,
    type: "deep",
    description:
      "Multi-layer perceptron treating each CV point independently. Smallest model (176 KB) with fastest inference, but does not capture sweep direction sequentiality.",
    architecture: "Dense(128) → Dropout(0.15) → Dense(64) → Dropout(0.1) → Dense(32) → Dense(1)",
    strengths: [
      "Smallest model: 176 KB",
      "Fastest inference",
      "Simple deployment",
      "No sequence dependency",
    ],
    weaknesses: [
      "Worst validation RMSE (49.93 µA)",
      "No temporal modelling",
      "Lower extrapolation capability",
    ],
    deployScore: 60,
    size: "176 KB",
  },
];

export const MODEL_MAP: Record<string, ModelConfig> = Object.fromEntries(
  MODELS.map((m) => [m.id, m])
);

// ── Material definitions ──────────────────────────────────────────────────
export const MATERIALS = [
  { id: "NM1", label: "NM1 — ZnO baseline", color: "#3b82f6" },
  { id: "NM2", label: "NM2 — ZnO/rGO", color: "#22d3ee" },
  { id: "NM3", label: "NM3 — ZnO/MnO₂", color: "#a78bfa" },
  { id: "NM4", label: "NM4 — ZnO/Co₃O₄ (test-MAT)", color: "#f472b6" },
];

// ── Valid scan rates ──────────────────────────────────────────────────────
export const SCAN_RATES = [10, 20, 30, 50, 70, 100];
export const SCAN_RATE_MIN = 10;
export const SCAN_RATE_MAX = 100;
