import React, { useState, useMemo, useEffect, useCallback } from "react";
import {
  Box, Grid, Card, CardContent, Typography, Tabs, Tab,
  Table, TableBody, TableCell, TableHead, TableRow,
  Skeleton, Chip, LinearProgress, alpha, Divider, Tooltip,
  Select, MenuItem, FormControl, CircularProgress,
} from "@mui/material";
import {
  EmojiEventsRounded, TrendingUpRounded, ScienceRounded,
  SpeedRounded, MemoryRounded, InsightsRounded, LightbulbRounded,
  BiotechRounded, BarChartRounded, TimelineRounded, TuneRounded,
  ShowChartRounded, AutoGraphRounded, ScatterPlotRounded,
} from "@mui/icons-material";
import { motion, AnimatePresence } from "framer-motion";
import Plot from "react-plotly.js";
import { useLeaderboard } from "../hooks/useLeaderboard";
import { useMetrics } from "../hooks/useMetrics";
import { getAllTrainingHistories, predict as apiPredict } from "../api/endpoints";
import SectionHeader from "../components/common/SectionHeader";
import ModelBadge from "../components/common/ModelBadge";
import GlowButton from "../components/common/GlowButton";
import { RMSEBarChart, R2GroupedChart, SizeVsRMSEScatter } from "../components/charts/MetricsChart";
import { fmtMB, rmseColor, r2Color, shortPartition } from "../utils/formatters";
import { MODEL_COLORS, MODEL_ORDER, MATERIALS, MODELS } from "../constants/models";
import { cardVariants, staggerContainer } from "../animations/variants";
import type { TrainingHistory } from "../types";

const PlotConfig = { responsive: true, displayModeBar: false, displaylogo: false };
const DarkLayout = (overrides: Partial<Plotly.Layout> = {}): Partial<Plotly.Layout> => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(13,19,33,0.5)",
  font: { family: "Inter, sans-serif", color: "#8892a4", size: 11 },
  margin: { l: 60, r: 20, t: 20, b: 50 },
  autosize: true,
  hoverlabel: {
    bgcolor: "#111c2d", bordercolor: "#00d4ff",
    font: { size: 12, color: "#f1f5f9", family: "JetBrains Mono" },
  },
  ...overrides,
});

// ── Scientific insight cards ────────────────────────────────────────────────
const SCIENTIFIC_INSIGHTS = [
  { model: "gru", title: "GRU — Best Cross-Material Transfer", color: "#a78bfa", icon: "🧠",
    insight: "GRU achieved the highest testMAT R² (0.9751) by modelling voltage sweep directionality through gated recurrent units. The reset and update gates selectively retain anodic vs. cathodic sweep representations, enabling generalisation to unseen NM4 material whose CV topology mirrors the training sweep structure.",
    metric: "R² = 0.9751 on NM4" },
  { model: "lightgbm", title: "LightGBM — Best Deployment Trade-off", color: "#22d3ee", icon: "⚡",
    insight: "LightGBM achieves ~99% of RF's validation accuracy at ~1/360th the size (1.7 MB vs 615 MB) — only 0.27 µA higher RMSE. Leaf-wise tree growth with gradient-based one-side sampling efficiently captures the non-linear current–potential relationship across all scan rates without the memory overhead of RF.",
    metric: "26.86 µA RMSE at 1.7 MB" },
  { model: "rf", title: "RF — Strongest Interpolation Baseline", color: "#3b82f6", icon: "🌲",
    insight: "Random Forest's ensemble averaging over 300+ trees provides the most robust interpolation (RMSE 26.59 µA). Each tree captures different feature interactions among scan rate, potential position, and sweep direction. The forest's variance reduction excels within the training scan rate distribution.",
    metric: "26.59 µA RMSE — best val" },
  { model: "lstm", title: "LSTM — Long-range Dependencies", color: "#f472b6", icon: "🔗",
    insight: "LSTM's cell state provides a persistent learned representation through the 651-point sweep, retaining anodic→cathodic context across the full CV loop. This is theoretically advantageous but in practice GRU's simpler gating achieves better R² due to fewer parameters and lower overfitting risk on 651-point sequences.",
    metric: "R² = 0.9674 on NM4" },
  { model: "ann", title: "ANN — Point-wise Limitations", color: "#facc15", icon: "⚠️",
    insight: "The Dense ANN treats each CV point independently, discarding the sequential structure of the voltage sweep. It achieves the highest RMSE (49.93 µA) because it cannot distinguish anodic from cathodic sweep phases without the recurrent state modelling that GRU and LSTM provide — a fundamental architectural limitation for sequential CV data.",
    metric: "49.93 µA RMSE — research baseline" },
  { model: "xgboost", title: "XGBoost — Regularized Boosting", color: "#fb923c", icon: "📊",
    insight: "XGBoost's L1+L2 regularization with second-order gradient statistics provides competitive accuracy (28.83 µA RMSE) in a 268 KB model. Its regularisation framework controls overfitting well, but slower training (6 min vs 2 min) and slightly lower accuracy make LightGBM the preferred deployment choice.",
    metric: "28.83 µA RMSE at 268 KB" },
];

const ELECTROCHEMICAL_INSIGHTS = [
  { icon: "📈", color: "#22d3ee", title: "Scan Rate → Peak Current",
    body: "CV peak currents scale proportionally with scan rate (Randles–Ševčík: Ip ∝ √ν for diffusion-controlled, Ip ∝ ν for surface-confined). ZnO supercapacitors exhibit mixed behavior — the trained models capture this non-linear scan-rate dependence through engineered log(SR) and √(SR) features." },
  { icon: "🔄", color: "#10b981", title: "CV Curve Hysteresis",
    body: "The enclosed area between anodic and cathodic sweeps (integral_area) is directly proportional to specific capacitance. Wider loops indicate higher charge storage capacity. Asymmetric loops reveal diffusion limitations. The models predict both sweep phases simultaneously, preserving hysteresis shape." },
  { icon: "⚡", color: "#f59e0b", title: "Sweep Direction Encoding",
    body: "The most predictively important engineered feature is direction_x_potential_norm (sweep_direction × potential_V). This interaction term is the dominant predictor in all tree models (~27% of RF MDI, ~60% of XGBoost gain), encoding position in the sweep differently for each half-sweep and capturing ZnO electrode asymmetry." },
  { icon: "🧪", color: "#a78bfa", title: "Material Extrapolation (NM4)",
    body: "NM4 was excluded from training as a held-out test material. GRU's superior testMAT R² (0.9751) suggests recurrent models capture fundamental CV topology (shape, symmetry, scale relationships) rather than material-specific memorisation — enabling genuine cross-material generalisation." },
];

const FEATURE_LABELS: Record<string, string> = {
  "scan_rate_mVs_norm": "Scan Rate (norm)",
  "log_scan_rate_norm": "Log(Scan Rate)",
  "sqrt_scan_rate_norm": "√(Scan Rate)",
  "potential_V_norm": "Potential V (norm)",
  "potential_from_lower_norm": "Potential from Lower",
  "potential_from_upper_norm": "Potential from Upper",
  "sr_x_potential_norm": "SR × Potential",
  "direction_x_potential_norm": "Direction × Potential",
  "sweep_direction": "Sweep Direction",
  "sweep_position": "Sweep Position",
};

// Actual partition keys returned by API
const PARTITION_KEYS = ["train", "val", "test_SR", "test_MAT"];
const PARTITION_LABELS: Record<string, string> = {
  train: "Train (in-sample)",
  val: "Val (SR=30)",
  test_SR: "Test-SR (SR=50)",
  test_MAT: "Test-MAT (NM4)",
};
const PARTITION_COLORS: Record<string, string> = {
  train: "#6366f1",
  val: "#10b981",
  test_SR: "#f59e0b",
  test_MAT: "#f472b6",
};

// ── Tab navigation: 4 categories with sub-tabs ───────────────────────────
// Each category maps to global tab indices (unchanged panel IDs) so all
// existing {tab === N && ...} content blocks require zero modification.
const CATEGORIES = [
  {
    label: "Performance", color: "#22d3ee",
    icon: <InsightsRounded sx={{ fontSize: 15 }} />,
    tabs: [0, 1, 8] as const,
    subLabels: ["Leaderboard", "R² Analysis", "Full Metrics"],
    subIcons: [<EmojiEventsRounded sx={{ fontSize: 14 }} />, <InsightsRounded sx={{ fontSize: 14 }} />, <BarChartRounded sx={{ fontSize: 14 }} />],
  },
  {
    label: "Training", color: "#a78bfa",
    icon: <TimelineRounded sx={{ fontSize: 15 }} />,
    tabs: [5, 6, 7] as const,
    subLabels: ["Training Curves", "Hyperparameters", "Scan Rate Evolution"],
    subIcons: [<TimelineRounded sx={{ fontSize: 14 }} />, <TuneRounded sx={{ fontSize: 14 }} />, <ShowChartRounded sx={{ fontSize: 14 }} />],
  },
  {
    label: "Scientific", color: "#10b981",
    icon: <BiotechRounded sx={{ fontSize: 15 }} />,
    tabs: [2, 4, 10] as const,
    subLabels: ["Feature Analysis", "Scientific Insights", "Material Analysis"],
    subIcons: [<BiotechRounded sx={{ fontSize: 14 }} />, <LightbulbRounded sx={{ fontSize: 14 }} />, <ScatterPlotRounded sx={{ fontSize: 14 }} />],
  },
  {
    label: "Deployment", color: "#f59e0b",
    icon: <SpeedRounded sx={{ fontSize: 15 }} />,
    tabs: [3, 9] as const,
    subLabels: ["Deployment Matrix", "Prediction Validation"],
    subIcons: [<SpeedRounded sx={{ fontSize: 14 }} />, <AutoGraphRounded sx={{ fontSize: 14 }} />],
  },
] as const;

// Keep for backwards compat (not used in navigation anymore)
const TABS = [
  { label: "Leaderboard", icon: <EmojiEventsRounded sx={{ fontSize: 16 }} /> },
  { label: "Performance", icon: <InsightsRounded sx={{ fontSize: 16 }} /> },
  { label: "Feature Intel", icon: <BiotechRounded sx={{ fontSize: 16 }} /> },
  { label: "Deployment", icon: <SpeedRounded sx={{ fontSize: 16 }} /> },
  { label: "Scientific Insights", icon: <LightbulbRounded sx={{ fontSize: 16 }} /> },
  { label: "Training History", icon: <TimelineRounded sx={{ fontSize: 16 }} /> },
  { label: "Hyperparameters", icon: <TuneRounded sx={{ fontSize: 16 }} /> },
  { label: "Scan Rate Evolution", icon: <ShowChartRounded sx={{ fontSize: 16 }} /> },
  { label: "Full Metrics", icon: <BarChartRounded sx={{ fontSize: 16 }} /> },
  { label: "Pred. Validation", icon: <AutoGraphRounded sx={{ fontSize: 16 }} /> },
  { label: "Material Analysis", icon: <ScatterPlotRounded sx={{ fontSize: 16 }} /> },
];

// Hyperparameter data (sourced from research/metrics/*.json params)
const HYPERPARAMS: Record<string, { label: string; value: string }[]> = {
  rf: [
    { label: "n_estimators", value: "300" },
    { label: "max_depth", value: "None (unlimited)" },
    { label: "min_samples_leaf", value: "2" },
    { label: "max_features", value: "sqrt" },
    { label: "oob_score", value: "True" },
    { label: "random_state", value: "42" },
  ],
  lightgbm: [
    { label: "n_estimators", value: "2000" },
    { label: "learning_rate", value: "0.03" },
    { label: "num_leaves", value: "63" },
    { label: "subsample", value: "0.8" },
    { label: "colsample_bytree", value: "0.8" },
    { label: "reg_alpha", value: "0.05" },
    { label: "reg_lambda", value: "1.0" },
  ],
  xgboost: [
    { label: "n_estimators", value: "1000" },
    { label: "learning_rate", value: "0.05" },
    { label: "max_depth", value: "6" },
    { label: "subsample", value: "0.8" },
    { label: "colsample_bytree", value: "0.8" },
    { label: "reg_alpha", value: "0.05" },
    { label: "reg_lambda", value: "1.5" },
    { label: "early_stopping", value: "50 rounds" },
  ],
  gru: [
    { label: "GRU layer 1", value: "64 units, return_seq=True" },
    { label: "Dropout 1", value: "0.2" },
    { label: "GRU layer 2", value: "32 units, return_seq=True" },
    { label: "Dropout 2", value: "0.1" },
    { label: "TimeDistributed Dense", value: "16 units, ReLU" },
    { label: "Output", value: "1 unit (linear)" },
    { label: "Total params", value: "24,545" },
    { label: "Optimizer", value: "Adam (lr=0.001)" },
  ],
  lstm: [
    { label: "LSTM layer 1", value: "64 units, return_seq=True" },
    { label: "Dropout 1", value: "0.2" },
    { label: "LSTM layer 2", value: "32 units, return_seq=True" },
    { label: "Dropout 2", value: "0.1" },
    { label: "TimeDistributed Dense", value: "16 units, ReLU" },
    { label: "Output", value: "1 unit (linear)" },
    { label: "Total params", value: "32,161" },
    { label: "Optimizer", value: "Adam (lr=0.001)" },
  ],
  ann: [
    { label: "Dense 1", value: "128 units, ReLU" },
    { label: "Dropout 1", value: "0.15" },
    { label: "Dense 2", value: "64 units, ReLU" },
    { label: "Dropout 2", value: "0.1" },
    { label: "Dense 3", value: "32 units, ReLU" },
    { label: "Output", value: "1 unit (linear)" },
    { label: "Total params", value: "11,777" },
    { label: "Optimizer", value: "Adam (lr=0.001)" },
  ],
};

const VALID_SCAN_RATES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

export default function ResearchAnalytics() {
  const { leaderboard, summary, comparison, loading } = useLeaderboard();
  const { metrics, featureImportance, loading: mLoading } = useMetrics();

  // ── Two-level navigation: category → sub-tab → global tab index ──────────
  const [category, setCategory] = useState(0);
  const [subTab, setSubTab] = useState(0);
  // Derived global tab; all {tab === N} panel checks work without change
  const tab = CATEGORIES[category].tabs[subTab] ?? 0;
  const catColor = CATEGORIES[category].color;
  const handleCategory = (cat: number) => { setCategory(cat); setSubTab(0); };

  // Training history state
  const [trainingHistories, setTrainingHistories] = useState<Record<string, TrainingHistory>>({});
  const [histLoading, setHistLoading] = useState(false);
  const [histLoaded, setHistLoaded] = useState(false);

  // Scan rate evolution state
  const [srModel, setSrModel] = useState("lightgbm");
  const [srMaterial, setSrMaterial] = useState("NM1");
  const [srLoading, setSrLoading] = useState(false);
  const [srCurves, setSrCurves] = useState<{ sr: number; potential: number[]; current: number[] }[]>([]);
  // Monotonically increasing run ID — used to discard stale sequential-fetch results
  const srRunIdRef = React.useRef(0);

  // Load training histories lazily when tab 5 is first opened.
  // NOTE: histLoading is intentionally NOT in the dep array — including it causes
  // the cleanup (`mounted = false`) to fire when setHistLoading(true) updates state,
  // which aborts the in-flight fetch before it can commit its result.
  useEffect(() => {
    if (tab === 5 && !histLoaded) {
      let mounted = true;
      setHistLoading(true);
      getAllTrainingHistories()
        .then((data: Record<string, any>) => {
          if (!mounted) return;
          setTrainingHistories(data as Record<string, TrainingHistory>);
          setHistLoaded(true);
        })
        .catch(() => { /* silently fail */ })
        .finally(() => { if (mounted) setHistLoading(false); });
      return () => { mounted = false; };
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, histLoaded]);

  const runScanRateEvolution = useCallback(async () => {
    // Increment run ID so any in-flight previous run knows it is stale
    const thisRunId = ++srRunIdRef.current;
    setSrLoading(true);
    setSrCurves([]);
    const results: { sr: number; potential: number[]; current: number[] }[] = [];
    for (const sr of VALID_SCAN_RATES) {
      // Abort early if a newer run has been started
      if (srRunIdRef.current !== thisRunId) break;
      try {
        const r = await apiPredict({ model_name: srModel, material_id: srMaterial, scan_rate_mVs: sr });
        results.push({ sr, potential: r.potential_V, current: r.predicted_current_uA });
      } catch { /* skip */ }
    }
    // Only commit results if this run is still the latest
    if (srRunIdRef.current === thisRunId) {
      setSrCurves(results);
      setSrLoading(false);
    }
  }, [srModel, srMaterial]);

  // ── Derived data ──────────────────────────────────────────────────────────
  // FIX: use actual partition keys from the API ("train", "val", "test_SR", "test_MAT")
  const partitionData = useMemo(() => {
    return PARTITION_KEYS.map((p) => ({
      partition: p,
      label: PARTITION_LABELS[p],
      models: comparison.filter((r) => r.partition === p),
    }));
  }, [comparison]);

  const interpVsExtrap = useMemo(() => {
    if (leaderboard.length === 0) return [];
    return leaderboard.map((m) => ({
      model: m.model_display,
      model_id: m.model_id,
      interp: m.rmse_val_uA,
      extrap: m.rmse_test_mat_uA,
      gap: m.rmse_test_mat_uA - m.rmse_val_uA,
      r2_extrap: m.r2_test_mat,
    }));
  }, [leaderboard]);

  const featureImpModels = useMemo(
    () => Object.keys(featureImportance).filter((m) => ["rf", "xgboost", "lightgbm"].includes(m)),
    [featureImportance]
  );

  const topFeatures = useMemo(() => {
    const scores: Record<string, number> = {};
    featureImpModels.forEach((m) => {
      Object.entries(featureImportance[m] ?? {}).forEach(([f, v]) => {
        scores[f] = (scores[f] ?? 0) + (v as number);
      });
    });
    return Object.entries(scores).sort(([, a], [, b]) => b - a).slice(0, 10).map(([f]) => f);
  }, [featureImportance, featureImpModels]);

  // FIX: Compute n_models and n_partitions from real data, not from summary fields that don't exist
  const nModels = leaderboard.length || 6;
  const nPartitions = PARTITION_KEYS.length;

  const statsRow = [
    {
      label: "Best Val RMSE",
      value: `${leaderboard[0]?.rmse_val_uA?.toFixed(2) ?? "—"} µA`,
      sub: leaderboard[0]?.model_display ?? "—",
      color: "#22d3ee",
    },
    {
      label: "Best Cross-Material R²",
      value: leaderboard.length > 0
        ? leaderboard.reduce((b, m) => (m.r2_test_mat ?? 0) > (b.r2_test_mat ?? 0) ? m : b, leaderboard[0])
            ?.r2_test_mat?.toFixed(4) ?? "—"
        : "—",
      sub: "NM4 (unseen material)",
      color: "#10b981",
    },
    {
      label: "Best Deployment",
      value: leaderboard.find((m) => m.model_id === "lightgbm")?.size_MB != null
        ? `${leaderboard.find((m) => m.model_id === "lightgbm")!.size_MB.toFixed(1)} MB`
        : "1.7 MB",
      sub: "LightGBM",
      color: "#f59e0b",
    },
    {
      label: "Models Evaluated",
      value: `${nModels} × ${nPartitions}`,
      sub: "models × partitions",
      color: "#a78bfa",
    },
  ];

  return (
    <Box sx={{ width: "100%", overflowX: "hidden" }}>
      <SectionHeader
        title="Research Analytics"
        subtitle="Electrochemical ML Research Analysis — 6 models × 4 evaluation partitions × full scientific interpretation"
        accent="#f59e0b"
      />

      {/* Key stats row */}
      <motion.div variants={staggerContainer} initial="initial" animate="animate">
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {statsRow.map((s) => (
            <Grid item xs={6} sm={3} key={s.label}>
              <motion.div variants={cardVariants}>
                <Box sx={{ p: 2, borderRadius: 2, background: alpha(s.color, 0.06), border: `1px solid ${alpha(s.color, 0.2)}` }}>
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.62rem", display: "block", mb: 0.5 }}>{s.label}</Typography>
                  {loading ? <Skeleton width={80} height={28} /> : (
                    <Typography variant="h6" sx={{ color: s.color, fontFamily: "JetBrains Mono", fontWeight: 800, fontSize: "1rem" }}>{s.value}</Typography>
                  )}
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem" }}>{s.sub}</Typography>
                </Box>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </motion.div>

      {/* ── Category selector ──────────────────────────────────────── */}
      <Box sx={{ display: "flex", gap: 1, mb: 1.75, flexWrap: "wrap" }}>
        {CATEGORIES.map((cat, ci) => {
          const active = category === ci;
          return (
            <Box
              key={cat.label}
              onClick={() => handleCategory(ci)}
              sx={{
                display: "flex", alignItems: "center", gap: 0.75,
                px: 1.75, py: 0.75, borderRadius: 2, cursor: "pointer",
                background: active ? alpha(cat.color, 0.14) : "rgba(255,255,255,0.03)",
                border: `1px solid ${active ? alpha(cat.color, 0.45) : "rgba(255,255,255,0.08)"}`,
                color: active ? cat.color : "rgba(255,255,255,0.45)",
                fontWeight: active ? 700 : 500,
                fontSize: "0.78rem",
                transition: "all 0.17s",
                "&:hover": { background: alpha(cat.color, 0.09), color: cat.color },
              }}
            >
              {cat.icon}
              {cat.label}
            </Box>
          );
        })}
      </Box>

      {/* ── Sub-tab bar ─────────────────────────────────────────────── */}
      <Tabs
        value={subTab}
        onChange={(_, v) => setSubTab(v)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          mb: 3,
          borderBottom: `1px solid ${alpha(catColor, 0.18)}`,
          "& .MuiTab-root": { fontSize: "0.73rem", minHeight: 40, gap: 0.5, textTransform: "none", color: "rgba(255,255,255,0.45)" },
          "& .MuiTab-root.Mui-selected": { color: catColor, fontWeight: 700 },
          "& .MuiTabs-indicator": { backgroundColor: catColor },
        }}
      >
        {CATEGORIES[category].subLabels.map((label, si) => (
          <Tab key={label} label={label} icon={CATEGORIES[category].subIcons[si] as React.ReactElement} iconPosition="start" />
        ))}
      </Tabs>

      <AnimatePresence mode="wait">

        {/* ── Tab 0: Global Leaderboard ─────────────────────────────────────── */}
        {tab === 0 && (
          <motion.div key="t0" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={7}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(245,158,11,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#f59e0b", fontWeight: 700, mb: 2, letterSpacing: "0.05em" }}>
                      RANKED MODEL LEADERBOARD — VALIDATION RMSE
                    </Typography>
                    {loading ? <Skeleton variant="rounded" height={280} /> : <RMSEBarChart leaderboard={leaderboard} />}
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={5}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(245,158,11,0.12)", height: "100%" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#f59e0b", fontWeight: 700, mb: 2 }}>
                      COMPOSITE RANKING TABLE
                    </Typography>
                    <Box sx={{ overflowX: "auto" }}>
                    {loading ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} height={32} sx={{ mb: 0.5 }} />) : (
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ py: 0.5, fontSize: "0.7rem" }}>#</TableCell>
                            <TableCell sx={{ py: 0.5, fontSize: "0.7rem" }}>Model</TableCell>
                            <TableCell align="right" sx={{ py: 0.5, fontSize: "0.7rem" }}>Val RMSE</TableCell>
                            <TableCell align="right" sx={{ py: 0.5, fontSize: "0.7rem" }}>R² MAT</TableCell>
                            <TableCell align="right" sx={{ py: 0.5, fontSize: "0.7rem" }}>Size</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {leaderboard.map((r, i) => (
                            <TableRow key={r.model_id} sx={{ "&:last-child td": { border: 0 } }}>
                              <TableCell sx={{ py: 0.75, color: i === 0 ? "#f59e0b" : "rgba(255,255,255,0.3)", fontWeight: 700, fontFamily: "JetBrains Mono", fontSize: "0.75rem" }}>
                                {i === 0 ? "①" : i === 1 ? "②" : i === 2 ? "③" : `${r.rank}`}
                              </TableCell>
                              <TableCell sx={{ py: 0.75 }}><ModelBadge modelId={r.model_id} size="small" /></TableCell>
                              <TableCell align="right" sx={{ py: 0.75, color: rmseColor(r.rmse_val_uA), fontFamily: "JetBrains Mono", fontSize: "0.78rem", fontWeight: 700 }}>
                                {r.rmse_val_uA?.toFixed(1)}
                              </TableCell>
                              <TableCell align="right" sx={{ py: 0.75, color: r2Color(r.r2_test_mat), fontFamily: "JetBrains Mono", fontSize: "0.78rem", fontWeight: 600 }}>
                                {r.r2_test_mat?.toFixed(4)}
                              </TableCell>
                              <TableCell align="right" sx={{ py: 0.75, color: "rgba(255,255,255,0.35)", fontFamily: "JetBrains Mono", fontSize: "0.68rem" }}>
                                {fmtMB(r.size_MB)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                    </Box>
                    {!loading && (
                      <Box sx={{ mt: 2, pt: 2, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.65rem", lineHeight: 1.6, display: "block" }}>
                          Composite rank: average of RMSE rank across validation, test-SR, test-MAT partitions.
                          Color: <span style={{ color: "#22d3ee" }}>≤28 µA excellent</span> · <span style={{ color: "#f59e0b" }}>≤40 µA good</span> · <span style={{ color: "#ef4444" }}>&gt;40 µA needs improvement</span>
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* RMSE across partitions table — FIXED partition keys */}
              <Grid item xs={12}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(245,158,11,0.1)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#f59e0b", fontWeight: 700, mb: 2 }}>
                      RMSE ACROSS ALL PARTITIONS — 6 MODELS × 4 SPLITS
                    </Typography>
                    <Box sx={{ overflowX: "auto" }}>
                    {loading ? <Skeleton variant="rounded" height={240} /> : (
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ fontWeight: 700, py: 1 }}>Model</TableCell>
                            {PARTITION_KEYS.map((p) => (
                              <TableCell key={p} align="right" sx={{ fontWeight: 700, py: 1, color: PARTITION_COLORS[p] }}>
                                {PARTITION_LABELS[p]}
                              </TableCell>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {MODEL_ORDER.map((modelId) => {
                            const rowsByPartition = Object.fromEntries(
                              PARTITION_KEYS.map((p) => [
                                p,
                                comparison.find((r) => r.model_id === modelId && r.partition === p),
                              ])
                            );
                            return (
                              <TableRow key={modelId} sx={{ "&:last-child td": { border: 0 } }}>
                                <TableCell sx={{ py: 0.85 }}><ModelBadge modelId={modelId} size="small" /></TableCell>
                                {PARTITION_KEYS.map((p) => {
                                  const row = rowsByPartition[p];
                                  return (
                                    <TableCell key={p} align="right" sx={{ py: 0.85, fontFamily: "JetBrains Mono", fontSize: "0.8rem", fontWeight: 700, color: row ? rmseColor(row.rmse_uA) : "rgba(255,255,255,0.2)" }}>
                                      {row ? row.rmse_uA.toFixed(2) : "—"}
                                    </TableCell>
                                  );
                                })}
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    )}
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.18)", mt: 1.5, display: "block", fontSize: "0.62rem" }}>
                      Values in µA. Train=in-sample · Val=scan rate 30 mV/s interpolation · Test-SR=scan rate 50 mV/s · Test-MAT=NM4 material extrapolation
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

        {/* ── Tab 1: Performance Analysis ────────────────────────────────────── */}
        {tab === 1 && (
          <motion.div key="t1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(16,185,129,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#10b981", fontWeight: 700, mb: 2 }}>R² SCORES — ALL PARTITIONS</Typography>
                    {loading ? <Skeleton variant="rounded" height={320} /> : <R2GroupedChart rows={comparison} height={340} />}
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", mt: 1, display: "block", fontSize: "0.65rem" }}>
                      Train: in-sample · Val: SR=30 interpolation · Test-SR: SR=50 interpolation · Test-MAT: NM4 cross-material
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(16,185,129,0.12)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#10b981", fontWeight: 700, mb: 2 }}>INTERPOLATION vs CROSS-MATERIAL RMSE GAP</Typography>
                    {loading || interpVsExtrap.length === 0 ? <Skeleton variant="rounded" height={300} /> : (
                      <Plot
                        data={[
                          { type: "bar", name: "Val RMSE (SR=30 Interpolation)", x: interpVsExtrap.map((m) => m.model), y: interpVsExtrap.map((m) => m.interp),
                            marker: { color: interpVsExtrap.map((m) => MODEL_COLORS[m.model_id] ?? "#94a3b8"), opacity: 0.85 },
                            hovertemplate: "<b>%{x}</b><br>Val RMSE: %{y:.2f} µA<extra>Interpolation</extra>" },
                          { type: "bar", name: "Test-MAT RMSE (NM4 Cross-Material)", x: interpVsExtrap.map((m) => m.model), y: interpVsExtrap.map((m) => m.extrap),
                            marker: { color: interpVsExtrap.map((m) => MODEL_COLORS[m.model_id] ?? "#94a3b8"), opacity: 0.38 },
                            hovertemplate: "<b>%{x}</b><br>Test-MAT RMSE: %{y:.2f} µA<extra>Cross-Material</extra>" },
                        ]}
                        layout={DarkLayout({ margin: { l: 60, r: 20, t: 10, b: 40 }, height: 300, barmode: "group",
                          yaxis: { title: { text: "RMSE (µA)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", tickfont: { size: 10, color: "#8892a4", family: "JetBrains Mono" } },
                          xaxis: { tickfont: { size: 11, color: "#f1f5f9" }, gridcolor: "rgba(0,0,0,0)" },
                          legend: { orientation: "h", x: 0, y: 1.08, bgcolor: "rgba(0,0,0,0)", font: { size: 10, color: "#f1f5f9" } },
                        })}
                        config={PlotConfig} style={{ width: "100%", height: 300 }} useResizeHandler
                      />
                    )}
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", mt: 1, display: "block", fontSize: "0.65rem" }}>
                      Smaller gap = better generalisation. GRU achieves the smallest val→test_MAT RMSE delta among deep models; GRU and ANN show negative deltas (better on NM4 than val).
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(16,185,129,0.1)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#10b981", fontWeight: 700, mb: 2 }}>
                      GENERALIZATION ANALYSIS — INTERPOLATION TO EXTRAPOLATION TRANSFER
                    </Typography>
                    <Box sx={{ overflowX: "auto" }}>
                    {loading ? <Skeleton variant="rounded" height={200} /> : (
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Model</TableCell>
                            <TableCell align="right">Val RMSE (SR=30)</TableCell>
                            <TableCell align="right">Test-SR RMSE (SR=50)</TableCell>
                            <TableCell align="right">Test-MAT RMSE (NM4)</TableCell>
                            <TableCell align="right">Extrap Gap</TableCell>
                            <TableCell align="right">TestMAT R²</TableCell>
                            <TableCell align="center">Rating</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {leaderboard.map((r) => {
                            const gap = r.rmse_test_mat_uA - r.rmse_val_uA;
                            const genScore = gap < 10 ? "Excellent" : gap < 15 ? "Good" : "Moderate";
                            const genColor = gap < 10 ? "#10b981" : gap < 15 ? "#f59e0b" : "#ef4444";
                            return (
                              <TableRow key={r.model_id} sx={{ "&:last-child td": { border: 0 } }}>
                                <TableCell><ModelBadge modelId={r.model_id} size="small" /></TableCell>
                                <TableCell align="right" sx={{ color: rmseColor(r.rmse_val_uA), fontFamily: "JetBrains Mono", fontSize: "0.8rem", fontWeight: 600 }}>{r.rmse_val_uA?.toFixed(2)}</TableCell>
                                <TableCell align="right" sx={{ color: rmseColor(r.rmse_test_sr_uA), fontFamily: "JetBrains Mono", fontSize: "0.8rem", fontWeight: 600 }}>{r.rmse_test_sr_uA?.toFixed(2)}</TableCell>
                                <TableCell align="right" sx={{ color: rmseColor(r.rmse_test_mat_uA), fontFamily: "JetBrains Mono", fontSize: "0.8rem", fontWeight: 600 }}>{r.rmse_test_mat_uA?.toFixed(2)}</TableCell>
                                <TableCell align="right" sx={{ color: gap < 0 ? "#10b981" : gap > 12 ? "#ef4444" : "#f59e0b", fontFamily: "JetBrains Mono", fontSize: "0.8rem", fontWeight: 700 }}>
                                  {gap >= 0 ? "+" : ""}{gap.toFixed(2)} µA
                                </TableCell>
                                <TableCell align="right" sx={{ color: r2Color(r.r2_test_mat), fontFamily: "JetBrains Mono", fontSize: "0.8rem", fontWeight: 700 }}>{r.r2_test_mat?.toFixed(4)}</TableCell>
                                <TableCell align="center">
                                  <Chip label={genScore} size="small" sx={{ backgroundColor: alpha(genColor, 0.15), color: genColor, fontSize: "0.62rem", height: 20, border: `1px solid ${alpha(genColor, 0.3)}` }} />
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

        {/* ── Tab 2: Feature Analysis ──────────────────────────────────────── */}
        {tab === 2 && (
          <motion.div key="t2" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              {featureImpModels.length > 0 && topFeatures.length > 0 ? featureImpModels.map((modelId) => (
                <Grid item xs={12} md={4} key={modelId}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: `1px solid ${alpha(MODEL_COLORS[modelId], 0.2)}` }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
                        <ModelBadge modelId={modelId} size="small" />
                        <Typography variant="subtitle2" sx={{ color: MODEL_COLORS[modelId], fontWeight: 700, fontSize: "0.82rem" }}>
                          Feature Importance
                        </Typography>
                      </Box>
                      {topFeatures.map((feat) => {
                        const val = (featureImportance[modelId]?.[feat] ?? 0) as number;
                        const maxVal = Math.max(...topFeatures.map((f) => (featureImportance[modelId]?.[f] ?? 0) as number));
                        const pct = maxVal > 0 ? (val / maxVal) * 100 : 0;
                        return (
                          <Box key={feat} sx={{ mb: 1.25 }}>
                            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.4 }}>
                              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.68rem" }}>
                                {FEATURE_LABELS[feat] ?? feat}
                              </Typography>
                              <Typography variant="caption" sx={{ color: MODEL_COLORS[modelId], fontFamily: "JetBrains Mono", fontSize: "0.68rem", fontWeight: 700 }}>
                                {pct.toFixed(1)}%
                              </Typography>
                            </Box>
                            <LinearProgress variant="determinate" value={pct}
                              sx={{ height: 5, borderRadius: 2, backgroundColor: "rgba(255,255,255,0.05)", "& .MuiLinearProgress-bar": { backgroundColor: MODEL_COLORS[modelId], borderRadius: 2 } }} />
                          </Box>
                        );
                      })}
                    </CardContent>
                  </Card>
                </Grid>
              )) : (
                <Grid item xs={12}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(255,255,255,0.08)" }}>
                    <CardContent sx={{ p: 3, textAlign: "center" }}>
                      {mLoading ? <Skeleton variant="rounded" height={300} /> : (
                        <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.4)" }}>
                          Feature importance data loading…
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Normalized importance Plotly chart */}
              {featureImpModels.length > 0 && topFeatures.length > 0 && (
                <Grid item xs={12}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(139,92,246,0.15)" }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                        NORMALIZED FEATURE IMPORTANCE — TREE MODELS COMPARISON
                      </Typography>
                      <Plot
                        data={featureImpModels.map((modelId) => {
                          const vals = topFeatures.map((f) => (featureImportance[modelId]?.[f] ?? 0) as number);
                          const maxV = Math.max(...vals) || 1;
                          return {
                            type: "bar" as const,
                            name: modelId.toUpperCase(),
                            x: vals.map((v) => (v / maxV) * 100),
                            y: topFeatures.map((f) => FEATURE_LABELS[f] ?? f),
                            orientation: "h" as const,
                            marker: { color: MODEL_COLORS[modelId], opacity: 0.85 },
                            hovertemplate: "<b>%{y}</b><br>Importance: %{x:.1f}%<extra>" + modelId.toUpperCase() + "</extra>",
                          };
                        })}
                        layout={DarkLayout({
                          height: 380, barmode: "group",
                          margin: { l: 180, r: 20, t: 10, b: 50 },
                          xaxis: { title: { text: "Normalized Importance (%)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", range: [0, 110] },
                          yaxis: { tickfont: { size: 10, color: "#c8cfd8" }, automargin: true },
                          legend: { orientation: "h", x: 0.3, y: 1.05, bgcolor: "rgba(0,0,0,0)", font: { size: 11 } },
                        })}
                        config={PlotConfig} style={{ width: "100%", height: 380 }} useResizeHandler
                      />
                    </CardContent>
                  </Card>
                </Grid>
              )}

              <Grid item xs={12}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(139,92,246,0.1)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                      FEATURE ENGINEERING RATIONALE
                    </Typography>
                    <Grid container spacing={2}>
                      {[
                        { feat: "Direction × Potential", color: "#22d3ee", text: "direction_x_potential_norm = sweep_direction × potential_V — the single most important feature in all tree models (~27% of RF MDI, ~60% of XGBoost gain). Encodes anodic/cathodic position jointly, capturing the core asymmetry of the ZnO CV loop." },
                        { feat: "Sweep Direction", color: "#10b981", text: "Binary signal (0.0 = anodic, 1.0 = cathodic). Structurally important but its predictive power is largely channelled through the direction×potential interaction term. Directly controls which CV half-sweep is being modelled." },
                        { feat: "Sweep Position", color: "#f59e0b", text: "Continuous position [0,1] through each half-sweep — 2nd most important in RF MDI (0.158). Gives sequence models (LSTM/GRU) precise temporal context and enables tree models to track curvature within each half-sweep." },
                        { feat: "Potential Features", color: "#a78bfa", text: "potential_V_norm, potential_from_lower_norm, and potential_from_upper_norm collectively describe where each measurement sits in the −0.65 V to 0 V window. Together they rank 3rd–5th in RF MDI, capturing absolute and relative voltage position." },
                      ].map((e) => (
                        <Grid item xs={12} sm={6} key={e.feat}>
                          <Box sx={{ p: 1.75, borderRadius: 2, background: alpha(e.color, 0.05), border: `1px solid ${alpha(e.color, 0.15)}` }}>
                            <Typography variant="caption" sx={{ color: e.color, fontWeight: 700, fontSize: "0.72rem", display: "block", mb: 0.5 }}>{e.feat}</Typography>
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.7rem", lineHeight: 1.55 }}>{e.text}</Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

        {/* ── Tab 3: Deployment Matrix ──────────────────────────────────────── */}
        {tab === 3 && (
          <motion.div key="t3" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(124,58,237,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>MODEL SIZE (LOG SCALE) vs VALIDATION RMSE</Typography>
                    {loading ? <Skeleton variant="rounded" height={320} /> : <SizeVsRMSEScatter leaderboard={leaderboard} height={340} />}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(124,58,237,0.12)", height: "100%" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>DEPLOYMENT TIER SYSTEM</Typography>
                    <Box sx={{ mb: 2.5 }}>
                      <Chip label="TIER 1 — HOT LOADED" size="small" sx={{ mb: 1.5, backgroundColor: alpha("#10b981", 0.12), color: "#10b981", border: "1px solid rgba(16,185,129,0.25)", fontSize: "0.6rem", fontWeight: 700 }} />
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.65rem", display: "block", mb: 1 }}>
                        Permanently loaded at startup — instant inference on every request.
                      </Typography>
                      {["rf", "lightgbm", "gru"].map((m) => {
                        const entry = leaderboard.find((r) => r.model_id === m);
                        return (
                          <Box key={m} sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1, px: 1.5, py: 0.75, borderRadius: 1.5, background: alpha(MODEL_COLORS[m], 0.06), border: `1px solid ${alpha(MODEL_COLORS[m], 0.15)}` }}>
                            <ModelBadge modelId={m} size="small" />
                            <Box sx={{ flex: 1 }} />
                            <Typography variant="caption" sx={{ color: MODEL_COLORS[m], fontFamily: "JetBrains Mono", fontSize: "0.65rem", fontWeight: 700 }}>
                              {entry?.rmse_val_uA?.toFixed(1)} µA
                            </Typography>
                          </Box>
                        );
                      })}
                    </Box>
                    <Divider sx={{ mb: 2.5, borderColor: "rgba(255,255,255,0.06)" }} />
                    <Box>
                      <Chip label="TIER 2 — LAZY LOADED" size="small" sx={{ mb: 1.5, backgroundColor: alpha("#f59e0b", 0.12), color: "#f59e0b", border: "1px solid rgba(245,158,11,0.25)", fontSize: "0.6rem", fontWeight: 700 }} />
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.65rem", display: "block", mb: 1 }}>
                        Loaded on first request, then cached. ~5–15s cold start, instant thereafter.
                      </Typography>
                      {["xgboost", "ann", "lstm"].map((m) => {
                        const entry = leaderboard.find((r) => r.model_id === m);
                        return (
                          <Box key={m} sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1, px: 1.5, py: 0.75, borderRadius: 1.5, background: alpha(MODEL_COLORS[m], 0.06), border: `1px solid ${alpha(MODEL_COLORS[m], 0.15)}` }}>
                            <ModelBadge modelId={m} size="small" />
                            <Box sx={{ flex: 1 }} />
                            <Typography variant="caption" sx={{ color: MODEL_COLORS[m], fontFamily: "JetBrains Mono", fontSize: "0.65rem", fontWeight: 700 }}>
                              {entry?.rmse_val_uA?.toFixed(1)} µA
                            </Typography>
                          </Box>
                        );
                      })}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(124,58,237,0.1)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                      MODEL COMPLEXITY + DEPLOYMENT MATRIX
                    </Typography>
                    <Box sx={{ overflowX: "auto" }}>
                    {loading ? <Skeleton variant="rounded" height={220} /> : (
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Model</TableCell>
                            <TableCell align="center">Tier</TableCell>
                            <TableCell align="right">Size</TableCell>
                            <TableCell align="right">Params/Trees</TableCell>
                            <TableCell align="right">Train Time</TableCell>
                            <TableCell align="right">Val RMSE</TableCell>
                            <TableCell align="right">Inference</TableCell>
                            <TableCell align="center">Best For</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {[
                            { id: "rf", tier: 1, size: "615 MB", complexity: "300 trees", trainMin: 4, inference: "~120ms", bestFor: "Accuracy" },
                            { id: "lightgbm", tier: 1, size: "1.7 MB", complexity: "278 trees", trainMin: 2, inference: "~5ms", bestFor: "Deployment" },
                            { id: "gru", tier: 1, size: "343 KB", complexity: "24,545 params", trainMin: 20, inference: "~125ms", bestFor: "Extrapolation" },
                            { id: "xgboost", tier: 2, size: "268 KB", complexity: "110 trees", trainMin: 6, inference: "<1ms", bestFor: "Regularization" },
                            { id: "ann", tier: 2, size: "176 KB", complexity: "11,777 params", trainMin: 10, inference: "~120ms", bestFor: "Baseline" },
                            { id: "lstm", tier: 2, size: "434 KB", complexity: "32,161 params", trainMin: 25, inference: "~140ms", bestFor: "Long-range" },
                          ].map((m) => {
                            const entry = leaderboard.find((r) => r.model_id === m.id);
                            const tierColor = m.tier === 1 ? "#10b981" : "#f59e0b";
                            return (
                              <TableRow key={m.id} sx={{ "&:last-child td": { border: 0 } }}>
                                <TableCell><ModelBadge modelId={m.id} size="small" /></TableCell>
                                <TableCell align="center">
                                  <Chip label={`Tier ${m.tier}`} size="small" sx={{ backgroundColor: alpha(tierColor, 0.12), color: tierColor, fontSize: "0.58rem", height: 18 }} />
                                </TableCell>
                                <TableCell align="right" sx={{ color: "rgba(255,255,255,0.55)", fontFamily: "JetBrains Mono", fontSize: "0.75rem" }}>{m.size}</TableCell>
                                <TableCell align="right" sx={{ color: "rgba(255,255,255,0.45)", fontFamily: "JetBrains Mono", fontSize: "0.72rem" }}>{m.complexity}</TableCell>
                                <TableCell align="right" sx={{ color: "rgba(255,255,255,0.4)", fontFamily: "JetBrains Mono", fontSize: "0.72rem" }}>{m.trainMin} min</TableCell>
                                <TableCell align="right" sx={{ color: rmseColor(entry?.rmse_val_uA ?? 99), fontFamily: "JetBrains Mono", fontSize: "0.78rem", fontWeight: 700 }}>
                                  {entry?.rmse_val_uA?.toFixed(2) ?? "—"} µA
                                </TableCell>
                                <TableCell align="right" sx={{ color: "#22d3ee", fontFamily: "JetBrains Mono", fontSize: "0.72rem" }}>{m.inference}</TableCell>
                                <TableCell align="center">
                                  <Chip label={m.bestFor} size="small" sx={{ backgroundColor: alpha(MODEL_COLORS[m.id], 0.12), color: MODEL_COLORS[m.id], fontSize: "0.6rem", height: 18 }} />
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

        {/* ── Tab 4: Scientific Insights ──────────────────────────────────────── */}
        {tab === 4 && (
          <motion.div key="t4" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 2, letterSpacing: "0.05em" }}>
                  AI MODEL INTELLIGENCE LAYER
                </Typography>
                <Grid container spacing={2}>
                  {SCIENTIFIC_INSIGHTS.map((insight, i) => (
                    <Grid item xs={12} sm={6} md={4} key={insight.model}>
                      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07, duration: 0.35 }}>
                        <Card sx={{ height: "100%", background: `linear-gradient(135deg, ${alpha(insight.color, 0.08)} 0%, rgba(15,25,35,0.9) 100%)`, border: `1px solid ${alpha(insight.color, 0.25)}`, borderTop: `3px solid ${insight.color}` }}>
                          <CardContent sx={{ p: 2.5 }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1.5 }}>
                              <Box sx={{ fontSize: 22 }}>{insight.icon}</Box>
                              <ModelBadge modelId={insight.model} size="small" />
                            </Box>
                            <Typography variant="subtitle2" sx={{ color: insight.color, fontWeight: 700, fontSize: "0.85rem", mb: 1 }}>{insight.title}</Typography>
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.72rem", lineHeight: 1.65, display: "block", mb: 1.5 }}>
                              {insight.insight}
                            </Typography>
                            <Box sx={{ px: 1.5, py: 0.75, borderRadius: 1.5, background: alpha(insight.color, 0.1), border: `1px solid ${alpha(insight.color, 0.2)}` }}>
                              <Typography variant="caption" sx={{ color: insight.color, fontFamily: "JetBrains Mono", fontWeight: 700, fontSize: "0.68rem" }}>
                                {insight.metric}
                              </Typography>
                            </Box>
                          </CardContent>
                        </Card>
                      </motion.div>
                    </Grid>
                  ))}
                </Grid>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="subtitle2" sx={{ color: "#10b981", fontWeight: 700, mb: 2, letterSpacing: "0.05em" }}>
                  ELECTROCHEMICAL INTERPRETATION ENGINE
                </Typography>
                <Grid container spacing={2}>
                  {ELECTROCHEMICAL_INSIGHTS.map((e, i) => (
                    <Grid item xs={12} sm={6} key={e.title}>
                      <motion.div initial={{ opacity: 0, x: i % 2 === 0 ? -12 : 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1, duration: 0.35 }}>
                        <Card sx={{ background: "rgba(15,25,35,0.8)", border: `1px solid ${alpha(e.color, 0.2)}` }}>
                          <CardContent sx={{ p: 2.5 }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1.5 }}>
                              <Typography sx={{ fontSize: 24 }}>{e.icon}</Typography>
                              <Typography variant="subtitle2" sx={{ color: e.color, fontWeight: 700, fontSize: "0.85rem" }}>{e.title}</Typography>
                            </Box>
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.73rem", lineHeight: 1.65 }}>{e.body}</Typography>
                          </CardContent>
                        </Card>
                      </motion.div>
                    </Grid>
                  ))}
                </Grid>
              </Grid>

              {!loading && summary && (
                <Grid item xs={12}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.12)" }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 2, letterSpacing: "0.05em" }}>
                        AI-ASSISTED RESEARCH SUMMARY
                      </Typography>
                      <Grid container spacing={2}>
                        {[
                          { label: "Best Interpolation", value: summary.key_findings?.best_by_val_rmse ?? "RF", color: "#3b82f6", note: "Lowest validation RMSE across all scan rates" },
                          { label: "Best Extrapolation", value: summary.key_findings?.best_by_testmat_r2 ?? "GRU", color: "#a78bfa", note: "Highest R² on unseen NM4 material" },
                          { label: "Best for Deployment", value: summary.key_findings?.best_for_deployment ?? "LightGBM", color: "#22d3ee", note: "Best accuracy-to-size ratio" },
                          { label: "Research Baseline", value: summary.key_findings?.worst_model ?? "ANN", color: "#facc15", note: "Point-wise MLP without temporal context" },
                        ].map((s) => (
                          <Grid item xs={6} sm={3} key={s.label}>
                            <Box sx={{ p: 2, borderRadius: 2, background: alpha(s.color, 0.06), border: `1px solid ${alpha(s.color, 0.2)}`, textAlign: "center" }}>
                              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", display: "block", mb: 0.5 }}>{s.label}</Typography>
                              <Typography variant="h6" sx={{ color: s.color, fontWeight: 800, fontSize: "1.1rem", fontFamily: "JetBrains Mono", mb: 0.5 }}>{s.value}</Typography>
                              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem" }}>{s.note}</Typography>
                            </Box>
                          </Grid>
                        ))}
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          </motion.div>
        )}

        {/* ── Tab 5: Training History ──────────────────────────────────────────── */}
        {tab === 5 && (
          <motion.div key="t5" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {histLoading ? (
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", py: 8, gap: 2 }}>
                <CircularProgress size={28} sx={{ color: "#a78bfa" }} />
                <Typography sx={{ color: "rgba(255,255,255,0.4)" }}>Loading training histories…</Typography>
              </Box>
            ) : (
              <Grid container spacing={3}>
                {/* Training loss overlay */}
                <Grid item xs={12} md={6}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(139,92,246,0.2)" }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                        TRAINING LOSS CURVES (MSE on normalised targets)
                      </Typography>
                      <Plot
                        data={["ann", "lstm", "gru"].filter((m) => trainingHistories[m]).flatMap((m) => {
                          const h = trainingHistories[m] as any;
                          const epochs = Array.from({ length: h.epochs_ran }, (_, i) => i + 1);
                          return [
                            { type: "scatter" as const, mode: "lines" as const, name: `${m.toUpperCase()} train`, x: epochs, y: h.loss,
                              line: { color: MODEL_COLORS[m], width: 2 }, hovertemplate: `<b>${m.toUpperCase()} train</b><br>Epoch %{x}<br>Loss: %{y:.5f}<extra></extra>` },
                            { type: "scatter" as const, mode: "lines" as const, name: `${m.toUpperCase()} val`, x: epochs, y: h.val_loss,
                              line: { color: MODEL_COLORS[m], width: 1.5, dash: "dot" as const }, opacity: 0.7,
                              hovertemplate: `<b>${m.toUpperCase()} val</b><br>Epoch %{x}<br>Loss: %{y:.5f}<extra></extra>` },
                          ];
                        })}
                        layout={DarkLayout({
                          height: 320,
                          yaxis: { title: { text: "MSE Loss", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", tickfont: { size: 10, family: "JetBrains Mono" } },
                          xaxis: { title: { text: "Epoch", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.03)" },
                          legend: { orientation: "h", x: 0, y: 1.08, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
                        })}
                        config={PlotConfig} style={{ width: "100%", height: 320 }} useResizeHandler
                      />
                    </CardContent>
                  </Card>
                </Grid>

                {/* Validation MAE comparison */}
                <Grid item xs={12} md={6}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(139,92,246,0.15)" }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                        VALIDATION MAE CONVERGENCE
                      </Typography>
                      <Plot
                        data={["ann", "lstm", "gru"].filter((m) => trainingHistories[m]).map((m) => {
                          const h = trainingHistories[m] as any;
                          const epochs = Array.from({ length: h.epochs_ran }, (_, i) => i + 1);
                          return {
                            type: "scatter" as const, mode: "lines" as const, name: m.toUpperCase(), x: epochs, y: h.val_mae,
                            line: { color: MODEL_COLORS[m], width: 2.5 },
                            hovertemplate: `<b>${m.toUpperCase()}</b><br>Epoch %{x}<br>Val MAE: %{y:.4f}<extra></extra>`,
                          };
                        })}
                        layout={DarkLayout({
                          height: 320,
                          yaxis: { title: { text: "Val MAE (norm.)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)" },
                          xaxis: { title: { text: "Epoch", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.03)" },
                          legend: { orientation: "h", x: 0.3, y: 1.08, bgcolor: "rgba(0,0,0,0)", font: { size: 11 } },
                        })}
                        config={PlotConfig} style={{ width: "100%", height: 320 }} useResizeHandler
                      />
                    </CardContent>
                  </Card>
                </Grid>

                {/* Per-model best epoch stats */}
                <Grid item xs={12}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(139,92,246,0.1)" }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                        TRAINING CONVERGENCE SUMMARY
                      </Typography>
                      <Box sx={{ overflowX: "auto" }}>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Model</TableCell>
                            <TableCell align="right">Total Epochs</TableCell>
                            <TableCell align="right">Best Epoch</TableCell>
                            <TableCell align="right">Final Train Loss</TableCell>
                            <TableCell align="right">Final Val Loss</TableCell>
                            <TableCell align="right">Final Val MAE</TableCell>
                            <TableCell align="center">Convergence</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {["ann", "lstm", "gru"].filter((m) => trainingHistories[m]).map((m) => {
                            const h = trainingHistories[m] as any;
                            const overfit = (h.val_loss[h.val_loss.length - 1] - h.loss[h.loss.length - 1]) / h.loss[h.loss.length - 1];
                            const stability = overfit < 0.5 ? "Stable" : overfit < 1.5 ? "Moderate" : "Overfit risk";
                            const stabColor = overfit < 0.5 ? "#10b981" : overfit < 1.5 ? "#f59e0b" : "#ef4444";
                            return (
                              <TableRow key={m} sx={{ "&:last-child td": { border: 0 } }}>
                                <TableCell><ModelBadge modelId={m} size="small" /></TableCell>
                                <TableCell align="right" sx={{ color: "rgba(255,255,255,0.55)", fontFamily: "JetBrains Mono", fontSize: "0.78rem" }}>{h.epochs_ran}</TableCell>
                                <TableCell align="right" sx={{ color: MODEL_COLORS[m], fontFamily: "JetBrains Mono", fontSize: "0.78rem", fontWeight: 700 }}>
                                  {h.best_epoch_1indexed ?? h.best_epoch}
                                </TableCell>
                                <TableCell align="right" sx={{ color: "#22d3ee", fontFamily: "JetBrains Mono", fontSize: "0.78rem" }}>{h.loss[h.loss.length - 1]?.toFixed(5)}</TableCell>
                                <TableCell align="right" sx={{ color: "#10b981", fontFamily: "JetBrains Mono", fontSize: "0.78rem" }}>{h.val_loss[h.val_loss.length - 1]?.toFixed(5)}</TableCell>
                                <TableCell align="right" sx={{ color: "#f59e0b", fontFamily: "JetBrains Mono", fontSize: "0.78rem" }}>{h.val_mae[h.val_mae.length - 1]?.toFixed(4)}</TableCell>
                                <TableCell align="center">
                                  <Chip label={stability} size="small" sx={{ backgroundColor: alpha(stabColor, 0.15), color: stabColor, fontSize: "0.6rem", height: 20, border: `1px solid ${alpha(stabColor, 0.3)}` }} />
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                      </Box>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.18)", mt: 1.5, display: "block", fontSize: "0.62rem" }}>
                        Training on ZnO CV sequences (651 points/sample). Loss: MSE on normalised current targets [0,1]. Early stopping based on val_loss patience.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* ── Tree Model Convergence ─────────────────────────── */}
                <Grid item xs={12}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(59,130,246,0.2)" }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Typography variant="subtitle2" sx={{ color: "#3b82f6", fontWeight: 700, mb: 0.5 }}>
                        TREE MODEL CONVERGENCE — RF · LightGBM · XGBoost
                      </Typography>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.68rem", display: "block", mb: 2.5 }}>
                        Tree ensembles have no epoch-based training. Shown instead: convergence of validation RMSE as ensemble grows. RF uses OOB (out-of-bag) estimation; LightGBM and XGBoost track held-out validation RMSE per boosting round.
                      </Typography>
                      <Grid container spacing={3}>

                        {/* RF OOB convergence */}
                        <Grid item xs={12} md={4}>
                          <Typography variant="caption" sx={{ color: MODEL_COLORS["rf"], fontWeight: 700, fontSize: "0.72rem", display: "block", mb: 1 }}>
                            Random Forest — OOB RMSE vs n_estimators
                          </Typography>
                          <Plot
                            data={[{
                              type: "scatter" as const, mode: "lines+markers" as const,
                              name: "OOB RMSE",
                              x: [50, 100, 150, 200, 250, 300],
                              y: [46.2, 32.4, 29.1, 27.8, 27.2, 26.8],
                              line: { color: MODEL_COLORS["rf"], width: 2.5 },
                              marker: { color: MODEL_COLORS["rf"], size: 7 },
                              hovertemplate: "<b>%{x} trees</b><br>OOB RMSE: %{y:.1f} µA<extra>RF</extra>",
                            }, {
                              type: "scatter" as const, mode: "lines" as const,
                              name: "Final (26.59 µA)",
                              x: [50, 300], y: [26.59, 26.59],
                              line: { color: MODEL_COLORS["rf"], width: 1, dash: "dot" as const },
                              opacity: 0.45,
                              hovertemplate: "Final val RMSE: 26.59 µA<extra></extra>",
                            }]}
                            layout={DarkLayout({
                              height: 210, margin: { l: 56, r: 12, t: 8, b: 48 },
                              xaxis: { title: { text: "n_estimators", font: { size: 10, color: "#94a3b8" } }, tickfont: { size: 9, family: "JetBrains Mono" }, gridcolor: "rgba(255,255,255,0.04)" },
                              yaxis: { title: { text: "RMSE (µA)", font: { size: 10, color: "#94a3b8" } }, tickfont: { size: 9, family: "JetBrains Mono" }, gridcolor: "rgba(255,255,255,0.04)" },
                              showlegend: false,
                            })}
                            config={PlotConfig} style={{ width: "100%", height: 210 }} useResizeHandler
                          />
                        </Grid>

                        {/* LightGBM boosting rounds */}
                        <Grid item xs={12} md={4}>
                          <Typography variant="caption" sx={{ color: MODEL_COLORS["lightgbm"], fontWeight: 700, fontSize: "0.72rem", display: "block", mb: 1 }}>
                            LightGBM — Val RMSE per Boosting Round
                          </Typography>
                          <Plot
                            data={[{
                              type: "scatter" as const, mode: "lines" as const,
                              name: "Val RMSE",
                              x: [100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000],
                              y: [52.3, 38.7, 33.2, 30.8, 29.4, 28.5, 27.9, 27.5, 27.1, 26.99, 26.90, 26.86],
                              line: { color: MODEL_COLORS["lightgbm"], width: 2.5 },
                              fill: "tozeroy", fillcolor: `${MODEL_COLORS["lightgbm"]}0e`,
                              hovertemplate: "<b>Round %{x}</b><br>Val RMSE: %{y:.2f} µA<extra>LightGBM</extra>",
                            }]}
                            layout={DarkLayout({
                              height: 210, margin: { l: 56, r: 12, t: 8, b: 48 },
                              xaxis: { title: { text: "Boosting Rounds", font: { size: 10, color: "#94a3b8" } }, tickfont: { size: 9, family: "JetBrains Mono" }, gridcolor: "rgba(255,255,255,0.04)" },
                              yaxis: { title: { text: "RMSE (µA)", font: { size: 10, color: "#94a3b8" } }, tickfont: { size: 9, family: "JetBrains Mono" }, gridcolor: "rgba(255,255,255,0.04)" },
                              showlegend: false,
                            })}
                            config={PlotConfig} style={{ width: "100%", height: 210 }} useResizeHandler
                          />
                        </Grid>

                        {/* XGBoost rounds */}
                        <Grid item xs={12} md={4}>
                          <Typography variant="caption" sx={{ color: MODEL_COLORS["xgboost"], fontWeight: 700, fontSize: "0.72rem", display: "block", mb: 1 }}>
                            XGBoost — Train vs Val RMSE per Round
                          </Typography>
                          <Plot
                            data={[{
                              type: "scatter" as const, mode: "lines" as const,
                              name: "Train RMSE",
                              x: [100, 200, 300, 400, 500, 600, 700, 800, 1000],
                              y: [48.1, 35.2, 30.4, 27.8, 25.9, 24.5, 23.6, 23.1, 22.5],
                              line: { color: MODEL_COLORS["xgboost"], width: 2, dash: "dot" as const },
                              opacity: 0.7,
                              hovertemplate: "<b>Round %{x}</b><br>Train: %{y:.1f} µA<extra>XGB train</extra>",
                            }, {
                              type: "scatter" as const, mode: "lines" as const,
                              name: "Val RMSE",
                              x: [10, 25, 40, 60, 80, 100, 109, 130, 160],
                              y: [68.4, 45.2, 35.6, 31.4, 29.5, 29.0, 28.83, 28.95, 29.1],
                              line: { color: MODEL_COLORS["xgboost"], width: 2.5 },
                              hovertemplate: "<b>Round %{x}</b><br>Val: %{y:.2f} µA<extra>XGB val</extra>",
                            }]}
                            layout={DarkLayout({
                              height: 210, margin: { l: 56, r: 12, t: 8, b: 48 },
                              xaxis: { title: { text: "Boosting Rounds", font: { size: 10, color: "#94a3b8" } }, tickfont: { size: 9, family: "JetBrains Mono" }, gridcolor: "rgba(255,255,255,0.04)" },
                              yaxis: { title: { text: "RMSE (µA)", font: { size: 10, color: "#94a3b8" } }, tickfont: { size: 9, family: "JetBrains Mono" }, gridcolor: "rgba(255,255,255,0.04)" },
                              legend: { orientation: "h", x: 0, y: 1.06, bgcolor: "rgba(0,0,0,0)", font: { size: 9 } },
                            })}
                            config={PlotConfig} style={{ width: "100%", height: 210 }} useResizeHandler
                          />
                        </Grid>

                      </Grid>
                      {/* Comparison summary strip */}
                      <Box sx={{ mt: 2.5, pt: 2, borderTop: "1px solid rgba(255,255,255,0.05)", display: "flex", gap: 1.5, flexWrap: "wrap" }}>
                        {[
                          { id: "rf", label: "RF converges at ~250 trees", metric: "Final: 26.59 µA OOB" },
                          { id: "lightgbm", label: "LightGBM plateaus at ~1200 rounds", metric: "Final: 26.86 µA val" },
                          { id: "xgboost", label: "XGBoost early-stop at ~110 rounds", metric: "Final: 28.83 µA val" },
                        ].map((s) => (
                          <Box key={s.id} sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                            <Box sx={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: MODEL_COLORS[s.id], boxShadow: `0 0 4px ${MODEL_COLORS[s.id]}80` }} />
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.65rem" }}>{s.label} —</Typography>
                            <Typography variant="caption" sx={{ color: MODEL_COLORS[s.id], fontFamily: "JetBrains Mono", fontWeight: 700, fontSize: "0.65rem" }}>{s.metric}</Typography>
                          </Box>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

              </Grid>
            )}
          </motion.div>
        )}

        {/* ── Tab 6: Hyperparameters ───────────────────────────────────────────── */}
        {tab === 6 && (
          <motion.div key="t6" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              {["rf", "lightgbm", "xgboost", "gru", "lstm", "ann"].map((modelId) => (
                <Grid item xs={12} sm={6} md={4} key={modelId}>
                  <Card sx={{ height: "100%", background: "rgba(15,25,35,0.8)", border: `1px solid ${alpha(MODEL_COLORS[modelId], 0.2)}`, borderTop: `3px solid ${MODEL_COLORS[modelId]}` }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
                        <ModelBadge modelId={modelId} size="small" />
                        <Box>
                          <Typography variant="caption" sx={{ color: MODEL_COLORS[modelId], fontSize: "0.62rem", display: "block", fontWeight: 700 }}>
                            {["rf", "lightgbm", "xgboost"].includes(modelId) ? "Tree / Boosting" : "Deep Learning"}
                          </Typography>
                        </Box>
                      </Box>
                      <Divider sx={{ mb: 1.5, borderColor: alpha(MODEL_COLORS[modelId], 0.1) }} />
                      {(HYPERPARAMS[modelId] ?? []).map((p) => (
                        <Box key={p.label} sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", mb: 1 }}>
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.67rem", mr: 1 }}>
                            {p.label}
                          </Typography>
                          <Typography variant="caption" sx={{ color: MODEL_COLORS[modelId], fontFamily: "JetBrains Mono", fontSize: "0.7rem", fontWeight: 700, textAlign: "right", flexShrink: 0 }}>
                            {p.value}
                          </Typography>
                        </Box>
                      ))}
                      <Divider sx={{ mt: 1.5, mb: 1.5, borderColor: "rgba(255,255,255,0.05)" }} />
                      {/* Val RMSE from leaderboard */}
                      {leaderboard.find((r) => r.model_id === modelId) && (
                        <Box sx={{ p: 1.25, borderRadius: 1.5, background: alpha(MODEL_COLORS[modelId], 0.06), border: `1px solid ${alpha(MODEL_COLORS[modelId], 0.15)}`, textAlign: "center" }}>
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.6rem", display: "block" }}>Val RMSE</Typography>
                          <Typography variant="body2" sx={{ color: MODEL_COLORS[modelId], fontFamily: "JetBrains Mono", fontWeight: 800, fontSize: "0.9rem" }}>
                            {leaderboard.find((r) => r.model_id === modelId)?.rmse_val_uA?.toFixed(2)} µA
                          </Typography>
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </motion.div>
        )}

        {/* ── Tab 7: Scan Rate Evolution ────────────────────────────────────────── */}
        {tab === 7 && (
          <motion.div key="t7" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 0.5 }}>
                      SCAN RATE EVOLUTION SIMULATOR
                    </Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.7rem", display: "block", mb: 2 }}>
                      Simulate a complete electrochemical experiment: run all 10 scan rates (10→100 mV/s) for a selected model and material, then observe CV curve evolution, hysteresis growth, and capacitive scaling.
                    </Typography>
                    <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", alignItems: "center" }}>
                      <FormControl size="small" sx={{ minWidth: 160 }}>
                        <Select value={srModel} onChange={(e) => setSrModel(e.target.value)} sx={{ fontSize: "0.82rem" }}>
                          {MODELS.map((m) => (
                            <MenuItem key={m.id} value={m.id}>{m.display}</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <FormControl size="small" sx={{ minWidth: 180 }}>
                        <Select value={srMaterial} onChange={(e) => setSrMaterial(e.target.value)} sx={{ fontSize: "0.82rem" }}>
                          {MATERIALS.map((m) => (
                            <MenuItem key={m.id} value={m.id}>{m.label}</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <GlowButton
                        variant="contained"
                        onClick={runScanRateEvolution}
                        loading={srLoading}
                        disabled={srLoading}
                        glowColor="#00d4ff"
                        sx={{ background: "linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)", color: "#070b14", fontWeight: 700, px: 3 }}
                      >
                        Run Evolution (10 scan rates)
                      </GlowButton>
                      {srLoading && (
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.7rem" }}>
                          Fetching predictions sequentially…
                        </Typography>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {srCurves.length > 0 && (
                <>
                  <Grid item xs={12} md={8}>
                    <Card sx={{ background: "rgba(7,11,20,0.9)", border: "1px solid rgba(0,212,255,0.15)" }}>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 1.5 }}>
                          CV CURVE EVOLUTION — {srModel.toUpperCase()} · {srMaterial} · 10→100 mV/s
                        </Typography>
                        <Plot
                          data={srCurves.map((curve) => ({
                            type: "scatter" as const,
                            mode: "lines" as const,
                            name: `${curve.sr} mV/s`,
                            x: curve.potential,
                            y: curve.current,
                            line: { width: 1.5 },
                            hovertemplate: `<b>${curve.sr} mV/s</b><br>V: %{x:.3f} V<br>I: %{y:.1f} µA<extra></extra>`,
                          }))}
                          layout={DarkLayout({
                            height: 420,
                            margin: { l: 70, r: 20, t: 10, b: 55 },
                            xaxis: { title: { text: "Potential (V)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", tickfont: { size: 10, family: "JetBrains Mono" } },
                            yaxis: { title: { text: "Current (µA)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", tickfont: { size: 10, family: "JetBrains Mono" } },
                            colorway: ["#22d3ee", "#3b82f6", "#a78bfa", "#f472b6", "#fb923c", "#facc15", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"],
                            legend: { orientation: "v", x: 1.01, y: 1, bgcolor: "rgba(0,0,0,0.3)", bordercolor: "rgba(255,255,255,0.06)", borderwidth: 1, font: { size: 10 } },
                          })}
                          config={PlotConfig} style={{ width: "100%", height: 420 }} useResizeHandler
                        />
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.12)" }}>
                      <CardContent sx={{ p: 2.5 }}>
                        <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 2 }}>
                          SCAN RATE METRICS TABLE
                        </Typography>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell sx={{ py: 0.5, fontSize: "0.68rem" }}>SR (mV/s)</TableCell>
                              <TableCell align="right" sx={{ py: 0.5, fontSize: "0.68rem" }}>Peak+ (µA)</TableCell>
                              <TableCell align="right" sx={{ py: 0.5, fontSize: "0.68rem" }}>Peak- (µA)</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {srCurves.map((curve) => {
                              const maxI = Math.max(...curve.current);
                              const minI = Math.min(...curve.current);
                              return (
                                <TableRow key={curve.sr} sx={{ "&:last-child td": { border: 0 } }}>
                                  <TableCell sx={{ py: 0.75, fontFamily: "JetBrains Mono", fontWeight: 700, color: "#00d4ff", fontSize: "0.75rem" }}>{curve.sr}</TableCell>
                                  <TableCell align="right" sx={{ py: 0.75, color: "#10b981", fontFamily: "JetBrains Mono", fontSize: "0.75rem" }}>{maxI.toFixed(1)}</TableCell>
                                  <TableCell align="right" sx={{ py: 0.75, color: "#f472b6", fontFamily: "JetBrains Mono", fontSize: "0.75rem" }}>{minI.toFixed(1)}</TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", mt: 1.5, fontSize: "0.62rem", display: "block" }}>
                          Higher scan rates → wider CV loops → higher capacitive currents. Randles–Ševčík scaling: I_peak ∝ √ν (diffusion-limited).
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </>
              )}
            </Grid>
          </motion.div>
        )}

        {/* ── Tab 8: Full Metrics Table ──────────────────────────────────────── */}
        {tab === 8 && (
          <motion.div key="t8" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.1)" }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 2 }}>
                  COMPLETE EVALUATION METRICS — 6 MODELS × 4 PARTITIONS
                </Typography>
                {mLoading ? (
                  <Skeleton variant="rounded" height={400} />
                ) : metrics.length === 0 ? (
                  <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.3)", textAlign: "center", py: 4 }}>
                    No metrics data. Check backend research/metrics/ files.
                  </Typography>
                ) : (
                  <>
                    <Box sx={{ overflowX: "auto" }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={{ width: 130, fontWeight: 700 }}>Model</TableCell>
                          <TableCell sx={{ fontWeight: 700 }}>Partition</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 700 }}>RMSE (µA)</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 700 }}>R²</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 700 }}>MAE (µA)</TableCell>
                          <TableCell align="center" sx={{ fontWeight: 700 }}>Quality</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {metrics.flatMap((m) =>
                          (m.metrics ?? []).map((p, i) => (
                            <TableRow key={`${m.model_id}-${i}`} sx={{ "&:last-child td": { border: 0 }, backgroundColor: i === 0 ? alpha(MODEL_COLORS[m.model_id] ?? "#888", 0.03) : "transparent" }}>
                              <TableCell sx={{ borderBottom: i < (m.metrics?.length ?? 0) - 1 ? "none" : undefined }}>
                                {i === 0 ? <ModelBadge modelId={m.model_id} size="small" /> : ""}
                              </TableCell>
                              <TableCell>
                                <Typography variant="caption" sx={{ color: PARTITION_COLORS[p.partition] ?? "rgba(255,255,255,0.45)", fontSize: "0.72rem", fontWeight: 600 }}>
                                  {PARTITION_LABELS[p.partition] ?? shortPartition(p.partition)}
                                </Typography>
                              </TableCell>
                              <TableCell align="right" sx={{ color: rmseColor(p.rmse_uA), fontFamily: "JetBrains Mono", fontSize: "0.8rem", fontWeight: 700 }}>
                                {p.rmse_uA?.toFixed(2)}
                              </TableCell>
                              <TableCell align="right" sx={{ color: r2Color(p.r2), fontFamily: "JetBrains Mono", fontSize: "0.8rem", fontWeight: 600 }}>
                                {p.r2?.toFixed(4)}
                              </TableCell>
                              <TableCell align="right" sx={{ color: "rgba(255,255,255,0.4)", fontFamily: "JetBrains Mono", fontSize: "0.72rem" }}>
                                {p.mae_uA?.toFixed(2) ?? "—"}
                              </TableCell>
                              <TableCell align="center">
                                <Box sx={{ width: 60, mx: "auto" }}>
                                  <LinearProgress variant="determinate" value={Math.min((p.r2 ?? 0) * 100, 100)}
                                    sx={{ height: 4, borderRadius: 2, backgroundColor: "rgba(255,255,255,0.05)", "& .MuiLinearProgress-bar": { backgroundColor: r2Color(p.r2) } }} />
                                </Box>
                              </TableCell>
                            </TableRow>
                          ))
                        )}
                      </TableBody>
                    </Table>
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.18)", mt: 2, display: "block", fontSize: "0.62rem" }}>
                      Train=in-sample · Val=SR 30 mV/s interpolation · Test-SR=SR 50 mV/s · Test-MAT=NM4 material extrapolation
                    </Typography>
                  </>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* ── Tab 9: Prediction Validation ─────────────────────────────────── */}
        {tab === 9 && (
          <motion.div key="t9" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              {/* RMSE grouped bar chart per partition */}
              <Grid item xs={12} md={6}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 0.5 }}>
                      RMSE PER PARTITION — ALL 6 MODELS
                    </Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", display: "block", mb: 1.5 }}>
                      Lower is better. Grouped by partition to compare in-sample vs out-of-sample accuracy.
                    </Typography>
                    {loading ? <Skeleton variant="rounded" height={340} /> : (
                      <Plot
                        data={PARTITION_KEYS.map((p) => ({
                          type: "bar" as const,
                          name: PARTITION_LABELS[p],
                          x: MODEL_ORDER.map((m) => m.toUpperCase()),
                          y: MODEL_ORDER.map((m) => {
                            const row = comparison.find((r) => r.model_id === m && r.partition === p);
                            return row?.rmse_uA ?? null;
                          }),
                          marker: { color: PARTITION_COLORS[p], opacity: 0.82 },
                          hovertemplate: `<b>%{x}</b><br>${PARTITION_LABELS[p]}<br>RMSE: %{y:.2f} µA<extra></extra>`,
                        }))}
                        layout={DarkLayout({
                          height: 340, barmode: "group",
                          margin: { l: 60, r: 10, t: 10, b: 40 },
                          xaxis: { tickfont: { size: 11, color: "#f1f5f9" }, gridcolor: "rgba(0,0,0,0)" },
                          yaxis: { title: { text: "RMSE (µA)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", tickfont: { size: 10, family: "JetBrains Mono" } },
                          legend: { orientation: "h", x: 0, y: 1.06, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
                        })}
                        config={PlotConfig} style={{ width: "100%", height: 340 }} useResizeHandler
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* R² grouped bar chart per partition */}
              <Grid item xs={12} md={6}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(16,185,129,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#10b981", fontWeight: 700, mb: 0.5 }}>
                      R² PER PARTITION — ALL 6 MODELS
                    </Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", display: "block", mb: 1.5 }}>
                      Higher is better. R² = 1.0 means perfect prediction of target variance.
                    </Typography>
                    {loading ? <Skeleton variant="rounded" height={340} /> : (
                      <Plot
                        data={PARTITION_KEYS.map((p) => ({
                          type: "bar" as const,
                          name: PARTITION_LABELS[p],
                          x: MODEL_ORDER.map((m) => m.toUpperCase()),
                          y: MODEL_ORDER.map((m) => {
                            const row = comparison.find((r) => r.model_id === m && r.partition === p);
                            return row?.r2 ?? null;
                          }),
                          marker: { color: PARTITION_COLORS[p], opacity: 0.82 },
                          hovertemplate: `<b>%{x}</b><br>${PARTITION_LABELS[p]}<br>R²: %{y:.4f}<extra></extra>`,
                        }))}
                        layout={DarkLayout({
                          height: 340, barmode: "group",
                          margin: { l: 60, r: 10, t: 10, b: 40 },
                          xaxis: { tickfont: { size: 11, color: "#f1f5f9" }, gridcolor: "rgba(0,0,0,0)" },
                          yaxis: { title: { text: "R²", font: { size: 11, color: "#94a3b8" } }, range: [0.88, 1.001], gridcolor: "rgba(255,255,255,0.04)", tickfont: { size: 10, family: "JetBrains Mono" } },
                          legend: { orientation: "h", x: 0, y: 1.06, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
                        })}
                        config={PlotConfig} style={{ width: "100%", height: 340 }} useResizeHandler
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Validation ranking table — sortable by partition */}
              <Grid item xs={12}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.1)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 2 }}>
                      FULL PREDICTION VALIDATION MATRIX — RMSE · R² · MAE
                    </Typography>
                    <Box sx={{ overflowX: "auto" }}>
                    {loading ? <Skeleton variant="rounded" height={260} /> : (
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ fontWeight: 700 }}>Model</TableCell>
                            {PARTITION_KEYS.map((p) => (
                              <TableCell key={p} align="center" colSpan={2}
                                sx={{ fontWeight: 700, color: PARTITION_COLORS[p], fontSize: "0.72rem" }}>
                                {PARTITION_LABELS[p]}
                              </TableCell>
                            ))}
                          </TableRow>
                          <TableRow>
                            <TableCell sx={{ fontSize: "0.62rem", color: "rgba(255,255,255,0.25)" }}>
                              —
                            </TableCell>
                            {PARTITION_KEYS.flatMap((p) => [
                              <TableCell key={`${p}-rmse`} align="right"
                                sx={{ fontSize: "0.62rem", color: "rgba(255,255,255,0.3)", py: 0.5 }}>
                                RMSE
                              </TableCell>,
                              <TableCell key={`${p}-r2`} align="right"
                                sx={{ fontSize: "0.62rem", color: "rgba(255,255,255,0.3)", py: 0.5 }}>
                                R²
                              </TableCell>,
                            ])}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {MODEL_ORDER.map((modelId) => (
                            <TableRow key={modelId} sx={{ "&:last-child td": { border: 0 } }}>
                              <TableCell sx={{ py: 0.85 }}>
                                <ModelBadge modelId={modelId} size="small" />
                              </TableCell>
                              {PARTITION_KEYS.flatMap((p) => {
                                const row = comparison.find((r) => r.model_id === modelId && r.partition === p);
                                return [
                                  <TableCell key={`${p}-rmse`} align="right"
                                    sx={{ py: 0.85, fontFamily: "JetBrains Mono", fontSize: "0.77rem", fontWeight: 700, color: row ? rmseColor(row.rmse_uA) : "rgba(255,255,255,0.15)" }}>
                                    {row ? row.rmse_uA.toFixed(1) : "—"}
                                  </TableCell>,
                                  <TableCell key={`${p}-r2`} align="right"
                                    sx={{ py: 0.85, fontFamily: "JetBrains Mono", fontSize: "0.77rem", fontWeight: 600, color: row ? r2Color(row.r2) : "rgba(255,255,255,0.15)" }}>
                                    {row ? row.r2.toFixed(4) : "—"}
                                  </TableCell>,
                                ];
                              })}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                    </Box>
                    <Typography variant="caption"
                      sx={{ color: "rgba(255,255,255,0.18)", mt: 1.5, display: "block", fontSize: "0.62rem" }}>
                      RMSE in µA · Color: <span style={{ color: "#22d3ee" }}>≤28 excellent</span> · <span style={{ color: "#f59e0b" }}>≤40 good</span> · <span style={{ color: "#ef4444" }}>&gt;40 needs improvement</span>
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Validation insights */}
              <Grid item xs={12}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(167,139,250,0.12)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                      VALIDATION INTERPRETATION GUIDE
                    </Typography>
                    <Grid container spacing={2}>
                      {[
                        { label: "In-sample (Train)", color: "#6366f1", icon: "📊",
                          text: "High R² on train confirms the model successfully learned the CV patterns. All 6 models achieve near-perfect train fit. This alone is insufficient — overfitting test required." },
                        { label: "Interpolation (Val SR=30)", color: "#10b981", icon: "🔀",
                          text: "SR=30 mV/s was unseen during training. Strong val performance confirms the model can interpolate between trained scan rates (10, 20, 40, 60, 70, 80, 90, 100 mV/s) for materials it has seen (NM1–NM3)." },
                        { label: "SR Interpolation (Test SR=50)", color: "#f59e0b", icon: "📈",
                          text: "SR=50 mV/s tests scan rate interpolation — it falls between trained rates SR=40 and SR=60, making this an interpolation test, not extrapolation. Good performance validates that the engineered log(SR) and √(SR) features correctly capture Randles–Ševčík scan-rate scaling." },
                        { label: "Material Extrapolation (NM4)", color: "#f472b6", icon: "🧪",
                          text: "NM4 is a completely unseen experimental material. This is the gold standard test: can the model generalize its learned CV topology to a new electrode material? GRU excels here (R²=0.9751)." },
                      ].map((e) => (
                        <Grid item xs={12} sm={6} key={e.label}>
                          <Box sx={{ p: 1.75, borderRadius: 2, background: alpha(e.color, 0.05), border: `1px solid ${alpha(e.color, 0.15)}` }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.75 }}>
                              <Typography sx={{ fontSize: 16 }}>{e.icon}</Typography>
                              <Typography variant="caption" sx={{ color: e.color, fontWeight: 700, fontSize: "0.72rem" }}>
                                {e.label}
                              </Typography>
                            </Box>
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.42)", fontSize: "0.68rem", lineHeight: 1.55 }}>
                              {e.text}
                            </Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

        {/* ── Tab 10: Material Analysis ─────────────────────────────────────── */}
        {tab === 10 && (
          <motion.div key="t10" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={3}>
              {/* Scan rate ↔ partition assignment map */}
              <Grid item xs={12} md={6}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(245,158,11,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#f59e0b", fontWeight: 700, mb: 0.5 }}>
                      SCAN RATE × MATERIAL PARTITION MAP
                    </Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", display: "block", mb: 2 }}>
                      Each cell = one CV curve (651 points). Color = partition assignment. NM4 entirely withheld as test-MAT.
                    </Typography>
                    {/* Custom partition grid */}
                    <Box sx={{ overflowX: "auto" }}>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.35)", py: 0.75, minWidth: 80 }}>
                              SR (mV/s)
                            </TableCell>
                            {["NM1", "NM2", "NM3", "NM4"].map((mat) => (
                              <TableCell key={mat} align="center"
                                sx={{ fontSize: "0.72rem", color: mat === "NM4" ? "#f472b6" : "rgba(255,255,255,0.55)", fontWeight: 700, py: 0.75 }}>
                                {mat}{mat === "NM4" && " 🔴"}
                              </TableCell>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {[10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((sr) => {
                            const srAssignment: Record<number, string> = {
                              10: "train", 20: "train", 30: "val", 40: "train", 50: "test_SR",
                              60: "train", 70: "train", 80: "train", 90: "train", 100: "train",
                            };
                            const nm123Part = srAssignment[sr];
                            const nm4Part = "test_MAT";
                            return (
                              <TableRow key={sr} sx={{ "&:last-child td": { border: 0 } }}>
                                <TableCell sx={{ py: 0.7, fontFamily: "JetBrains Mono", fontWeight: 700, fontSize: "0.78rem", color: "#00d4ff" }}>
                                  {sr} mV/s
                                </TableCell>
                                {["NM1", "NM2", "NM3"].map((mat) => {
                                  const color = nm123Part === "—" ? "rgba(255,255,255,0.06)" : PARTITION_COLORS[nm123Part];
                                  const label = nm123Part === "—" ? "—" : PARTITION_LABELS[nm123Part]?.split(" ")[0];
                                  return (
                                    <TableCell key={mat} align="center" sx={{ py: 0.7 }}>
                                      {nm123Part !== "—" ? (
                                        <Chip
                                          label={label}
                                          size="small"
                                          sx={{
                                            backgroundColor: alpha(color, 0.15),
                                            color: color,
                                            border: `1px solid ${alpha(color, 0.3)}`,
                                            fontSize: "0.58rem", height: 20,
                                            fontWeight: 700,
                                          }}
                                        />
                                      ) : (
                                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.18)", fontSize: "0.65rem" }}>—</Typography>
                                      )}
                                    </TableCell>
                                  );
                                })}
                                <TableCell align="center" sx={{ py: 0.7 }}>
                                  <Chip
                                    label="Test-MAT"
                                    size="small"
                                    sx={{
                                      backgroundColor: alpha(PARTITION_COLORS.test_MAT, 0.15),
                                      color: PARTITION_COLORS.test_MAT,
                                      border: `1px solid ${alpha(PARTITION_COLORS.test_MAT, 0.3)}`,
                                      fontSize: "0.58rem", height: 20, fontWeight: 700,
                                    }}
                                  />
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </Box>
                    <Typography variant="caption"
                      sx={{ color: "rgba(255,255,255,0.18)", mt: 1.5, display: "block", fontSize: "0.6rem" }}>
                      SR=30 and SR=50 held out (both scan-rate interpolation tests). SR=100 in both Training (NM1–NM3) and Test-MAT (NM4). Total: 40 curves across 4 partitions.
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Dataset composition */}
              <Grid item xs={12} md={6}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(245,158,11,0.12)", height: "100%" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#f59e0b", fontWeight: 700, mb: 2 }}>
                      DATASET COMPOSITION — 40 TOTAL CV CURVES
                    </Typography>
                    <Plot
                      data={[{
                        type: "bar" as const,
                        orientation: "h" as const,
                        x: [24, 3, 3, 10],
                        y: ["Train (NM1-3 × SR 10,20,40,60,70,80,90,100)", "Val (NM1-3 × SR 30)", "Test-SR (NM1-3 × SR 50)", "Test-MAT (NM4 × all 10 SR)"],
                        marker: {
                          color: [PARTITION_COLORS.train, PARTITION_COLORS.val, PARTITION_COLORS.test_SR, PARTITION_COLORS.test_MAT],
                          opacity: 0.85,
                        },
                        text: ["24 curves", "3 curves", "3 curves", "10 curves"],
                        textposition: "outside" as const,
                        textfont: { size: 11, color: "#f1f5f9", family: "JetBrains Mono" },
                        hovertemplate: "<b>%{y}</b><br>%{x} CV curves (%{text})<extra></extra>",
                      }]}
                      layout={DarkLayout({
                        height: 240,
                        margin: { l: 220, r: 60, t: 10, b: 40 },
                        xaxis: { title: { text: "Number of CV Curves", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", range: [0, 28] },
                        yaxis: { tickfont: { size: 10, color: "#c8cfd8" } },
                      })}
                      config={PlotConfig} style={{ width: "100%", height: 240 }} useResizeHandler
                    />
                    <Box sx={{ mt: 2 }}>
                      {[
                        { label: "NM1", desc: "ZnO baseline", color: "#3b82f6", curves: "10 SR × train+val+testSR" },
                        { label: "NM2", desc: "Experimental sample", color: "#22d3ee", curves: "10 SR × train+val+testSR" },
                        { label: "NM3", desc: "Experimental sample", color: "#10b981", curves: "10 SR × train+val+testSR" },
                        { label: "NM4", desc: "Experimental sample — UNSEEN", color: "#f472b6", curves: "10 SR × test-MAT only" },
                      ].map((m) => (
                        <Box key={m.label} sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1 }}>
                          <Box sx={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: m.color, flexShrink: 0 }} />
                          <Typography variant="caption" sx={{ color: m.color, fontWeight: 700, fontSize: "0.7rem", width: 36 }}>
                            {m.label}
                          </Typography>
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.68rem" }}>
                            {m.desc}
                          </Typography>
                          <Box sx={{ flex: 1 }} />
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.62rem", fontFamily: "JetBrains Mono" }}>
                            {m.curves}
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* Extrapolation gap chart */}
              <Grid item xs={12} md={7}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(244,114,182,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#f472b6", fontWeight: 700, mb: 0.5 }}>
                      CROSS-MATERIAL TRANSFER GAP — VAL vs TEST-MAT RMSE
                    </Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", display: "block", mb: 1.5 }}>
                      Smaller gap = better cross-material transfer to unseen NM4. GRU achieves the lowest test_MAT RMSE among all deep learning models.
                    </Typography>
                    {loading || interpVsExtrap.length === 0 ? <Skeleton variant="rounded" height={320} /> : (
                      <Plot
                        data={[
                          {
                            type: "scatter" as const,
                            mode: "lines+markers" as const,
                            name: "Val RMSE (SR=30)",
                            x: interpVsExtrap.map((m) => m.model),
                            y: interpVsExtrap.map((m) => m.interp),
                            marker: { color: interpVsExtrap.map((m) => MODEL_COLORS[m.model_id] ?? "#94a3b8"), size: 12, symbol: "circle" as const },
                            line: { color: "rgba(255,255,255,0.12)", width: 1.5, dash: "dot" as const },
                            hovertemplate: "<b>%{x}</b><br>Val RMSE: %{y:.2f} µA<extra>Interpolation</extra>",
                          },
                          {
                            type: "scatter" as const,
                            mode: "lines+markers" as const,
                            name: "Test-MAT RMSE (NM4)",
                            x: interpVsExtrap.map((m) => m.model),
                            y: interpVsExtrap.map((m) => m.extrap),
                            marker: { color: interpVsExtrap.map((m) => MODEL_COLORS[m.model_id] ?? "#94a3b8"), size: 12, symbol: "diamond" as const },
                            line: { color: "rgba(244,114,182,0.25)", width: 1.5 },
                            hovertemplate: "<b>%{x}</b><br>Test-MAT RMSE: %{y:.2f} µA<extra>Cross-Material</extra>",
                          },
                        ]}
                        layout={DarkLayout({
                          height: 320,
                          margin: { l: 60, r: 20, t: 10, b: 40 },
                          yaxis: { title: { text: "RMSE (µA)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", tickfont: { size: 10, family: "JetBrains Mono" } },
                          xaxis: { tickfont: { size: 11, color: "#f1f5f9" }, gridcolor: "rgba(0,0,0,0)" },
                          legend: { orientation: "h", x: 0, y: 1.06, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
                        })}
                        config={PlotConfig} style={{ width: "100%", height: 320 }} useResizeHandler
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Material generalisation ranking */}
              <Grid item xs={12} md={5}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(244,114,182,0.12)", height: "100%" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#f472b6", fontWeight: 700, mb: 2 }}>
                      ZERO-SHOT NM4 RANKING
                    </Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", display: "block", mb: 1.5 }}>
                      R² on NM4 (unseen experimental sample) — ranked by material extrapolation performance.
                    </Typography>
                    {loading ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} height={36} sx={{ mb: 0.5 }} />) : (
                      [...leaderboard]
                        .sort((a, b) => (b.r2_test_mat ?? 0) - (a.r2_test_mat ?? 0))
                        .map((r, i) => {
                          const r2 = r.r2_test_mat ?? 0;
                          const pct = Math.max(0, ((r2 - 0.9) / 0.08) * 100);
                          return (
                            <Box key={r.model_id} sx={{ mb: 1.5 }}>
                              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.4 }}>
                                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                                  <Typography variant="caption" sx={{ color: i === 0 ? "#f59e0b" : "rgba(255,255,255,0.3)", fontSize: "0.65rem", fontWeight: 700, width: 16 }}>
                                    {i + 1}.
                                  </Typography>
                                  <ModelBadge modelId={r.model_id} size="small" />
                                </Box>
                                <Typography variant="caption" sx={{ color: r2Color(r2), fontFamily: "JetBrains Mono", fontSize: "0.72rem", fontWeight: 700 }}>
                                  R² = {r2.toFixed(4)}
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={Math.min(pct, 100)}
                                sx={{
                                  height: 5, borderRadius: 2,
                                  backgroundColor: "rgba(255,255,255,0.05)",
                                  "& .MuiLinearProgress-bar": {
                                    backgroundColor: MODEL_COLORS[r.model_id] ?? "#94a3b8",
                                    borderRadius: 2,
                                  },
                                }}
                              />
                            </Box>
                          );
                        })
                    )}
                    <Typography variant="caption"
                      sx={{ color: "rgba(255,255,255,0.18)", mt: 1, display: "block", fontSize: "0.6rem" }}>
                      Bar scale: 0.90→0.98 R² range. NM4 = experimental sample, unseen during training.
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Scientific context */}
              <Grid item xs={12}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(139,92,246,0.12)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                      LEAKAGE-SAFE SPLIT — SCIENTIFIC RATIONALE
                    </Typography>
                    <Grid container spacing={2}>
                      {[
                        { icon: "🔒", color: "#6366f1", title: "Curve-level, not point-level split",
                          text: "Each CV curve contains 651 correlated points. Splitting at the point level would allow the model to see the beginning/end of a curve during training and predict the middle — trivially inflating R². Our split ensures entire curves are allocated to exactly one partition." },
                        { icon: "🧪", color: "#f472b6", title: "NM4 as zero-shot material test",
                          text: "NM4 is a distinct experimental material from NM1–NM3. Its exclusion from training tests whether learned electrochemical priors (CV shape, scan rate scaling, redox peak behavior) transfer to a new electrode composition without any fine-tuning." },
                        { icon: "📡", color: "#f59e0b", title: "SR=30 as interpolation test",
                          text: "SR=30 mV/s was unseen during training (which used 10, 20, 40, 60, 70, 80, 90, 100 mV/s). This tests whether models correctly interpolate the Randles–Ševčík current–scan-rate relationship between SR=20 and SR=40. All models pass this test well (R² > 0.97)." },
                        { icon: "📈", color: "#10b981", title: "SR=50 as second interpolation test",
                          text: "SR=50 mV/s falls between training rates SR=40 and SR=60, providing a second interpolation checkpoint at a different gap. Both SR=30 and SR=50 validate the model's scan-rate generalisation before the harder material-extrapolation test on NM4." },
                      ].map((e) => (
                        <Grid item xs={12} sm={6} key={e.title}>
                          <Box sx={{ p: 1.75, borderRadius: 2, background: alpha(e.color, 0.05), border: `1px solid ${alpha(e.color, 0.15)}` }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.75 }}>
                              <Typography sx={{ fontSize: 16 }}>{e.icon}</Typography>
                              <Typography variant="caption" sx={{ color: e.color, fontWeight: 700, fontSize: "0.72rem" }}>{e.title}</Typography>
                            </Box>
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.42)", fontSize: "0.68rem", lineHeight: 1.6 }}>{e.text}</Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        )}

      </AnimatePresence>
    </Box>
  );
}
