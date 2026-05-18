import React from "react";
import {
  Box, Grid, Card, CardContent, Typography, Chip, alpha,
} from "@mui/material";
import {
  ScienceRounded, DatasetRounded, InsightsRounded, BuildRounded,
  CheckCircleRounded, AccountTreeRounded, LockRounded,
  WarningAmberRounded, RocketLaunchRounded, ElectricCarRounded,
  DevicesRounded, BiotechRounded, AutoGraphRounded, FactoryRounded,
  MenuBookRounded,
} from "@mui/icons-material";
import { motion } from "framer-motion";
import Plot from "react-plotly.js";
import SectionHeader from "../components/common/SectionHeader";
import GlossaryTooltip from "../components/common/GlossaryTooltip";
import { GLOSSARY } from "../constants/glossary";
import { cardVariants, staggerContainer, fadeUp } from "../animations/variants";

// ── Methodology pipeline steps ────────────────────────────────────────────
const METHODOLOGY_STEPS = [
  { icon: "🧪", title: "ZnO Nanocomposite Synthesis", color: "#22d3ee",
    content: "Four ZnO-based nanocomposite electrode materials were synthesised: NM1 (ZnO baseline), NM2 (ZnO/reduced Graphene Oxide), NM3 (ZnO/MnO₂), and NM4 (ZnO/Co₃O₄). Cyclic voltammetry experiments were conducted at ten scan rates (10, 20, 30, 40, 50, 60, 70, 80, 90, 100 mV/s)." },
  { icon: "📊", title: "Dataset Construction", color: "#10b981",
    content: "CV data was digitised into 651-point voltage sweeps (−0.65V → 0V → −0.65V, step 0.002V). Each training point has 10 engineered features: potential_V, scan_rate_mVs, log_scan_rate, sqrt_scan_rate, potential_from_lower, potential_from_upper, sr×potential, direction×potential, sweep_direction, and sweep_position." },
  { icon: "🔒", title: "Leakage-Safe Data Splitting", color: "#a78bfa",
    content: "Critical design decision: splits performed at CV-curve level (not point level) to prevent data leakage. Train: NM1–NM3 at SR=10,20,40,60,70,80,90,100 mV/s. Validation: NM1–NM3 at SR=30 mV/s (interpolation). Test-SR: NM1–NM3 at SR=50 mV/s. Test-MAT: NM4 all rates (material extrapolation)." },
  { icon: "⚙️", title: "Feature Normalisation", color: "#fb923c",
    content: "Per-group MinMax normalisation: scalers fitted on each (material_id, scan_rate) combination independently, then stored in scalers.json. Current values denormalised using the same group key at inference. This design avoids cross-group leakage between materials and scan rates." },
  { icon: "🤖", title: "Model Training & Tuning", color: "#f472b6",
    content: "Six ML architectures: Random Forest (300 trees), LightGBM (leaf-wise GBDT), XGBoost (regularised boosting), Dense ANN (4-layer MLP), Stacked LSTM (2× LSTM layers), Stacked GRU (2× GRU layers). Deep learning models used sequences of length 651 with 10 features per timestep." },
  { icon: "📈", title: "Evaluation Protocol", color: "#facc15",
    content: "Primary metrics: RMSE (µA) and R². Four evaluation partitions allow measuring in-sample fit, scan-rate interpolation at SR=30 (Val), scan-rate interpolation at SR=50 (Test-SR), and material extrapolation to unseen NM4 (Test-MAT) — providing a complete multi-dimensional picture of generalisation capability." },
];

const FINDINGS = [
  { text: "RF achieves lowest validation RMSE (26.59 µA) — best accuracy overall", color: "#3b82f6" },
  { text: "LightGBM is ~360× smaller than RF with only 1% higher RMSE — clear deployment winner", color: "#22d3ee" },
  { text: "GRU achieves highest test-MAT R² (0.9751) — best generalisation to unseen ZnO/Co₃O₄", color: "#a78bfa" },
  { text: "All 6 models achieve R² > 0.96 on test-MAT, validating the approach for ZnO composites", color: "#10b981" },
  { text: "Leakage-safe splitting prevents overly optimistic metrics — results are reliable", color: "#f59e0b" },
  { text: "Scan rate features (log, sqrt, sr×potential) are the most important predictors across all tree models", color: "#fb923c" },
];

// ── Research pipeline flow steps ─────────────────────────────────────────
const PIPELINE_STEPS = [
  { label: "Material\nSynthesis",    color: "#22d3ee", icon: "🧪", desc: "ZnO nanocomposites NM1–NM4" },
  { label: "CV\nAcquisition",        color: "#10b981", icon: "📡", desc: "10 scan rates × 4 materials" },
  { label: "Feature\nEngineering",   color: "#a78bfa", icon: "⚙️", desc: "10 electrochemical features" },
  { label: "Leakage-Safe\nSplitting",color: "#f59e0b", icon: "🔒", desc: "Curve-level, not point-level" },
  { label: "Model\nTraining",        color: "#f472b6", icon: "🤖", desc: "6 architectures, 4 partitions" },
  { label: "Evaluation\n& Deploy",   color: "#fb923c", icon: "📈", desc: "RMSE + R² across all splits" },
];

export default function About() {
  return (
    <Box sx={{ width: "100%", overflowX: "hidden" }}>
      <SectionHeader
        title="About This Research"
        subtitle="AI-based prediction of ZnO supercapacitor CV curves — methodology, dataset, leakage-safe evaluation, and scientific findings"
        accent="#a78bfa"
      />

      {/* ── Abstract ─────────────────────────────────────────────────── */}
      <motion.div variants={cardVariants} initial="initial" animate="animate">
        <Card sx={{ mb: 3.5, background: "linear-gradient(135deg, rgba(124,58,237,0.07) 0%, rgba(0,212,255,0.04) 100%)", border: "1px solid rgba(124,58,237,0.2)", borderLeft: "4px solid #a78bfa" }}>
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
              <ScienceRounded sx={{ color: "#a78bfa", fontSize: 22 }} />
              <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Project Abstract</Typography>
            </Box>
            <Typography variant="body1" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.8, mb: 2 }}>
              This platform implements an end-to-end AI system for predicting Cyclic Voltammetry (CV) trajectories of ZnO-based supercapacitor electrodes. Six machine learning models (Random Forest, XGBoost, LightGBM, Dense ANN, Stacked LSTM, Stacked GRU) were trained on experimentally measured CV data from four ZnO nanocomposites.
            </Typography>
            <Typography variant="body1" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.8 }}>
              The platform enables researchers to predict complete CV profiles (651 points per curve) at arbitrary scan rates for known and unseen materials, without needing to run physical electrochemical experiments. All models achieve R² {">"} 0.96 on unseen material extrapolation, validating the approach for accelerated supercapacitor research.
            </Typography>
          </CardContent>
        </Card>
      </motion.div>

      {/* ══ ANIMATED RESEARCH PIPELINE ══════════════════════════════ */}
      <motion.div variants={fadeUp} initial="initial" animate="animate">
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
          <AccountTreeRounded sx={{ color: "#00d4ff", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Research Pipeline</Typography>
        </Box>
      </motion.div>

      <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ marginBottom: 28 }}>
        <Card sx={{ background: "rgba(15,25,35,0.85)", border: "1px solid rgba(0,212,255,0.12)" }}>
          <CardContent sx={{ p: 2.5 }}>
            {/* Desktop flow */}
            <Box sx={{ display: { xs: "none", sm: "flex" }, alignItems: "center", justifyContent: "space-between", gap: 0 }}>
              {PIPELINE_STEPS.map((step, i) => (
                <React.Fragment key={step.label}>
                  <motion.div
                    initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.12, duration: 0.35 }}
                    style={{ flex: 1, textAlign: "center" }}
                  >
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
                      <Box sx={{ width: 54, height: 54, borderRadius: "50%", background: `radial-gradient(circle, ${alpha(step.color, 0.25)} 0%, ${alpha(step.color, 0.08)} 100%)`, border: `2px solid ${alpha(step.color, 0.5)}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, boxShadow: `0 0 16px ${alpha(step.color, 0.2)}` }}>
                        {step.icon}
                      </Box>
                      <Typography variant="caption" sx={{ color: step.color, fontWeight: 700, fontSize: "0.7rem", textAlign: "center", whiteSpace: "pre-line", lineHeight: 1.3 }}>{step.label}</Typography>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.28)", fontSize: "0.62rem", textAlign: "center", lineHeight: 1.3 }}>{step.desc}</Typography>
                    </Box>
                  </motion.div>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <motion.div initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ delay: i * 0.12 + 0.2, duration: 0.3 }} style={{ transformOrigin: "left" }}>
                      <Box sx={{ display: "flex", alignItems: "center", color: "rgba(255,255,255,0.2)", fontSize: 18, mx: 0.5, flexShrink: 0 }}>→</Box>
                    </motion.div>
                  )}
                </React.Fragment>
              ))}
            </Box>
            {/* Mobile flow */}
            <Box sx={{ display: { xs: "flex", sm: "none" }, flexDirection: "column", gap: 1.5 }}>
              {PIPELINE_STEPS.map((step, i) => (
                <Box key={step.label} sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                  <Box sx={{ width: 36, height: 36, borderRadius: "50%", background: alpha(step.color, 0.15), border: `1.5px solid ${alpha(step.color, 0.4)}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0 }}>
                    {step.icon}
                  </Box>
                  <Box>
                    <Typography variant="caption" sx={{ color: step.color, fontWeight: 700, fontSize: "0.72rem" }}>{step.label.replace("\n", " ")}</Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", display: "block" }}>{step.desc}</Typography>
                  </Box>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </motion.div>

      {/* ══ LEAKAGE-SAFE SPLIT VISUALIZATION ═══════════════════════ */}
      <motion.div variants={fadeUp} initial="initial" animate="animate">
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
          <LockRounded sx={{ color: "#a78bfa", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Leakage-Safe Evaluation Design</Typography>
          <Chip label="SCIENTIFICALLY CRITICAL" size="small" sx={{ fontSize: "0.58rem", height: 20, fontWeight: 700, backgroundColor: alpha("#a78bfa", 0.12), color: "#a78bfa", border: "1px solid rgba(139,92,246,0.3)", letterSpacing: "0.05em" }} />
        </Box>
      </motion.div>

      <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ marginBottom: 28 }}>
        <Card sx={{ background: "rgba(15,25,35,0.85)", border: "1px solid rgba(139,92,246,0.2)" }}>
          <CardContent sx={{ p: 2.5 }}>
            <Grid container spacing={3}>
              {/* Left: split diagram */}
              <Grid item xs={12} md={7}>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.68rem", display: "block", mb: 2 }}>
                  CV curves are grouped by (material × scan_rate). Splits are made at the curve level — no single curve appears in two partitions.
                </Typography>
                {/* Visual partition diagram */}
                {[
                  { label: "🔵 TRAIN",    desc: "NM1, NM2, NM3  ×  SR = 10,20,40,60,70,80,90,100 mV/s", detail: "24 CV curves · 62,400 points · in-sample fitting", color: "#6366f1", pct: 60 },
                  { label: "🟢 VALIDATION", desc: "NM1, NM2, NM3  ×  SR = 30 mV/s", detail: "3 CV curves · 7,800 points · scan-rate interpolation", color: "#10b981", pct: 8 },
                  { label: "🟡 TEST-SR",  desc: "NM1, NM2, NM3  ×  SR = 50 mV/s", detail: "3 CV curves · 7,800 points · scan-rate interpolation (SR=50 between 40 & 60)", color: "#f59e0b", pct: 8 },
                  { label: "🔴 TEST-MAT", desc: "NM4 (ZnO/Co₃O₄)  ×  all 10 scan rates", detail: "10 CV curves · 26,000 points · material extrapolation", color: "#f472b6", pct: 25 },
                ].map((split, i) => (
                  <motion.div key={split.label} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1, duration: 0.35 }}>
                    <Box sx={{ mb: 1.75 }}>
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", mb: 0.5 }}>
                        <Typography variant="caption" sx={{ color: split.color, fontWeight: 700, fontSize: "0.72rem" }}>{split.label}</Typography>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.62rem" }}>{split.detail}</Typography>
                      </Box>
                      <Box sx={{ height: 20, borderRadius: 1.5, background: "rgba(255,255,255,0.04)", overflow: "hidden", position: "relative", border: `1px solid ${alpha(split.color, 0.15)}` }}>
                        <motion.div initial={{ width: 0 }} animate={{ width: `${split.pct}%` }} transition={{ delay: i * 0.1 + 0.3, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                          style={{ height: "100%", background: `linear-gradient(90deg, ${split.color}55 0%, ${split.color}22 100%)`, display: "flex", alignItems: "center", paddingLeft: 8 }}>
                          <Typography variant="caption" sx={{ color: split.color, fontSize: "0.62rem", fontFamily: "JetBrains Mono", fontWeight: 700, whiteSpace: "nowrap" }}>
                            {split.desc}
                          </Typography>
                        </motion.div>
                      </Box>
                    </Box>
                  </motion.div>
                ))}
              </Grid>

              {/* Right: explanation */}
              <Grid item xs={12} md={5}>
                <Box sx={{ p: 2, borderRadius: 2, background: alpha("#a78bfa", 0.05), border: "1px solid rgba(139,92,246,0.15)", mb: 2 }}>
                  <Typography variant="caption" sx={{ color: "#a78bfa", fontWeight: 700, fontSize: "0.7rem", display: "block", mb: 1 }}>⚠ Why leakage-safe splitting matters</Typography>
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>
                    Naively splitting 651-point CV sweeps at the <em>point level</em> would allow points from the same experimental curve to appear in both train and test sets. This creates data leakage — the model sees partial information about the curve it's being tested on, producing artificially inflated R² scores.
                  </Typography>
                </Box>
                <Box sx={{ p: 2, borderRadius: 2, background: alpha("#10b981", 0.05), border: "1px solid rgba(16,185,129,0.15)", mb: 2 }}>
                  <Typography variant="caption" sx={{ color: "#10b981", fontWeight: 700, fontSize: "0.7rem", display: "block", mb: 1 }}>✓ Our approach: Curve-level grouping</Typography>
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>
                    Each complete CV curve (651 points at a specific material+SR combination) is assigned to exactly one partition. This ensures the model cannot "remember" any part of a test curve, providing honest estimates of generalisation ability.
                  </Typography>
                </Box>
                <Box sx={{ p: 2, borderRadius: 2, background: alpha("#f472b6", 0.05), border: "1px solid rgba(244,114,182,0.15)" }}>
                  <Typography variant="caption" sx={{ color: "#f472b6", fontWeight: 700, fontSize: "0.7rem", display: "block", mb: 1 }}>🔬 Test-MAT: Zero-shot material extrapolation</Typography>
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>
                    NM4 (ZnO/Co₃O₄) was never seen during training. Test-MAT R² measures whether models truly learn CV physics — not just material-specific memorisation. GRU achieves R²=0.9751 on this zero-shot test.
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </motion.div>

      {/* ── Dataset charts ───────────────────────────────────────────── */}
      <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ marginBottom: 28 }}>
        <Grid container spacing={3}>
          {/* Scan rate distribution */}
          <Grid item xs={12} md={6}>
            <Card sx={{ background: "rgba(15,25,35,0.85)", border: "1px solid rgba(16,185,129,0.15)" }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="subtitle2" sx={{ color: "#10b981", fontWeight: 700, mb: 1.5 }}>SCAN RATE DISTRIBUTION ACROSS PARTITIONS</Typography>
                <Plot
                  data={[
                    { type: "bar", name: "Train",    x: [10, 20, 40, 60, 70, 80, 90, 100], y: [3, 3, 3, 3, 3, 3, 3, 3], marker: { color: "#6366f1", opacity: 0.85 } },
                    { type: "bar", name: "Val",      x: [30],        y: [3],       marker: { color: "#10b981", opacity: 0.85 } },
                    { type: "bar", name: "Test-SR",  x: [50],        y: [3],       marker: { color: "#f59e0b", opacity: 0.85 } },
                    { type: "bar", name: "Test-MAT", x: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], y: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], marker: { color: "#f472b6", opacity: 0.85 } },
                  ]}
                  layout={{
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(13,19,33,0.4)",
                    height: 230, margin: { l: 50, r: 20, t: 10, b: 50 }, barmode: "stack",
                    font: { family: "Inter", color: "#8892a4", size: 10 },
                    xaxis: { title: { text: "Scan Rate (mV/s)", font: { size: 11, color: "#94a3b8" } }, tickvals: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], gridcolor: "rgba(255,255,255,0.03)", tickfont: { size: 10, family: "JetBrains Mono" } },
                    yaxis: { title: { text: "# Curves", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.03)", tickfont: { size: 10 } },
                    legend: { orientation: "h", x: 0, y: 1.05, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
                    hoverlabel: { bgcolor: "#111c2d", bordercolor: "#10b981", font: { size: 11, color: "#f1f5f9", family: "JetBrains Mono" } },
                  }}
                  config={{ responsive: true, displayModeBar: false }} style={{ width: "100%" }} useResizeHandler
                />
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.62rem" }}>
                  SR=30 and SR=50 are held-out interpolation test rates (both fall within the training SR range). SR=100 appears in both Training (NM1–NM3) and Test-MAT (NM4).
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Material distribution donut */}
          <Grid item xs={12} md={6}>
            <Card sx={{ background: "rgba(15,25,35,0.85)", border: "1px solid rgba(139,92,246,0.15)" }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 1.5 }}>MATERIAL × PARTITION ALLOCATION</Typography>
                <Plot
                  data={[{
                    type: "pie",
                    labels: ["Train NM1", "Train NM2", "Train NM3", "Val NM1–NM3", "Test-SR NM1–NM3", "Test-MAT NM4"],
                    values: [8, 8, 8, 3, 3, 10],
                    hole: 0.55,
                    marker: { colors: ["#3b82f6", "#22d3ee", "#a78bfa", "#10b981", "#f59e0b", "#f472b6"], line: { color: "rgba(0,0,0,0.3)", width: 1.5 } },
                    textfont: { size: 9, color: "#f1f5f9" },
                    hovertemplate: "<b>%{label}</b><br>%{value} curves<br>%{percent}<extra></extra>",
                    textposition: "inside",
                  }]}
                  layout={{
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                    height: 230, margin: { l: 10, r: 10, t: 10, b: 10 },
                    font: { family: "Inter", color: "#8892a4", size: 10 },
                    legend: { orientation: "v", x: 1.02, y: 0.5, bgcolor: "rgba(0,0,0,0)", font: { size: 9, color: "#c8cfd8" } },
                    annotations: [{ text: "40<br>curves", x: 0.5, y: 0.5, font: { size: 13, color: "#f1f5f9", family: "JetBrains Mono" }, showarrow: false }],
                    hoverlabel: { bgcolor: "#111c2d", bordercolor: "#a78bfa", font: { size: 11, color: "#f1f5f9", family: "JetBrains Mono" } },
                  }}
                  config={{ responsive: true, displayModeBar: false }} style={{ width: "100%", height: 230 }} useResizeHandler
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </motion.div>

      {/* ── Dataset summary ──────────────────────────────────────────── */}
      <motion.div variants={cardVariants} initial="initial" animate="animate">
        <Card sx={{ mb: 3.5, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(16,185,129,0.15)" }}>
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}>
              <DatasetRounded sx={{ color: "#10b981", fontSize: 20 }} />
              <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Dataset Summary</Typography>
            </Box>
            <Grid container spacing={3}>
              {[
                { label: "Materials", value: "4", desc: "NM1 (ZnO), NM2 (ZnO/rGO), NM3 (ZnO/MnO₂), NM4 (ZnO/Co₃O₄)", color: "#22d3ee" },
                { label: "Scan Rates", value: "10", desc: "10, 20, 30, 40, 50, 60, 70, 80, 90, 100 mV/s", color: "#10b981" },
                { label: "Points per Curve", value: "651", desc: "−0.65V → 0V → −0.65V, Δ=0.002V", color: "#a78bfa" },
                { label: "Total CV Curves", value: "40", desc: "4 materials × 10 scan rates", color: "#fb923c" },
                { label: "Feature Dimensions", value: "10", desc: "8 engineered + sweep_direction + sweep_position", color: "#f472b6" },
                { label: "Sequence Length", value: "651", desc: "Time steps for LSTM/GRU sequence models", color: "#facc15" },
              ].map((item) => (
                <Grid item xs={6} sm={4} key={item.label}>
                  <Box sx={{ p: 2, borderRadius: 2, backgroundColor: alpha(item.color, 0.06), border: `1px solid ${alpha(item.color, 0.18)}` }}>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.65rem", display: "block", mb: 0.5 }}>{item.label}</Typography>
                    <Typography variant="h5" sx={{ fontWeight: 800, color: item.color, fontFamily: "JetBrains Mono", fontSize: "1.6rem", mb: 0.5 }}>{item.value}</Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem" }}>{item.desc}</Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      </motion.div>

      {/* ── Methodology steps ────────────────────────────────────────── */}
      <motion.div variants={fadeUp} initial="initial" animate="animate">
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2.5 }}>
          <BuildRounded sx={{ color: "#00d4ff", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Research Methodology</Typography>
        </Box>
      </motion.div>

      <motion.div variants={staggerContainer} initial="initial" animate="animate" style={{ marginBottom: 28 }}>
        <Grid container spacing={2.5}>
          {METHODOLOGY_STEPS.map((step, i) => (
            <Grid item xs={12} md={6} key={step.title}>
              <motion.div variants={cardVariants}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: `1px solid ${alpha(step.color, 0.2)}`, height: "100%", "&:hover": { borderColor: alpha(step.color, 0.4) }, transition: "border-color 0.2s ease" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1.5 }}>
                      <Box sx={{ fontSize: 22 }}>{step.icon}</Box>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: step.color, fontSize: "0.88rem" }}>{step.title}</Typography>
                      <Chip label={`Step ${i + 1}`} size="small" sx={{ ml: "auto", height: 18, fontSize: "0.58rem", fontWeight: 700, backgroundColor: alpha(step.color, 0.1), color: alpha(step.color, 0.8), border: `1px solid ${alpha(step.color, 0.2)}`, "& .MuiChip-label": { px: 0.6 } }} />
                    </Box>
                    <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.55)", lineHeight: 1.65, fontSize: "0.8rem" }}>{step.content}</Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </motion.div>

      {/* ── Key findings ─────────────────────────────────────────────── */}
      <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ marginBottom: 28 }}>
        <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.1)" }}>
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}>
              <InsightsRounded sx={{ color: "#00d4ff", fontSize: 20 }} />
              <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Scientific Findings</Typography>
            </Box>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
              {FINDINGS.map((f) => (
                <Box key={f.text} sx={{ display: "flex", alignItems: "flex-start", gap: 1.5 }}>
                  <CheckCircleRounded sx={{ color: f.color, fontSize: 18, mt: 0.2, flexShrink: 0 }} />
                  <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.6)", lineHeight: 1.6, fontSize: "0.85rem" }}>{f.text}</Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </motion.div>

      {/* ══ RESEARCH LIMITATIONS & HONEST SCOPE ═════════════════════ */}
      <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ marginBottom: 16 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
          <WarningAmberRounded sx={{ color: "#f59e0b", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Research Limitations & Honest Scope</Typography>
          <Chip label="SCIENTIFIC TRANSPARENCY" size="small"
            sx={{ fontSize: "0.58rem", height: 20, fontWeight: 700, backgroundColor: alpha("#f59e0b", 0.1), color: "#f59e0b", border: "1px solid rgba(245,158,11,0.3)", letterSpacing: "0.04em" }} />
        </Box>
        <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.35)", mb: 2.5, fontSize: "0.8rem", lineHeight: 1.7 }}>
          We present these limitations transparently so results can be interpreted correctly. Strong ML metrics do not imply unlimited applicability — context matters in materials science.
        </Typography>
      </motion.div>

      <motion.div variants={staggerContainer} initial="initial" animate="animate" style={{ marginBottom: 32 }}>
        <Grid container spacing={2.5}>
          {[
            {
              icon: "📦",
              title: "Small Training Dataset",
              color: "#f59e0b",
              severity: "Moderate",
              text: "The full dataset contains 40 CV curves across 4 materials (24 training, 3 validation, 3 test-SR, 10 test-MAT). This is standard for experimental supercapacitor research but limits generalisation to exotic or complex multi-component ZnO composites not represented in the training distribution.",
            },
            {
              icon: "⚗️",
              title: "Single Electrolyte System",
              color: "#fb923c",
              severity: "High",
              text: "All experiments used KOH (potassium hydroxide) aqueous electrolyte. The trained models cannot reliably predict CV behaviour in LiOH, Na₂SO₄, organic, or ionic liquid electrolytes, where electrode-electrolyte interaction fundamentally differs.",
            },
            {
              icon: "📐",
              title: "Electrode Geometry Not Normalised",
              color: "#f472b6",
              severity: "Moderate",
              text: "Models predict current (µA) for the specific electrode dimensions used experimentally. Absolute capacitance per unit area (mF/cm²) or per unit mass (F/g) cannot be directly extracted — only relative trends between materials and scan rates.",
            },
            {
              icon: "⏱️",
              title: "Scan Rate Range 10–100 mV/s",
              color: "#a78bfa",
              severity: "Low-Moderate",
              text: "Training covers scan rates from 10 to 100 mV/s. Predictions at very low rates (< 5 mV/s — true quasi-equilibrium capacitance) or very high rates (> 200 mV/s — power-optimised fast-charge) are extrapolation and should be treated with caution.",
            },
            {
              icon: "🔄",
              title: "No Cycling Degradation Modelling",
              color: "#22d3ee",
              severity: "High",
              text: "Models predict the CV response of fresh electrodes at a fixed state-of-health. Degradation over charge-discharge cycles, surface passivation, electrolyte depletion, and ageing effects are not captured — predictions represent T=0 (pristine electrode) behaviour only.",
            },
            {
              icon: "📉",
              title: "CV Prediction Only",
              color: "#3b82f6",
              severity: "Informational",
              text: "The platform models cyclic voltammetry exclusively. Electrochemical Impedance Spectroscopy (EIS), galvanostatic charge-discharge (GCD), rate capability curves, and long-term stability measurements are outside the current model scope.",
            },
          ].map((lim, i) => (
            <Grid item xs={12} sm={6} md={4} key={lim.title}>
              <motion.div variants={cardVariants}>
                <Card sx={{ height: "100%", background: `linear-gradient(135deg, ${alpha(lim.color, 0.06)} 0%, rgba(15,25,35,0.9) 100%)`, border: `1px solid ${alpha(lim.color, 0.2)}`, borderLeft: `4px solid ${lim.color}` }}>
                  <CardContent sx={{ p: 2.25 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.25 }}>
                      <Typography sx={{ fontSize: 18 }}>{lim.icon}</Typography>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="subtitle2" sx={{ color: lim.color, fontWeight: 700, fontSize: "0.8rem", lineHeight: 1.3 }}>{lim.title}</Typography>
                      </Box>
                      <Chip label={lim.severity} size="small"
                        sx={{ fontSize: "0.55rem", height: 16, backgroundColor: alpha(lim.color, 0.1), color: lim.color, border: `1px solid ${alpha(lim.color, 0.25)}` }} />
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>
                      {lim.text}
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </motion.div>

      {/* ══ REAL-WORLD APPLICATIONS & IMPACT ════════════════════════ */}
      <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ marginBottom: 16 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
          <RocketLaunchRounded sx={{ color: "#10b981", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Real-World Applications & Impact</Typography>
        </Box>
        <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.35)", mb: 2.5, fontSize: "0.8rem", lineHeight: 1.7 }}>
          AI-driven CV prediction has direct applications across energy storage industries, reducing experiment time from days to milliseconds.
        </Typography>
      </motion.div>

      <motion.div variants={staggerContainer} initial="initial" animate="animate">
        <Grid container spacing={2.5}>
          {[
            {
              Icon: ElectricCarRounded,
              title: "Electric Vehicle Supercapacitors",
              color: "#22d3ee",
              text: "Hybrid EVs use supercapacitors for regenerative braking and peak power delivery. AI-predicted CVs enable rapid screening of electrode materials for high-power density applications without manufacturing physical test cells.",
              impact: "10–100× faster screening",
            },
            {
              Icon: DevicesRounded,
              title: "Wearable Energy Storage",
              color: "#a78bfa",
              text: "Flexible ZnO-based supercapacitors power IoT sensors and health monitors. LightGBM's 1.7 MB footprint enables on-device CV prediction for self-monitoring energy systems with fast inference.",
              impact: "Edge deployment ready",
            },
            {
              Icon: BiotechRounded,
              title: "New Material Discovery",
              color: "#f472b6",
              text: "GRU's cross-material transfer to NM4 (R²=0.9751) demonstrates that recurrent models capture fundamental CV topology from ZnO composites. This enables model-assisted pre-screening of novel dopants and heterostructures before physical synthesis.",
              impact: "Zero-shot new materials",
            },
            {
              Icon: FactoryRounded,
              title: "Lab Automation & High-Throughput Screening",
              color: "#f59e0b",
              text: "Automated electrode libraries can use predicted CVs for first-pass quality control and performance estimation. Only the top candidates proceed to physical electrochemical characterisation, dramatically reducing lab time and reagent costs.",
              impact: "Lab cost reduction",
            },
            {
              Icon: AutoGraphRounded,
              title: "Electrochemical Process Monitoring",
              color: "#10b981",
              text: "Comparing predicted vs. measured CVs during manufacturing provides real-time quality metrics. Deviation from predicted morphology signals synthesis defects, degradation, or electrolyte contamination in production environments.",
              impact: "Real-time QC",
            },
            {
              Icon: MenuBookRounded,
              title: "Academic Research Support",
              color: "#3b82f6",
              text: "AI-predicted CVs can supplement experimental data in publications, fill missing scan rates in datasets, and provide theoretical reference curves for comparison studies. The platform supports reproducibility by making predictions openly available.",
              impact: "Research acceleration",
            },
          ].map(({ Icon, title, color, text, impact }, i) => (
            <Grid item xs={12} sm={6} md={4} key={title}>
              <motion.div variants={cardVariants}>
                <Card sx={{ height: "100%", background: `linear-gradient(135deg, ${alpha(color, 0.07)} 0%, rgba(15,25,35,0.9) 100%)`, border: `1px solid ${alpha(color, 0.2)}`, borderTop: `3px solid ${color}`, "&:hover": { borderColor: alpha(color, 0.4) }, transition: "border-color 0.2s" }}>
                  <CardContent sx={{ p: 2.25 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.25 }}>
                      <Box sx={{ width: 36, height: 36, borderRadius: "10px", background: alpha(color, 0.12), border: `1px solid ${alpha(color, 0.25)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <Icon sx={{ fontSize: 18, color }} />
                      </Box>
                      <Typography variant="subtitle2" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.82rem", lineHeight: 1.3 }}>{title}</Typography>
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.7rem", lineHeight: 1.65, display: "block", mb: 1.25 }}>
                      {text}
                    </Typography>
                    <Chip label={impact} size="small"
                      sx={{ fontSize: "0.6rem", height: 18, backgroundColor: alpha(color, 0.1), color, border: `1px solid ${alpha(color, 0.25)}` }} />
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </motion.div>

      {/* ══ ELECTROCHEMISTRY GLOSSARY ════════════════════════════════ */}
      <Box sx={{ mt: 5 }}>
        <motion.div variants={cardVariants} initial="initial" animate="animate">
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}>
            <MenuBookRounded sx={{ color: "#22d3ee", fontSize: 22 }} />
            <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Electrochemistry Glossary</Typography>
            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.72rem", ml: 1 }}>
              Hover any underlined term throughout the platform for instant definitions
            </Typography>
          </Box>
        </motion.div>

        <motion.div variants={staggerContainer} initial="initial" animate="animate">
          <Grid container spacing={2}>
            {[
              { key: "cv", color: "#22d3ee" },
              { key: "scan_rate", color: "#10b981" },
              { key: "anodic", color: "#10b981" },
              { key: "cathodic", color: "#f472b6" },
              { key: "edlc", color: "#a78bfa" },
              { key: "faradaic", color: "#f59e0b" },
              { key: "symmetry_factor", color: "#22d3ee" },
              { key: "capacitance", color: "#fb923c" },
              { key: "charge_storage_index", color: "#00d4ff" },
              { key: "rmse", color: "#3b82f6" },
              { key: "r2", color: "#10b981" },
              { key: "zero_shot", color: "#f472b6" },
              { key: "integral_area", color: "#a78bfa" },
              { key: "peak_separation", color: "#22d3ee" },
              { key: "nanocomposite", color: "#f59e0b" },
            ].map(({ key, color }) => {
              const entry = GLOSSARY[key];
              if (!entry) return null;
              return (
                <Grid item xs={12} sm={6} md={4} key={key}>
                  <motion.div variants={cardVariants}>
                    <Box sx={{ p: 1.75, borderRadius: 2, background: alpha(color, 0.05), border: `1px solid ${alpha(color, 0.18)}`, height: "100%" }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.75 }}>
                        <Box sx={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: color, flexShrink: 0 }} />
                        <Typography variant="caption" sx={{ color, fontWeight: 700, fontSize: "0.72rem" }}>
                          <GlossaryTooltip termKey={key}>{entry.term}</GlossaryTooltip>
                        </Typography>
                      </Box>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.67rem", lineHeight: 1.6, display: "block" }}>
                        {entry.short}
                      </Typography>
                    </Box>
                  </motion.div>
                </Grid>
              );
            })}
          </Grid>
        </motion.div>
      </Box>
    </Box>
  );
}
