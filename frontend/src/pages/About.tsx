import React, { useState, useRef } from "react";
import {
  Box, Grid, Card, CardContent, Typography, Chip, alpha,
  Accordion, AccordionSummary, AccordionDetails,
  TextField, InputAdornment, Paper, ClickAwayListener,
} from "@mui/material";
import {
  ExpandMoreRounded, SearchRounded, SchoolRounded,
  HomeRounded, LightbulbRounded, ScienceRounded, FlashOnRounded,
  ShowChartRounded, TouchAppRounded, FunctionsRounded, DatasetRounded,
  TuneRounded, PsychologyRounded, AccountTreeRounded, RouteRounded,
  WarningAmberRounded, RocketLaunchRounded, ExploreRounded, MenuBookRounded,
  CheckCircleRounded, LockRounded, InsightsRounded, BuildRounded,
  TimelineRounded, ElectricCarRounded, DevicesRounded, BiotechRounded,
  AutoGraphRounded, FactoryRounded,
} from "@mui/icons-material";
import { motion } from "framer-motion";
import Plot from "react-plotly.js";
import SectionHeader from "../components/common/SectionHeader";
import GlossaryTooltip from "../components/common/GlossaryTooltip";
import { GLOSSARY } from "../constants/glossary";
import { cardVariants, staggerContainer } from "../animations/variants";

// Subtle note box for contextual callouts
function NoteBox({ children }: { children: React.ReactNode }) {
  return (
    <Box sx={{ mt: 1.5, p: 2, borderRadius: 2, background: "rgba(167,139,250,0.05)", border: "1px solid rgba(167,139,250,0.15)", borderLeft: "3px solid rgba(167,139,250,0.5)" }}>
      {children}
    </Box>
  );
}

function FormulaBox({ label, formula, desc }: { label: string; formula: string; desc: string }) {
  return (
    <Box sx={{ p: 2, borderRadius: 2, background: "rgba(244,114,182,0.05)", border: "1px solid rgba(244,114,182,0.2)", mb: 1.5 }}>
      <Typography variant="caption" sx={{ color: "#f472b6", fontWeight: 700, fontSize: "0.68rem", display: "block", mb: 0.75 }}>{label}</Typography>
      <Typography sx={{ fontFamily: "JetBrains Mono", color: "#f1f5f9", fontSize: "0.85rem", mb: 1, display: "block" }}>{formula}</Typography>
      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.7rem", lineHeight: 1.6, display: "block" }}>{desc}</Typography>
    </Box>
  );
}

// ── Synthetic CV demo data ────────────────────────────────────────────────────
const DEMO_CV = (() => {
  const fwdV: number[] = [], fwdI: number[] = [];
  const rvV: number[] = [], rvI: number[] = [];
  for (let i = 0; i <= 325; i++) {
    const v = -0.65 + i * 0.002;
    fwdV.push(+v.toFixed(4));
    const base = 80 + 50 * ((v + 0.65) / 0.65);
    fwdI.push(+(base + 38 * Math.exp(-((v + 0.35) ** 2) / (2 * 0.055 ** 2))).toFixed(1));
  }
  for (let i = 1; i <= 325; i++) {
    const v = -(i * 0.002);
    rvV.push(+v.toFixed(4));
    const base = -(80 + 50 * ((v + 0.65) / 0.65));
    rvI.push(+(base - 38 * Math.exp(-((v + 0.28) ** 2) / (2 * 0.055 ** 2))).toFixed(1));
  }
  return { fwdV, fwdI, rvV, rvI };
})();

// ── Existing data ─────────────────────────────────────────────────────────────
const METHODOLOGY_STEPS = [
  { icon: "🧪", title: "Electrode Sample Preparation", color: "#22d3ee", content: "Four ZnO-based electrode material groups (NM1–NM4) were used in the experimental study. NM1 is the ZnO baseline electrode. NM2, NM3, and NM4 are additional experimental samples. Cyclic voltammetry experiments were conducted at ten scan rates (10, 20, 30, 40, 50, 60, 70, 80, 90, 100 mV/s)." },
  { icon: "📊", title: "Dataset Construction", color: "#10b981", content: "CV data was digitised into 651-point voltage sweeps (−0.65V → 0V → −0.65V, step 0.002V). Each training point has 10 engineered features: potential_V, scan_rate_mVs, log_scan_rate, sqrt_scan_rate, potential_from_lower, potential_from_upper, sr×potential, direction×potential, sweep_direction, and sweep_position." },
  { icon: "🔒", title: "Leakage-Safe Data Splitting", color: "#a78bfa", content: "Critical design decision: splits performed at CV-curve level (not point level) to prevent data leakage. Train: NM1–NM3 at SR=10,20,40,60,70,80,90,100 mV/s. Validation: NM1–NM3 at SR=30 mV/s (interpolation). Test-SR: NM1–NM3 at SR=50 mV/s. Test-MAT: NM4 all rates (material extrapolation)." },
  { icon: "⚙️", title: "Feature Normalisation", color: "#fb923c", content: "Per-group MinMax normalisation: scalers fitted on each (material_id, scan_rate) combination independently, then stored in scalers.json. Current values denormalised using the same group key at inference. This design avoids cross-group leakage between materials and scan rates." },
  { icon: "🤖", title: "Model Training & Tuning", color: "#f472b6", content: "Six ML architectures: Random Forest (300 trees), LightGBM (leaf-wise GBDT), XGBoost (regularised boosting), Dense ANN (4-layer MLP), Stacked LSTM (2× LSTM layers), Stacked GRU (2× GRU layers). Deep learning models used sequences of length 651 with 10 features per timestep." },
  { icon: "📈", title: "Evaluation Protocol", color: "#facc15", content: "Primary metrics: RMSE (µA) and R². Four evaluation partitions allow measuring in-sample fit, scan-rate interpolation at SR=30 (Val), scan-rate interpolation at SR=50 (Test-SR), and material extrapolation to unseen NM4 (Test-MAT) — providing a complete multi-dimensional picture of generalisation capability." },
];
const FINDINGS = [
  { text: "RF achieves lowest validation RMSE (26.59 µA) — best accuracy overall", color: "#3b82f6" },
  { text: "LightGBM is ~360× smaller than RF with only 1% higher RMSE — clear deployment winner", color: "#22d3ee" },
  { text: "GRU achieves highest test-MAT R² (0.9751) — best generalisation to unseen NM4 sample", color: "#a78bfa" },
  { text: "All 6 models achieve R² > 0.96 on test-MAT, validating the approach for ZnO composites", color: "#10b981" },
  { text: "Leakage-safe splitting prevents overly optimistic metrics — results are reliable", color: "#f59e0b" },
  { text: "Scan rate features (log, sqrt, sr×potential) are the most important predictors across all tree models", color: "#fb923c" },
];
const PIPELINE_STEPS = [
  { label: "Material\nSynthesis", color: "#22d3ee", icon: "🧪", desc: "ZnO-based electrode samples NM1–NM4" },
  { label: "CV\nAcquisition", color: "#10b981", icon: "📡", desc: "10 scan rates × 4 materials" },
  { label: "Feature\nEngineering", color: "#a78bfa", icon: "⚙️", desc: "10 electrochemical features" },
  { label: "Leakage-Safe\nSplitting", color: "#f59e0b", icon: "🔒", desc: "Curve-level, not point-level" },
  { label: "Model\nTraining", color: "#f472b6", icon: "🤖", desc: "6 architectures, 4 partitions" },
  { label: "Evaluation\n& Deploy", color: "#fb923c", icon: "📈", desc: "RMSE + R² across all splits" },
];

// ── Extra glossary terms ──────────────────────────────────────────────────────
const EXTRA_TERMS = [
  { term: "Electrode", short: "Electrical conductor immersed in electrolyte to exchange electrons with chemical species.", color: "#22d3ee" },
  { term: "Electrolyte", short: "Ionic conductor (1 M Na₂SO₄ here) that carries charge between electrodes without electron transfer.", color: "#10b981" },
  { term: "Redox Reaction", short: "Combined oxidation–reduction: one species loses electrons while another gains them simultaneously.", color: "#a78bfa" },
  { term: "Na₂SO₄", short: "Sodium sulfate — the aqueous electrolyte (1 M) used in all experiments, as specified in the original study.", color: "#f472b6" },
  { term: "Working Electrode", short: "Electrode under study in a 3-electrode cell — the ZnO-coated Nickel sheet.", color: "#3b82f6" },
  { term: "Reference Electrode", short: "Stable-potential electrode (SCE — Saturated Calomel Electrode) used as voltage reference in this study.", color: "#f59e0b" },
  { term: "Counter Electrode", short: "Graphite electrode completing the circuit. Current flows through it; its chemistry is not analysed.", color: "#fb923c" },
  { term: "Electric Double Layer", short: "Two parallel charge sheets at electrode–electrolyte interface — the physical basis of EDLC capacitance.", color: "#22d3ee" },
  { term: "Energy Density", short: "Energy per unit mass (Wh/kg). Higher = longer runtime. Batteries beat supercapacitors here.", color: "#10b981" },
  { term: "Power Density", short: "Power per unit mass (W/kg). Higher = faster charge/discharge. Supercapacitors outperform batteries.", color: "#a78bfa" },
  { term: "rGO", short: "Reduced Graphene Oxide — a high-surface-area carbon nanomaterial used in composite supercapacitor electrodes to enhance electric double-layer capacitance.", color: "#f472b6" },
  { term: "MnO₂", short: "Manganese dioxide — a pseudocapacitive metal oxide that stores charge via Mn²⁺↔Mn⁴⁺ surface redox reactions. Common composite additive in supercapacitor research.", color: "#3b82f6" },
  { term: "Co₃O₄", short: "Cobalt oxide — a pseudocapacitive material with Co²⁺↔Co³⁺ redox chemistry. Used in composite electrode research for enhanced capacitance.", color: "#f59e0b" },
  { term: "MinMax Normalisation", short: "Scales features to [0,1]: x_norm = (x − x_min)/(x_max − x_min). Applied per (material, scan_rate) group.", color: "#fb923c" },
  { term: "Overfitting", short: "Model memorises training data and fails on new inputs. Prevented by leakage-safe multi-partition evaluation.", color: "#22d3ee" },
  { term: "Gradient Boosting", short: "Ensemble: sequentially adds decision trees, each correcting the previous tree's residual errors.", color: "#10b981" },
  { term: "Decision Tree", short: "Single model: recursively splits data by feature thresholds. Weak alone; powerful in ensembles.", color: "#a78bfa" },
  { term: "Bagging", short: "Bootstrap Aggregating in Random Forest: each tree trains on a random data subset; predictions averaged.", color: "#f472b6" },
  { term: "Hyperparameter", short: "Model setting not learned from data (e.g., n_estimators=300). Chosen before training.", color: "#3b82f6" },
  { term: "Sequence Model", short: "Neural network processing inputs as ordered time steps (LSTM, GRU) — natural for 651-point CV sweeps.", color: "#f59e0b" },
  { term: "Hidden State", short: "LSTM/GRU internal memory vector passed between time steps, encoding context from earlier sweep points.", color: "#fb923c" },
  { term: "Inference", short: "Running a trained model on new input (prediction). RF: slow; LightGBM: fastest; RNNs: moderate.", color: "#22d3ee" },
  { term: "Overpotential", short: "Extra voltage required beyond equilibrium for a reaction to proceed at a measurable rate.", color: "#10b981" },
  { term: "Diffusion", short: "Ion movement through electrolyte driven by concentration gradient. Dominant at low scan rates.", color: "#a78bfa" },
  { term: "Charge–Discharge Cycle", short: "One complete store-and-release of energy. Supercapacitors survive 10⁵–10⁶ cycles vs ~10³ for batteries.", color: "#f472b6" },
];

// ─────────────────────────────────────────────────────────────────────────────
// Module components
// ─────────────────────────────────────────────────────────────────────────────

function M01_Welcome() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        This platform is an end-to-end AI system that predicts complete Cyclic Voltammetry (CV) curves
        for ZnO-based supercapacitor electrodes. Six machine learning models trained on real experimental
        data can predict 651-point CV profiles at arbitrary scan rates — without running physical experiments.
        All models achieve R² {">"} 0.96 on unseen material extrapolation.
      </Typography>
      <Grid container spacing={2}>
        {[
          { icon: "🔬", title: "Researchers", desc: "Predict CV profiles for new ZnO compositions without physical experiments. Screen new electrode materials in milliseconds." },
          { icon: "🎓", title: "Students", desc: "Learn electrochemistry and ML together through this interactive platform. Use Learning Mode for guided explanations." },
          { icon: "⚡", title: "Engineers", desc: "Pre-screen electrode materials for energy storage applications. LightGBM (1.7 MB) is edge-deployment ready." },
        ].map(({ icon, title, desc }) => (
          <Grid item xs={12} sm={4} key={title}>
            <Box sx={{ p: 2, borderRadius: 2, background: "rgba(167,139,250,0.06)", border: "1px solid rgba(167,139,250,0.15)", height: "100%" }}>
              <Typography sx={{ fontSize: 24, mb: 1 }}>{icon}</Typography>
              <Typography variant="subtitle2" sx={{ color: "#f1f5f9", fontWeight: 700, mb: 0.5 }}>{title}</Typography>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>{desc}</Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
      <Box>
        <Typography variant="caption" sx={{ color: "#a78bfa", fontWeight: 700, fontSize: "0.68rem", letterSpacing: "0.06em", display: "block", mb: 1.5 }}>KEY FEATURES</Typography>
        <Grid container spacing={1.25}>
          {[
            ["Predict CV curves at 10–100 mV/s for 4 ZnO materials", "#22d3ee"],
            ["6 ML models: RF, LightGBM, XGBoost, ANN, LSTM, GRU", "#10b981"],
            ["Real-time model comparison with cosine similarity matrix", "#a78bfa"],
            ["Electrochemical metrics: RMSE, R², CSI, symmetry factor", "#f59e0b"],
            ["Leakage-safe evaluation on 4 dataset partitions", "#f472b6"],
            ["Zero-shot material extrapolation tested on NM4 (unseen experimental sample)", "#3b82f6"],
          ].map(([feat, color]) => (
            <Grid item xs={12} sm={6} key={feat as string}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Box sx={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: color as string, flexShrink: 0 }} />
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.73rem" }}>{feat as string}</Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Box>
      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>New here? Start with these modules in order:</strong> "The Problem" → "Electrochemistry Fundamentals" → "Supercapacitors" → "Cyclic Voltammetry Course" → "Interactive CV Graph" → then the ML modules. Each section builds on the previous one.
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M02_Problem() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        Global demand for clean energy storage is accelerating. Supercapacitors are fast, durable, and safe —
        but discovering new electrode materials remains slow, expensive, and labour-intensive. This platform
        uses AI to collapse weeks of experimentation into milliseconds of computation.
      </Typography>
      <Grid container spacing={2}>
        {[
          { label: "⚡ The Energy Storage Gap", color: "#f59e0b", text: "Batteries store lots of energy but charge slowly and degrade over ~1,000 cycles. Supercapacitors charge instantly and last 100,000+ cycles but store less energy. Hybrid designs using materials like ZnO composites aim to bridge this gap." },
          { label: "🧪 The Experimentation Bottleneck", color: "#f472b6", text: "Testing one electrode material requires: synthesising the nanocomposite (days), measuring CV at 10 scan rates (hours), analysing results (hours). Testing 100 candidate materials takes months in the lab." },
          { label: "🤖 The AI Solution", color: "#10b981", text: "Train ML models once on experimental data from 4 ZnO electrode samples. Then predict CV behavior instantly. The GRU model correctly extrapolates to the never-seen NM4 sample with R²=0.9751." },
        ].map(({ label, color, text }) => (
          <Grid item xs={12} md={4} key={label}>
            <Box sx={{ p: 2, borderRadius: 2, background: alpha(color, 0.05), border: `1px solid ${alpha(color, 0.2)}`, height: "100%" }}>
              <Typography variant="subtitle2" sx={{ color, fontWeight: 700, fontSize: "0.8rem", mb: 1 }}>{label}</Typography>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.7, display: "block" }}>{text}</Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>Analogy:</strong> Think of a supercapacitor like a water tank with a very wide opening (fast to fill/empty) vs a battery like a narrow-necked bottle (holds more but slow). ZnO nanocomposites are new tank designs. Testing each design physically takes weeks — this AI predicts how they behave before building them.
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M03_Electrochemistry() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        Electrochemistry is the science of electrical-chemical energy conversion. Understanding these fundamentals
        helps you interpret CV curves and what the AI models are predicting.
      </Typography>
      <Grid container spacing={2}>
        {[
          { term: "Ions", color: "#22d3ee", def: "Atoms that have gained or lost electrons, giving them a net electric charge. Na⁺ and SO₄²⁻ ions in 1 M Na₂SO₄ electrolyte carry charge between electrodes." },
          { term: "Electrode", color: "#10b981", def: "Electrically conductive material immersed in electrolyte. Our working electrode is a ZnO-coated Nickel sheet. Electrons enter/leave here." },
          { term: "Electrolyte", color: "#a78bfa", def: "Ion-conducting solution (1 M Na₂SO₄ in this study) between electrodes. Ions move through it but electrons cannot — electrons must flow through the external circuit." },
          { term: "Oxidation", color: "#f59e0b", def: "Loss of electrons from a species. Mn²⁺ → Mn⁴⁺ + 2e⁻ is oxidation. Occurs at the electrode during the anodic (forward) CV sweep." },
          { term: "Reduction", color: "#f472b6", def: "Gain of electrons by a species. Mn⁴⁺ + 2e⁻ → Mn²⁺ is reduction. Occurs during the cathodic (reverse) CV sweep." },
          { term: "Electric Double Layer", color: "#3b82f6", def: "When a charged electrode is immersed in electrolyte, ions of opposite charge accumulate on its surface forming two parallel charge layers. This is where EDLC energy is stored — no chemistry, just electrostatics." },
        ].map(({ term, color, def }) => (
          <Grid item xs={12} sm={6} key={term}>
            <Box sx={{ p: 2, borderRadius: 2, background: alpha(color, 0.05), border: `1px solid ${alpha(color, 0.18)}` }}>
              <Typography variant="subtitle2" sx={{ color, fontWeight: 700, fontSize: "0.8rem", mb: 0.75 }}>{term}</Typography>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.7, display: "block" }}>{def}</Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>Simple analogy:</strong> Think of electrochemistry like a swimming pool party. The pool is the electrolyte. The walls (electrodes) attract people (ions) when charged. "Oxidation" = someone hands their ticket (electron) to the wall. "Reduction" = the wall gives a ticket back. The CV machine records how many tickets move at each moment.
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M04_Supercapacitors() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        Supercapacitors (also called ultracapacitors or electrochemical capacitors) bridge batteries and conventional capacitors.
        They charge/discharge in seconds, survive 100,000+ cycles, and operate across wide temperatures.
      </Typography>
      <Grid container spacing={2}>
        {[
          { t: "EDLC", c: "#22d3ee", d: "Electric Double-Layer Capacitor. Stores charge purely electrostatically — no chemical reactions. CV signature: nearly rectangular. Fast, reversible, and highly stable over cycles." },
          { t: "Pseudocapacitor", c: "#f59e0b", d: "Stores charge via fast, reversible Faradaic (redox) reactions at the electrode surface. CV signature: distinct oxidation/reduction peaks. Higher capacitance than EDLC but less stable." },
          { t: "Hybrid / Battery-type", c: "#f472b6", d: "Combines EDLC and Faradaic mechanisms. ZnO-based experimental samples may exhibit hybrid behaviour — ZnO provides EDLC character while some samples show additional pseudocapacitive contributions based on their CV profiles." },
        ].map(({ t, c, d }) => (
          <Grid item xs={12} sm={4} key={t}>
            <Box sx={{ p: 2, borderRadius: 2, background: alpha(c, 0.06), border: `1px solid ${alpha(c, 0.22)}`, height: "100%" }}>
              <Typography variant="subtitle2" sx={{ color: c, fontWeight: 700, mb: 0.75 }}>{t}</Typography>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.7, display: "block" }}>{d}</Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
      <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, fontSize: "0.8rem", mt: 0.5 }}>THE FOUR EXPERIMENTAL MATERIAL GROUPS IN THIS STUDY</Typography>
      <Grid container spacing={1.5}>
        {[
          { nm: "NM1", mat: "ZnO Baseline", c: "#3b82f6", role: "ZnO baseline electrode sample. EDLC-dominant behaviour. Used as the reference against which other samples are compared. Included in Train, Val, and Test-SR partitions." },
          { nm: "NM2", mat: "Experimental Sample", c: "#22d3ee", role: "NM2 is an experimental material group with a CV profile distinct from NM1. Specific composition is not defined in the source dataset. Included in Train, Val, and Test-SR partitions." },
          { nm: "NM3", mat: "Experimental Sample", c: "#10b981", role: "NM3 is an experimental material group showing CV behaviour different from NM1 and NM2. Specific composition is not defined in the source dataset. Included in Train, Val, and Test-SR partitions." },
          { nm: "NM4", mat: "Experimental Sample (unseen)", c: "#f472b6", role: "NM4 is an experimental material group NEVER seen during training — used exclusively as the zero-shot material extrapolation test (Test-MAT). Specific composition is not defined in the source dataset." },
        ].map(({ nm, mat, c, role }) => (
          <Grid item xs={12} sm={6} key={nm}>
            <Box sx={{ p: 2, borderRadius: 2, background: alpha(c, 0.05), border: `1px solid ${alpha(c, 0.2)}`, display: "flex", gap: 1.5 }}>
              <Chip label={nm} size="small" sx={{ height: 22, fontSize: "0.68rem", fontWeight: 800, backgroundColor: alpha(c, 0.15), color: c, border: `1px solid ${alpha(c, 0.3)}`, fontFamily: "JetBrains Mono", flexShrink: 0 }} />
              <Box>
                <Typography variant="caption" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.72rem", display: "block" }}>{mat}</Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem", lineHeight: 1.65, display: "block" }}>{role}</Typography>
              </Box>
            </Box>
          </Grid>
        ))}
      </Grid>
      {/* Electrode Fabrication — Original Experimental Study */}
      <Box sx={{ mt: 1 }}>
        <Typography variant="subtitle2" sx={{ color: "#22d3ee", fontWeight: 700, fontSize: "0.8rem", mb: 1 }}>ELECTRODE FABRICATION — ORIGINAL EXPERIMENTAL STUDY</Typography>
        <Box sx={{ p: 2, borderRadius: 2, background: "rgba(34,211,238,0.04)", border: "1px solid rgba(34,211,238,0.15)", borderLeft: "3px solid rgba(34,211,238,0.4)", mb: 1.5 }}>
          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.7, display: "block" }}>
            The experimental dataset used in this application originates from ZnO-based electrodes fabricated using the following procedure (as described in the source study). <strong style={{ color: "#f1f5f9" }}>This project did not fabricate any electrodes</strong> — we apply AI/ML to data from the original experiment.
          </Typography>
        </Box>
        <Grid container spacing={1.5}>
          {[
            { step: "1", label: "Electrode Slurry", color: "#22d3ee", text: "90 wt% ZnO (active material — stores charge), 5 wt% Carbon Black (conductive additive — improves electron transport), 5 wt% PVDF — Polyvinylidene Fluoride (binder — adheres the mixture to the current collector). Each component is essential: ZnO stores charge, carbon black conducts electrons, PVDF holds the layer together." },
            { step: "2", label: "Solvent & Mixing", color: "#a78bfa", text: "The dry mixture is dispersed in NMP (1-Methyl-2-Pyrrolidinone) to form a homogeneous slurry. A liquid slurry is used rather than dry powder because it enables a uniform, adherent thin-film coating. Dry powder would not bond to the metal sheet and would crack on drying." },
            { step: "3", label: "Coating & Drying", color: "#10b981", text: "The slurry is coated uniformly onto a 0.1 mm Nickel sheet (current collector). Nickel is used because it is chemically inert in Na₂SO₄ electrolyte and conducts electrons efficiently. The coated sheet is dried overnight at 70°C to evaporate NMP and form a stable, mechanically robust ZnO electrode film." },
            { step: "4", label: "3-Electrode Cell", color: "#f59e0b", text: "The dried electrode is used as the Working Electrode in a 3-electrode electrochemical cell. Reference Electrode: SCE — Saturated Calomel Electrode (stable voltage reference). Counter Electrode: Graphite (completes the circuit without interfering with measurements). Electrolyte: 1 M Na₂SO₄ aqueous solution (neutral pH, sodium sulfate)." },
            { step: "5", label: "CV → Dataset → AI", color: "#f472b6", text: "Cyclic voltammetry is performed at multiple scan rates on the fabricated electrode. The resulting current–voltage curves are digitised into the dataset used to train all six ML models in this application. Every prediction on this platform is grounded in these physical measurements." },
          ].map(({ step, label, color, text }) => (
            <Grid item xs={12} md={6} key={step}>
              <Box sx={{ p: 2, borderRadius: 2, background: alpha(color, 0.04), border: `1px solid ${alpha(color, 0.18)}`, display: "flex", gap: 1.5, height: "100%" }}>
                <Box sx={{ width: 26, height: 26, borderRadius: "50%", background: alpha(color, 0.14), border: `1.5px solid ${alpha(color, 0.38)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, mt: 0.25 }}>
                  <Typography sx={{ color, fontWeight: 800, fontSize: "0.7rem", fontFamily: "JetBrains Mono" }}>{step}</Typography>
                </Box>
                <Box>
                  <Typography variant="caption" sx={{ color, fontWeight: 700, fontSize: "0.72rem", display: "block", mb: 0.5 }}>{label}</Typography>
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.43)", fontSize: "0.68rem", lineHeight: 1.65, display: "block" }}>{text}</Typography>
                </Box>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Box>
      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>Think of it like this:</strong> NM1 is a plain sponge (just soaks up water). NM2, NM3, and NM4 are different sponge varieties — each with a distinct shape and texture determining how much water (charge) they can hold. NM4 is the hardest to predict because the AI never trained on it — but it still achieves R²=0.9751!
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M05_CVCourse() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        Cyclic Voltammetry is the most common technique in electrochemical characterisation. It reveals an electrode's
        charge storage mechanism, peak redox potentials, reversibility, and relative capacitance — all from a single experiment.
      </Typography>
      {[
        { step: "1", title: "The 3-Electrode Cell Setup", color: "#22d3ee", content: "Three electrodes immersed in 1 M Na₂SO₄ electrolyte: Working Electrode (ZnO-coated Nickel sheet — what we study), Reference Electrode (SCE — Saturated Calomel Electrode, stable voltage reference), Counter Electrode (Graphite — completes the circuit). A potentiostat controls the voltage between working and reference electrodes." },
        { step: "2", title: "The Voltage Sweep", color: "#10b981", content: "The potentiostat sweeps the working electrode potential from V_lower (−0.65 V) → V_upper (0 V) → back to V_lower at a fixed scan rate ν (mV/s). This is the anodic sweep (forward) followed by the cathodic sweep (reverse). One complete forward+reverse sweep = one CV cycle." },
        { step: "3", title: "The Resulting I–V Curve", color: "#a78bfa", content: "The potentiostat records the resulting current I (µA) at each voltage. Plotting I vs V gives the CV curve — a closed loop. X-axis = voltage (V), Y-axis = current (µA). Positive I = anodic (oxidation). Negative I = cathodic (reduction)." },
        { step: "4", title: "Reading the Curve", color: "#f59e0b", content: "A rectangular loop ≈ pure EDLC (capacitive, no reactions). Distinct humps/peaks = Faradaic reactions (pseudocapacitive). Larger loop area = higher capacitance. Symmetric loop (|I_anodic| ≈ |I_cathodic|) = reversible behaviour. The anodic peak voltage and cathodic peak voltage reveal what reactions occur." },
        { step: "5", title: "Effect of Scan Rate", color: "#f472b6", content: "Higher scan rate ν → higher currents (I ∝ ν for EDLC, I ∝ √ν for diffusion-controlled). But faster sweeps give less time for ions to diffuse, so effective capacitance decreases at high scan rates. This scan rate dependence is why we train on multiple scan rates (10–100 mV/s)." },
      ].map(({ step, title, color, content }) => (
        <Box key={step} sx={{ display: "flex", gap: 2 }}>
          <Box sx={{ width: 28, height: 28, borderRadius: "50%", background: alpha(color, 0.15), border: `1.5px solid ${alpha(color, 0.4)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, mt: 0.25 }}>
            <Typography sx={{ color, fontWeight: 800, fontSize: "0.72rem", fontFamily: "JetBrains Mono" }}>{step}</Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ color, fontWeight: 700, fontSize: "0.8rem", mb: 0.5 }}>{title}</Typography>
            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.72rem", lineHeight: 1.7, display: "block" }}>{content}</Typography>
          </Box>
        </Box>
      ))}
      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>CV in one sentence:</strong> We slowly push and pull electrons from the electrode while measuring how hard it resists — the pattern of resistance tells us exactly how it stores charge. Open the "Interactive CV Graph" module next to see this visually with annotations.
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M06_InteractiveCV() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        The diagram below is a synthetic CV curve representing a ZnO-based hybrid electrode at 50 mV/s.
        Hover over the curve for details. Key regions are annotated.
      </Typography>
      <Card sx={{ background: "rgba(13,19,33,0.6)", border: "1px solid rgba(167,139,250,0.15)" }}>
        <CardContent sx={{ p: 2 }}>
          <Plot
            data={[
              {
                x: DEMO_CV.fwdV, y: DEMO_CV.fwdI,
                mode: "lines", name: "Anodic sweep →",
                line: { color: "#22d3ee", width: 2.5 },
                hovertemplate: "V = %{x:.3f} V<br>I = %{y:.1f} µA<br><i>Anodic sweep</i><extra></extra>",
              },
              {
                x: DEMO_CV.rvV, y: DEMO_CV.rvI,
                mode: "lines", name: "← Cathodic sweep",
                line: { color: "#f472b6", width: 2.5 },
                hovertemplate: "V = %{x:.3f} V<br>I = %{y:.1f} µA<br><i>Cathodic sweep</i><extra></extra>",
              },
            ]}
            layout={{
              paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(13,19,33,0.4)",
              height: 320, margin: { l: 55, r: 20, t: 50, b: 50 },
              font: { family: "Inter", color: "#8892a4", size: 10 },
              xaxis: { title: { text: "Potential (V vs Ref)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, tickfont: { size: 10, family: "JetBrains Mono" } },
              yaxis: { title: { text: "Current (µA)", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.04)", zeroline: true, zerolinecolor: "rgba(255,255,255,0.1)", zerolinewidth: 1, tickfont: { size: 10 } },
              legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)", font: { size: 10, color: "#c8cfd8" } },
              hoverlabel: { bgcolor: "#111c2d", bordercolor: "#a78bfa", font: { size: 11, color: "#f1f5f9", family: "JetBrains Mono" } },
              annotations: [
                { x: -0.35, y: 141, text: "Anodic Peak<br>(Oxidation, ~141 µA)", showarrow: true, arrowhead: 2, arrowcolor: "#22d3ee", ax: 55, ay: -35, font: { size: 9, color: "#22d3ee" }, bgcolor: "rgba(13,19,33,0.85)", bordercolor: "#22d3ee", borderwidth: 1 },
                { x: -0.28, y: -147, text: "Cathodic Peak<br>(Reduction, ~−147 µA)", showarrow: true, arrowhead: 2, arrowcolor: "#f472b6", ax: 55, ay: 35, font: { size: 9, color: "#f472b6" }, bgcolor: "rgba(13,19,33,0.85)", bordercolor: "#f472b6", borderwidth: 1 },
                { x: -0.60, y: 88, text: "EDLC region<br>(capacitive background)", showarrow: false, font: { size: 9, color: "rgba(255,255,255,0.4)" } },
                { x: -0.05, y: 125, text: "Scan rate 50 mV/s<br>(synthetic illustration)", showarrow: false, font: { size: 9, color: "rgba(255,255,255,0.3)" }, xanchor: "right" },
              ],
              shapes: [
                { type: "line", x0: -0.65, y0: 0, x1: 0, y1: 0, line: { color: "rgba(255,255,255,0.08)", width: 1, dash: "dot" } },
              ],
            } as object}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: "100%" }}
            useResizeHandler
          />
        </CardContent>
      </Card>
      <Grid container spacing={1.5}>
        {[
          { label: "Anodic Peak", color: "#22d3ee", text: "The bump in the upper trace. Marks where electrons are being pulled OUT of the electrode fastest — an oxidation reaction happening at ~−0.35 V." },
          { label: "Cathodic Peak", color: "#f472b6", text: "The dip in the lower trace. Marks where electrons return to the electrode — the reverse reduction reaction at ~−0.28 V." },
          { label: "EDLC Background", color: "#a78bfa", text: "The 'platform' current on which peaks sit. This is pure capacitive charging — ions accumulating on the surface, no chemical reactions." },
          { label: "Loop Area", color: "#10b981", text: "Area enclosed by the whole CV loop ∝ total charge stored per cycle. Larger area = more energy stored = higher capacitance." },
        ].map(({ label, color, text }) => (
          <Grid item xs={12} sm={6} key={label}>
            <Box sx={{ p: 1.5, borderRadius: 2, background: alpha(color, 0.05), border: `1px solid ${alpha(color, 0.2)}` }}>
              <Typography variant="caption" sx={{ color, fontWeight: 700, fontSize: "0.68rem", display: "block", mb: 0.5 }}>● {label}</Typography>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem", lineHeight: 1.6, display: "block" }}>{text}</Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}

function M07_Mathematics() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85, mb: 2 }}>
        Every metric shown in this platform is computed from these formulas. All formulas operate on the 651-point
        current arrays predicted by the models.
      </Typography>
      <FormulaBox label="RMSE — Root Mean Square Error"
        formula="RMSE = √( Σ(I_pred − I_true)² / n )"
        desc="Prediction error in µA. Lower is better. Penalises large errors more than small ones. n=651 points per curve. Range in this study: 26–50 µA on currents of ±200–600 µA." />
      <FormulaBox label="R² — Coefficient of Determination"
        formula="R² = 1 − SS_res / SS_tot  =  1 − Σ(y_true−y_pred)² / Σ(y_true−ȳ)²"
        desc="Fraction of variance in the current explained by the model. R²=1.0 is perfect. R²=0 means the model is no better than the mean. All models here achieve R²>0.96 on the zero-shot test." />
      <FormulaBox label="CSI — Charge Storage Index"
        formula="CSI = ∮ I dV / ( 2 · ν · ΔV )"
        desc="Proxy for relative capacitance. ∮I dV is the loop area (trapezoidal integration over voltage). ν = scan rate (V/s). ΔV = voltage window (0.65 V). Not absolute without electrode geometry." />
      <FormulaBox label="Cosine Similarity — Model Agreement"
        formula="cos(A, B) = (A · B) / ( ‖A‖ · ‖B‖ )"
        desc="Compares two model prediction vectors A and B (each 651 points). 1.0 = identical predictions. Used in the similarity matrix on the Compare Models page." />
      <FormulaBox label="Symmetry Factor"
        formula="SF = | I_anodic_peak | / | I_cathodic_peak |"
        desc="Ratio of anodic to cathodic peak currents. Ideal = 1.000 for perfectly reversible capacitive electrode. Deviations indicate Faradaic contributions or diffusion limitations." />
      <FormulaBox label="Integral Area — Charge per Cycle"
        formula="Area = | ∫_anodic I dV | + | ∫_cathodic I dV |  (µA·V)"
        desc="Total area enclosed by the CV loop via trapezoidal integration over the voltage axis. Proportional to charge Q stored per cycle. Not integrated over array indices — voltage axis only." />
      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>Which metric matters most?</strong> For prediction accuracy use RMSE and R². For electrode quality use CSI, integral area, and symmetry factor. For comparing ML models use cosine similarity. R² is the most intuitive — 0.97 means the model explains 97% of the variation in the current.
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M08_Dataset() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        The dataset consists of 40 experimentally measured CV curves across 4 ZnO nanocomposites and 10 scan rates.
        Each curve has 651 digitised current measurements — 26,100 data points total.
      </Typography>
      {/* Charts */}
      <Grid container spacing={3}>
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
                  height: 265, margin: { l: 50, r: 20, t: 48, b: 50 }, barmode: "stack",
                  font: { family: "Inter", color: "#8892a4", size: 10 },
                  xaxis: { title: { text: "Scan Rate (mV/s)", font: { size: 11, color: "#94a3b8" } }, tickvals: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], gridcolor: "rgba(255,255,255,0.03)", tickfont: { size: 10, family: "JetBrains Mono" } },
                  yaxis: { title: { text: "# Curves", font: { size: 11, color: "#94a3b8" } }, gridcolor: "rgba(255,255,255,0.03)", tickfont: { size: 10 } },
                  legend: { orientation: "h", x: 0.5, xanchor: "center", y: 1.0, yanchor: "bottom", bgcolor: "rgba(0,0,0,0)", font: { size: 10, color: "#c8cfd8" } },
                  hoverlabel: { bgcolor: "#111c2d", bordercolor: "#10b981", font: { size: 11, color: "#f1f5f9", family: "JetBrains Mono" } },
                }}
                config={{ responsive: true, displayModeBar: false }} style={{ width: "100%" }} useResizeHandler
              />
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.62rem" }}>
                SR=30 and SR=50 are completely absent from training — model performance at these rates measures true scan-rate interpolation. SR=100 appears in both Train and Test-MAT.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
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
                  height: 265, margin: { l: 10, r: 10, t: 10, b: 10 },
                  font: { family: "Inter", color: "#8892a4", size: 10 },
                  legend: { orientation: "v", x: 1.02, y: 0.5, bgcolor: "rgba(0,0,0,0)", font: { size: 9, color: "#c8cfd8" } },
                  annotations: [{ text: "40<br>curves", x: 0.5, y: 0.5, font: { size: 13, color: "#f1f5f9", family: "JetBrains Mono" }, showarrow: false }],
                  hoverlabel: { bgcolor: "#111c2d", bordercolor: "#a78bfa", font: { size: 11, color: "#f1f5f9", family: "JetBrains Mono" } },
                }}
                config={{ responsive: true, displayModeBar: false }} style={{ width: "100%", height: 265 }} useResizeHandler
              />
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.62rem" }}>
                NM4 is entirely absent from training — its 10 curves (25%) form a strict out-of-distribution test for material generalization.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      {/* Dataset summary */}
      <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(16,185,129,0.15)" }}>
        <CardContent sx={{ p: 2.5 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
            <DatasetRounded sx={{ color: "#10b981", fontSize: 20 }} />
            <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Dataset Summary</Typography>
          </Box>
          <Grid container spacing={2.5}>
            {[
              { label: "Materials", value: "4", desc: "NM1 (ZnO baseline), NM2, NM3, NM4 (experimental samples — compositions not specified in source dataset)", color: "#22d3ee" },
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
    </Box>
  );
}

function M09_Features() {
  const features = [
    { name: "potential_V", formula: "V (raw)", color: "#22d3ee", why: "The fundamental electrochemical variable. Directly determines what reactions occur and at what rate." },
    { name: "scan_rate_mVs", formula: "ν (raw)", color: "#10b981", why: "Controls current magnitude. Essential feature — models at different ν produce different current magnitudes." },
    { name: "log_scan_rate", formula: "log₁₀(ν)", color: "#a78bfa", why: "Linearises the power-law relationship between ν and current. Helps tree models learn the ν-dependence efficiently." },
    { name: "sqrt_scan_rate", formula: "√ν", color: "#f59e0b", why: "Directly encodes the Randles–Ševčík diffusion term (I ∝ √ν for Faradaic processes). Physically motivated feature." },
    { name: "potential_from_lower", formula: "V − V_lower = V + 0.65", color: "#f472b6", why: "Distance from the lower voltage limit. Models often need to know how far into the sweep we are, not just the absolute potential." },
    { name: "potential_from_upper", formula: "V_upper − V = 0 − V", color: "#3b82f6", why: "Distance from the upper voltage limit. Together with potential_from_lower, forms a symmetric position encoding." },
    { name: "sr×potential", formula: "ν × V", color: "#fb923c", why: "Interaction term. Captures how the relationship between potential and current changes at different scan rates." },
    { name: "direction×potential", formula: "d × V  (d = ±1)", color: "#22d3ee", why: "Encodes which direction the sweep is going AND where we are. Gives the model full polarity context." },
    { name: "sweep_direction", formula: "+1 (anodic) / −1 (cathodic)", color: "#10b981", why: "Boolean flag distinguishing forward from reverse sweep. Without this, anodic and cathodic currents at the same voltage would be indistinguishable." },
    { name: "sweep_position", formula: "index / 650 ∈ [0, 1]", color: "#a78bfa", why: "Normalised position within the full 651-point sweep. Lets sequence models know where they are in the sweep without magnitude bias." },
  ];
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        Each of the 651 points in a CV curve is represented by 10 features. These were engineered from the raw
        voltage value and scan rate, incorporating domain knowledge from electrochemistry physics.
      </Typography>
      <Grid container spacing={1.5}>
        {features.map((f, i) => (
          <Grid item xs={12} sm={6} key={f.name}>
            <Box sx={{ p: 1.75, borderRadius: 2, background: alpha(f.color, 0.05), border: `1px solid ${alpha(f.color, 0.2)}` }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.75 }}>
                <Chip label={`F${i + 1}`} size="small" sx={{ height: 18, fontSize: "0.58rem", fontWeight: 800, fontFamily: "JetBrains Mono", backgroundColor: alpha(f.color, 0.15), color: f.color, border: `1px solid ${alpha(f.color, 0.3)}` }} />
                <Typography sx={{ fontFamily: "JetBrains Mono", color: "#f1f5f9", fontSize: "0.75rem", fontWeight: 700 }}>{f.name}</Typography>
              </Box>
              <Typography variant="caption" sx={{ color: f.color, fontSize: "0.67rem", fontFamily: "JetBrains Mono", display: "block", mb: 0.5 }}>{f.formula}</Typography>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.67rem", lineHeight: 1.6, display: "block" }}>{f.why}</Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>Why not just use voltage and scan rate?</strong> These two raw values are insufficient because ML models (especially trees) need help learning non-linear physics. log_scan_rate makes the 10× scan rate range linear. sqrt_scan_rate matches Faradaic theory. sweep_direction tells the model which half of the loop we're on. Together they let tree models learn in ~minutes what would otherwise require deep physics priors.
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M10_Models() {
  const models = [
    {
      name: "Random Forest", abbr: "RF", color: "#6366f1",
      category: "Tree Ensemble",
      tagline: "Accuracy champion — best validation RMSE",
      metric: "Val RMSE 26.59 µA",
      size: "85 MB",
      what: "A Random Forest is a collection (or 'forest') of many independent decision trees. Each tree is like a set of if-then rules: 'If scan rate > 50 mV/s AND voltage < −0.4 V AND sweep is anodic, then current ≈ 180 µA.' One tree alone is fragile and noisy — but 300 trees together, each trained on a different random slice of the data, vote together for a stable, accurate answer.",
      how: [
        { label: "Step 1 — Bootstrap Sampling", text: "For each of the 300 trees, randomly draw ~63% of the training points (with replacement). Each tree therefore trains on a slightly different version of the dataset." },
        { label: "Step 2 — Random Feature Selection", text: "At every node split, only a random subset of the 10 features is considered (typically √10 ≈ 3). This prevents all trees from making the same splits and keeps them diverse." },
        { label: "Step 3 — Grow Deep Trees", text: "Each tree is grown fully — no early stopping. A deep tree memorises its training slice perfectly but that's fine, because other trees memorise different slices." },
        { label: "Step 4 — Average the Predictions", text: "All 300 trees independently predict the current at each voltage point. The final prediction is simply the average. Errors from individual trees cancel each other out — the ensemble is far more accurate than any single tree." },
      ],
      insight: "Feature importance falls out naturally: features that cause the best splits across all trees (like log_scan_rate and sr×potential) rank highest. This is why RF is the most interpretable tree ensemble for understanding which electrochemical variables matter most.",
      strength: "Best validation RMSE (26.59 µA). Highly stable. Built-in feature importance ranking.",
      weakness: "85 MB on disk — the heaviest model. Slowest prediction (must query 300 trees sequentially).",
    },
    {
      name: "LightGBM", abbr: "LGB", color: "#22d3ee",
      category: "Gradient Boosting",
      tagline: "Deployment champion — 360× smaller than RF",
      metric: "Val RMSE 27.2 µA · 1.7 MB",
      size: "1.7 MB",
      what: "LightGBM is a Gradient Boosted Decision Tree (GBDT) model. Unlike Random Forest where all trees are independent, in gradient boosting trees are built sequentially — each new tree is specifically designed to fix the errors of all previous trees. LightGBM adds a key innovation: leaf-wise tree growth, which makes it exceptionally fast and compact.",
      how: [
        { label: "Step 1 — Start with a simple guess", text: "Tree #1 predicts the average current for all 651 points. This is a terrible prediction but gives us a starting point and a set of residual errors (how wrong we were)." },
        { label: "Step 2 — Build a corrector tree", text: "Tree #2 is trained not on the original target, but on the residual errors from Tree #1. It learns: 'where did Tree #1 go wrong, and by how much?' Then we add its predictions to Tree #1's output." },
        { label: "Step 3 — Keep boosting", text: "This continues: each new tree targets the remaining errors. After hundreds of rounds, the cumulative prediction steadily improves. A learning rate (e.g. 0.05) scales each tree's contribution to prevent overshooting." },
        { label: "Step 4 — Leaf-wise growth", text: "Traditional boosting grows trees level by level (all nodes at depth 1, then depth 2, etc.). LightGBM always splits the single leaf that reduces the error the most — creating asymmetric trees that converge faster with fewer trees, resulting in a tiny 1.7 MB model." },
      ],
      insight: "LightGBM is 360× smaller than RF (1.7 MB vs 85 MB) with only ~2% higher RMSE. For edge deployment — embedding the model in a device, an API, or a mobile app — this tradeoff is decisive. It's the recommended production model.",
      strength: "Smallest model (1.7 MB), fastest inference, edge-deployment ready. Nearly matches RF accuracy.",
      weakness: "Leaf-wise growth can overfit on very small or noisy datasets if learning rate and regularisation are not tuned carefully.",
    },
    {
      name: "XGBoost", abbr: "XGB", color: "#10b981",
      category: "Gradient Boosting",
      tagline: "Consistent all-rounder with built-in regularisation",
      metric: "Test-MAT R² > 0.97",
      size: "~5 MB",
      what: "XGBoost (eXtreme Gradient Boosting) is the same sequential error-correction idea as LightGBM but uses level-wise tree growth and adds explicit mathematical regularisation (L1 + L2 penalties) into its training objective. This makes XGBoost more conservative and less likely to overfit, especially on the small training sets common in experimental material science.",
      how: [
        { label: "Step 1 — Regularised objective", text: "XGBoost minimises: Loss(actual, predicted) + λ·Σ(leaf weights²) + α·Σ|leaf weights|. The λ (L2) and α (L1) terms penalise extreme leaf values, forcing the model to be simpler and more generalisable." },
        { label: "Step 2 — Level-wise growth", text: "XGBoost expands all leaves at the same depth before going deeper. This produces more symmetric, balanced trees compared to LightGBM's asymmetric leaf-wise trees. Safer on small datasets but slightly slower." },
        { label: "Step 3 — Second-order gradient", text: "XGBoost uses both the gradient (first derivative) and the Hessian (second derivative) of the loss function to determine the best splits — a more precise mathematical step size than first-order methods like basic GBDT." },
        { label: "Step 4 — Column and row subsampling", text: "Like Random Forest, XGBoost randomly samples features and data rows at each tree, adding extra randomness that improves generalisation to new scan rates and materials." },
      ],
      insight: "XGBoost and LightGBM produce similar accuracy, but XGBoost's regularisation makes it more robust when the training set is small (24 curves here). Its Test-MAT R² > 0.97 confirms that the regularised approach generalises well to the unseen NM4 sample.",
      strength: "Strong across all 4 evaluation partitions. Regularisation prevents overfitting on small experimental datasets.",
      weakness: "Larger than LightGBM. Level-wise growth is slower to train. Not as compact for edge deployment.",
    },
    {
      name: "Dense ANN", abbr: "ANN", color: "#f59e0b",
      category: "Neural Network",
      tagline: "Universal approximator — learns any non-linear pattern",
      metric: "R² > 0.96 all splits",
      size: "~2 MB",
      what: "A Dense ANN (Artificial Neural Network) is a stack of layers, each containing many artificial neurons. Unlike tree models, which make decisions through if-then rules, neurons compute weighted sums of their inputs and apply a non-linear activation function. Through training (backpropagation), the network automatically discovers which patterns in the 10 electrochemical features best predict the current — no manual feature engineering decisions needed.",
      how: [
        { label: "Layer 1 — Input (10 features)", text: "The 10 engineered features for one voltage point (potential_V, scan_rate, log_scan_rate, etc.) enter the network simultaneously. Each feature is just a number." },
        { label: "Layer 2 — Dense(256) + ReLU", text: "256 neurons each compute: output = ReLU(w₁·f₁ + w₂·f₂ + … + w₁₀·f₁₀ + bias). ReLU zeroes out negative values. This layer learns low-level patterns like 'high scan rate + anodic direction → increase current'." },
        { label: "Layer 3 — Dense(128) + ReLU, Layer 4 — Dense(64) + ReLU", text: "Each deeper layer combines patterns from the previous layer into more abstract representations. Layer 3 might learn 'this voltage is near the anodic peak region at this scan rate'. Layer 4 might learn 'this is the plateau region between anodic peak and upper reversal'." },
        { label: "Layer 5 — Output (1 value)", text: "A single neuron with no activation outputs the predicted current in µA. The full 651-point CV curve is reconstructed by running all 651 points through the network independently." },
      ],
      insight: "Each of the 651 voltage points is predicted independently — the ANN has no memory of adjacent points. This means it cannot leverage the fact that current at −0.30 V follows causally from current at −0.31 V. LSTM and GRU fix this limitation by processing the sweep as a sequence.",
      strength: "Universal approximator — can fit any smooth function. Compact (2 MB). Good generalisation across all partitions.",
      weakness: "No sequence awareness — predicts each of the 651 points in isolation, ignoring sweep order and context from neighbouring points.",
    },
    {
      name: "Stacked LSTM", abbr: "LSTM", color: "#a78bfa",
      category: "Recurrent Neural Network",
      tagline: "Sequence-aware — reads the CV sweep like a story",
      metric: "Strong on all partitions",
      size: "~3 MB",
      what: "LSTM (Long Short-Term Memory) is a recurrent neural network that processes the 651-point CV sweep as an ordered sequence — exactly like reading a sentence word by word rather than reading all words simultaneously. At each voltage step, the LSTM maintains a 'memory' (called the hidden state) of everything it has processed so far, letting it know: 'I already passed the anodic peak, so I'm now on the cathodic descent.'",
      how: [
        { label: "The Cell State — Long-term memory", text: "The LSTM maintains a cell state vector (128 numbers) that flows through the entire sequence like a conveyor belt, carrying information from the start of the sweep to the end with minimal modification at each step." },
        { label: "Forget Gate", text: "At each voltage step, the forget gate looks at the current input + previous hidden state and outputs values between 0 and 1. Multiplied with the cell state, it decides what old memory to erase. Example: entering the reversal region might signal 'forget the anodic peak position, we've passed it'." },
        { label: "Input Gate", text: "The input gate decides what new information to write into the cell state. It creates a candidate memory (tanh layer) then selectively gates how much of it to actually store. Example: 'this is the cathodic peak position — remember its exact voltage'." },
        { label: "Output Gate", text: "The output gate decides what part of the current cell state to expose as the hidden state (the working output passed to the next step and to the Dense layer). The Dense layer at the end converts the 64-dimensional hidden state into one current value in µA." },
      ],
      insight: "Two LSTM layers are stacked: LSTM₁(128 units) processes the raw input sequence and passes its hidden states to LSTM₂(64 units), which learns higher-level temporal patterns across the sweep. This is analogous to a two-stage reading comprehension — first parse the words, then understand the sentence structure.",
      strength: "Understands the entire sweep context when predicting each point. Captures long-range dependencies like peak-to-peak relationships.",
      weakness: "More trainable parameters than GRU. Slower training. Can overfit on small datasets, making it slightly less generalisable than GRU to unseen materials.",
    },
    {
      name: "Stacked GRU", abbr: "GRU", color: "#f472b6",
      category: "Recurrent Neural Network",
      tagline: "Best zero-shot generalisation — wins on unseen material",
      metric: "Test-MAT R² 0.9751 ★",
      size: "~2 MB",
      what: "GRU (Gated Recurrent Unit) is a streamlined version of LSTM. It achieves the same goal — processing the CV sweep as a sequence with memory — but uses just 2 gates instead of LSTM's 3, with no separate cell state. Fewer parameters means it trains faster, generalises better on small datasets, and is the top-performing model for zero-shot material extrapolation in this study.",
      how: [
        { label: "Reset Gate", text: "The reset gate decides how much of the previous hidden state (memory from earlier sweep points) to consider when computing the new candidate state. A value near 0 means 'ignore the past — treat this as a fresh start'. This is useful near the voltage reversal point." },
        { label: "Update Gate", text: "The update gate decides how much of the old hidden state to keep vs how much of the new candidate to blend in. It simultaneously acts as both LSTM's forget gate and input gate — one operation instead of two. Values near 1 = 'mostly keep old memory'. Values near 0 = 'mostly use new input'." },
        { label: "Candidate Hidden State", text: "The new candidate state = tanh(W·[reset_gate ⊙ h_prev, x_t]). This combines the (selectively reset) past memory with the current input to propose an update." },
        { label: "Final Hidden State", text: "h_t = (1 − update_gate) ⊙ h_{t−1} + update_gate ⊙ candidate. This elegant single formula replaces LSTM's three-gate mechanism. The result is passed to GRU₂(64 units) and finally to a Dense layer for current prediction." },
      ],
      insight: "GRU beats LSTM on Test-MAT (unseen NM4 sample) despite being simpler. This is a classic bias-variance tradeoff: fewer parameters = higher bias but lower variance = better generalisation when training data is limited (only 24 curves here). GRU learned the underlying CV physics, not just the training materials' specific signatures.",
      strength: "Best Test-MAT R² (0.9751) — strongest zero-shot material generalisation. Fewer parameters, faster training, smaller than LSTM.",
      weakness: "Slightly less expressive than LSTM for very complex long-range dependencies — though this is rarely a limitation in practice.",
    },
  ];

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        Six machine learning architectures were trained and evaluated: three tree-based ensemble methods and three neural networks.
        Each is explained below — what it is, how it works step by step, and what makes it useful for CV prediction.
      </Typography>

      {/* Category divider */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mt: 0.5 }}>
        <Typography variant="caption" sx={{ color: "#6366f1", fontWeight: 700, fontSize: "0.68rem", letterSpacing: "0.08em", flexShrink: 0 }}>🌳 TREE-BASED MODELS</Typography>
        <Box sx={{ flex: 1, height: "1px", background: "rgba(99,102,241,0.2)" }} />
      </Box>

      {models.slice(0, 3).map((m) => (
        <Card key={m.abbr} sx={{ background: `linear-gradient(135deg, ${alpha(m.color, 0.06)} 0%, rgba(10,16,32,0.95) 100%)`, border: `1px solid ${alpha(m.color, 0.22)}`, borderLeft: `4px solid ${m.color}` }}>
          <CardContent sx={{ p: 2.5 }}>
            {/* Header */}
            <Box sx={{ display: "flex", alignItems: "flex-start", gap: 1.5, mb: 2, flexWrap: "wrap" }}>
              <Chip label={m.abbr} size="small" sx={{ height: 22, fontSize: "0.68rem", fontWeight: 800, fontFamily: "JetBrains Mono", backgroundColor: alpha(m.color, 0.18), color: m.color, border: `1px solid ${alpha(m.color, 0.35)}` }} />
              <Box sx={{ flex: 1 }}>
                <Typography variant="h6" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.92rem", lineHeight: 1.2 }}>{m.name}</Typography>
                <Typography variant="caption" sx={{ color: m.color, fontSize: "0.68rem", fontStyle: "italic" }}>{m.tagline}</Typography>
              </Box>
              <Chip label={m.metric} size="small" sx={{ fontSize: "0.6rem", height: 20, backgroundColor: alpha(m.color, 0.12), color: m.color, border: `1px solid ${alpha(m.color, 0.3)}`, fontFamily: "JetBrains Mono" }} />
            </Box>

            {/* What it is */}
            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.08em", display: "block", mb: 0.75 }}>WHAT IT IS</Typography>
            <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.62)", fontSize: "0.76rem", lineHeight: 1.8, mb: 2 }}>{m.what}</Typography>

            {/* How it works */}
            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.08em", display: "block", mb: 1 }}>HOW IT WORKS — STEP BY STEP</Typography>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1, mb: 2 }}>
              {m.how.map((step, i) => (
                <Box key={i} sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
                  <Box sx={{ width: 22, height: 22, borderRadius: "6px", background: alpha(m.color, 0.14), border: `1px solid ${alpha(m.color, 0.3)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, mt: 0.1 }}>
                    <Typography sx={{ color: m.color, fontSize: "0.6rem", fontWeight: 800, fontFamily: "JetBrains Mono" }}>{i + 1}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" sx={{ color: m.color, fontWeight: 700, fontSize: "0.68rem", display: "block", mb: 0.3 }}>{step.label}</Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.7, display: "block" }}>{step.text}</Typography>
                  </Box>
                </Box>
              ))}
            </Box>

            {/* Key insight */}
            <Box sx={{ p: 1.5, borderRadius: 2, background: alpha(m.color, 0.06), border: `1px solid ${alpha(m.color, 0.15)}`, mb: 1.5 }}>
              <Typography variant="caption" sx={{ color: m.color, fontWeight: 700, fontSize: "0.65rem", display: "block", mb: 0.5 }}>💡 KEY INSIGHT</Typography>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.7rem", lineHeight: 1.7, display: "block" }}>{m.insight}</Typography>
            </Box>

            {/* Strength / Weakness */}
            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
              <Box sx={{ flex: 1, minWidth: 180, p: 1.25, borderRadius: 1.5, background: "rgba(16,185,129,0.05)", border: "1px solid rgba(16,185,129,0.15)" }}>
                <Typography variant="caption" sx={{ color: "#10b981", fontSize: "0.65rem", fontWeight: 700, display: "block", mb: 0.4 }}>✓ STRENGTHS</Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem", lineHeight: 1.6, display: "block" }}>{m.strength}</Typography>
              </Box>
              <Box sx={{ flex: 1, minWidth: 180, p: 1.25, borderRadius: 1.5, background: "rgba(245,158,11,0.05)", border: "1px solid rgba(245,158,11,0.15)" }}>
                <Typography variant="caption" sx={{ color: "#f59e0b", fontSize: "0.65rem", fontWeight: 700, display: "block", mb: 0.4 }}>⚠ LIMITATIONS</Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem", lineHeight: 1.6, display: "block" }}>{m.weakness}</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      ))}

      {/* Neural network divider */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mt: 0.5 }}>
        <Typography variant="caption" sx={{ color: "#f59e0b", fontWeight: 700, fontSize: "0.68rem", letterSpacing: "0.08em", flexShrink: 0 }}>🧠 NEURAL NETWORK MODELS</Typography>
        <Box sx={{ flex: 1, height: "1px", background: "rgba(245,158,11,0.2)" }} />
      </Box>

      {models.slice(3).map((m) => (
        <Card key={m.abbr} sx={{ background: `linear-gradient(135deg, ${alpha(m.color, 0.06)} 0%, rgba(10,16,32,0.95) 100%)`, border: `1px solid ${alpha(m.color, 0.22)}`, borderLeft: `4px solid ${m.color}` }}>
          <CardContent sx={{ p: 2.5 }}>
            <Box sx={{ display: "flex", alignItems: "flex-start", gap: 1.5, mb: 2, flexWrap: "wrap" }}>
              <Chip label={m.abbr} size="small" sx={{ height: 22, fontSize: "0.68rem", fontWeight: 800, fontFamily: "JetBrains Mono", backgroundColor: alpha(m.color, 0.18), color: m.color, border: `1px solid ${alpha(m.color, 0.35)}` }} />
              <Box sx={{ flex: 1 }}>
                <Typography variant="h6" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.92rem", lineHeight: 1.2 }}>{m.name}</Typography>
                <Typography variant="caption" sx={{ color: m.color, fontSize: "0.68rem", fontStyle: "italic" }}>{m.tagline}</Typography>
              </Box>
              <Chip label={m.metric} size="small" sx={{ fontSize: "0.6rem", height: 20, backgroundColor: alpha(m.color, 0.12), color: m.color, border: `1px solid ${alpha(m.color, 0.3)}`, fontFamily: "JetBrains Mono" }} />
            </Box>

            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.08em", display: "block", mb: 0.75 }}>WHAT IT IS</Typography>
            <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.62)", fontSize: "0.76rem", lineHeight: 1.8, mb: 2 }}>{m.what}</Typography>

            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.08em", display: "block", mb: 1 }}>HOW IT WORKS — STEP BY STEP</Typography>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1, mb: 2 }}>
              {m.how.map((step, i) => (
                <Box key={i} sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
                  <Box sx={{ width: 22, height: 22, borderRadius: "6px", background: alpha(m.color, 0.14), border: `1px solid ${alpha(m.color, 0.3)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, mt: 0.1 }}>
                    <Typography sx={{ color: m.color, fontSize: "0.6rem", fontWeight: 800, fontFamily: "JetBrains Mono" }}>{i + 1}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" sx={{ color: m.color, fontWeight: 700, fontSize: "0.68rem", display: "block", mb: 0.3 }}>{step.label}</Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.7, display: "block" }}>{step.text}</Typography>
                  </Box>
                </Box>
              ))}
            </Box>

            <Box sx={{ p: 1.5, borderRadius: 2, background: alpha(m.color, 0.06), border: `1px solid ${alpha(m.color, 0.15)}`, mb: 1.5 }}>
              <Typography variant="caption" sx={{ color: m.color, fontWeight: 700, fontSize: "0.65rem", display: "block", mb: 0.5 }}>💡 KEY INSIGHT</Typography>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.7rem", lineHeight: 1.7, display: "block" }}>{m.insight}</Typography>
            </Box>

            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
              <Box sx={{ flex: 1, minWidth: 180, p: 1.25, borderRadius: 1.5, background: "rgba(16,185,129,0.05)", border: "1px solid rgba(16,185,129,0.15)" }}>
                <Typography variant="caption" sx={{ color: "#10b981", fontSize: "0.65rem", fontWeight: 700, display: "block", mb: 0.4 }}>✓ STRENGTHS</Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem", lineHeight: 1.6, display: "block" }}>{m.strength}</Typography>
              </Box>
              <Box sx={{ flex: 1, minWidth: 180, p: 1.25, borderRadius: 1.5, background: "rgba(245,158,11,0.05)", border: "1px solid rgba(245,158,11,0.15)" }}>
                <Typography variant="caption" sx={{ color: "#f59e0b", fontSize: "0.65rem", fontWeight: 700, display: "block", mb: 0.4 }}>⚠ LIMITATIONS</Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem", lineHeight: 1.6, display: "block" }}>{m.weakness}</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      ))}

      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>Which model should you use?</strong> For highest raw accuracy use RF. For deployment or edge devices use LightGBM (1.7 MB, nearly as accurate). For predicting a completely new ZnO material nobody has seen before, use GRU (highest Test-MAT R² = 0.9751). All 6 models agree strongly (cosine similarity {">"} 0.94), so the choice of model matters less than you might think — the feature engineering and leakage-safe splits are the real drivers of quality.
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M11_Training() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
      {/* Research Pipeline */}
      <Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
          <TimelineRounded sx={{ color: "#00d4ff", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9", fontSize: "0.95rem" }}>Research Pipeline</Typography>
        </Box>
        <Card sx={{ background: "rgba(15,25,35,0.85)", border: "1px solid rgba(0,212,255,0.12)" }}>
          <CardContent sx={{ p: 2.5 }}>
            <Box sx={{ display: { xs: "none", sm: "flex" }, alignItems: "center", justifyContent: "space-between", gap: 0 }}>
              {PIPELINE_STEPS.map((step, i) => (
                <React.Fragment key={step.label}>
                  <motion.div
                    initial={{ opacity: 0, y: 18, scale: 0.92 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ delay: i * 0.13, duration: 0.42, ease: [0.16, 1, 0.3, 1] }}
                    style={{ flex: 1, textAlign: "center" }}
                  >
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
                      <Box sx={{ width: 48, height: 48, borderRadius: "50%", background: `radial-gradient(circle, ${alpha(step.color, 0.25)} 0%, ${alpha(step.color, 0.08)} 100%)`, border: `2px solid ${alpha(step.color, 0.5)}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>{step.icon}</Box>
                      <Typography variant="caption" sx={{ color: step.color, fontWeight: 700, fontSize: "0.68rem", textAlign: "center", whiteSpace: "pre-line", lineHeight: 1.3 }}>{step.label}</Typography>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.28)", fontSize: "0.6rem", textAlign: "center", lineHeight: 1.3 }}>{step.desc}</Typography>
                    </Box>
                  </motion.div>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.13 + 0.25, duration: 0.3 }}
                    >
                      <Box sx={{ color: "rgba(255,255,255,0.2)", fontSize: 18, mx: 0.5, flexShrink: 0 }}>→</Box>
                    </motion.div>
                  )}
                </React.Fragment>
              ))}
            </Box>
            <Box sx={{ display: { xs: "flex", sm: "none" }, flexDirection: "column", gap: 1.5 }}>
              {PIPELINE_STEPS.map((step) => (
                <Box key={step.label} sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                  <Box sx={{ width: 34, height: 34, borderRadius: "50%", background: alpha(step.color, 0.15), border: `1.5px solid ${alpha(step.color, 0.4)}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0 }}>{step.icon}</Box>
                  <Box>
                    <Typography variant="caption" sx={{ color: step.color, fontWeight: 700, fontSize: "0.7rem" }}>{step.label.replace("\n", " ")}</Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.6rem", display: "block" }}>{step.desc}</Typography>
                  </Box>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </Box>
      {/* Leakage-safe split */}
      <Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
          <LockRounded sx={{ color: "#a78bfa", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9", fontSize: "0.95rem" }}>Leakage-Safe Evaluation Design</Typography>
          <Chip label="SCIENTIFICALLY CRITICAL" size="small" sx={{ fontSize: "0.56rem", height: 18, fontWeight: 700, backgroundColor: alpha("#a78bfa", 0.12), color: "#a78bfa", border: "1px solid rgba(139,92,246,0.3)" }} />
        </Box>
        <Card sx={{ background: "rgba(15,25,35,0.85)", border: "1px solid rgba(139,92,246,0.2)" }}>
          <CardContent sx={{ p: 2.5 }}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={7}>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.68rem", display: "block", mb: 2 }}>CV curves grouped by (material × scan_rate). Splits at curve level — no curve appears in two partitions.</Typography>
                {[
                  { label: "🔵 TRAIN",    desc: "NM1, NM2, NM3  ×  SR = 10,20,40,60,70,80,90,100 mV/s", detail: "24 curves · 15,624 pts · in-sample fitting", color: "#6366f1", pct: 60 },
                  { label: "🟢 VALIDATION", desc: "NM1, NM2, NM3  ×  SR = 30 mV/s", detail: "3 curves · 1,953 pts · scan-rate interpolation", color: "#10b981", pct: 8 },
                  { label: "🟡 TEST-SR",  desc: "NM1, NM2, NM3  ×  SR = 50 mV/s", detail: "3 curves · 1,953 pts · scan-rate interpolation", color: "#f59e0b", pct: 8 },
                  { label: "🔴 TEST-MAT", desc: "NM4 (unseen sample)  ×  all 10 scan rates", detail: "10 curves · 6,510 pts · material extrapolation", color: "#f472b6", pct: 25 },
                ].map((split, i) => (
                  <motion.div key={split.label} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1, duration: 0.3 }}>
                    <Box sx={{ mb: 1.75 }}>
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", mb: 0.5 }}>
                        <Typography variant="caption" sx={{ color: split.color, fontWeight: 700, fontSize: "0.7rem" }}>{split.label}</Typography>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.6rem" }}>{split.detail}</Typography>
                      </Box>
                      <Box sx={{ height: 18, borderRadius: 1.5, background: "rgba(255,255,255,0.04)", overflow: "hidden", border: `1px solid ${alpha(split.color, 0.15)}` }}>
                        <motion.div initial={{ width: 0 }} animate={{ width: `${split.pct}%` }} transition={{ delay: i * 0.1 + 0.3, duration: 0.6, ease: [0.16, 1, 0.3, 1] }} style={{ height: "100%", background: `linear-gradient(90deg, ${split.color}55 0%, ${split.color}22 100%)`, display: "flex", alignItems: "center", paddingLeft: 8 }}>
                          <Typography variant="caption" sx={{ color: split.color, fontSize: "0.6rem", fontFamily: "JetBrains Mono", fontWeight: 700, whiteSpace: "nowrap" }}>{split.desc}</Typography>
                        </motion.div>
                      </Box>
                    </Box>
                  </motion.div>
                ))}
              </Grid>
              <Grid item xs={12} md={5}>
                <Box sx={{ p: 2, borderRadius: 2, background: alpha("#a78bfa", 0.05), border: "1px solid rgba(139,92,246,0.15)", mb: 2 }}>
                  <Typography variant="caption" sx={{ color: "#a78bfa", fontWeight: 700, fontSize: "0.7rem", display: "block", mb: 1 }}>⚠ Why leakage-safe splitting matters</Typography>
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>Splitting at the point level would let the model see partial information about the curve it's being tested on — inflating R² artificially. Each of our 651-point curves is assigned to exactly one partition.</Typography>
                </Box>
                <Box sx={{ p: 2, borderRadius: 2, background: alpha("#f472b6", 0.05), border: "1px solid rgba(244,114,182,0.15)" }}>
                  <Typography variant="caption" sx={{ color: "#f472b6", fontWeight: 700, fontSize: "0.7rem", display: "block", mb: 1 }}>🔬 Test-MAT: Zero-shot material extrapolation</Typography>
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>NM4 was never seen during training. Test-MAT R² measures whether models truly learn CV physics. GRU achieves R²=0.9751 on this zero-shot test.</Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>
      {/* Methodology steps */}
      <Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
          <BuildRounded sx={{ color: "#00d4ff", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9", fontSize: "0.95rem" }}>Research Methodology</Typography>
        </Box>
        <motion.div variants={staggerContainer} initial="initial" animate="animate">
          <Grid container spacing={2}>
            {METHODOLOGY_STEPS.map((step, i) => (
              <Grid item xs={12} md={6} key={step.title}>
                <motion.div variants={cardVariants}>
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: `1px solid ${alpha(step.color, 0.2)}`, height: "100%" }}>
                    <CardContent sx={{ p: 2.25 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1.25 }}>
                        <Box sx={{ fontSize: 20 }}>{step.icon}</Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: step.color, fontSize: "0.82rem" }}>{step.title}</Typography>
                        <Chip label={`Step ${i + 1}`} size="small" sx={{ ml: "auto", height: 16, fontSize: "0.56rem", fontWeight: 700, backgroundColor: alpha(step.color, 0.1), color: alpha(step.color, 0.8), border: `1px solid ${alpha(step.color, 0.2)}` }} />
                      </Box>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", lineHeight: 1.65, fontSize: "0.72rem", display: "block" }}>{step.content}</Typography>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </motion.div>
      </Box>
      {/* Findings */}
      <Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
          <InsightsRounded sx={{ color: "#00d4ff", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9", fontSize: "0.95rem" }}>Scientific Findings</Typography>
        </Box>
        <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.1)" }}>
          <CardContent sx={{ p: 2.5 }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
              {FINDINGS.map((f) => (
                <Box key={f.text} sx={{ display: "flex", alignItems: "flex-start", gap: 1.5 }}>
                  <CheckCircleRounded sx={{ color: f.color, fontSize: 17, mt: 0.25, flexShrink: 0 }} />
                  <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.6)", lineHeight: 1.6, fontSize: "0.82rem" }}>{f.text}</Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
}

function M12_Architecture() {
  const layers = [
    { label: "React + Vite Frontend", color: "#22d3ee", items: ["Material UI dark theme", "Plotly.js interactive charts", "React Router + Framer Motion", "PWA with service worker"] },
    { label: "FastAPI Backend (Python)", color: "#a78bfa", items: ["REST endpoints: /predict, /compare", "CORS, error handling, pydantic validation", "Uvicorn ASGI server", "Health check + model status"] },
    { label: "ML Model Layer", color: "#f472b6", items: ["RF (joblib pickle, 85 MB)", "LightGBM + XGBoost (pkl, 1–5 MB)", "ANN / LSTM / GRU (Keras .h5)", "Lazy-loaded at first request"] },
    { label: "Data / Scaler Layer", color: "#10b981", items: ["scalers.json — 40 MinMax scalers", "One scaler per (material, scan_rate)", "Feature engineering in Python", "651 × 10 input matrix per prediction"] },
  ];
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        The platform is a two-process architecture: a React frontend served by Vite communicates with a
        FastAPI Python backend via REST. The backend loads all 6 ML models once and keeps them in memory for fast inference.
      </Typography>
      {/* Flow */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1.5, py: 1.5, flexWrap: "wrap" }}>
        {["User Browser", "React Frontend", "FastAPI Backend", "ML Models", "CV Prediction"].map((node, i, arr) => (
          <React.Fragment key={node}>
            <Box sx={{ px: 2, py: 1, borderRadius: 1.5, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.12)", textAlign: "center" }}>
              <Typography variant="caption" sx={{ color: "#f1f5f9", fontSize: "0.72rem", fontWeight: 600 }}>{node}</Typography>
            </Box>
            {i < arr.length - 1 && <Typography sx={{ color: "rgba(255,255,255,0.25)", fontSize: "1.1rem", flexShrink: 0 }}>→</Typography>}
          </React.Fragment>
        ))}
      </Box>
      <Grid container spacing={2}>
        {layers.map(({ label, color, items }) => (
          <Grid item xs={12} sm={6} key={label}>
            <Box sx={{ p: 2, borderRadius: 2, background: alpha(color, 0.05), border: `1px solid ${alpha(color, 0.2)}`, height: "100%" }}>
              <Typography variant="subtitle2" sx={{ color, fontWeight: 700, fontSize: "0.8rem", mb: 1 }}>{label}</Typography>
              {items.map((item) => (
                <Box key={item} sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                  <Box sx={{ width: 4, height: 4, borderRadius: "50%", backgroundColor: color, flexShrink: 0 }} />
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem" }}>{item}</Typography>
                </Box>
              ))}
            </Box>
          </Grid>
        ))}
      </Grid>
      <NoteBox>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", lineHeight: 1.75, display: "block" }}>
          <strong style={{ color: "#f1f5f9" }}>How a prediction request works:</strong> You select a material + scan rate + model → the frontend sends a POST /api/v1/predict → backend generates a 651×10 feature matrix → applies the stored MinMax scaler for that (material, scan_rate) → runs the model → returns 651 current values → frontend plots them with Plotly.
        </Typography>
      </NoteBox>
    </Box>
  );
}

function M13_Walkthrough() {
  const steps = [
    { n: "1", title: "Navigate to Predict", icon: "🏠", desc: "Click 'Predict' in the navigation. This is the main prediction interface where you select inputs and run models." },
    { n: "2", title: "Select a Material", icon: "🧪", desc: "Choose one of the 4 experimental electrode samples: NM1 (ZnO baseline), NM2, NM3, or NM4 (experimental samples — compositions not defined in source dataset). NM4 was never used in training — it's the hardest test." },
    { n: "3", title: "Set the Scan Rate", icon: "⚡", desc: "Choose a scan rate from 10–100 mV/s. SR=30 and SR=50 are excluded from training — these test interpolation ability. Others (10, 20, 40, 60, 70, 80, 90, 100) are in-distribution." },
    { n: "4", title: "Select Models to Run", icon: "🤖", desc: "Check which models you want to compare. Select all 6 to see how they agree. A toggle selects/deselects all at once." },
    { n: "5", title: "Run Prediction", icon: "▶️", desc: "Click Predict. The backend generates the 651-point CV curve. Typical response time: 200ms–2s depending on model." },
    { n: "6", title: "Interpret the CV Plot", icon: "📊", desc: "The predicted CV curve shows current (µA) vs potential (V). Hover for exact values. Toggle models on/off in the legend. Look for anodic/cathodic peaks and EDLC regions." },
    { n: "7", title: "Check Electrochemical Metrics", icon: "📐", desc: "Below the chart: RMSE and R² (if ground truth available), CSI (charge storage index), integral area (µA·V), symmetry factor, and peak current values." },
    { n: "8", title: "Compare Models", icon: "🔍", desc: "Navigate to 'Compare Models'. Run the same prediction across multiple models. See the cosine similarity matrix — values > 0.96 mean strong agreement. The radar chart compares performance across 5 dimensions." },
  ];
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        A step-by-step guide for using the platform. Follow these steps for your first prediction.
      </Typography>
      <Grid container spacing={1.5}>
        {steps.map(({ n, title, icon, desc }) => (
          <Grid item xs={12} sm={6} key={n}>
            <Box sx={{ p: 2, borderRadius: 2, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", display: "flex", gap: 1.5 }}>
              <Box sx={{ width: 30, height: 30, borderRadius: "50%", background: "rgba(167,139,250,0.12)", border: "1.5px solid rgba(167,139,250,0.3)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, mt: 0.25 }}>
                <Typography sx={{ color: "#a78bfa", fontWeight: 800, fontSize: "0.7rem", fontFamily: "JetBrains Mono" }}>{n}</Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.78rem", mb: 0.4 }}>{icon} {title}</Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.69rem", lineHeight: 1.65, display: "block" }}>{desc}</Typography>
              </Box>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}

function M14_Limitations() {
  const limits = [
    { icon: "📦", title: "Small Training Dataset", color: "#f59e0b", severity: "Moderate", text: "The full dataset contains 40 CV curves across 4 materials (24 training, 3 validation, 3 test-SR, 10 test-MAT). This is standard for experimental supercapacitor research but limits generalisation to exotic or complex multi-component ZnO composites not represented in the training distribution." },
    { icon: "⚗️", title: "Single Electrolyte System", color: "#fb923c", severity: "High", text: "All experiments used 1 M Na₂SO₄ (sodium sulfate) aqueous electrolyte. The trained models cannot reliably predict CV behaviour in KOH, LiOH, organic, or ionic liquid electrolytes, where electrode-electrolyte interaction fundamentally differs." },
    { icon: "📐", title: "Electrode Geometry Not Normalised", color: "#f472b6", severity: "Moderate", text: "Models predict current (µA) for the specific electrode dimensions used experimentally. Absolute capacitance per unit area (mF/cm²) or per unit mass (F/g) cannot be directly extracted — only relative trends between materials and scan rates." },
    { icon: "⏱️", title: "Scan Rate Range 10–100 mV/s", color: "#a78bfa", severity: "Low-Moderate", text: "Training covers scan rates from 10 to 100 mV/s. Predictions at very low rates (< 5 mV/s) or very high rates (> 200 mV/s) are extrapolation and should be treated with caution." },
    { icon: "🔄", title: "No Cycling Degradation Modelling", color: "#22d3ee", severity: "High", text: "Models predict the CV response of fresh electrodes at a fixed state-of-health. Degradation over charge-discharge cycles, surface passivation, electrolyte depletion, and ageing effects are not captured — predictions represent T=0 (pristine electrode) behaviour only." },
    { icon: "📉", title: "CV Prediction Only", color: "#3b82f6", severity: "Informational", text: "The platform models cyclic voltammetry exclusively. Electrochemical Impedance Spectroscopy (EIS), galvanostatic charge-discharge (GCD), rate capability curves, and long-term stability measurements are outside the current model scope." },
  ];
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.35)", lineHeight: 1.8, fontSize: "0.82rem" }}>
        Strong ML metrics do not imply unlimited applicability. We present these limitations transparently so results can be interpreted correctly. Context matters in materials science.
      </Typography>
      <motion.div variants={staggerContainer} initial="initial" animate="animate">
        <Grid container spacing={2.5}>
          {limits.map((lim) => (
            <Grid item xs={12} sm={6} md={4} key={lim.title}>
              <motion.div variants={cardVariants}>
                <Card sx={{ height: "100%", background: `linear-gradient(135deg, ${alpha(lim.color, 0.06)} 0%, rgba(15,25,35,0.9) 100%)`, border: `1px solid ${alpha(lim.color, 0.2)}`, borderLeft: `4px solid ${lim.color}` }}>
                  <CardContent sx={{ p: 2.25 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.25 }}>
                      <Typography sx={{ fontSize: 18 }}>{lim.icon}</Typography>
                      <Typography variant="subtitle2" sx={{ color: lim.color, fontWeight: 700, fontSize: "0.8rem", flex: 1, lineHeight: 1.3 }}>{lim.title}</Typography>
                      <Chip label={lim.severity} size="small" sx={{ fontSize: "0.55rem", height: 16, backgroundColor: alpha(lim.color, 0.1), color: lim.color, border: `1px solid ${alpha(lim.color, 0.25)}` }} />
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>{lim.text}</Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </motion.div>
    </Box>
  );
}

function M15_Applications() {
  const apps = [
    { Icon: ElectricCarRounded, title: "Electric Vehicle Supercapacitors", color: "#22d3ee", text: "Hybrid EVs use supercapacitors for regenerative braking and peak power delivery. AI-predicted CVs enable rapid screening of electrode materials for high-power density applications without manufacturing physical test cells.", impact: "10–100× faster screening" },
    { Icon: DevicesRounded, title: "Wearable Energy Storage", color: "#a78bfa", text: "Flexible ZnO-based supercapacitors power IoT sensors and health monitors. LightGBM's 1.7 MB footprint enables on-device CV prediction for self-monitoring energy systems with fast inference.", impact: "Edge deployment ready" },
    { Icon: BiotechRounded, title: "New Material Discovery", color: "#f472b6", text: "GRU's cross-material transfer to NM4 (R²=0.9751) demonstrates that recurrent models capture fundamental CV topology from ZnO-based electrodes. This enables model-assisted pre-screening of novel electrode compositions before physical synthesis.", impact: "Zero-shot new materials" },
    { Icon: FactoryRounded, title: "Lab Automation & High-Throughput Screening", color: "#f59e0b", text: "Automated electrode libraries can use predicted CVs for first-pass quality control and performance estimation. Only the top candidates proceed to physical electrochemical characterisation, reducing lab time and reagent costs.", impact: "Lab cost reduction" },
    { Icon: AutoGraphRounded, title: "Electrochemical Process Monitoring", color: "#10b981", text: "Comparing predicted vs. measured CVs during manufacturing provides real-time quality metrics. Deviation from predicted morphology signals synthesis defects, degradation, or electrolyte contamination in production environments.", impact: "Real-time QC" },
    { Icon: MenuBookRounded, title: "Academic Research Support", color: "#3b82f6", text: "AI-predicted CVs can supplement experimental data in publications, fill missing scan rates in datasets, and provide theoretical reference curves for comparison studies. The platform supports reproducibility by making predictions openly available.", impact: "Research acceleration" },
  ];
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.35)", lineHeight: 1.8, fontSize: "0.82rem" }}>
        AI-driven CV prediction has direct applications across energy storage industries, reducing experiment time from days to milliseconds.
      </Typography>
      <motion.div variants={staggerContainer} initial="initial" animate="animate">
        <Grid container spacing={2.5}>
          {apps.map(({ Icon, title, color, text, impact }) => (
            <Grid item xs={12} sm={6} md={4} key={title}>
              <motion.div variants={cardVariants}>
                <Card sx={{ height: "100%", background: `linear-gradient(135deg, ${alpha(color, 0.07)} 0%, rgba(15,25,35,0.9) 100%)`, border: `1px solid ${alpha(color, 0.2)}`, borderTop: `3px solid ${color}` }}>
                  <CardContent sx={{ p: 2.25 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.25 }}>
                      <Box sx={{ width: 34, height: 34, borderRadius: "10px", background: alpha(color, 0.12), border: `1px solid ${alpha(color, 0.25)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <Icon sx={{ fontSize: 18, color }} />
                      </Box>
                      <Typography variant="subtitle2" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.82rem", lineHeight: 1.3 }}>{title}</Typography>
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.7rem", lineHeight: 1.65, display: "block", mb: 1.25 }}>{text}</Typography>
                    <Chip label={impact} size="small" sx={{ fontSize: "0.6rem", height: 18, backgroundColor: alpha(color, 0.1), color, border: `1px solid ${alpha(color, 0.25)}` }} />
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </motion.div>
    </Box>
  );
}

function M16_Roadmap() {
  const items = [
    { icon: "🧪", title: "Multi-Electrolyte Support", color: "#22d3ee", desc: "Extend the dataset and models to support KOH, LiOH, and organic electrolytes. This would require new experimental data but dramatically expand applicability beyond the current 1 M Na₂SO₄ system." },
    { icon: "📡", title: "EIS & GCD Prediction", color: "#10b981", desc: "Train separate models to predict Electrochemical Impedance Spectroscopy (EIS) spectra and galvanostatic charge-discharge curves — the two other major electrochemical characterisation techniques." },
    { icon: "🔄", title: "Degradation Modelling", color: "#a78bfa", desc: "Incorporate cycle-count as a feature. Train on time-series CV data from aging experiments to predict how electrode performance evolves over 10,000+ charge-discharge cycles." },
    { icon: "🌐", title: "Public REST API", color: "#f59e0b", desc: "Expose a public API for programmatic predictions. Researchers could integrate AI-predicted CVs into their own data pipelines and analysis notebooks." },
    { icon: "🔬", title: "More Electrode Samples", color: "#f472b6", desc: "Fabricate and measure additional ZnO-based electrode groups (NM5–NM10) with varying synthesis conditions. More training materials = better zero-shot generalisation for truly novel electrode compositions." },
    { icon: "🧠", title: "Physics-Informed Neural Networks", color: "#3b82f6", desc: "Incorporate Randles–Ševčík and Butler–Volmer equations directly into the loss function. This would constrain predictions to be physically plausible and improve extrapolation at extreme scan rates." },
  ];
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        Planned research directions that would extend the platform's scientific scope, generalisability, and real-world utility.
      </Typography>
      <Grid container spacing={2}>
        {items.map(({ icon, title, color, desc }) => (
          <Grid item xs={12} sm={6} key={title}>
            <Box sx={{ p: 2, borderRadius: 2, background: alpha(color, 0.05), border: `1px solid ${alpha(color, 0.2)}`, display: "flex", gap: 1.5 }}>
              <Typography sx={{ fontSize: 22, flexShrink: 0 }}>{icon}</Typography>
              <Box>
                <Typography variant="subtitle2" sx={{ color, fontWeight: 700, fontSize: "0.8rem", mb: 0.5 }}>{title}</Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.7rem", lineHeight: 1.65, display: "block" }}>{desc}</Typography>
              </Box>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}

function M17_Glossary() {
  const existing15 = [
    { key: "cv", color: "#22d3ee" }, { key: "scan_rate", color: "#10b981" }, { key: "anodic", color: "#10b981" },
    { key: "cathodic", color: "#f472b6" }, { key: "edlc", color: "#a78bfa" }, { key: "faradaic", color: "#f59e0b" },
    { key: "symmetry_factor", color: "#22d3ee" }, { key: "capacitance", color: "#fb923c" },
    { key: "charge_storage_index", color: "#00d4ff" }, { key: "rmse", color: "#3b82f6" }, { key: "r2", color: "#10b981" },
    { key: "zero_shot", color: "#f472b6" }, { key: "integral_area", color: "#a78bfa" },
    { key: "peak_separation", color: "#22d3ee" }, { key: "nanocomposite", color: "#f59e0b" },
  ];
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.85 }}>
        {existing15.length + EXTRA_TERMS.length} terms covering electrochemistry, machine learning, and platform-specific concepts.
        Hover any underlined term throughout the platform for instant definitions.
      </Typography>
      {/* Existing 15 terms from GLOSSARY */}
      <Typography variant="caption" sx={{ color: "#22d3ee", fontWeight: 700, fontSize: "0.68rem", letterSpacing: "0.06em" }}>ELECTROCHEMISTRY & ML CORE TERMS</Typography>
      <motion.div variants={staggerContainer} initial="initial" animate="animate">
        <Grid container spacing={1.5}>
          {existing15.map(({ key, color }) => {
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
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.67rem", lineHeight: 1.6, display: "block" }}>{entry.short}</Typography>
                  </Box>
                </motion.div>
              </Grid>
            );
          })}
        </Grid>
      </motion.div>
      {/* Extra terms */}
      <Typography variant="caption" sx={{ color: "#a78bfa", fontWeight: 700, fontSize: "0.68rem", letterSpacing: "0.06em", mt: 1 }}>EXTENDED REFERENCE TERMS</Typography>
      <Grid container spacing={1.5}>
        {EXTRA_TERMS.map(({ term, short, color }) => (
          <Grid item xs={12} sm={6} md={4} key={term}>
            <Box sx={{ p: 1.75, borderRadius: 2, background: alpha(color, 0.04), border: `1px solid ${alpha(color, 0.14)}`, height: "100%" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.75 }}>
                <Box sx={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: color, flexShrink: 0 }} />
                <Typography variant="caption" sx={{ color, fontWeight: 700, fontSize: "0.72rem" }}>{term}</Typography>
              </Box>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.67rem", lineHeight: 1.6, display: "block" }}>{short}</Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registry
// ─────────────────────────────────────────────────────────────────────────────
interface ModuleDef {
  id: string; title: string; color: string; tag: string;
  Icon: React.ElementType; Component: React.FC;
}

const MODULES: ModuleDef[] = [
  { id: "welcome",          title: "Welcome to the Platform",             color: "#a78bfa", tag: "Introduction",    Icon: HomeRounded,          Component: M01_Welcome },
  { id: "problem",          title: "The Problem We Solve",                color: "#f59e0b", tag: "Research Context", Icon: LightbulbRounded,     Component: M02_Problem },
  { id: "electrochemistry", title: "Electrochemistry Fundamentals",       color: "#22d3ee", tag: "Science",          Icon: ScienceRounded,       Component: M03_Electrochemistry },
  { id: "supercapacitors",  title: "Supercapacitors & ZnO Materials",    color: "#10b981", tag: "Materials Science", Icon: FlashOnRounded,       Component: M04_Supercapacitors },
  { id: "cv_course",        title: "Cyclic Voltammetry — Full Course",    color: "#22d3ee", tag: "Technique",        Icon: ShowChartRounded,     Component: M05_CVCourse },
  { id: "interactive_cv",   title: "Interactive CV Graph Guide",          color: "#a78bfa", tag: "Interactive",      Icon: TouchAppRounded,      Component: M06_InteractiveCV },
  { id: "mathematics",      title: "Mathematics & Formulas",              color: "#f472b6", tag: "Math",             Icon: FunctionsRounded,     Component: M07_Mathematics },
  { id: "dataset",          title: "Dataset Construction & Statistics",   color: "#10b981", tag: "Data",             Icon: DatasetRounded,       Component: M08_Dataset },
  { id: "features",         title: "Feature Engineering",                 color: "#fb923c", tag: "ML",              Icon: TuneRounded,          Component: M09_Features },
  { id: "models",           title: "Model Explanations",                  color: "#3b82f6", tag: "ML",              Icon: PsychologyRounded,    Component: M10_Models },
  { id: "training",         title: "Model Training & Evaluation",         color: "#a78bfa", tag: "Methodology",     Icon: SchoolRounded,        Component: M11_Training },
  { id: "architecture",     title: "Application Architecture",            color: "#22d3ee", tag: "System",          Icon: AccountTreeRounded,   Component: M12_Architecture },
  { id: "walkthrough",      title: "How to Use This Application",         color: "#10b981", tag: "Guide",           Icon: RouteRounded,         Component: M13_Walkthrough },
  { id: "limitations",      title: "Research Limitations",                color: "#f59e0b", tag: "Transparency",    Icon: WarningAmberRounded,  Component: M14_Limitations },
  { id: "applications",     title: "Real-World Applications & Impact",    color: "#10b981", tag: "Impact",          Icon: RocketLaunchRounded,  Component: M15_Applications },
  { id: "roadmap",          title: "Future Research Roadmap",             color: "#fb923c", tag: "Future",          Icon: ExploreRounded,       Component: M16_Roadmap },
  { id: "glossary",         title: "Complete Electrochemistry Glossary",  color: "#22d3ee", tag: "Reference",       Icon: MenuBookRounded,      Component: M17_Glossary },
];

// ─────────────────────────────────────────────────────────────────────────────
// Main About component
// ─────────────────────────────────────────────────────────────────────────────
export default function About() {
  const [expanded, setExpanded] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  const toggle = (id: string) =>
    setExpanded((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);

  const suggestions = search.trim()
    ? MODULES.filter((m) => {
        const q = search.toLowerCase();
        return m.title.toLowerCase().includes(q) || m.tag.toLowerCase().includes(q) || m.id.includes(q);
      })
    : MODULES;

  const handleJump = (id: string) => {
    setExpanded((prev) => prev.includes(id) ? prev : [...prev, id]);
    setSearch("");
    setDropdownOpen(false);
    setTimeout(() => {
      document.getElementById(`module-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 150);
  };

  return (
      <Box sx={{ width: "100%", overflowX: "hidden" }}>
        <SectionHeader
          title="About This Research"
          subtitle="Interactive Documentation — 17 modules covering the science, data, models, and implementation behind ZnO supercapacitor CV prediction"
          accent="#a78bfa"
        />

        {/* Search bar with jump-to dropdown */}
        <Box sx={{ mb: 3, maxWidth: 480, position: "relative" }} ref={searchRef}>
          <ClickAwayListener onClickAway={() => setDropdownOpen(false)}>
            <Box>
              <TextField
                value={search}
                onChange={(e) => { setSearch(e.target.value); setDropdownOpen(true); }}
                onFocus={() => setDropdownOpen(true)}
                placeholder="Jump to a module…"
                size="small"
                fullWidth
                InputProps={{ startAdornment: <InputAdornment position="start"><SearchRounded sx={{ color: "rgba(255,255,255,0.3)", fontSize: 18 }} /></InputAdornment> }}
                sx={{
                  "& .MuiOutlinedInput-root": {
                    color: "#f1f5f9", background: "rgba(255,255,255,0.04)",
                    "& fieldset": { borderColor: "rgba(255,255,255,0.1)" },
                    "&:hover fieldset": { borderColor: "rgba(167,139,250,0.3)" },
                    "&.Mui-focused fieldset": { borderColor: "#a78bfa" },
                  },
                  "& input::placeholder": { color: "rgba(255,255,255,0.25)" },
                }}
              />
              {dropdownOpen && (
                <Paper
                  elevation={8}
                  sx={{
                    position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, zIndex: 1400,
                    background: "#0d1321", border: "1px solid rgba(167,139,250,0.25)", borderRadius: "10px",
                    overflow: "hidden",
                  }}
                >
                  <Box sx={{ px: 2, py: 1, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", fontWeight: 700, letterSpacing: "0.07em" }}>
                      {search.trim() ? `${suggestions.length} RESULT${suggestions.length !== 1 ? "S" : ""}` : "ALL MODULES — CLICK TO JUMP"}
                    </Typography>
                  </Box>
                  <Box sx={{ maxHeight: 360, overflowY: "auto", py: 0.5,
                    "&::-webkit-scrollbar": { width: 4 },
                    "&::-webkit-scrollbar-track": { background: "transparent" },
                    "&::-webkit-scrollbar-thumb": { background: "rgba(167,139,250,0.3)", borderRadius: 2 },
                  }}>
                    {suggestions.length > 0 ? suggestions.map((mod) => (
                      <Box
                        key={mod.id}
                        onClick={() => handleJump(mod.id)}
                        sx={{
                          px: 2, py: 1.1, display: "flex", alignItems: "center", gap: 1.5, cursor: "pointer",
                          "&:hover": { background: "rgba(167,139,250,0.08)" },
                          transition: "background 0.12s",
                        }}
                      >
                        <Box sx={{ width: 28, height: 28, borderRadius: "8px", background: alpha(mod.color, 0.12), border: `1px solid ${alpha(mod.color, 0.25)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                          <mod.Icon sx={{ fontSize: 15, color: mod.color }} />
                        </Box>
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Typography sx={{ color: "#f1f5f9", fontSize: "0.8rem", fontWeight: 600, lineHeight: 1.3 }}>{mod.title}</Typography>
                          <Typography variant="caption" sx={{ color: mod.color, fontSize: "0.62rem" }}>{mod.tag}</Typography>
                        </Box>
                      </Box>
                    )) : (
                      <Box sx={{ px: 2, py: 2, textAlign: "center" }}>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.72rem" }}>No modules match "{search}"</Typography>
                      </Box>
                    )}
                  </Box>
                </Paper>
              )}
            </Box>
          </ClickAwayListener>
        </Box>

        {/* Module accordions — all 17 always visible */}
        {MODULES.map((mod) => {
          const isOpen = expanded.includes(mod.id);
          return (
            <Accordion
              key={mod.id}
              id={`module-${mod.id}`}
              disableGutters
              elevation={0}
              expanded={isOpen}
              onChange={() => toggle(mod.id)}
              sx={{
                background: isOpen
                  ? `linear-gradient(135deg, ${alpha(mod.color, 0.07)} 0%, rgba(10,16,32,0.92) 100%)`
                  : "rgba(10,16,32,0.82)",
                border: `1px solid ${isOpen ? alpha(mod.color, 0.38) : alpha(mod.color, 0.14)}`,
                borderRadius: "12px !important",
                mb: 1.5,
                transition: "all 0.18s ease",
                "&:before": { display: "none" },
                "&.Mui-expanded": { margin: "0 0 12px 0" },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreRounded sx={{ color: mod.color, fontSize: 20 }} />}
                sx={{
                  px: 2.5, minHeight: 56,
                  "& .MuiAccordionSummary-content": { my: 1.25 },
                  "& .MuiAccordionSummary-content.Mui-expanded": { my: 1.25 },
                }}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, width: "100%" }}>
                  <Box sx={{ width: 36, height: 36, borderRadius: "10px", background: alpha(mod.color, 0.12), border: `1px solid ${alpha(mod.color, 0.25)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <mod.Icon sx={{ fontSize: 18, color: mod.color }} />
                  </Box>
                  <Typography variant="subtitle1" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.9rem", flex: 1, lineHeight: 1.3 }}>
                    {mod.title}
                  </Typography>
                  <Chip label={mod.tag} size="small" sx={{ fontSize: "0.58rem", height: 18, backgroundColor: alpha(mod.color, 0.1), color: mod.color, border: `1px solid ${alpha(mod.color, 0.25)}`, mr: 0.5, flexShrink: 0 }} />
                </Box>
              </AccordionSummary>
              <AccordionDetails sx={{ px: 2.5, pt: 0, pb: 2.5 }}>
                <mod.Component />
              </AccordionDetails>
            </Accordion>
          );
        })}
      </Box>
  );
}
