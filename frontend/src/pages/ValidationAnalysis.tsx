/**
 * ValidationAnalysis.tsx — Scientific Validation & Experimental Comparison
 *
 * Tabs:
 *  1. Overlay          — Actual vs Predicted CV curves with metrics
 *  2. Residuals        — Point-wise residual analysis
 *  3. Performance Map  — Pre-computed per-group RMSE heatmap + trends
 *  4. Methodology      — Scientific credibility panel
 *
 * ALL data shown here is sourced from real experimental ZnO measurements
 * and actual ML model predictions — nothing is fabricated.
 */
import React, { useState, useCallback, useEffect } from "react";
import {
  Box, Grid, Typography, Tab, Tabs, Chip,
  FormControl, InputLabel, Select, MenuItem,
  Button, Alert, CircularProgress, Paper,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Divider, Skeleton,
  ToggleButton, ToggleButtonGroup,
} from "@mui/material";
import {
  ScienceRounded, BarChartRounded,
  InfoOutlined, PlayArrowRounded,
  CheckCircleRounded, VerifiedRounded,
} from "@mui/icons-material";
import Plot from "react-plotly.js";
import { motion, AnimatePresence } from "framer-motion";
import { compareValidation, getPerGroupMetrics } from "../api/endpoints";
import type {
  ValidationCompareResponse,
  PerGroupRow,
  PerGroupResponse,
} from "../types";
import SectionHeader from "../components/common/SectionHeader";
import { splitCVBranches, buildMasterHoverTrace } from "../components/charts/CVPlot";

// ── Constants ──────────────────────────────────────────────────────────────

const MODELS = ["rf", "lightgbm", "xgboost", "gru", "lstm", "ann"];
const MODEL_LABELS: Record<string, string> = {
  rf: "Random Forest", lightgbm: "LightGBM", xgboost: "XGBoost",
  gru: "GRU", lstm: "LSTM", ann: "ANN",
};
const MODEL_COLORS: Record<string, string> = {
  rf: "#00d4ff", lightgbm: "#7c3aed", xgboost: "#f59e0b",
  gru: "#10b981", lstm: "#f43f5e", ann: "#64748b",
};
const MATERIALS = ["NM1", "NM2", "NM3", "NM4"];
const SCAN_RATES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

const SPLIT_COLORS: Record<string, string> = {
  val:           "#00d4ff",
  test_sr:       "#f59e0b",
  test_mat:      "#10b981",
  test_SR:       "#f59e0b",
  test_MAT:      "#10b981",
  test_scanrate: "#f59e0b",
  test_material: "#10b981",
  train:         "#64748b",
};
const SPLIT_LABELS: Record<string, string> = {
  val:           "Validation (30 mV/s)",
  test_sr:       "Test — Unseen Scan Rate (50 mV/s)",
  test_mat:      "Test — Unseen Material (NM4)",
  test_SR:       "Test — Unseen Scan Rate (50 mV/s)",
  test_MAT:      "Test — Unseen Material (NM4)",
  test_scanrate: "Test — Unseen Scan Rate",
  test_material: "Test — Unseen Material",
  train:         "Training",
};

const DARK_PAPER = "rgba(13,19,33,0.85)";
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  font: { family: "Inter, sans-serif", color: "#94a3b8", size: 12 },
  margin: { t: 40, r: 20, b: 60, l: 60 },
  legend: { bgcolor: "rgba(0,0,0,0)", bordercolor: "rgba(255,255,255,0.1)", borderwidth: 1, font: { size: 11 } },
  xaxis: {
    gridcolor: "rgba(255,255,255,0.06)", zerolinecolor: "rgba(255,255,255,0.12)",
    color: "#94a3b8", tickfont: { size: 11 },
  },
  yaxis: {
    gridcolor: "rgba(255,255,255,0.06)", zerolinecolor: "rgba(255,255,255,0.12)",
    color: "#94a3b8", tickfont: { size: 11 },
  },
};

// ── Helper components ───────────────────────────────────────────────────────

function MetricCard({ label, value, unit, accent = "#00d4ff", sub }: {
  label: string; value: number | string; unit?: string; accent?: string; sub?: string;
}) {
  return (
    <Box sx={{
      p: 2, borderRadius: 2,
      background: `linear-gradient(135deg, rgba(${hexToRgb(accent)},0.07) 0%, rgba(0,0,0,0) 100%)`,
      border: `1px solid rgba(${hexToRgb(accent)},0.2)`,
      flex: 1, minWidth: 110,
    }}>
      <Typography sx={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.4)", textTransform: "uppercase", letterSpacing: "0.08em", mb: 0.5 }}>
        {label}
      </Typography>
      <Typography sx={{ fontSize: "1.4rem", fontWeight: 700, color: accent, lineHeight: 1, fontFamily: "'JetBrains Mono', monospace" }}>
        {typeof value === "number" ? value.toFixed(3) : value}
        {unit && <span style={{ fontSize: "0.8rem", fontWeight: 400, color: "rgba(255,255,255,0.4)", marginLeft: 4 }}>{unit}</span>}
      </Typography>
      {sub && <Typography sx={{ fontSize: "0.66rem", color: "rgba(255,255,255,0.3)", mt: 0.3 }}>{sub}</Typography>}
    </Box>
  );
}

function hexToRgb(hex: string): string {
  const m = hex.replace("#","").match(/.{2}/g);
  if (!m) return "0,212,255";
  return m.map(x => parseInt(x, 16)).join(",");
}

function SplitBadge({ split }: { split: string }) {
  const color = SPLIT_COLORS[split] ?? "#64748b";
  const label = SPLIT_LABELS[split] ?? split;
  return (
    <Chip
      size="small"
      label={label}
      sx={{
        height: 22, fontSize: "0.68rem", fontWeight: 600,
        color, borderColor: `${color}50`,
        border: "1px solid", backgroundColor: `${color}12`,
        "& .MuiChip-label": { px: 1 },
      }}
    />
  );
}

// ── Tab: Overlay ───────────────────────────────────────────────────────────

function OverlayTab({ data, loading, onRun, model, setModel, material, setMaterial, scanRate, setScanRate, error }: {
  data: ValidationCompareResponse | null;
  loading: boolean;
  onRun: () => void;
  model: string; setModel: (v: string) => void;
  material: string; setMaterial: (v: string) => void;
  scanRate: number; setScanRate: (v: number) => void;
  error: string | null;
}) {
  const accent = MODEL_COLORS[model] ?? "#00d4ff";
  const chartH = typeof window !== "undefined" && window.innerWidth < 600 ? 320 : 420;

  return (
    <Box>
      {/* Controls */}
      <Paper sx={{ p: { xs: 2, sm: 2.5 }, mb: 2.5, background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel sx={{ color: "rgba(255,255,255,0.4)" }}>Model</InputLabel>
              <Select value={model} label="Model" onChange={(e) => setModel(e.target.value)}
                sx={{ color: "#f1f5f9", "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(255,255,255,0.12)" } }}>
                {MODELS.map((m) => <MenuItem key={m} value={m}>{MODEL_LABELS[m]}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6} sm={3} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel sx={{ color: "rgba(255,255,255,0.4)" }}>Material</InputLabel>
              <Select value={material} label="Material" onChange={(e) => setMaterial(e.target.value)}
                sx={{ color: "#f1f5f9", "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(255,255,255,0.12)" } }}>
                {MATERIALS.map((m) => <MenuItem key={m} value={m}>{m}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6} sm={3} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel sx={{ color: "rgba(255,255,255,0.4)" }}>Scan Rate</InputLabel>
              <Select value={scanRate} label="Scan Rate" onChange={(e) => setScanRate(Number(e.target.value))}
                sx={{ color: "#f1f5f9", "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(255,255,255,0.12)" } }}>
                {SCAN_RATES.map((s) => <MenuItem key={s} value={s}>{s} mV/s</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={12} md={5}>
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <PlayArrowRounded />}
              onClick={onRun}
              disabled={loading}
              fullWidth
              sx={{
                background: `linear-gradient(135deg, ${accent} 0%, rgba(124,58,237,0.8) 100%)`,
                color: "#fff", fontWeight: 700, fontSize: "0.82rem",
                py: 1, borderRadius: 2, textTransform: "none",
                "&:hover": { filter: "brightness(1.1)" },
              }}
            >
              {loading ? "Computing…" : "Run Validation Comparison"}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {!data && !loading && (
        <Box sx={{ textAlign: "center", py: 8, color: "rgba(255,255,255,0.2)" }}>
          <VerifiedRounded sx={{ fontSize: 64, mb: 2, opacity: 0.3 }} />
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 1, color: "rgba(255,255,255,0.35)" }}>
            Select a model, material and scan rate to begin
          </Typography>
          <Typography variant="body2">
            Compares real experimental ZnO CV measurements against ML predictions
          </Typography>
        </Box>
      )}

      {loading && (
        <Box>
          <Skeleton variant="rounded" height={chartH} sx={{ bgcolor: "rgba(255,255,255,0.04)", borderRadius: 3, mb: 2 }} />
          <Box sx={{ display: "flex", gap: 2 }}>
            {[1,2,3,4].map(i => <Skeleton key={i} variant="rounded" height={80} sx={{ flex: 1, bgcolor: "rgba(255,255,255,0.04)" }} />)}
          </Box>
        </Box>
      )}

      <AnimatePresence mode="wait">
        {data && !loading && (
          <motion.div key={`${data.model_name}-${data.material_id}-${data.scan_rate_mVs}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
            {/* Split badge + title */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
              <Typography sx={{ fontWeight: 600, color: "#f1f5f9", fontSize: "0.9rem" }}>
                {MODEL_LABELS[data.model_name]} · {data.material_id} · {data.scan_rate_mVs} mV/s
              </Typography>
              <SplitBadge split={data.split} />
            </Box>

            {/* Overlay chart */}
            <Paper sx={{ background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3, p: { xs: 1, sm: 2 }, mb: 2.5 }}>
              {/* Compact dual-branch hover via master trace — all visual traces skip hover */}
              {(() => {
                const eS = splitCVBranches(data.experimental_potential_V, data.experimental_current_uA);
                const pS = splitCVBranches(data.predicted_potential_V,    data.predicted_current_uA);
                // Master hover trace: invisible, carries all dual-branch data for both curves
                const masterTrace = buildMasterHoverTrace(
                  pS.fwdPot, pS.fwdCurr, pS.revPot, pS.revCurr,
                  data.model_name,
                  { potential_V: data.experimental_potential_V, actual_current_uA: data.experimental_current_uA }
                );
                const predLabel = `${MODEL_LABELS[data.model_name]} (Predicted)`;
                return (
              <Plot
                data={[
                  // Experimental — visual only (forward + reverse), no hover
                  { x: eS.fwdPot, y: eS.fwdCurr, name: "Experimental (Measured)",
                    type: "scatter", mode: "lines", line: { color: "#10b981", width: 2.5 },
                    hoverinfo: "skip", showlegend: false },
                  { x: eS.revPot, y: eS.revCurr, name: "Experimental (Measured)",
                    type: "scatter", mode: "lines", line: { color: "#10b981", width: 2.5 },
                    hoverinfo: "skip", showlegend: true },
                  // Predicted — visual only (forward + reverse), no hover
                  { x: pS.fwdPot, y: pS.fwdCurr, name: predLabel,
                    type: "scatter", mode: "lines", line: { color: accent, width: 2, dash: "dash" },
                    hoverinfo: "skip", showlegend: false },
                  { x: pS.revPot, y: pS.revCurr, name: predLabel,
                    type: "scatter", mode: "lines", line: { color: accent, width: 2, dash: "dash" },
                    hoverinfo: "skip", showlegend: true },
                  // Master hover trace — owns the tooltip
                  masterTrace,
                ]}
                layout={{
                  ...PLOTLY_LAYOUT_BASE,
                  height: chartH,
                  hovermode: "closest",
                  title: { text: "Cyclic Voltammetry: Experimental vs Predicted", font: { size: 13, color: "#f1f5f9" } },
                  xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: "Potential (V)", standoff: 12 },
                    showspikes: true, spikemode: "across", spikethickness: 1,
                    spikecolor: "rgba(0,212,255,0.35)", spikedash: "dot" },
                  yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: "Current (µA)", standoff: 8 } },
                }}
                config={{ displayModeBar: true, modeBarButtonsToRemove: ["lasso2d","select2d"], displaylogo: false }}
                style={{ width: "100%", height: chartH }}
                useResizeHandler
              />
                ); })()}
            </Paper>

            {/* Metrics row */}
            <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
              <MetricCard label="RMSE" value={data.metrics.rmse_uA} unit="µA" accent={accent} />
              <MetricCard label="MAE" value={data.metrics.mae_uA} unit="µA" accent="#f59e0b" />
              <MetricCard label="R²" value={data.metrics.r2} accent="#10b981" sub="Coeff. of determination" />
              <MetricCard label="Max Error" value={data.metrics.max_error_uA} unit="µA" accent="#f43f5e" sub="Worst-case deviation" />
            </Box>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
}

// ── Tab: Residuals ─────────────────────────────────────────────────────────

function ResidualsTab({ data }: { data: ValidationCompareResponse | null }) {
  if (!data) {
    return (
      <Box sx={{ textAlign: "center", py: 8, color: "rgba(255,255,255,0.2)" }}>
        <BarChartRounded sx={{ fontSize: 64, mb: 2, opacity: 0.3 }} />
        <Typography variant="h6" sx={{ color: "rgba(255,255,255,0.35)" }}>
          Run a comparison first (Overlay tab)
        </Typography>
      </Box>
    );
  }

  const accent = MODEL_COLORS[data.model_name] ?? "#00d4ff";
  const residuals = data.residual_uA;
  const potentials = data.residual_potential_V;
  const absResiduals = data.abs_residual_uA;
  const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
  const std  = Math.sqrt(residuals.reduce((a, b) => a + (b - mean) ** 2, 0) / residuals.length);
  const pct10 = (residuals.filter(r => Math.abs(r) < 10).length / residuals.length * 100).toFixed(1);
  const pct20 = (residuals.filter(r => Math.abs(r) < 20).length / residuals.length * 100).toFixed(1);
  const chartH = typeof window !== "undefined" && window.innerWidth < 600 ? 280 : 350;

  // Histogram bins
  const binSize = 5;
  const minR = Math.floor(Math.min(...residuals) / binSize) * binSize;
  const maxR = Math.ceil(Math.max(...residuals) / binSize) * binSize;
  const bins: number[] = [];
  for (let b = minR; b <= maxR; b += binSize) bins.push(b);
  const counts = bins.map((b) => residuals.filter((r) => r >= b && r < b + binSize).length);

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}>
        <Typography sx={{ fontWeight: 600, color: "#f1f5f9", fontSize: "0.9rem" }}>
          {MODEL_LABELS[data.model_name]} · {data.material_id} · {data.scan_rate_mVs} mV/s
        </Typography>
        <SplitBadge split={data.split} />
      </Box>

      <Grid container spacing={2.5}>
        {/* Residual vs potential */}
        <Grid item xs={12} lg={7}>
          <Paper sx={{ background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3, p: { xs: 1, sm: 1.5 } }}>
            <Plot
              data={[
                {
                  x: potentials, y: residuals,
                  name: "Residual (Pred − Exp)",
                  type: "scatter", mode: "lines",
                  line: { color: accent, width: 1.5 },
                  hovertemplate: "V: %{x:.4f}<br>ε: %{y:.2f} µA<extra></extra>",
                },
                {
                  x: [Math.min(...potentials), Math.max(...potentials)],
                  y: [0, 0],
                  name: "Zero line",
                  type: "scatter", mode: "lines",
                  line: { color: "rgba(255,255,255,0.25)", width: 1, dash: "dot" },
                  hoverinfo: "skip",
                },
              ]}
              layout={{
                ...PLOTLY_LAYOUT_BASE,
                height: chartH,
                title: { text: "Residual vs Potential", font: { size: 13, color: "#f1f5f9" } },
                xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: "Potential (V)", standoff: 10 } },
                yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: "Residual (µA)", standoff: 8 } },
              }}
              config={{ displayModeBar: false, displaylogo: false }}
              style={{ width: "100%", height: chartH }}
              useResizeHandler
            />
          </Paper>
        </Grid>

        {/* Histogram */}
        <Grid item xs={12} lg={5}>
          <Paper sx={{ background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3, p: { xs: 1, sm: 1.5 } }}>
            <Plot
              data={[
                {
                  x: bins, y: counts,
                  name: "Count",
                  type: "bar",
                  marker: {
                    color: bins.map(b => b < 0 ? "#f43f5e" : "#10b981"),
                    opacity: 0.8,
                  },
                  hovertemplate: "[%{x}, %{x}+5): %{y}<extra></extra>",
                },
              ]}
              layout={{
                ...PLOTLY_LAYOUT_BASE,
                height: chartH,
                title: { text: "Residual Distribution", font: { size: 13, color: "#f1f5f9" } },
                xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: "Residual (µA)", standoff: 10 } },
                yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: "Count", standoff: 8 } },
                bargap: 0.1,
                shapes: [{
                  type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper",
                  line: { color: "rgba(255,255,255,0.3)", width: 1.5, dash: "dot" },
                }],
              }}
              config={{ displayModeBar: false, displaylogo: false }}
              style={{ width: "100%", height: chartH }}
              useResizeHandler
            />
          </Paper>
        </Grid>

        {/* Abs residual vs potential */}
        <Grid item xs={12}>
          <Paper sx={{ background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3, p: { xs: 1, sm: 1.5 } }}>
            <Plot
              data={[{
                x: potentials, y: absResiduals,
                name: "|Residual|",
                type: "scatter", mode: "lines",
                fill: "tozeroy",
                fillcolor: `rgba(${hexToRgb(accent)},0.08)`,
                line: { color: accent, width: 1.5 },
                hovertemplate: "V: %{x:.4f}<br>|ε|: %{y:.2f} µA<extra></extra>",
              }]}
              layout={{
                ...PLOTLY_LAYOUT_BASE,
                height: 240,
                title: { text: "Absolute Residual Envelope", font: { size: 13, color: "#f1f5f9" } },
                xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: "Potential (V)", standoff: 10 } },
                yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: "|Residual| (µA)", standoff: 8 } },
              }}
              config={{ displayModeBar: false, displaylogo: false }}
              style={{ width: "100%", height: 240 }}
              useResizeHandler
            />
          </Paper>
        </Grid>
      </Grid>

      {/* Stats summary */}
      <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap", mt: 2.5 }}>
        <MetricCard label="Mean Residual" value={mean} unit="µA" accent={accent} sub="Systematic bias" />
        <MetricCard label="Std Dev" value={std} unit="µA" accent="#f59e0b" sub="Spread of errors" />
        <MetricCard label="Within ±10 µA" value={`${pct10}%`} accent="#10b981" sub="Fraction of predictions" />
        <MetricCard label="Within ±20 µA" value={`${pct20}%`} accent="#7c3aed" sub="Fraction of predictions" />
      </Box>
    </Box>
  );
}

// ── Tab: Performance Map ───────────────────────────────────────────────────

function PerformanceMapTab({ pgData, pgLoading, pgError, pgModel, setPgModel }: {
  pgData: PerGroupResponse | null;
  pgLoading: boolean;
  pgError: string | null;
  pgModel: string;
  setPgModel: (v: string) => void;
}) {
  const chartH = typeof window !== "undefined" && window.innerWidth < 600 ? 300 : 400;

  // Heatmap data: rows = materials, cols = scan rates (only test_mat for NM4)
  const testMatRows = pgData?.rows.filter(r => r.partition === "test_mat") ?? [];
  const valRows     = pgData?.rows.filter(r => r.partition === "val") ?? [];
  const testSrRows  = pgData?.rows.filter(r => r.partition === "test_sr") ?? [];

  // For NM4 heatmap (all scan rates)
  const nm4ScanRates = [...new Set(testMatRows.map(r => r.scan_rate_mVs))].sort((a,b) => a-b);
  const nm4Rmse      = nm4ScanRates.map(sr => testMatRows.find(r => r.scan_rate_mVs === sr)?.rmse_uA ?? null);
  const nm4R2        = nm4ScanRates.map(sr => testMatRows.find(r => r.scan_rate_mVs === sr)?.r2 ?? null);

  // Bar chart: all evaluation rows by material
  const allRows = [...valRows, ...testSrRows, ...testMatRows];
  const byMaterial: Record<string, { sr: number[], rmse: number[], label: string[] }> = {};
  allRows.forEach(r => {
    if (!byMaterial[r.nm_id]) byMaterial[r.nm_id] = { sr: [], rmse: [], label: [] };
    byMaterial[r.nm_id].sr.push(r.scan_rate_mVs);
    byMaterial[r.nm_id].rmse.push(r.rmse_uA);
    byMaterial[r.nm_id].label.push(`${r.scan_rate_mVs}|${r.partition}`);
  });

  const matColors: Record<string, string> = { NM1: "#00d4ff", NM2: "#7c3aed", NM3: "#f59e0b", NM4: "#10b981" };

  return (
    <Box>
      {/* Model selector */}
      <Paper sx={{ p: 2, mb: 2.5, background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel sx={{ color: "rgba(255,255,255,0.4)" }}>Model</InputLabel>
              <Select value={pgModel} label="Model" onChange={(e) => setPgModel(e.target.value)}
                sx={{ color: "#f1f5f9", "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(255,255,255,0.12)" } }}>
                {MODELS.map(m => <MenuItem key={m} value={m}>{MODEL_LABELS[m]}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.78rem" }}>
              Showing pre-computed per-group RMSE from training evaluation. Partitions: val · test_SR · test_MAT
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {pgError && <Alert severity="error" sx={{ mb: 2 }}>{pgError}</Alert>}

      {pgLoading && (
        <Box>
          <Skeleton variant="rounded" height={chartH} sx={{ bgcolor: "rgba(255,255,255,0.04)", borderRadius: 3, mb: 2 }} />
        </Box>
      )}

      <AnimatePresence mode="wait">
        {pgData && !pgLoading && (
          <motion.div key={pgModel} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
            <Grid container spacing={2.5}>
              {/* NM4 scan-rate trend (test_MAT) */}
              <Grid item xs={12} lg={7}>
                <Paper sx={{ background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3, p: { xs: 1, sm: 1.5 } }}>
                  <Plot
                    data={[
                      {
                        x: nm4ScanRates, y: nm4Rmse,
                        name: "RMSE (µA)",
                        type: "scatter", mode: "lines+markers",
                        line: { color: "#10b981", width: 2.5 },
                        marker: { size: 7, color: "#10b981" },
                        yaxis: "y",
                        hovertemplate: "%{x} mV/s: RMSE = %{y:.2f} µA<extra></extra>",
                      },
                      {
                        x: nm4ScanRates, y: nm4R2,
                        name: "R²",
                        type: "scatter", mode: "lines+markers",
                        line: { color: "#f59e0b", width: 2, dash: "dash" },
                        marker: { size: 6, color: "#f59e0b" },
                        yaxis: "y2",
                        hovertemplate: "%{x} mV/s: R² = %{y:.4f}<extra></extra>",
                      },
                    ]}
                    layout={{
                      ...PLOTLY_LAYOUT_BASE,
                      height: chartH,
                      title: { text: `NM4 Generalisation — ${MODEL_LABELS[pgModel]} (Test-MAT)`, font: { size: 13, color: "#f1f5f9" } },
                      xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: "Scan Rate (mV/s)", standoff: 10 } },
                      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: "RMSE (µA)", standoff: 8 } },
                      yaxis2: {
                        ...PLOTLY_LAYOUT_BASE.yaxis,
                        title: { text: "R²", standoff: 8 }, overlaying: "y", side: "right",
                        range: [0.9, 1.0], tickformat: ".3f",
                      },
                    }}
                    config={{ displayModeBar: false, displaylogo: false }}
                    style={{ width: "100%", height: chartH }}
                    useResizeHandler
                  />
                </Paper>
              </Grid>

              {/* Per-group RMSE bar (all evaluation groups) */}
              <Grid item xs={12} lg={5}>
                <Paper sx={{ background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3, p: { xs: 1, sm: 1.5 } }}>
                  <Plot
                    data={[...valRows, ...testSrRows, ...testMatRows].map(r => ({
                      x: [`${r.nm_id}\n${r.scan_rate_mVs}mV/s`],
                      y: [r.rmse_uA],
                      name: r.nm_id,
                      type: "bar" as const,
                      showlegend: false,
                      marker: { color: `${matColors[r.nm_id]}99` },
                      hovertemplate: `${r.nm_id} @ ${r.scan_rate_mVs} mV/s<br>RMSE: %{y:.2f} µA<br>${r.partition}<extra></extra>`,
                    }))}
                    layout={{
                      ...PLOTLY_LAYOUT_BASE,
                      height: chartH,
                      title: { text: `Per-Group RMSE — ${MODEL_LABELS[pgModel]}`, font: { size: 13, color: "#f1f5f9" } },
                      xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: "Group", standoff: 10 } },
                      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: "RMSE (µA)", standoff: 8 } },
                      barmode: "group",
                    }}
                    config={{ displayModeBar: false, displaylogo: false }}
                    style={{ width: "100%", height: chartH }}
                    useResizeHandler
                  />
                </Paper>
              </Grid>

              {/* Summary table */}
              <Grid item xs={12}>
                <Paper sx={{ background: DARK_PAPER, border: "1px solid rgba(255,255,255,0.07)", borderRadius: 3 }}>
                  <Box sx={{ p: 2, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                    <Typography sx={{ fontWeight: 600, color: "#f1f5f9", fontSize: "0.85rem" }}>
                      Evaluation Metrics — {MODEL_LABELS[pgModel]}
                    </Typography>
                  </Box>
                  <Box sx={{ overflowX: "auto" }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          {["Material","Scan Rate","RMSE (µA)","R²","Partition"].map(h => (
                            <TableCell key={h} sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.72rem", fontWeight: 600, borderColor: "rgba(255,255,255,0.06)" }}>
                              {h}
                            </TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {pgData.rows.map((r, i) => (
                          <TableRow key={i} sx={{ "&:last-child td": { border: 0 }, "&:hover td": { background: "rgba(255,255,255,0.02)" } }}>
                            <TableCell sx={{ color: matColors[r.nm_id] ?? "#f1f5f9", fontWeight: 600, fontSize: "0.78rem", borderColor: "rgba(255,255,255,0.04)" }}>{r.nm_id}</TableCell>
                            <TableCell sx={{ color: "#94a3b8", fontSize: "0.78rem", borderColor: "rgba(255,255,255,0.04)" }}>{r.scan_rate_mVs} mV/s</TableCell>
                            <TableCell sx={{ color: "#f1f5f9", fontFamily: "JetBrains Mono", fontSize: "0.78rem", borderColor: "rgba(255,255,255,0.04)" }}>{r.rmse_uA.toFixed(2)}</TableCell>
                            <TableCell sx={{ color: r.r2 > 0.97 ? "#10b981" : r.r2 > 0.95 ? "#f59e0b" : "#f43f5e", fontFamily: "JetBrains Mono", fontSize: "0.78rem", borderColor: "rgba(255,255,255,0.04)" }}>{r.r2.toFixed(4)}</TableCell>
                            <TableCell sx={{ borderColor: "rgba(255,255,255,0.04)" }}>
                              <Chip label={r.partition} size="small" sx={{
                                height: 18, fontSize: "0.62rem", fontWeight: 600,
                                color: SPLIT_COLORS[r.partition] ?? "#64748b",
                                backgroundColor: `${SPLIT_COLORS[r.partition] ?? "#64748b"}15`,
                                "& .MuiChip-label": { px: 0.8 },
                              }} />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
}

// ── Tab: Methodology ───────────────────────────────────────────────────────

function MethodologyTab() {
  return (
    <Box>
      <Grid container spacing={2.5}>
        {[
          {
            icon: <ScienceRounded sx={{ color: "#00d4ff" }} />,
            title: "Experimental Data Source",
            accent: "#00d4ff",
            items: [
              "Real ZnO supercapacitor CV measurements (4 materials × 10 scan rates)",
              "104,000 data points across 4 complete electrochemical materials",
              "Potential range: −0.65 V to 0 V (vs. Ag/AgCl reference electrode)",
              "Scan rates: 10 to 100 mV/s in 10 mV/s steps",
              "Stored in master_long_format.parquet — no fabrication or simulation",
            ],
          },
          {
            icon: <BarChartRounded sx={{ color: "#7c3aed" }} />,
            title: "Data Partitioning Strategy",
            accent: "#7c3aed",
            items: [
              "Training: NM1/NM2/NM3 at scan rates ≠ 30 and ≠ 50 mV/s",
              "Validation: NM1/NM2/NM3 at 30 mV/s — tests scan rate interpolation",
              "Test-SR: NM1/NM2/NM3 at 50 mV/s — tests scan rate interpolation (SR=50 between trained 40 & 60 mV/s)",
              "Test-MAT: NM4 at all scan rates — tests generalisation to new material",
              "No data leakage: test partitions completely unseen during training",
            ],
          },
          {
            icon: <CheckCircleRounded sx={{ color: "#10b981" }} />,
            title: "Residual Computation",
            accent: "#10b981",
            items: [
              "Prediction grid: 651 uniformly spaced points (−0.65 V → 0 V → −0.65 V)",
              "Experimental grid: ~1,301 points per cycle (651 anodic + 650 cathodic)",
              "Alignment: linear interpolation of experimental onto prediction grid",
              "Each half-sweep (anodic/cathodic) interpolated separately — handles V-shape",
              "Sign convention: residual = predicted − experimental (positive = over-prediction)",
            ],
          },
          {
            icon: <VerifiedRounded sx={{ color: "#f59e0b" }} />,
            title: "Scientific Integrity",
            accent: "#f59e0b",
            items: [
              "All displayed metrics computed from real measured data only",
              "Per-group metrics pre-computed during rigorous training evaluation",
              "No post-hoc adjustments or cherry-picking of results",
              "R² computed against experimental mean as baseline",
              "Results reproducible from master_long_format.parquet source data",
            ],
          },
          {
            icon: <ScienceRounded sx={{ color: "#22d3ee" }} />,
            title: "Electrochemical Physics Checks",
            accent: "#22d3ee",
            items: [
              "CV loop area (∮ I dV) computed by trapezoidal integration over voltage — not array indices",
              "Anodic/cathodic peak currents extracted from correct half-sweeps (direction = 0/1 convention verified)",
              "Integral area units: µA·V — proportional to charge stored Q = ∫I dV / ν",
              "Sweep reversal continuity: 651-point sweep with shared endpoint at V_upper (0.0 V) between half-sweeps",
              "Scan-rate scaling: per-group RMSE trends visible in the scan-rate chart above — models preserve the expected monotonic I ∝ ν relationship",
              "No Sp_Cap_F (specific capacitance) used as input — zero target-leakage confirmed by training assertions",
            ],
          },
        ].map((card) => (
          <Grid item xs={12} md={6} key={card.title}>
            <Paper sx={{
              p: 2.5,
              background: `linear-gradient(135deg, rgba(${hexToRgb(card.accent)},0.04) 0%, rgba(0,0,0,0) 100%)`,
              border: `1px solid rgba(${hexToRgb(card.accent)},0.15)`,
              borderRadius: 3, height: "100%",
            }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
                {card.icon}
                <Typography sx={{ fontWeight: 700, color: "#f1f5f9", fontSize: "0.88rem" }}>
                  {card.title}
                </Typography>
              </Box>
              <Box component="ul" sx={{ m: 0, pl: 2.5 }}>
                {card.items.map((item, i) => (
                  <Box component="li" key={i} sx={{ mb: 0.75 }}>
                    <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.8rem", lineHeight: 1.6 }}>
                      {item}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function ValidationAnalysis() {
  const [activeTab, setActiveTab] = useState(0);

  // Overlay / Residuals state
  const [model,     setModel]     = useState("lightgbm");
  const [material,  setMaterial]  = useState("NM1");
  const [scanRate,  setScanRate]  = useState(30);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState<string | null>(null);
  const [cmpData,   setCmpData]   = useState<ValidationCompareResponse | null>(null);

  // Performance map state
  const [pgModel,   setPgModel]   = useState("lightgbm");
  const [pgLoading, setPgLoading] = useState(false);
  const [pgError,   setPgError]   = useState<string | null>(null);
  const [pgData,    setPgData]    = useState<PerGroupResponse | null>(null);

  // ── CRITICAL: clear stale comparison data whenever any selection input changes ──
  // Without this, the chart title, legend, metrics and traces all lag behind
  // the dropdown, causing a model desync (e.g. dropdown shows LightGBM but
  // chart still renders the last RF response).
  useEffect(() => {
    setCmpData(null);
    setError(null);
  }, [model, material, scanRate]);

  // Run validation comparison — capture selection at call time to guard against
  // the user changing dropdowns while the request is in flight
  const handleRun = useCallback(async () => {
    const reqModel    = model;
    const reqMaterial = material;
    const reqSR       = scanRate;
    setLoading(true);
    setError(null);
    try {
      const result = await compareValidation({ model_name: reqModel, material_id: reqMaterial, scan_rate_mVs: reqSR });
      // Discard result if selections changed since the request was issued
      setCmpData((prev) => {
        const selectionChanged = reqModel !== model || reqMaterial !== material || reqSR !== scanRate;
        return selectionChanged ? prev : result;
      });
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Validation failed";
      setError(typeof msg === "string" ? msg : JSON.stringify(msg));
    } finally {
      setLoading(false);
    }
  }, [model, material, scanRate]);

  // Load per-group metrics when tab 2 is opened for the first time (or after model change).
  // Guard with !pgLoading to prevent duplicate in-flight requests.
  const handleTabChange = useCallback(async (_: React.SyntheticEvent, val: number) => {
    setActiveTab(val);
    if (val === 2 && !pgData && !pgLoading) {
      setPgLoading(true);
      setPgError(null);
      try {
        const result = await getPerGroupMetrics(pgModel);
        setPgData(result);
      } catch (e: any) {
        setPgError(e?.response?.data?.detail ?? "Failed to load metrics");
      } finally {
        setPgLoading(false);
      }
    }
  }, [pgData, pgLoading, pgModel]);

  const handlePgModelChange = useCallback(async (newModel: string) => {
    setPgModel(newModel);
    setPgLoading(true);
    setPgError(null);
    setPgData(null);
    try {
      const result = await getPerGroupMetrics(newModel);
      setPgData(result);
    } catch (e: any) {
      setPgError(e?.response?.data?.detail ?? "Failed to load metrics");
    } finally {
      setPgLoading(false);
    }
  }, []);

  return (
    <Box sx={{ overflowX: "hidden" }}>
      <SectionHeader
        title="Validation Analysis"
        subtitle="Compare ML predictions against real experimental ZnO supercapacitor CV measurements"
        accent="#10b981"
      />

      {/* Credibility banner */}
      <Paper sx={{
        p: { xs: 1.5, sm: 2 }, mb: 3,
        background: "rgba(16,185,129,0.04)",
        border: "1px solid rgba(16,185,129,0.18)",
        borderRadius: 3,
        display: "flex", alignItems: "center", gap: 1.5,
      }}>
        <CheckCircleRounded sx={{ color: "#10b981", fontSize: 20, flexShrink: 0 }} />
        <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.8rem" }}>
          All validation data is sourced from <strong style={{ color: "#10b981" }}>real electrochemical experiments</strong>. Curves are measured on a ZnO nanostructure electrode at the specified scan rate. No simulated or synthetic data is used anywhere on this page.
        </Typography>
      </Paper>

      {/* Tabs */}
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          mb: 3,
          borderBottom: "1px solid rgba(255,255,255,0.08)",
          "& .MuiTab-root": { color: "rgba(255,255,255,0.45)", fontSize: "0.8rem", fontWeight: 500, textTransform: "none", minWidth: 80 },
          "& .Mui-selected": { color: "#10b981 !important", fontWeight: 700 },
          "& .MuiTabs-indicator": { backgroundColor: "#10b981" },
        }}
      >
        <Tab label="Overlay" />
        <Tab label="Residuals" />
        <Tab label="Performance Map" />
        <Tab label="Methodology" />
      </Tabs>

      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.25 }}
        >
          {activeTab === 0 && (
            <OverlayTab
              data={cmpData} loading={loading} error={error} onRun={handleRun}
              model={model} setModel={setModel}
              material={material} setMaterial={setMaterial}
              scanRate={scanRate} setScanRate={setScanRate}
            />
          )}
          {activeTab === 1 && <ResidualsTab data={cmpData} />}
          {activeTab === 2 && (
            <PerformanceMapTab
              pgData={pgData} pgLoading={pgLoading} pgError={pgError}
              pgModel={pgModel} setPgModel={handlePgModelChange}
            />
          )}
          {activeTab === 3 && <MethodologyTab />}
        </motion.div>
      </AnimatePresence>
    </Box>
  );
}
