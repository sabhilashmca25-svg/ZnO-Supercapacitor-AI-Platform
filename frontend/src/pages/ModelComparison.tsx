import React, { useState, useCallback, useMemo } from "react";
import {
  Box, Grid, Card, CardContent, Typography, Select, MenuItem,
  FormControl, InputLabel, Slider, Table, TableBody, TableCell,
  TableHead, TableRow, Alert, Chip, TextField, InputAdornment, alpha,
} from "@mui/material";
import {
  CompareArrowsRounded, TimerRounded, ShowChartRounded,
  GridOnRounded, BarChartRounded, LeaderboardRounded,
  RadarRounded, BubbleChartRounded, InsightsRounded,
} from "@mui/icons-material";
import { motion, AnimatePresence } from "framer-motion";
import Plot from "react-plotly.js";
import { useComparison } from "../hooks/usePrediction";
import { useLeaderboard } from "../hooks/useLeaderboard";
import { MultiCVPlot } from "../components/charts/CVPlot";
import GlowButton from "../components/common/GlowButton";
import SectionHeader from "../components/common/SectionHeader";
import LoadingOverlay from "../components/common/LoadingOverlay";
import ModelBadge from "../components/common/ModelBadge";
import { MODELS, MATERIALS, MODEL_COLORS } from "../constants/models";
import { fmtUa, fmtMB, rmseColor, r2Color } from "../utils/formatters";
import { cardVariants, staggerContainer } from "../animations/variants";

const SCAN_RATE_MARKS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((v) => ({
  value: v,
  label: v % 20 === 0 || v === 10 ? `${v}` : "",
}));
const VALID_SR = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
const LAZY_MODELS = new Set(["xgboost", "ann", "lstm"]);

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

function nearestSR(val: number): number {
  const r = Math.round(val);
  if (VALID_SR.includes(r)) return r;
  return VALID_SR.reduce((a, b) => (Math.abs(b - val) < Math.abs(a - val) ? b : a));
}

/** Cosine similarity between two equal-length number arrays */
function cosSim(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return normA > 0 && normB > 0 ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
}

type ViewMode = "overlay" | "residual" | "similarity";

const VIEW_MODES: { mode: ViewMode; label: string; icon: React.ReactElement }[] = [
  { mode: "overlay",    label: "CV Overlay",  icon: <ShowChartRounded sx={{ fontSize: 13 }} /> },
  { mode: "residual",   label: "Residual Δ",  icon: <BarChartRounded sx={{ fontSize: 13 }} /> },
  { mode: "similarity", label: "Similarity",  icon: <GridOnRounded sx={{ fontSize: 13 }} /> },
];

export default function ModelComparison() {
  const [selectedModels, setSelectedModels] = useState<string[]>(["rf", "lightgbm", "gru"]);
  const [materialId, setMaterialId] = useState("NM1");
  const [scanRate, setScanRate] = useState<number>(30);
  const [scanRateText, setScanRateText] = useState("30");
  const [viewMode, setViewMode] = useState<ViewMode>("overlay");

  const { compare, compareResult, comparing, error } = useComparison();
  const { leaderboard } = useLeaderboard();

  const snappedSR = useMemo(() => nearestSR(scanRate), [scanRate]);
  const isSnapped = snappedSR !== Math.round(scanRate * 10) / 10;
  const hasLazySelected = selectedModels.some((m) => LAZY_MODELS.has(m));
  const lazyNames = selectedModels
    .filter((m) => LAZY_MODELS.has(m))
    .map((m) => m.toUpperCase())
    .join(", ");

  /** Leaderboard rows for currently-selected models */
  const selectedPerf = useMemo(
    () => leaderboard.filter((r) => selectedModels.includes(r.model_id)),
    [leaderboard, selectedModels]
  );

  /** Residual = each model current − ensemble mean current */
  const residualData = useMemo(() => {
    if (!compareResult || Object.keys(compareResult.predictions).length < 2) return null;
    const potential = compareResult.potential_V;
    const modelIds = Object.keys(compareResult.predictions);
    const allCurrents = modelIds.map((m) => compareResult.predictions[m].predicted_current_uA);
    const mean = potential.map((_, i) =>
      allCurrents.reduce((s, c) => s + c[i], 0) / allCurrents.length
    );
    return {
      potential,
      models: modelIds,
      residuals: allCurrents.map((c) => c.map((v, i) => v - mean[i])),
    };
  }, [compareResult]);

  /** Pairwise cosine similarity matrix */
  const similarityMatrix = useMemo(() => {
    if (!compareResult) return null;
    const modelIds = Object.keys(compareResult.predictions);
    const currs = modelIds.map((m) => compareResult.predictions[m].predicted_current_uA);
    const n = modelIds.length;
    const matrix: number[][] = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => cosSim(currs[i], currs[j]))
    );
    return { modelIds, matrix };
  }, [compareResult]);

  /** Deploy scores lookup from MODELS config */
  const deployScoreMap = useMemo(
    () => Object.fromEntries(MODELS.map((m) => [m.id, m.deployScore])),
    []
  );

  /** Radar chart traces for selected models */
  const radarTraces = useMemo<Plotly.Data[]>(() => {
    if (selectedPerf.length === 0) return [];
    const axes = ["Accuracy", "NM4 R²", "Deployment", "Size Eff.", "SR Interp.", "Accuracy"];
    return selectedPerf.map((r) => {
      const deployScore = deployScoreMap[r.model_id] ?? 70;
      const accuracy = Math.max(0, Math.min(100, 100 * (55 - r.rmse_val_uA) / (55 - 26)));
      const nm4R2 = Math.max(0, Math.min(100, 100 * (r.r2_test_mat - 0.9) / 0.1));
      const srExtrap = Math.max(0, Math.min(100, 100 * (120 - (r.rmse_test_sr_uA ?? 80)) / (120 - 28)));
      const sizeEff = Math.max(0, Math.min(100,
        100 * (1 - (Math.log10((r.size_MB ?? 1) + 0.01) - Math.log10(0.17)) / (Math.log10(615) - Math.log10(0.17)))
      ));
      const color = MODEL_COLORS[r.model_id] ?? "#94a3b8";
      return {
        type: "scatterpolar" as const,
        r: [accuracy, nm4R2, deployScore, sizeEff, srExtrap, accuracy],
        theta: axes,
        name: r.model_id.toUpperCase(),
        fill: "toself" as const,
        opacity: 0.75,
        line: { color, width: 2 },
        fillcolor: `${color}22`,
        hovertemplate: "<b>%{theta}</b><br>Score: %{r:.1f} / 100<extra></extra>",
      };
    });
  }, [selectedPerf, deployScoreMap]);

  /** Pareto frontier: models not dominated on RMSE × size */
  const paretoFrontier = useMemo(() => {
    if (leaderboard.length === 0) return [];
    const sorted = [...leaderboard].sort((a, b) => a.rmse_val_uA - b.rmse_val_uA);
    const frontier: typeof leaderboard = [];
    let minSize = Infinity;
    for (const r of sorted) {
      if ((r.size_MB ?? Infinity) < minSize) {
        frontier.push(r);
        minSize = r.size_MB ?? Infinity;
      }
    }
    return frontier.sort((a, b) => (a.size_MB ?? 0) - (b.size_MB ?? 0));
  }, [leaderboard]);

  // ── Event handlers ────────────────────────────────────────────────────────
  const handleSliderChange = useCallback((_: Event, v: number | number[]) => {
    const val = v as number;
    setScanRate(val);
    setScanRateText(Number.isInteger(val) ? String(val) : val.toFixed(1));
  }, []);

  const handleTextChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setScanRateText(raw);
    const parsed = parseFloat(raw);
    if (!isNaN(parsed) && parsed >= 10 && parsed <= 100)
      setScanRate(Math.round(parsed * 10) / 10);
  }, []);

  const handleTextBlur = useCallback(() => {
    const parsed = parseFloat(scanRateText);
    if (isNaN(parsed) || parsed < 10 || parsed > 100) {
      setScanRate(30);
      setScanRateText("30");
    } else {
      const clamped = Math.min(100, Math.max(10, Math.round(parsed * 10) / 10));
      setScanRate(clamped);
      setScanRateText(Number.isInteger(clamped) ? String(clamped) : clamped.toFixed(1));
    }
  }, [scanRateText]);

  const toggleModel = useCallback((id: string) => {
    setSelectedModels((prev) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]
    );
  }, []);

  const handleCompare = useCallback(() => {
    if (selectedModels.length === 0) return;
    compare({ models: selectedModels, material_id: materialId, scan_rate_mVs: scanRate });
  }, [compare, selectedModels, materialId, scanRate]);

  return (
    <Box sx={{ width: "100%", overflowX: "hidden" }}>
      <SectionHeader
        title="Multi-Model Comparison"
        subtitle="Overlay CV curves · compare residual bias · analyse prediction curve similarity across all 6 architectures"
        accent="#7c3aed"
      />

      <Grid container spacing={{ xs: 2, md: 3 }}>
        {/* ── Left control panel ─────────────────────────────────────────── */}
        <Grid item xs={12} lg={3}>
          <motion.div variants={staggerContainer} initial="initial" animate="animate">

            {/* Model picker */}
            <motion.div variants={cardVariants}>
              <Card sx={{ mb: 2.5, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(124,58,237,0.2)" }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography variant="overline"
                    sx={{ color: "#a78bfa", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 2, display: "block" }}>
                    Select Models ({selectedModels.length})
                  </Typography>
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                    {MODELS.map((m) => {
                      const active = selectedModels.includes(m.id);
                      const color = MODEL_COLORS[m.id];
                      return (
                        <Box
                          key={m.id}
                          component={motion.div}
                          whileHover={{ x: 2 }}
                          onClick={() => toggleModel(m.id)}
                          role="checkbox"
                          aria-checked={active}
                          tabIndex={0}
                          onKeyDown={(e) => e.key === "Enter" && toggleModel(m.id)}
                          sx={{
                            display: "flex", alignItems: "center", gap: 1.5,
                            px: 1.5, py: 1, borderRadius: 2, cursor: "pointer",
                            backgroundColor: active ? alpha(color, 0.1) : "transparent",
                            border: `1px solid ${active ? alpha(color, 0.35) : "transparent"}`,
                            transition: "all 0.2s ease",
                            "&:hover": { backgroundColor: alpha(color, 0.08) },
                          }}
                        >
                          <Box sx={{
                            width: 14, height: 14, borderRadius: 1,
                            border: `2px solid ${color}`,
                            backgroundColor: active ? color : "transparent",
                            transition: "background-color 0.15s",
                            flexShrink: 0, display: "flex", alignItems: "center",
                            justifyContent: "center", fontSize: 9, color: "#070b14",
                          }}>
                            {active && "✓"}
                          </Box>
                          <Box sx={{ flex: 1 }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                              <Typography variant="body2"
                                sx={{ fontWeight: 600, color: active ? color : "rgba(255,255,255,0.6)", fontSize: "0.8rem" }}>
                                {m.display}
                              </Typography>
                              {LAZY_MODELS.has(m.id) && (
                                <Chip
                                  icon={<TimerRounded sx={{ fontSize: "9px !important" }} />}
                                  label="T2"
                                  size="small"
                                  sx={{
                                    fontSize: "0.55rem", height: 16,
                                    backgroundColor: "rgba(251,191,36,0.08)",
                                    color: "#fbbf24",
                                    border: "1px solid rgba(251,191,36,0.2)",
                                    "& .MuiChip-label": { px: 0.5 },
                                  }}
                                />
                              )}
                            </Box>
                            <Typography variant="caption"
                              sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.62rem" }}>
                              {m.size}
                            </Typography>
                          </Box>
                        </Box>
                      );
                    })}
                  </Box>
                </CardContent>
              </Card>
            </motion.div>

            {/* Experiment parameters */}
            <motion.div variants={cardVariants}>
              <Card sx={{ mb: 2.5, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.12)" }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography variant="overline"
                    sx={{ color: "#00d4ff", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 1.5, display: "block" }}>
                    Experiment
                  </Typography>
                  <FormControl fullWidth size="small" sx={{ mb: 2.5 }}>
                    <InputLabel sx={{ fontSize: "0.82rem" }}>Material</InputLabel>
                    <Select value={materialId} label="Material" onChange={(e) => setMaterialId(e.target.value)}>
                      {MATERIALS.map((m) => (
                        <MenuItem key={m.id} value={m.id}>{m.label}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <Box>
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.72rem" }}>
                        Scan Rate
                      </Typography>
                      <TextField
                        value={scanRateText}
                        onChange={handleTextChange}
                        onBlur={handleTextBlur}
                        onKeyDown={(e) => { if (e.key === "Enter") handleTextBlur(); }}
                        size="small"
                        variant="outlined"
                        inputProps={{
                          style: {
                            fontFamily: "JetBrains Mono, monospace", fontSize: "0.8rem",
                            color: "#00d4ff", fontWeight: 700, padding: "4px 6px",
                            textAlign: "right", width: 44,
                          },
                        }}
                        InputProps={{
                          endAdornment: (
                            <InputAdornment position="end">
                              <Typography sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.65rem" }}>mV/s</Typography>
                            </InputAdornment>
                          ),
                          sx: {
                            "& fieldset": { borderColor: "rgba(0,212,255,0.25)" },
                            "&:hover fieldset": { borderColor: "rgba(0,212,255,0.5) !important" },
                            "&.Mui-focused fieldset": { borderColor: "#00d4ff !important" },
                            backgroundColor: "rgba(0,212,255,0.04)", borderRadius: 1,
                          },
                        }}
                        sx={{ width: 110 }}
                      />
                    </Box>
                    <Slider
                      value={scanRate}
                      onChange={handleSliderChange}
                      min={10} max={100} step={0.5} marks={SCAN_RATE_MARKS}
                      valueLabelDisplay="auto"
                      valueLabelFormat={(v) => `${v} mV/s`}
                      sx={{
                        "& .MuiSlider-thumb": { width: { xs: 20, md: 16 }, height: { xs: 20, md: 16 } },
                        "& .MuiSlider-track": { height: { xs: 5, md: 4 } },
                        "& .MuiSlider-rail":  { height: { xs: 5, md: 4 } },
                        "& .MuiSlider-markLabel": { fontSize: "0.58rem", color: "rgba(255,255,255,0.3)" },
                        "& .MuiSlider-valueLabel": { fontSize: "0.65rem" },
                      }}
                    />
                    {isSnapped && (
                      <Typography variant="caption"
                        sx={{ color: "rgba(251,191,36,0.7)", fontSize: "0.62rem", display: "block", mt: 0.25 }}>
                        → denormalized as {snappedSR} mV/s
                      </Typography>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div variants={cardVariants}>
              <GlowButton
                variant="contained"
                color="secondary"
                fullWidth
                size="large"
                onClick={handleCompare}
                loading={comparing}
                disabled={comparing || selectedModels.length === 0}
                startIcon={<CompareArrowsRounded />}
                glowColor="#7c3aed"
                sx={{ py: 1.5, fontSize: "0.9rem", mb: 1 }}
              >
                Compare Models
              </GlowButton>
              {selectedModels.length === 0 && (
                <Typography variant="caption"
                  sx={{ color: "rgba(239,68,68,0.7)", fontSize: "0.65rem", display: "block", textAlign: "center" }}>
                  Select at least one model
                </Typography>
              )}
            </motion.div>

            <AnimatePresence>
              {error && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  style={{ marginTop: 12 }}>
                  <Alert severity="error" sx={{ fontSize: "0.8rem" }}>{error}</Alert>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Benchmark summary for selected models */}
            {selectedPerf.length > 0 && (
              <motion.div variants={cardVariants} style={{ marginTop: 16 }}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(124,58,237,0.15)" }}>
                  <CardContent sx={{ p: 2 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
                      <LeaderboardRounded sx={{ fontSize: 14, color: "#a78bfa" }} />
                      <Typography variant="overline"
                        sx={{ color: "#a78bfa", fontSize: "0.6rem", letterSpacing: "0.08em" }}>
                        Benchmark Overview
                      </Typography>
                    </Box>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={{ py: 0.4, fontSize: "0.6rem", color: "rgba(255,255,255,0.25)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                            Model
                          </TableCell>
                          <TableCell align="right" sx={{ py: 0.4, fontSize: "0.6rem", color: "rgba(255,255,255,0.25)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                            Val RMSE
                          </TableCell>
                          <TableCell align="right" sx={{ py: 0.4, fontSize: "0.6rem", color: "rgba(255,255,255,0.25)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                            R² MAT
                          </TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {selectedPerf.map((r) => (
                          <TableRow key={r.model_id} sx={{ "&:last-child td": { border: 0 } }}>
                            <TableCell sx={{ py: 0.55, borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                              <ModelBadge modelId={r.model_id} size="small" />
                            </TableCell>
                            <TableCell align="right" sx={{
                              py: 0.55, borderBottom: "1px solid rgba(255,255,255,0.04)",
                              color: rmseColor(r.rmse_val_uA), fontFamily: "JetBrains Mono",
                              fontSize: "0.7rem", fontWeight: 700,
                            }}>
                              {r.rmse_val_uA?.toFixed(1)}
                            </TableCell>
                            <TableCell align="right" sx={{
                              py: 0.55, borderBottom: "1px solid rgba(255,255,255,0.04)",
                              color: r2Color(r.r2_test_mat), fontFamily: "JetBrains Mono",
                              fontSize: "0.7rem", fontWeight: 600,
                            }}>
                              {r.r2_test_mat?.toFixed(4)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    <Typography variant="caption"
                      sx={{ color: "rgba(255,255,255,0.18)", mt: 1.25, display: "block", fontSize: "0.6rem" }}>
                      RMSE in µA · R² on unseen NM4 material
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </motion.div>
        </Grid>

        {/* ── Chart + analysis area ───────────────────────────────────────── */}
        <Grid item xs={12} lg={9}>
          <AnimatePresence mode="wait">
            {comparing && (
              <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(124,58,237,0.2)", minHeight: 420 }}>
                  <CardContent sx={{ p: 3 }}>
                    <LoadingOverlay
                      message={`Running ${selectedModels.length} model${selectedModels.length > 1 ? "s" : ""} in parallel…`}
                      subMessage={
                        hasLazySelected
                          ? `Tier-2 cold start for ${lazyNames} (~15s) — loading TensorFlow graph on first request`
                          : `Comparing ${selectedModels.map((m) => m.toUpperCase()).join(", ")} across 651 CV points`
                      }
                    />
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {!comparing && compareResult && (
              <motion.div key="result" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }} transition={{ duration: 0.45 }}>

                {/* ── View mode selector bar ── */}
                <Card sx={{ mb: 2, background: "rgba(10,18,28,0.75)", border: "1px solid rgba(124,58,237,0.15)" }}>
                  <CardContent sx={{ p: 1.5, "&:last-child": { pb: 1.5 } }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.68rem" }}>
                        View:
                      </Typography>
                      {VIEW_MODES.map(({ mode, icon, label }) => (
                        <Chip
                          key={mode}
                          icon={icon}
                          label={label}
                          size="small"
                          onClick={() => setViewMode(mode)}
                          sx={{
                            fontSize: "0.7rem", height: 28, cursor: "pointer",
                            backgroundColor: viewMode === mode ? alpha("#a78bfa", 0.18) : "transparent",
                            color: viewMode === mode ? "#a78bfa" : "rgba(255,255,255,0.35)",
                            border: `1px solid ${viewMode === mode ? alpha("#a78bfa", 0.4) : "rgba(255,255,255,0.08)"}`,
                            "& .MuiChip-icon": { color: "inherit" },
                            transition: "all 0.2s",
                            "&:hover": { backgroundColor: alpha("#a78bfa", 0.1), color: "#a78bfa" },
                          }}
                        />
                      ))}
                      <Box sx={{ flex: 1 }} />
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap" }}>
                        {Object.keys(compareResult.predictions).map((m) => (
                          <ModelBadge key={m} modelId={m} />
                        ))}
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.7rem", ml: 0.25 }}>
                          {compareResult.material_id} · {compareResult.scan_rate_mVs} mV/s
                        </Typography>
                        {compareResult.models_failed.length > 0 && (
                          <Chip
                            label={`${compareResult.models_failed.length} failed`}
                            size="small"
                            sx={{ backgroundColor: "rgba(239,68,68,0.1)", color: "#ef4444", fontSize: "0.6rem", height: 18 }}
                          />
                        )}
                      </Box>
                    </Box>
                  </CardContent>
                </Card>

                <AnimatePresence mode="wait">
                  {/* ── Overlay mode ── */}
                  {viewMode === "overlay" && (
                    <motion.div key="ov" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      transition={{ duration: 0.25 }}>
                      <Card sx={{ mb: 2.5, background: "rgba(7,11,20,0.9)", border: "1px solid rgba(124,58,237,0.2)" }}>
                        <CardContent sx={{ p: 2 }}>
                          <Typography variant="overline"
                            sx={{ color: "#a78bfa", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 1.5, display: "block" }}>
                            CV CURVE OVERLAY — ALL SELECTED MODELS
                          </Typography>
                          <MultiCVPlot result={compareResult} height={440} />
                        </CardContent>
                      </Card>
                    </motion.div>
                  )}

                  {/* ── Residual mode ── */}
                  {viewMode === "residual" && (
                    <motion.div key="res" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      transition={{ duration: 0.25 }}>
                      <Card sx={{ mb: 2.5, background: "rgba(7,11,20,0.9)", border: "1px solid rgba(251,146,60,0.2)" }}>
                        <CardContent sx={{ p: 2 }}>
                          <Typography variant="overline"
                            sx={{ color: "#fb923c", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 0.5, display: "block" }}>
                            RESIDUAL COMPARISON — MODEL CURRENT − ENSEMBLE MEAN
                          </Typography>
                          <Typography variant="caption"
                            sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", display: "block", mb: 1.5 }}>
                            Zero line = consensus mean of all selected models. Positive = model predicts higher than mean; negative = lower.
                          </Typography>
                          {residualData ? (
                            <Plot
                              data={residualData.models.map((m, i) => ({
                                type: "scatter" as const,
                                mode: "lines" as const,
                                name: m.toUpperCase(),
                                x: residualData.potential,
                                y: residualData.residuals[i],
                                line: { color: MODEL_COLORS[m] ?? "#94a3b8", width: 2 },
                                hovertemplate: `<b>${m.toUpperCase()}</b><br>V: %{x:.3f} V<br>Δ: %{y:.2f} µA<extra></extra>`,
                              }))}
                              layout={DarkLayout({
                                height: 440,
                                margin: { l: 72, r: 20, t: 10, b: 55 },
                                xaxis: {
                                  title: { text: "Potential (V)", font: { size: 11, color: "#94a3b8" } },
                                  gridcolor: "rgba(255,255,255,0.04)",
                                  tickfont: { size: 10, family: "JetBrains Mono" },
                                },
                                yaxis: {
                                  title: { text: "Residual (µA) — vs ensemble mean", font: { size: 11, color: "#94a3b8" } },
                                  gridcolor: "rgba(255,255,255,0.04)",
                                  zeroline: true,
                                  zerolinecolor: "rgba(255,255,255,0.28)",
                                  zerolinewidth: 1.5,
                                  tickfont: { size: 10, family: "JetBrains Mono" },
                                },
                                legend: { orientation: "h", x: 0.05, y: 1.06, bgcolor: "rgba(0,0,0,0)", font: { size: 11 } },
                              })}
                              config={PlotConfig}
                              style={{ width: "100%", height: 440 }}
                              useResizeHandler
                            />
                          ) : (
                            <Box sx={{ height: 440, display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.35)" }}>
                                Select 2+ models and run comparison to see residuals.
                              </Typography>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </motion.div>
                  )}

                  {/* ── Similarity mode ── */}
                  {viewMode === "similarity" && (
                    <motion.div key="sim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      transition={{ duration: 0.25 }}>
                      <Card sx={{ mb: 2.5, background: "rgba(7,11,20,0.9)", border: "1px solid rgba(0,212,255,0.2)" }}>
                        <CardContent sx={{ p: 2 }}>
                          <Typography variant="overline"
                            sx={{ color: "#00d4ff", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 0.5, display: "block" }}>
                            COSINE SIMILARITY MATRIX — PREDICTION CURVE ALIGNMENT
                          </Typography>
                          <Typography variant="caption"
                            sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", display: "block", mb: 1.5 }}>
                            1.0000 = identical curves · Values near 0.99+ indicate highly correlated model outputs across 651 potential points.
                          </Typography>
                          {similarityMatrix ? (
                            <Plot
                              data={[{
                                type: "heatmap" as const,
                                z: similarityMatrix.matrix,
                                x: similarityMatrix.modelIds.map((m) => m.toUpperCase()),
                                y: similarityMatrix.modelIds.map((m) => m.toUpperCase()),
                                colorscale: [
                                  [0,    "#1e1b4b"],
                                  [0.4,  "#312e81"],
                                  [0.7,  "#4f46e5"],
                                  [0.88, "#7c3aed"],
                                  [0.96, "#a78bfa"],
                                  [1,    "#ede9fe"],
                                ],
                                zmin: 0.8,
                                zmax: 1.0,
                                texttemplate: "%{z:.4f}",
                                textfont: { size: 13, color: "#f1f5f9", family: "JetBrains Mono" },
                                hovertemplate: "<b>%{y} vs %{x}</b><br>Cosine Similarity: %{z:.4f}<extra></extra>",
                                showscale: true,
                                colorbar: {
                                  thickness: 12, len: 0.85,
                                  title: { text: "Sim.", font: { size: 10, color: "#94a3b8" } },
                                  tickfont: { size: 9, family: "JetBrains Mono", color: "#94a3b8" },
                                  tickvals: [0.80, 0.85, 0.90, 0.95, 1.00],
                                },
                              }]}
                              layout={DarkLayout({
                                height: 440,
                                margin: { l: 80, r: 80, t: 20, b: 60 },
                                xaxis: {
                                  tickfont: { size: 12, color: "#e2e8f0", family: "JetBrains Mono" },
                                  side: "bottom" as const,
                                },
                                yaxis: {
                                  tickfont: { size: 12, color: "#e2e8f0", family: "JetBrains Mono" },
                                },
                              })}
                              config={PlotConfig}
                              style={{ width: "100%", height: 440 }}
                              useResizeHandler
                            />
                          ) : (
                            <Box sx={{ height: 440, display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.35)" }}>
                                Run comparison to compute cosine similarity matrix.
                              </Typography>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* ── Electrochemical stats table ── */}
                <Card sx={{ mb: 2.5, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(124,58,237,0.15)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="overline"
                      sx={{ color: "#a78bfa", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 2, display: "block" }}>
                      ELECTROCHEMICAL STATISTICS — THIS PREDICTION
                    </Typography>
                    <Box sx={{ overflowX: "auto" }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={{ fontWeight: 700 }}>Model</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 700 }}>Peak Anodic (µA)</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 700 }}>Peak Cathodic (µA)</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 700 }}>Range (µA)</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 700 }}>Integral Area</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(compareResult.predictions).map(([modelId, pred]) => (
                          <TableRow key={modelId} sx={{ "&:last-child td": { border: 0 } }}>
                            <TableCell><ModelBadge modelId={modelId} size="small" /></TableCell>
                            <TableCell align="right"
                              sx={{ color: "#10b981", fontFamily: "JetBrains Mono", fontSize: "0.8rem" }}>
                              {fmtUa(pred.peak_anodic_uA)}
                            </TableCell>
                            <TableCell align="right"
                              sx={{ color: "#f472b6", fontFamily: "JetBrains Mono", fontSize: "0.8rem" }}>
                              {fmtUa(pred.peak_cathodic_uA)}
                            </TableCell>
                            <TableCell align="right"
                              sx={{ color: "#00d4ff", fontFamily: "JetBrains Mono", fontSize: "0.8rem" }}>
                              {fmtUa(pred.current_range_uA)}
                            </TableCell>
                            <TableCell align="right"
                              sx={{ color: "#a78bfa", fontFamily: "JetBrains Mono", fontSize: "0.8rem" }}>
                              {pred.integral_area.toFixed(1)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    </Box>
                  </CardContent>
                </Card>

                {/* ── Benchmark performance for selected models ── */}
                {selectedPerf.length > 0 && (
                  <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(34,211,238,0.15)" }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                        <LeaderboardRounded sx={{ fontSize: 16, color: "#22d3ee" }} />
                        <Typography variant="overline"
                          sx={{ color: "#22d3ee", fontSize: "0.65rem", letterSpacing: "0.1em" }}>
                          BENCHMARK PERFORMANCE — SELECTED MODELS
                        </Typography>
                      </Box>
                      <Box sx={{ overflowX: "auto" }}>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ fontWeight: 700 }}>Model</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 700, color: "#10b981" }}>Val RMSE</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 700, color: "#f59e0b" }}>Test-SR RMSE</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 700, color: "#f472b6" }}>Test-MAT RMSE</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 700, color: "#a78bfa" }}>R² MAT</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 700 }}>Size</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {selectedPerf.map((r) => (
                            <TableRow key={r.model_id} sx={{ "&:last-child td": { border: 0 } }}>
                              <TableCell><ModelBadge modelId={r.model_id} size="small" /></TableCell>
                              <TableCell align="right" sx={{
                                color: rmseColor(r.rmse_val_uA), fontFamily: "JetBrains Mono",
                                fontSize: "0.8rem", fontWeight: 700,
                              }}>
                                {r.rmse_val_uA?.toFixed(2)} µA
                              </TableCell>
                              <TableCell align="right" sx={{
                                color: rmseColor(r.rmse_test_sr_uA), fontFamily: "JetBrains Mono",
                                fontSize: "0.8rem", fontWeight: 600,
                              }}>
                                {r.rmse_test_sr_uA?.toFixed(2)} µA
                              </TableCell>
                              <TableCell align="right" sx={{
                                color: rmseColor(r.rmse_test_mat_uA), fontFamily: "JetBrains Mono",
                                fontSize: "0.8rem", fontWeight: 600,
                              }}>
                                {r.rmse_test_mat_uA?.toFixed(2)} µA
                              </TableCell>
                              <TableCell align="right" sx={{
                                color: r2Color(r.r2_test_mat), fontFamily: "JetBrains Mono",
                                fontSize: "0.8rem", fontWeight: 700,
                              }}>
                                {r.r2_test_mat?.toFixed(4)}
                              </TableCell>
                              <TableCell align="right" sx={{
                                color: "rgba(255,255,255,0.4)", fontFamily: "JetBrains Mono", fontSize: "0.72rem",
                              }}>
                                {fmtMB(r.size_MB)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                      </Box>
                      <Typography variant="caption"
                        sx={{ color: "rgba(255,255,255,0.18)", mt: 1.5, display: "block", fontSize: "0.62rem" }}>
                        Val = SR 30 mV/s interpolation · Test-SR = SR 50 mV/s · Test-MAT = NM4 unseen material extrapolation
                      </Typography>
                    </CardContent>
                  </Card>
                )}
              </motion.div>
            )}

            {!comparing && !compareResult && (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Card sx={{
                  background: "rgba(15,25,35,0.6)", border: "1px dashed rgba(124,58,237,0.2)",
                  borderRadius: 3, minHeight: 440, display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <Box sx={{ textAlign: "center", p: 4 }}>
                    <Box
                      component={motion.div}
                      animate={{ rotate: [0, 360] }}
                      transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                      sx={{ fontSize: 44, mb: 2, display: "inline-block" }}
                    >
                      ⚖️
                    </Box>
                    <Typography variant="h6" sx={{ color: "rgba(255,255,255,0.45)", fontWeight: 600, mb: 1 }}>
                      Ready to Compare
                    </Typography>
                    <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.2)", maxWidth: 320 }}>
                      Select models from the left panel and click{" "}
                      <span style={{ color: "#a78bfa" }}>Compare Models</span>{" "}
                      to run multi-model CV prediction and compute curve similarity.
                    </Typography>
                  </Box>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </Grid>
      </Grid>

      {/* ══ MULTI-DIMENSIONAL BENCHMARK ANALYTICS ═══════════════════ */}
      {leaderboard.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}>
            <InsightsRounded sx={{ color: "#f59e0b", fontSize: 20 }} />
            <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>
              Multi-Dimensional Benchmark Analytics
            </Typography>
            <Chip label="ALL 6 MODELS" size="small"
              sx={{ fontSize: "0.6rem", height: 20, fontWeight: 700, backgroundColor: alpha("#f59e0b", 0.1), color: "#f59e0b", border: "1px solid rgba(245,158,11,0.3)" }} />
          </Box>

          <Grid container spacing={3}>
            {/* ── Radar chart ── */}
            <Grid item xs={12} md={7}>
              <motion.div variants={cardVariants} initial="initial" animate="animate">
                <Card sx={{ background: "rgba(7,11,20,0.9)", border: "1px solid rgba(167,139,250,0.2)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                      <RadarRounded sx={{ fontSize: 16, color: "#a78bfa" }} />
                      <Typography variant="overline" sx={{ color: "#a78bfa", fontSize: "0.65rem", letterSpacing: "0.1em" }}>
                        PERFORMANCE RADAR — SELECTED MODELS
                      </Typography>
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.63rem", display: "block", mb: 1.5 }}>
                      Normalised 0–100 scores per axis. Accuracy: val RMSE · NM4 R²: test-MAT · Deployment: size+speed · Size Eff.: inverse log-size · SR Interp.: test-SR RMSE
                    </Typography>
                    {radarTraces.length > 0 ? (
                      <Plot
                        data={radarTraces}
                        layout={{
                          paper_bgcolor: "rgba(0,0,0,0)",
                          polar: {
                            bgcolor: "rgba(13,19,33,0.5)",
                            radialaxis: {
                              visible: true,
                              range: [0, 100],
                              tickfont: { size: 9, color: "#8892a4", family: "JetBrains Mono" },
                              gridcolor: "rgba(255,255,255,0.06)",
                              linecolor: "rgba(255,255,255,0.1)",
                              tickvals: [0, 25, 50, 75, 100],
                            },
                            angularaxis: {
                              tickfont: { size: 11, color: "#c8cfd8", family: "Inter" },
                              gridcolor: "rgba(255,255,255,0.06)",
                              linecolor: "rgba(255,255,255,0.1)",
                            },
                          },
                          font: { family: "Inter, sans-serif", color: "#8892a4" },
                          height: 380,
                          margin: { l: 50, r: 50, t: 30, b: 55 },
                          legend: {
                            bgcolor: "rgba(13,19,33,0.8)", bordercolor: "rgba(167,139,250,0.2)",
                            borderwidth: 1, font: { size: 11, color: "#f1f5f9" },
                            orientation: "h", x: 0.1, y: -0.08,
                          },
                          hoverlabel: { bgcolor: "#111c2d", bordercolor: "#a78bfa", font: { size: 11, color: "#f1f5f9", family: "JetBrains Mono" } },
                        }}
                        config={PlotConfig}
                        style={{ width: "100%", height: 380 }}
                        useResizeHandler
                      />
                    ) : (
                      <Box sx={{ height: 380, display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.3)" }}>
                          Select models on the left to see radar comparison
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>

            {/* ── Pareto frontier ── */}
            <Grid item xs={12} md={5}>
              <motion.div variants={cardVariants} initial="initial" animate="animate">
                <Card sx={{ background: "rgba(7,11,20,0.9)", border: "1px solid rgba(34,211,238,0.2)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                      <BubbleChartRounded sx={{ fontSize: 16, color: "#22d3ee" }} />
                      <Typography variant="overline" sx={{ color: "#22d3ee", fontSize: "0.65rem", letterSpacing: "0.1em" }}>
                        PARETO FRONTIER — ACCURACY VS SIZE
                      </Typography>
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.63rem", display: "block", mb: 1.5 }}>
                      Dashed line = Pareto-optimal models (no other model has both lower RMSE and smaller size). Bottom-left = ideal.
                    </Typography>
                    <Plot
                      data={[
                        // All models as scatter
                        ...leaderboard.map((r) => ({
                          type: "scatter" as const,
                          mode: "text+markers" as const,
                          x: [r.size_MB],
                          y: [r.rmse_val_uA],
                          name: r.model_id.toUpperCase(),
                          text: [r.model_id.toUpperCase()],
                          textposition: "top center" as const,
                          textfont: { size: 9, color: MODEL_COLORS[r.model_id] ?? "#94a3b8", family: "JetBrains Mono" },
                          marker: {
                            color: MODEL_COLORS[r.model_id] ?? "#94a3b8",
                            size: selectedModels.includes(r.model_id) ? 14 : 10,
                            opacity: selectedModels.includes(r.model_id) ? 1 : 0.45,
                            line: {
                              color: selectedModels.includes(r.model_id) ? "#fff" : "transparent",
                              width: 1.5,
                            },
                            symbol: "circle",
                          },
                          hovertemplate: `<b>${r.model_id.toUpperCase()}</b><br>Size: %{x:.3g} MB<br>Val RMSE: %{y:.2f} µA<extra></extra>`,
                          showlegend: false,
                        })),
                        // Pareto frontier line
                        {
                          type: "scatter" as const,
                          mode: "lines" as const,
                          name: "Pareto frontier",
                          x: paretoFrontier.map((r) => r.size_MB),
                          y: paretoFrontier.map((r) => r.rmse_val_uA),
                          line: { color: "rgba(245,158,11,0.6)", width: 1.5, dash: "dot" },
                          hoverinfo: "skip" as const,
                          showlegend: true,
                        },
                      ]}
                      layout={DarkLayout({
                        height: 380,
                        margin: { l: 68, r: 24, t: 32, b: 56 },
                        xaxis: {
                          title: { text: "Model Size (MB)", font: { size: 11, color: "#94a3b8" } },
                          type: "log",
                          gridcolor: "rgba(255,255,255,0.04)",
                          tickfont: { size: 10, family: "JetBrains Mono" },
                          rangemode: "tozero",
                        },
                        yaxis: {
                          title: { text: "Val RMSE (µA)", font: { size: 11, color: "#94a3b8" } },
                          gridcolor: "rgba(255,255,255,0.04)",
                          tickfont: { size: 10, family: "JetBrains Mono" },
                        },
                        legend: {
                          bgcolor: "rgba(13,19,33,0.7)", bordercolor: "rgba(245,158,11,0.2)", borderwidth: 1,
                          font: { size: 10, color: "#f1f5f9" }, x: 0.45, y: 0.98,
                        },
                        shapes: [
                          {
                            type: "rect",
                            x0: -0.5, x1: 1.5, y0: 25, y1: 32,
                            fillcolor: "rgba(16,185,129,0.04)",
                            line: { width: 0 },
                            layer: "below",
                          } as Plotly.Shape,
                        ],
                        annotations: [
                          {
                            x: 0, y: 0.97, xref: "paper", yref: "paper",
                            text: "← smaller & more accurate = better",
                            font: { size: 9, color: "rgba(255,255,255,0.2)", family: "Inter" },
                            showarrow: false, align: "left",
                          } as Partial<Plotly.Annotations>,
                        ],
                      })}
                      config={PlotConfig}
                      style={{ width: "100%", height: 380 }}
                      useResizeHandler
                    />
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          </Grid>

          {/* Axis legend for radar */}
          <Box sx={{ mt: 2, p: 2, borderRadius: 2, background: "rgba(15,25,35,0.7)", border: "1px solid rgba(255,255,255,0.05)" }}>
            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.65rem", fontWeight: 600, display: "block", mb: 1 }}>
              Radar Axis Definitions
            </Typography>
            <Grid container spacing={1.5}>
              {[
                { axis: "Accuracy", desc: "Score from val RMSE — RF/LightGBM ~98, ANN ~55", color: "#10b981" },
                { axis: "NM4 R²", desc: "Test-MAT R² normalised to 0.90–1.00 range", color: "#a78bfa" },
                { axis: "Deployment", desc: "Composite size + speed + cold-start score from model metadata", color: "#22d3ee" },
                { axis: "Size Eff.", desc: "Inverse log-normalised model size (smaller file = higher score)", color: "#f59e0b" },
                { axis: "SR Interp.", desc: "Score from test-SR RMSE (scan rate interpolation at SR=50)", color: "#f472b6" },
              ].map((item) => (
                <Grid item xs={12} sm={6} md={4} lg="auto" key={item.axis} sx={{ flex: "1 1 200px" }}>
                  <Box sx={{ display: "flex", alignItems: "flex-start", gap: 0.75 }}>
                    <Box sx={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: item.color, mt: 0.5, flexShrink: 0 }} />
                    <Box>
                      <Typography variant="caption" sx={{ color: item.color, fontWeight: 700, fontSize: "0.65rem" }}>{item.axis}:</Typography>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", display: "block" }}>{item.desc}</Typography>
                    </Box>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Box>
      )}
    </Box>
  );
}
