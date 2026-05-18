import React, { useEffect, useMemo } from "react";
import {
  Box, Card, CardContent, Typography,
  Table, TableBody, TableCell, TableHead, TableRow,
  Chip, LinearProgress, Skeleton, alpha,
} from "@mui/material";
import {
  LeaderboardRounded, CheckCircleRounded, FlashOnRounded,
  HistoryRounded, BarChartRounded, ElectricBoltRounded,
  BiotechRounded, InsightsRounded, ScienceRounded,
  TrendingUpRounded, ShowChartRounded,
} from "@mui/icons-material";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Plot from "react-plotly.js";
import { useLeaderboard } from "../hooks/useLeaderboard";
import { useHealth } from "../hooks/useHealth";
import { usePrediction } from "../hooks/usePrediction";
import { predict as apiPredict } from "../api/endpoints";
import GlowButton from "../components/common/GlowButton";
import ModelBadge from "../components/common/ModelBadge";
import { MODEL_COLORS } from "../constants/models";
import { fmtMB, rmseColor, r2Color } from "../utils/formatters";
import { cardVariants } from "../animations/variants";
import { ROUTES } from "../constants/routes";

const MotionTableRow = motion(TableRow);

const QUICK_INSIGHTS = [
  { title: "Best Generalisation", model: "GRU",          color: "#a78bfa", icon: "🧠", metric: "R² = 0.9751 on NM4",          text: "Gated recurrent units model sweep directionality sequentially — highest cross-material transfer R² on unseen NM4." },
  { title: "Best Deployment",    model: "LightGBM",     color: "#22d3ee", icon: "⚡", metric: "1.7 MB · ~5 ms",               text: "Within 1% of RF validation RMSE at ~1/360th the size — optimal accuracy-footprint trade-off for deployment." },
  { title: "Best Accuracy",      model: "Random Forest",color: "#3b82f6", icon: "🌲", metric: "26.59 µA RMSE (val)",           text: "Ensemble averaging over 300 trees — lowest validation RMSE, best within-distribution interpolation performance." },
  { title: "Cross-Material",     model: "GRU",          color: "#f472b6", icon: "🔬", metric: "R² 0.9751 — unseen NM4",        text: "GRU predicts ZnO/Co₃O₄ with no NM4 training examples — generalises CV topology, not material-specific patterns." },
  { title: "Smallest Model",     model: "ANN",          color: "#facc15", icon: "💨", metric: "176 KB",                        text: "Smallest model at 176 KB — a point-wise MLP research baseline without sequential state representation." },
  { title: "Best Regularisation",model: "XGBoost",      color: "#fb923c", icon: "📊", metric: "28.83 µA at 268 KB",           text: "L1+L2 regularised boosting — competitive accuracy with excellent complexity control and 268 KB footprint." },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const { leaderboard, summary, loading: lbLoading } = useLeaderboard();
  const { health, loading: healthLoading } = useHealth(30_000);
  const { results: predResults } = usePrediction();

  const [sparkData, setSparkData] = React.useState<{ potential: number[]; current: number[] } | null>(null);
  const [sparkLoading, setSparkLoading] = React.useState(false);
  useEffect(() => {
    if (sparkData || sparkLoading) return;
    let mounted = true;
    setSparkLoading(true);
    apiPredict({ model_name: "lightgbm", material_id: "NM1", scan_rate_mVs: 30 })
      .then((r) => { if (mounted) setSparkData({ potential: r.potential_V, current: r.predicted_current_uA }); })
      .catch(() => {})
      .finally(() => { if (mounted) setSparkLoading(false); });
    return () => { mounted = false; };
  }, []); // eslint-disable-line

  const recentPredictions = useMemo(
    () => Object.values(predResults).slice(-5).reverse(),
    [predResults]
  );

  const best      = leaderboard[0];
  const bestDeploy = leaderboard.find((r) => r.model_id === "lightgbm");
  const bestExtrap = leaderboard.find((r) => r.model_id === "gru");

  /* ── Shared card style ─────────────────────────────────────────────── */
  const card = (border = "rgba(0,212,255,0.1)") => ({
    background: "rgba(15,25,35,0.82)",
    border: `1px solid ${border}`,
    height: "100%",
  });

  return (
    <Box sx={{ width: "100%", maxWidth: 1600, mx: "auto", minWidth: 0 }}>

      {/* ══ HERO BANNER ══════════════════════════════════════════════════ */}
      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}>
        <Card sx={{
          mb: 2, overflow: "hidden", position: "relative",
          background: "linear-gradient(135deg, rgba(0,10,20,0.97) 0%, rgba(8,4,26,0.97) 50%, rgba(0,14,22,0.97) 100%)",
          border: "1px solid rgba(0,212,255,0.18)",
        }}>
          {/* Decorative orbs */}
          <Box sx={{ position: "absolute", top: -28, left: -28, width: 160, height: 160, borderRadius: "50%", background: "radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 70%)", pointerEvents: "none" }} />
          <Box sx={{ position: "absolute", bottom: -18, right: 40, width: 130, height: 130, borderRadius: "50%", background: "radial-gradient(circle, rgba(139,92,246,0.055) 0%, transparent 70%)", pointerEvents: "none" }} />

          <CardContent sx={{ p: { xs: 2, md: 2.5 }, position: "relative", zIndex: 1 }}>
            {/* Two columns: title left, KPI grid right */}
            <Box sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "1fr auto" },
              gap: { xs: 2, md: 3 },
              alignItems: "center",
              minWidth: 0,
            }}>

              {/* Left: title + badges */}
              <Box sx={{ minWidth: 0 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                  <Box component={motion.div} animate={{ rotate: [0, 360] }} transition={{ duration: 20, repeat: Infinity, ease: "linear" }} sx={{ fontSize: 20, flexShrink: 0 }}>⚡</Box>
                  <Chip label="RESEARCH PLATFORM" size="small"
                    sx={{ fontSize: "0.55rem", fontWeight: 800, height: 18, backgroundColor: alpha("#00d4ff", 0.1), color: "#00d4ff", border: "1px solid rgba(0,212,255,0.25)", letterSpacing: "0.08em" }} />
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 800, color: "#f1f5f9", lineHeight: 1.22, mb: 0.75, fontSize: { xs: "1.05rem", md: "1.3rem" } }}>
                  AI-Driven ZnO Supercapacitor
                  <Box component="span" sx={{ color: "#00d4ff" }}> Research Platform</Box>
                </Typography>
                <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.38)", lineHeight: 1.6, mb: 1.5, fontSize: "0.78rem", maxWidth: 460 }}>
                  651-point cyclic voltammetry prediction across 6 ML architectures and 4 ZnO nanocomposites.
                </Typography>
                <Box sx={{ display: "flex", gap: 0.65, flexWrap: "wrap" }}>
                  {[
                    { label: "R² > 0.96", color: "#10b981" },
                    { label: "6 ML Models", color: "#22d3ee" },
                    { label: "24 CV Curves", color: "#a78bfa" },
                    { label: "Real-time", color: "#f59e0b" },
                    { label: "4 Nanocomposites", color: "#f472b6" },
                  ].map((b) => (
                    <Chip key={b.label} label={b.label} size="small"
                      sx={{ fontSize: "0.6rem", height: 18, backgroundColor: alpha(b.color, 0.1), color: b.color, border: `1px solid ${alpha(b.color, 0.2)}`, fontWeight: 600 }} />
                  ))}
                </Box>
              </Box>

              {/* Right: 4 live KPI tiles — 2×2 on all sizes */}
              <Box sx={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: { xs: 0.75, md: 1 },
                minWidth: 0,
              }}>
                {[
                  { label: "Best RMSE",    value: lbLoading ? "—" : `${best?.rmse_val_uA?.toFixed(1) ?? "—"} µA`,  sub: best?.model_display ?? "…",      color: "#22d3ee" },
                  { label: "Best Generalisation R²", value: lbLoading ? "—" : (bestExtrap?.r2_test_mat?.toFixed(4) ?? "—"), sub: "GRU · NM4 cross-material",         color: "#a78bfa" },
                  { label: "Deploy",       value: lbLoading ? "—" : fmtMB(bestDeploy?.size_MB),                   sub: "LightGBM · ~5 ms",                color: "#10b981" },
                  { label: "Loaded",       value: healthLoading ? "—" : `${health?.models_loaded?.length ?? 0}/6`, sub: "hot + lazy tiers",                color: "#f59e0b" },
                ].map((kpi) => (
                  <Box key={kpi.label} sx={{ p: 1.25, borderRadius: 1.5, background: alpha(kpi.color, 0.07), border: `1px solid ${alpha(kpi.color, 0.17)}`, textAlign: "center", minWidth: 0 }}>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.28)", fontSize: "0.55rem", display: "block", mb: 0.25 }}>{kpi.label}</Typography>
                    {lbLoading || healthLoading
                      ? <Skeleton width={52} height={20} sx={{ mx: "auto" }} />
                      : <Typography sx={{ color: kpi.color, fontFamily: "JetBrains Mono", fontWeight: 800, fontSize: { xs: "0.78rem", md: "0.9rem" }, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{kpi.value}</Typography>}
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.55rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>{kpi.sub}</Typography>
                  </Box>
                ))}
              </Box>
            </Box>
          </CardContent>
        </Card>
      </motion.div>

      {/* ══ MAIN GRID: CSS Grid — no MUI Grid overflow issues ════════════ */}
      <Box sx={{
        display: "grid",
        gridTemplateColumns: {
          xs: "1fr",
          lg: "minmax(0, 7fr) minmax(0, 5fr)",
        },
        gap: 2,
        mb: 2,
        alignItems: "start",
      }}>

        {/* ── LEFT: leaderboard + sparkline + RMSE snapshot ───────────── */}
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>

          {/* Leaderboard */}
          <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ minWidth: 0 }}>
            <Card sx={card()}>
              <CardContent sx={{ p: 2.5 }}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.75, minWidth: 0 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.85 }}>
                    <LeaderboardRounded sx={{ color: "#00d4ff", fontSize: 16 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, fontSize: "0.85rem" }}>Model Leaderboard</Typography>
                  </Box>
                  <GlowButton variant="outlined" size="small" onClick={() => navigate(ROUTES.BENCHMARKS)}
                    sx={{ fontSize: "0.68rem", py: 0.35, px: 1.1, flexShrink: 0 }}>
                    Full Report
                  </GlowButton>
                </Box>

                {/* Overflow wrapper prevents table from forcing card width */}
                <Box sx={{ overflowX: "auto" }}>
                  {lbLoading
                    ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} variant="rounded" height={32} sx={{ mb: 0.55 }} />)
                    : (
                      <Table size="small" sx={{ tableLayout: "fixed", width: "100%" }}>
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ width: 28, py: 0.6, fontSize: "0.67rem" }}>#</TableCell>
                            <TableCell sx={{ py: 0.6, fontSize: "0.67rem" }}>Model</TableCell>
                            <TableCell align="right" sx={{ width: 90, py: 0.6, fontSize: "0.67rem" }}>RMSE Val</TableCell>
                            {/* R² column hidden on xs — too wide for phone screens */}
                            <TableCell align="right" sx={{ width: 110, py: 0.6, fontSize: "0.67rem", display: { xs: "none", sm: "table-cell" } }}>R² TestMAT</TableCell>
                            {/* Size column hidden on xs/sm */}
                            <TableCell align="right" sx={{ width: 70, py: 0.6, fontSize: "0.67rem", display: { xs: "none", md: "table-cell" } }}>Size</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {leaderboard.map((row, i) => (
                            <MotionTableRow key={row.model_id}
                              initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                              sx={{ "&:last-child td": { border: 0 }, "&:hover": { background: "rgba(0,212,255,0.025)" }, transition: "background 0.12s" }}>
                              <TableCell sx={{ py: 0.7 }}>
                                <Typography variant="caption" sx={{ fontWeight: 700, color: i === 0 ? "#f59e0b" : i === 1 ? "#94a3b8" : "rgba(255,255,255,0.22)", fontFamily: "JetBrains Mono", fontSize: "0.78rem" }}>
                                  {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : `${row.rank}`}
                                </Typography>
                              </TableCell>
                              <TableCell sx={{ py: 0.7 }}>
                                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, overflow: "hidden" }}>
                                  <Box sx={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: MODEL_COLORS[row.model_id] ?? "#94a3b8", flexShrink: 0 }} />
                                  <Typography variant="body2" sx={{ fontWeight: 600, fontSize: "0.8rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                    {row.model_display}
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell align="right" sx={{ py: 0.7 }}>
                                <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.76rem", color: rmseColor(row.rmse_val_uA), fontWeight: 600, whiteSpace: "nowrap" }}>
                                  {row.rmse_val_uA.toFixed(1)} µA
                                </Typography>
                              </TableCell>
                              <TableCell align="right" sx={{ py: 0.7, display: { xs: "none", sm: "table-cell" } }}>
                                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 0.6 }}>
                                  <LinearProgress variant="determinate" value={row.r2_test_mat * 100}
                                    sx={{ width: 32, height: 3, borderRadius: 2, flexShrink: 0, backgroundColor: "rgba(255,255,255,0.06)", "& .MuiLinearProgress-bar": { backgroundColor: r2Color(row.r2_test_mat), borderRadius: 2 } }} />
                                  <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.74rem", color: r2Color(row.r2_test_mat), fontWeight: 600, whiteSpace: "nowrap" }}>
                                    {row.r2_test_mat.toFixed(4)}
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell align="right" sx={{ py: 0.7, display: { xs: "none", md: "table-cell" } }}>
                                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.32)", fontFamily: "JetBrains Mono", fontSize: "0.68rem", whiteSpace: "nowrap" }}>
                                  {fmtMB(row.size_MB)}
                                </Typography>
                              </TableCell>
                            </MotionTableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}

                  {/* Key findings inline strip */}
                  {!lbLoading && summary && (
                    <Box sx={{ mt: 1.75, pt: 1.5, borderTop: "1px solid rgba(255,255,255,0.05)", display: "flex", gap: 1.25, flexWrap: "wrap" }}>
                      {[
                        { label: "Accuracy",      value: summary.key_findings?.best_by_val_rmse   ?? "RF",        color: "#3b82f6" },
                        { label: "Generalisation", value: summary.key_findings?.best_by_testmat_r2 ?? "GRU",       color: "#a78bfa" },
                        { label: "Deployment",    value: summary.key_findings?.best_for_deployment ?? "LightGBM", color: "#22d3ee" },
                      ].map((f) => (
                        <Box key={f.label} sx={{ display: "flex", alignItems: "center", gap: 0.6 }}>
                          <CheckCircleRounded sx={{ color: f.color, fontSize: 12 }} />
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.6rem" }}>{f.label}:</Typography>
                          <Typography variant="caption" sx={{ color: f.color, fontWeight: 700, fontSize: "0.65rem", fontFamily: "JetBrains Mono" }}>{f.value}</Typography>
                        </Box>
                      ))}
                    </Box>
                  )}
                </Box>
              </CardContent>
            </Card>
          </motion.div>

          {/* Live CV Sparkline — wider left column = better chart */}
          <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ minWidth: 0 }}>
            <Card sx={{ background: "rgba(7,11,20,0.92)", border: "1px solid rgba(34,211,238,0.12)" }}>
              <CardContent sx={{ p: 1.75 }}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.65, minWidth: 0 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.7, overflow: "hidden" }}>
                    <ShowChartRounded sx={{ color: MODEL_COLORS["lightgbm"], fontSize: 13, flexShrink: 0 }} />
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.32)", fontSize: "0.62rem", overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
                      LightGBM · NM1 · 30 mV/s — Live Cyclic Voltammogram
                    </Typography>
                  </Box>
                  <GlowButton variant="outlined" size="small" startIcon={<FlashOnRounded sx={{ fontSize: 11 }} />}
                    onClick={() => navigate(ROUTES.PREDICTION_STUDIO)}
                    sx={{ fontSize: "0.6rem", py: 0.2, px: 0.85, borderColor: "rgba(34,211,238,0.18)", color: "#22d3ee", flexShrink: 0 }}>
                    Studio
                  </GlowButton>
                </Box>
                {sparkLoading
                  ? <Skeleton variant="rounded" height={140} />
                  : sparkData
                    ? (
                      <Plot
                        data={[{
                          x: sparkData.potential, y: sparkData.current,
                          type: "scatter", mode: "lines",
                          line: { color: MODEL_COLORS["lightgbm"], width: 1.75, shape: "spline", smoothing: 0.4 },
                          fill: "tozeroy", fillcolor: `${MODEL_COLORS["lightgbm"]}12`,
                          hovertemplate: "V: %{x:.3f} V<br>I: %{y:.1f} µA<extra></extra>",
                        }]}
                        layout={{
                          paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                          margin: { l: 42, r: 10, t: 4, b: 28 }, height: 140, autosize: true,
                          xaxis: { title: { text: "Potential (V)", font: { size: 9, color: "#94a3b8" } }, tickfont: { size: 9, color: "#8892a4", family: "JetBrains Mono" }, gridcolor: "rgba(255,255,255,0.022)", zeroline: false },
                          yaxis: { title: { text: "Current (µA)", font: { size: 9, color: "#94a3b8" } }, tickfont: { size: 9, color: "#8892a4", family: "JetBrains Mono" }, gridcolor: "rgba(255,255,255,0.022)", zeroline: true, zerolinecolor: "rgba(255,255,255,0.08)", zerolinewidth: 1 },
                          hoverlabel: { bgcolor: "#111c2d", bordercolor: "#22d3ee", font: { size: 10, color: "#f1f5f9", family: "JetBrains Mono" } },
                        }}
                        config={{ responsive: true, displayModeBar: false }}
                        style={{ width: "100%", height: 140 }} useResizeHandler
                      />
                    )
                    : (
                      <Box sx={{ height: 140, display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.16)", fontSize: "0.6rem" }}>Backend offline — start the API to see live CV</Typography>
                      </Box>
                    )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Val RMSE at a Glance — animated horizontal bars */}
          <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ minWidth: 0 }}>
            <Card sx={card()}>
              <CardContent sx={{ p: 1.75 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.85, mb: 1.5, minWidth: 0 }}>
                  <TrendingUpRounded sx={{ color: "#a78bfa", fontSize: 16 }} />
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, fontSize: "0.85rem" }}>Val RMSE at a Glance</Typography>
                  <Chip label="lower = better" size="small"
                    sx={{ fontSize: "0.55rem", height: 17, ml: "auto", backgroundColor: alpha("#a78bfa", 0.1), color: "#a78bfa", border: "1px solid rgba(167,139,250,0.2)" }} />
                </Box>
                {lbLoading
                  ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} variant="rounded" height={24} sx={{ mb: 0.75 }} />)
                  : leaderboard.map((row, i) => {
                      const maxRmse = Math.max(...leaderboard.map((r) => r.rmse_val_uA));
                      const pct = (row.rmse_val_uA / maxRmse) * 100;
                      const color = MODEL_COLORS[row.model_id] ?? "#94a3b8";
                      return (
                        <Box key={row.model_id} sx={{ mb: i < leaderboard.length - 1 ? 1 : 0, minWidth: 0 }}>
                          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.35 }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 0.65 }}>
                              <Box sx={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: color, flexShrink: 0 }} />
                              <Typography variant="caption" sx={{ color: "#f1f5f9", fontWeight: 600, fontSize: "0.71rem" }}>
                                {row.model_display}
                              </Typography>
                            </Box>
                            <Typography variant="caption" sx={{ color: rmseColor(row.rmse_val_uA), fontFamily: "JetBrains Mono", fontSize: "0.69rem", fontWeight: 700, flexShrink: 0 }}>
                              {row.rmse_val_uA.toFixed(2)} µA
                            </Typography>
                          </Box>
                          <Box sx={{ height: 5, borderRadius: 3, backgroundColor: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${pct}%` }}
                              transition={{ duration: 0.65, delay: 0.15 + i * 0.08, ease: [0.16, 1, 0.3, 1] }}
                              style={{ height: "100%", borderRadius: 3, background: `linear-gradient(90deg, ${color}99 0%, ${color} 100%)` }}
                            />
                          </Box>
                        </Box>
                      );
                    })
                }
              </CardContent>
            </Card>
          </motion.div>

        </Box>

        {/* ── RIGHT: stacked utility panels ──────────────────────────── */}
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>

          {/* Recent activity */}
          <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ minWidth: 0, flex: "1 1 auto" }}>
            <Card sx={card("rgba(255,255,255,0.07)")}>
              <CardContent sx={{ p: 1.75 }}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.1 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.7 }}>
                    <HistoryRounded sx={{ color: "rgba(255,255,255,0.25)", fontSize: 14 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, fontSize: "0.8rem" }}>Recent Activity</Typography>
                  </Box>
                  <GlowButton variant="outlined" size="small" startIcon={<FlashOnRounded sx={{ fontSize: 11 }} />}
                    onClick={() => navigate(ROUTES.PREDICTION_STUDIO)}
                    sx={{ fontSize: "0.6rem", py: 0.2, px: 0.85, flexShrink: 0 }}>
                    New
                  </GlowButton>
                </Box>
                {recentPredictions.length === 0 ? (
                  <Box sx={{ py: 2, textAlign: "center" }}>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.18)", fontSize: "0.66rem", display: "block", mb: 1.1 }}>
                      No predictions this session
                    </Typography>
                    <GlowButton variant="contained" size="small" startIcon={<FlashOnRounded sx={{ fontSize: 12 }} />}
                      onClick={() => navigate(ROUTES.PREDICTION_STUDIO)}
                      sx={{ fontSize: "0.68rem", background: "linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)", color: "#070b14", fontWeight: 700 }}>
                      Try a Prediction
                    </GlowButton>
                  </Box>
                ) : (
                  recentPredictions.map((r, i) => {
                    const color = MODEL_COLORS[r.model_name] ?? "#94a3b8";
                    const peak  = Math.max(r.statistics.peak_anodic_uA, Math.abs(r.statistics.peak_cathodic_uA));
                    return (
                      <Box key={`${r.model_name}_${r.material_id}_${r.scan_rate_mVs}`} onClick={() => navigate(ROUTES.PREDICTION_STUDIO)}
                        sx={{ display: "flex", alignItems: "center", gap: 1.1, py: 0.75, px: 1, borderRadius: 1.5, mb: 0.45, cursor: "pointer", background: i === 0 ? alpha(color, 0.055) : "transparent", border: `1px solid ${i === 0 ? alpha(color, 0.13) : "rgba(255,255,255,0.035)"}`, "&:hover": { background: alpha(color, 0.08) }, transition: "background 0.12s", minWidth: 0 }}>
                        <Box sx={{ width: 5, height: 5, borderRadius: "50%", backgroundColor: color, flexShrink: 0 }} />
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Typography variant="caption" sx={{ color: i === 0 ? color : "rgba(255,255,255,0.5)", fontWeight: 700, fontSize: "0.68rem", fontFamily: "JetBrains Mono", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>
                            {r.model_name.toUpperCase()} · {r.material_id} · {r.scan_rate_mVs} mV/s
                          </Typography>
                        </Box>
                        <Typography variant="caption" sx={{ color: "#10b981", fontFamily: "JetBrains Mono", fontSize: "0.65rem", fontWeight: 700, flexShrink: 0 }}>
                          {peak.toFixed(1)} µA
                        </Typography>
                      </Box>
                    );
                  })
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* System status */}
          <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ minWidth: 0 }}>
            <Card sx={card()}>
              <CardContent sx={{ p: 1.75 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.25, fontSize: "0.8rem" }}>System Status</Typography>
                {healthLoading
                  ? Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} variant="rounded" height={26} sx={{ mb: 0.65 }} />)
                  : (
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.65 }}>
                      {[
                        { label: "API Backend",    ok: health?.status === "ok",                        detail: `v${health?.version ?? "?"} · ${health?.environment ?? "?"}` },
                        { label: "Scaler Files",   ok: health?.scalers_ok,                             detail: "scalers.json loaded" },
                        { label: "Model Registry", ok: (health?.models_loaded?.length ?? 0) > 0,       detail: `${health?.models_loaded?.length ?? 0} active` },
                      ].map((item) => (
                        <Box key={item.label} sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", px: 1.1, py: 0.6, borderRadius: 1.25, backgroundColor: item.ok ? "rgba(16,185,129,0.045)" : "rgba(239,68,68,0.045)", border: `1px solid ${item.ok ? "rgba(16,185,129,0.13)" : "rgba(239,68,68,0.13)"}`, minWidth: 0 }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 0.65, overflow: "hidden" }}>
                            <CheckCircleRounded sx={{ fontSize: 11, color: item.ok ? "#10b981" : "#ef4444", flexShrink: 0 }} />
                            <Typography variant="body2" sx={{ fontWeight: 600, fontSize: "0.74rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.label}</Typography>
                          </Box>
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.6rem", flexShrink: 0, ml: 0.5 }}>{item.detail}</Typography>
                        </Box>
                      ))}
                      {health?.models_loaded && health.models_loaded.length > 0 && (
                        <Box sx={{ display: "flex", gap: 0.55, flexWrap: "wrap", mt: 0.2 }}>
                          {health.models_loaded.map((m) => <ModelBadge key={m} modelId={m} size="small" />)}
                        </Box>
                      )}
                    </Box>
                  )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Quick navigation 2×2 */}
          <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ minWidth: 0 }}>
            <Box sx={{
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: 1,
            }}>
              {[
                { label: "Research Analytics", icon: <BarChartRounded sx={{ fontSize: 16 }} />, color: "#f59e0b", route: ROUTES.RESEARCH_ANALYTICS, desc: "11 analysis tabs" },
                { label: "Model Comparison",   icon: <ScienceRounded  sx={{ fontSize: 16 }} />, color: "#a78bfa", route: ROUTES.MODEL_COMPARISON,  desc: "Side-by-side CV" },
                { label: "Encyclopedia",       icon: <BiotechRounded  sx={{ fontSize: 16 }} />, color: "#22d3ee", route: ROUTES.MODEL_ENCYCLOPEDIA, desc: "Architecture dives" },
                { label: "About Research",     icon: <InsightsRounded sx={{ fontSize: 16 }} />, color: "#10b981", route: ROUTES.ABOUT,              desc: "Methodology" },
              ].map((item) => (
                <Card key={item.label}
                  sx={{ background: alpha(item.color, 0.04), border: `1px solid ${alpha(item.color, 0.16)}`, cursor: "pointer", "&:hover": { border: `1px solid ${alpha(item.color, 0.35)}`, background: alpha(item.color, 0.07) }, transition: "all 0.16s", minWidth: 0 }}
                  onClick={() => navigate(item.route)}>
                  <CardContent sx={{ p: 1.4, textAlign: "center" }}>
                    <Box sx={{ color: item.color, mb: 0.5 }}>{item.icon}</Box>
                    <Typography variant="caption" sx={{ color: item.color, fontWeight: 700, fontSize: "0.64rem", display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.label}</Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.56rem" }}>{item.desc}</Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </motion.div>
        </Box>
      </Box>

      {/* ══ QUICK SCIENTIFIC INSIGHTS ════════════════════════════════════ */}
      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.26, duration: 0.42 }}>
        <Box sx={{ mb: 1.25, display: "flex", alignItems: "center", gap: 1 }}>
          <ElectricBoltRounded sx={{ color: "#f59e0b", fontSize: 16 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 700, color: "#f1f5f9", fontSize: "0.9rem" }}>Quick Research Insights</Typography>
          <Chip label="6 findings" size="small" sx={{ fontSize: "0.56rem", height: 17, backgroundColor: alpha("#f59e0b", 0.1), color: "#f59e0b", border: "1px solid rgba(245,158,11,0.2)" }} />
        </Box>
        <Box sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, minmax(0, 1fr))" },
          gap: 1.5,
        }}>
          {QUICK_INSIGHTS.map((ins, i) => (
            <motion.div key={ins.title} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 + i * 0.05, duration: 0.3 }} style={{ minWidth: 0 }}>
              <Card sx={{
                height: "100%",
                background: `linear-gradient(135deg, ${alpha(ins.color, 0.06)} 0%, rgba(15,25,35,0.9) 100%)`,
                border: `1px solid ${alpha(ins.color, 0.18)}`,
                borderTop: `3px solid ${ins.color}`,
                "&:hover": { borderColor: alpha(ins.color, 0.34) },
                transition: "border-color 0.16s",
                minWidth: 0,
              }}>
                <CardContent sx={{ p: 1.75 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.9 }}>
                    <Typography sx={{ fontSize: 16 }}>{ins.icon}</Typography>
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.56rem", display: "block" }}>{ins.title}</Typography>
                      <Typography variant="subtitle2" sx={{ color: ins.color, fontWeight: 700, fontSize: "0.76rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ins.model}</Typography>
                    </Box>
                  </Box>
                  <Box sx={{ px: 1, py: 0.45, borderRadius: 1.1, background: alpha(ins.color, 0.08), border: `1px solid ${alpha(ins.color, 0.15)}`, mb: 0.9 }}>
                    <Typography variant="caption" sx={{ color: ins.color, fontFamily: "JetBrains Mono", fontWeight: 700, fontSize: "0.62rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>{ins.metric}</Typography>
                  </Box>
                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.38)", fontSize: "0.66rem", lineHeight: 1.52 }}>{ins.text}</Typography>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </Box>
      </motion.div>
    </Box>
  );
}
