import React, { useMemo } from "react";
import {
  Box, Grid, Card, CardContent, Typography,
  Table, TableBody, TableCell, TableHead, TableRow,
  Skeleton, Chip, LinearProgress, alpha, Divider,
} from "@mui/material";
import {
  EmojiEventsRounded, BarChartRounded, RocketLaunchRounded,
  CompareArrowsRounded, MemoryRounded, TipsAndUpdatesRounded,
} from "@mui/icons-material";
import { motion } from "framer-motion";
import Plot from "react-plotly.js";
import { useLeaderboard } from "../hooks/useLeaderboard";
import SectionHeader from "../components/common/SectionHeader";
import ModelBadge from "../components/common/ModelBadge";
import { fmtMB, rmseColor, r2Color } from "../utils/formatters";
import { MODEL_COLORS } from "../constants/models";
import { cardVariants } from "../animations/variants";

const MotionTableRow = motion(TableRow);

// ── Correct partition keys from API ──────────────────────────────────────
const PARTITION_KEYS = ["train", "val", "test_SR", "test_MAT"] as const;
const PARTITION_LABELS: Record<string, string> = {
  train:    "Train (in-sample)",
  val:      "Val (SR=30 mV/s)",
  test_SR:  "Test-SR (SR=50 mV/s)",
  test_MAT: "Test-MAT (NM4)",
};
const PARTITION_COLORS: Record<string, string> = {
  train:    "#6366f1",
  val:      "#10b981",
  test_SR:  "#f59e0b",
  test_MAT: "#f472b6",
};

// ── Model metadata for publication table ─────────────────────────────────
const MODEL_METADATA: Record<string, {
  family: string; params: string; trainMin: number;
  inference: string; flops: string;
}> = {
  rf:       { family: "Ensemble / Tree",     params: "300 trees",    trainMin: 4,  inference: "~120 ms", flops: "~650K" },
  lightgbm: { family: "Gradient Boosting",   params: "278 trees",    trainMin: 2,  inference: "~5 ms",   flops: "~56K"  },
  xgboost:  { family: "Gradient Boosting",   params: "110 trees",    trainMin: 6,  inference: "<1 ms",   flops: "~40K"  },
  gru:      { family: "Recurrent (Deep)",     params: "24 545",       trainMin: 20, inference: "~125 ms", flops: "~32M"  },
  lstm:     { family: "Recurrent (Deep)",     params: "32 161",       trainMin: 25, inference: "~140 ms", flops: "~42M"  },
  ann:      { family: "Dense MLP (Deep)",     params: "11 777",       trainMin: 10, inference: "~120 ms", flops: "~12M"  },
};

const DarkLayout = (overrides: Partial<Plotly.Layout> = {}): Partial<Plotly.Layout> => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(13,19,33,0.5)",
  font: { family: "Inter, sans-serif", color: "#8892a4", size: 11 },
  margin: { l: 60, r: 30, t: 20, b: 55 },
  autosize: true,
  hoverlabel: { bgcolor: "#111c2d", bordercolor: "#00d4ff", font: { size: 12, color: "#f1f5f9", family: "JetBrains Mono" } },
  ...overrides,
});
const PlotCfg = { responsive: true, displayModeBar: false, displaylogo: false };

export default function BenchmarkResults() {
  const { leaderboard, comparison, loading } = useLeaderboard();

  // Partition lookup per model — uses CORRECT keys
  const partitionLookup = useMemo(() => {
    const map: Record<string, Record<string, { rmse: number; r2: number; mae: number | null }>> = {};
    comparison.forEach((row) => {
      if (!map[row.model_id]) map[row.model_id] = {};
      map[row.model_id][row.partition] = { rmse: row.rmse_uA, r2: row.r2, mae: row.mae_uA ?? null };
    });
    return map;
  }, [comparison]);

  // Efficiency frontier data for Plotly
  const efficiencyData = useMemo(() => leaderboard.map((r) => ({
    model: r.model_display,
    model_id: r.model_id,
    size_MB: r.size_MB,
    rmse_val: r.rmse_val_uA,
    r2_mat: r.r2_test_mat,
    color: MODEL_COLORS[r.model_id],
    params: MODEL_METADATA[r.model_id]?.params ?? "—",
  })), [leaderboard]);

  // Classical vs Deep split
  const classicalModels = leaderboard.filter((r) => ["rf", "lightgbm", "xgboost"].includes(r.model_id));
  const deepModels      = leaderboard.filter((r) => ["gru", "lstm", "ann"].includes(r.model_id));

  return (
    <Box sx={{ width: "100%", overflowX: "hidden" }}>
      <SectionHeader
        title="Benchmark Results"
        subtitle="Complete evaluation of 6 ML models across 4 data partitions — from in-sample training to cross-material transfer"
        accent="#22d3ee"
      />

      {/* ── Research headline stats ───────────────────────────────────── */}
      {!loading && leaderboard.length > 0 && (
        <motion.div variants={cardVariants} initial="initial" animate="animate">
          <Grid container spacing={2} sx={{ mb: 3 }}>
            {[
              { label: "Best Val RMSE", value: `${leaderboard[0]?.rmse_val_uA.toFixed(2)} µA`, sub: leaderboard[0]?.model_display, color: "#22d3ee", icon: <EmojiEventsRounded sx={{ fontSize: 18 }} /> },
              { label: "Best TestMAT R²", value: leaderboard.reduce((b, m) => (m.r2_test_mat > b.r2_test_mat ? m : b), leaderboard[0])?.r2_test_mat.toFixed(4), sub: "GRU — unseen NM4 material", color: "#a78bfa", icon: <BarChartRounded sx={{ fontSize: 18 }} /> },
              { label: "Smallest Model", value: "176 KB", sub: "Dense ANN — lightest footprint", color: "#10b981", icon: <MemoryRounded sx={{ fontSize: 18 }} /> },
              { label: "Fastest Inference", value: "<1 ms", sub: "XGBoost (per half-sweep)", color: "#f59e0b", icon: <RocketLaunchRounded sx={{ fontSize: 18 }} /> },
            ].map((s) => (
              <Grid item xs={6} sm={3} key={s.label}>
                <Box sx={{ p: 2, borderRadius: 2, background: alpha(s.color, 0.07), border: `1px solid ${alpha(s.color, 0.22)}`, display: "flex", alignItems: "center", gap: 1.5 }}>
                  <Box sx={{ color: s.color, opacity: 0.8 }}>{s.icon}</Box>
                  <Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.6rem", display: "block" }}>{s.label}</Typography>
                    <Typography sx={{ color: s.color, fontFamily: "JetBrains Mono", fontWeight: 800, fontSize: "0.95rem" }}>{s.value}</Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.28)", fontSize: "0.6rem" }}>{s.sub}</Typography>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>
        </motion.div>
      )}

      {/* ── Ranked leaderboard ───────────────────────────────────────── */}
      <motion.div variants={cardVariants} initial="initial" animate="animate">
        <Card sx={{ mb: 3, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.12)" }}>
          <CardContent sx={{ p: 2.5 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2.5 }}>
              <EmojiEventsRounded sx={{ color: "#00d4ff", fontSize: 18 }} />
              <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, letterSpacing: "0.05em" }}>
                RANKED LEADERBOARD — VALIDATION RMSE
              </Typography>
            </Box>
            {loading ? <Skeleton variant="rounded" height={260} /> : (
              <Box sx={{ overflowX: "auto" }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ width: 40 }}>Rank</TableCell>
                      <TableCell>Model</TableCell>
                      <TableCell align="right">RMSE Val (µA)</TableCell>
                      <TableCell align="right">R² Val</TableCell>
                      <TableCell align="right">RMSE Test-SR (µA)</TableCell>
                      <TableCell align="right">RMSE Test-MAT (µA)</TableCell>
                      <TableCell align="right">R² Test-MAT</TableCell>
                      <TableCell align="right">Size</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {leaderboard.map((row, i) => (
                      <MotionTableRow key={row.model_id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}
                        sx={{ "&:last-child td": { border: 0 }, backgroundColor: i === 0 ? alpha("#f59e0b", 0.03) : "transparent" }}>
                        <TableCell>
                          <Typography variant="caption" sx={{ fontWeight: 800, fontFamily: "JetBrains Mono", color: i === 0 ? "#f59e0b" : i < 3 ? "#94a3b8" : "rgba(255,255,255,0.3)" }}>
                            {i === 0 ? "①" : i === 1 ? "②" : i === 2 ? "③" : `${row.rank}`}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                            <Box sx={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: MODEL_COLORS[row.model_id], boxShadow: `0 0 6px ${MODEL_COLORS[row.model_id]}60`, flexShrink: 0 }} />
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>{row.model_display}</Typography>
                            {i === 0 && <Chip label="BEST VAL" size="small" sx={{ height: 16, fontSize: "0.55rem", fontWeight: 700, backgroundColor: alpha("#f59e0b", 0.15), color: "#f59e0b", border: "1px solid rgba(245,158,11,0.3)", "& .MuiChip-label": { px: 0.6 } }} />}
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.8rem", color: rmseColor(row.rmse_val_uA), fontWeight: 700 }}>{row.rmse_val_uA.toFixed(2)}</Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.8rem", color: r2Color(row.r2_val), fontWeight: 600 }}>{row.r2_val.toFixed(4)}</Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.78rem", color: rmseColor(row.rmse_test_sr_uA), fontWeight: 600 }}>{row.rmse_test_sr_uA.toFixed(2)}</Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.78rem", color: rmseColor(row.rmse_test_mat_uA), fontWeight: 600 }}>{row.rmse_test_mat_uA.toFixed(2)}</Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 1 }}>
                            <LinearProgress variant="determinate" value={Math.max(0, (row.r2_test_mat - 0.92) / 0.06 * 100)}
                              sx={{ width: 36, height: 4, borderRadius: 2, backgroundColor: "rgba(255,255,255,0.05)", "& .MuiLinearProgress-bar": { backgroundColor: r2Color(row.r2_test_mat) } }} />
                            <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.78rem", color: r2Color(row.r2_test_mat), fontWeight: 700 }}>{row.r2_test_mat.toFixed(4)}</Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontFamily: "JetBrains Mono" }}>{fmtMB(row.size_MB)}</Typography>
                        </TableCell>
                      </MotionTableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* ── RMSE cross-partition matrix — FIXED PARTITION KEYS ─────── */}
      <motion.div variants={cardVariants} initial="initial" animate="animate">
        <Card sx={{ mb: 3, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.1)" }}>
          <CardContent sx={{ p: 2.5, overflowX: "auto" }}>
            <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 2.5, letterSpacing: "0.05em" }}>
              RMSE (µA) ACROSS ALL PARTITIONS — 6 MODELS × 4 SPLITS
            </Typography>
            {loading ? <Skeleton variant="rounded" height={200} /> : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ minWidth: 130 }}>Model</TableCell>
                    {PARTITION_KEYS.map((p) => (
                      <TableCell key={p} align="center" sx={{ color: PARTITION_COLORS[p], fontWeight: 700, fontSize: "0.72rem" }}>
                        {PARTITION_LABELS[p]}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {leaderboard.map((row) => {
                    const pm = partitionLookup[row.model_id] ?? {};
                    return (
                      <TableRow key={row.model_id} sx={{ "&:last-child td": { border: 0 } }}>
                        <TableCell><ModelBadge modelId={row.model_id} size="small" /></TableCell>
                        {PARTITION_KEYS.map((p) => {
                          const v = pm[p]?.rmse ?? null;
                          return (
                            <TableCell key={p} align="center">
                              <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.82rem", color: v != null ? rmseColor(v) : "rgba(255,255,255,0.2)", fontWeight: 700 }}>
                                {v?.toFixed(2) ?? "—"}
                              </Typography>
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            )}
            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", mt: 1.5, display: "block", fontSize: "0.65rem" }}>
              Color: <span style={{ color: "#22d3ee" }}>≤28 µA excellent</span> · <span style={{ color: "#f59e0b" }}>≤40 µA good</span> · <span style={{ color: "#ef4444" }}>&gt;40 µA poor</span>
              {" | "}Train=in-sample · Val=SR=30 interpolation · Test-SR=SR=50 interpolation · Test-MAT=NM4 cross-material
            </Typography>
          </CardContent>
        </Card>
      </motion.div>

      {/* ── R² cross-partition matrix ────────────────────────────────── */}
      <motion.div variants={cardVariants} initial="initial" animate="animate">
        <Card sx={{ mb: 3, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(16,185,129,0.1)" }}>
          <CardContent sx={{ p: 2.5, overflowX: "auto" }}>
            <Typography variant="subtitle2" sx={{ color: "#10b981", fontWeight: 700, mb: 2.5, letterSpacing: "0.05em" }}>
              R² ACROSS ALL PARTITIONS — 6 MODELS × 4 SPLITS
            </Typography>
            {loading ? <Skeleton variant="rounded" height={200} /> : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ minWidth: 130 }}>Model</TableCell>
                    {PARTITION_KEYS.map((p) => (
                      <TableCell key={p} align="center" sx={{ color: PARTITION_COLORS[p], fontWeight: 700, fontSize: "0.72rem" }}>
                        {PARTITION_LABELS[p]}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {leaderboard.map((row) => {
                    const pm = partitionLookup[row.model_id] ?? {};
                    return (
                      <TableRow key={row.model_id} sx={{ "&:last-child td": { border: 0 } }}>
                        <TableCell><ModelBadge modelId={row.model_id} size="small" /></TableCell>
                        {PARTITION_KEYS.map((p) => {
                          const v = pm[p]?.r2 ?? null;
                          return (
                            <TableCell key={p} align="center">
                              <Typography variant="body2" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.82rem", color: v != null ? r2Color(v) : "rgba(255,255,255,0.2)", fontWeight: 700 }}>
                                {v?.toFixed(4) ?? "—"}
                              </Typography>
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* ── Efficiency frontier chart + architecture table ───────────── */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={7}>
          <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(124,58,237,0.15)", height: "100%" }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography variant="subtitle2" sx={{ color: "#a78bfa", fontWeight: 700, mb: 2 }}>
                ACCURACY–EFFICIENCY FRONTIER (size vs RMSE)
              </Typography>
              {loading || efficiencyData.length === 0 ? <Skeleton variant="rounded" height={320} /> : (
                <Plot
                  data={[{
                    type: "scatter",
                    mode: "text+markers",
                    x: efficiencyData.map((d) => Math.log10(Math.max(d.size_MB, 0.001))),
                    y: efficiencyData.map((d) => d.rmse_val),
                    text: efficiencyData.map((d) => d.model),
                    textposition: "top center" as const,
                    cliponaxis: false,
                    textfont: { size: 10, color: efficiencyData.map((d) => d.color) },
                    marker: {
                      color: efficiencyData.map((d) => d.color),
                      size: 14,
                      opacity: 0.9,
                      line: { color: "rgba(255,255,255,0.2)", width: 1 },
                    },
                    hovertemplate: "<b>%{text}</b><br>Size: %{customdata[0]}<br>Val RMSE: %{y:.2f} µA<br>R² TestMAT: %{customdata[1]}<extra></extra>",
                    customdata: efficiencyData.map((d) => [fmtMB(d.size_MB), d.r2_mat.toFixed(4)]),
                  }]}
                  layout={DarkLayout({
                    height: 320,
                    xaxis: {
                      title: { text: "Model Size (log₁₀ MB)", font: { size: 11, color: "#94a3b8" } },
                      gridcolor: "rgba(255,255,255,0.04)",
                      tickfont: { size: 10, color: "#8892a4", family: "JetBrains Mono" },
                      tickvals: [-1, 0, 1, 2, 3],
                      ticktext: ["100 KB", "1 MB", "10 MB", "100 MB", "1 GB"],
                    },
                    yaxis: {
                      title: { text: "Val RMSE (µA)", font: { size: 11, color: "#94a3b8" } },
                      gridcolor: "rgba(255,255,255,0.04)",
                      tickfont: { size: 10, color: "#8892a4", family: "JetBrains Mono" },
                    },
                    annotations: [{
                      x: 0.2, y: 0.92, xref: "paper" as const, yref: "paper" as const,
                      text: "← Better efficiency",
                      showarrow: false,
                      font: { size: 9, color: "rgba(255,255,255,0.3)" },
                    }],
                  })}
                  config={PlotCfg} style={{ width: "100%", height: 320 }} useResizeHandler
                />
              )}
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.62rem" }}>
                Lower-left = better (small model + low RMSE). LightGBM dominates the efficiency frontier.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={5}>
          <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.1)", height: "100%" }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography variant="subtitle2" sx={{ color: "#00d4ff", fontWeight: 700, mb: 2 }}>
                ARCHITECTURE & PARAMETER COMPARISON
              </Typography>
              {loading ? <Skeleton variant="rounded" height={280} /> : (
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontSize: "0.68rem" }}>Model</TableCell>
                      <TableCell sx={{ fontSize: "0.68rem" }}>Family</TableCell>
                      <TableCell align="right" sx={{ fontSize: "0.68rem" }}>Params</TableCell>
                      <TableCell align="right" sx={{ fontSize: "0.68rem" }}>Train</TableCell>
                      <TableCell align="right" sx={{ fontSize: "0.68rem" }}>Infer</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {leaderboard.map((row) => {
                      const meta = MODEL_METADATA[row.model_id];
                      return (
                        <TableRow key={row.model_id} sx={{ "&:last-child td": { border: 0 } }}>
                          <TableCell sx={{ py: 0.85 }}><ModelBadge modelId={row.model_id} size="small" /></TableCell>
                          <TableCell sx={{ py: 0.85, color: "rgba(255,255,255,0.45)", fontSize: "0.68rem" }}>{meta?.family}</TableCell>
                          <TableCell align="right" sx={{ py: 0.85, color: MODEL_COLORS[row.model_id], fontFamily: "JetBrains Mono", fontSize: "0.7rem", fontWeight: 700 }}>{meta?.params}</TableCell>
                          <TableCell align="right" sx={{ py: 0.85, color: "rgba(255,255,255,0.35)", fontFamily: "JetBrains Mono", fontSize: "0.68rem" }}>{meta?.trainMin} min</TableCell>
                          <TableCell align="right" sx={{ py: 0.85, color: "#22d3ee", fontFamily: "JetBrains Mono", fontSize: "0.68rem" }}>{meta?.inference}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* ── Classical vs Deep ML comparison ─────────────────────────── */}
      <motion.div variants={cardVariants} initial="initial" animate="animate">
        <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(34,211,238,0.12)" }}>
          <CardContent sx={{ p: 2.5 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2.5 }}>
              <CompareArrowsRounded sx={{ color: "#22d3ee", fontSize: 18 }} />
              <Typography variant="subtitle2" sx={{ color: "#22d3ee", fontWeight: 700, letterSpacing: "0.05em" }}>
                CLASSICAL ML vs DEEP LEARNING — HEAD-TO-HEAD
              </Typography>
            </Box>
            <Grid container spacing={3}>
              {[
                {
                  label: "Classical / Boosting",
                  color: "#3b82f6",
                  models: classicalModels,
                  icon: "🌲⚡",
                  pros: ["XGBoost <1 ms / LightGBM ~5 ms inference", "Compact models (268 KB – 1.7 MB excl. RF)", "No GPU required", "Deterministic, auditable predictions", "Best validation interpolation accuracy"],
                  cons: ["Cannot model sequential CV sweep direction", "Lower extrapolation R² than GRU", "RF is 615 MB, ~120 ms (deployment concern)"],
                },
                {
                  label: "Deep Learning (RNN)",
                  color: "#a78bfa",
                  models: deepModels,
                  icon: "🧠",
                  pros: ["GRU achieves best testMAT R² (0.9751)", "Captures sweep direction sequentially", "Best cross-material generalisation", "Compact weights (176–434 KB)"],
                  cons: ["Requires TensorFlow runtime", "~120–140 ms CPU inference per curve", "Cold-start latency for lazy loading", "ANN has worst RMSE (point-wise, no memory)"],
                },
              ].map((group) => (
                <Grid item xs={12} md={6} key={group.label}>
                  <Box sx={{ p: 2, borderRadius: 2, background: alpha(group.color, 0.05), border: `1px solid ${alpha(group.color, 0.2)}`, height: "100%" }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
                      <Typography sx={{ fontSize: 20 }}>{group.icon}</Typography>
                      <Typography variant="subtitle2" sx={{ color: group.color, fontWeight: 700, fontSize: "0.88rem" }}>{group.label}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap", mb: 1.75 }}>
                      {group.models.map((m) => (
                        <Box key={m.model_id} sx={{ display: "flex", alignItems: "center", gap: 0.75, px: 1, py: 0.4, borderRadius: 1.5, background: alpha(MODEL_COLORS[m.model_id], 0.1), border: `1px solid ${alpha(MODEL_COLORS[m.model_id], 0.2)}` }}>
                          <Box sx={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: MODEL_COLORS[m.model_id] }} />
                          <Typography variant="caption" sx={{ color: MODEL_COLORS[m.model_id], fontSize: "0.65rem", fontWeight: 700 }}>{m.model_display}</Typography>
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontFamily: "JetBrains Mono", fontSize: "0.62rem" }}>{m.rmse_val_uA.toFixed(1)} µA</Typography>
                        </Box>
                      ))}
                    </Box>
                    <Divider sx={{ mb: 1.5, borderColor: alpha(group.color, 0.12) }} />
                    {group.pros.map((p) => (
                      <Box key={p} sx={{ display: "flex", gap: 1, mb: 0.6 }}>
                        <Typography sx={{ color: "#10b981", fontSize: 12, mt: 0.2 }}>✓</Typography>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.7rem", lineHeight: 1.5 }}>{p}</Typography>
                      </Box>
                    ))}
                    {group.cons.map((c) => (
                      <Box key={c} sx={{ display: "flex", gap: 1, mb: 0.6 }}>
                        <Typography sx={{ color: "#f59e0b", fontSize: 12, mt: 0.2 }}>△</Typography>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.7rem", lineHeight: 1.5 }}>{c}</Typography>
                      </Box>
                    ))}
                  </Box>
                </Grid>
              ))}
            </Grid>

            <Box sx={{ mt: 2.5, p: 1.75, borderRadius: 2, background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.68rem", lineHeight: 1.7, display: "block" }}>
                <strong style={{ color: "#22d3ee" }}>Key finding:</strong>{" "}
                For <em>deployment</em>, LightGBM is the clear winner — ~360× smaller than RF with only 1% higher RMSE.
                For <em>scientific research</em> on new materials, GRU's recurrent memory achieves 0.9751 R² on the unseen NM4 electrode,
                outperforming all classical models on cross-material generalisation.
                Neither approach dominates universally — the choice depends on the specific electrochemical application.
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </motion.div>

      {/* ══ FEATURE IMPORTANCE ANALYSIS ═════════════════════════════ */}
      <motion.div variants={cardVariants} initial="initial" animate="animate" style={{ marginTop: 32 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}>
          <TipsAndUpdatesRounded sx={{ color: "#f59e0b", fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Feature Importance Analysis</Typography>
          <Chip label="FROM TRAINED MODELS" size="small"
            sx={{ fontSize: "0.58rem", height: 20, fontWeight: 700, backgroundColor: alpha("#f59e0b", 0.1), color: "#f59e0b", border: "1px solid rgba(245,158,11,0.3)", letterSpacing: "0.04em" }} />
        </Box>

        <Grid container spacing={3} sx={{ mb: 3 }}>
          {/* RF MDI + Permutation */}
          <Grid item xs={12} md={6}>
            <Card sx={{ background: "rgba(7,11,20,0.9)", border: "1px solid rgba(59,130,246,0.2)" }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="overline" sx={{ color: "#3b82f6", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 0.5, display: "block" }}>
                  RANDOM FOREST — PERMUTATION IMPORTANCE (Val Set)
                </Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", display: "block", mb: 1.5 }}>
                  RMSE increase when feature is permuted — generalisation-aware metric
                </Typography>
                <Plot
                  data={[{
                    type: "bar",
                    orientation: "h",
                    x: [0.110, 0.042, 0.041, 0.040, 0.039, 0.038, 0.013, 0.001, 0.001, 0.000],
                    y: [
                      "dir × potential",
                      "sweep position",
                      "sweep direction",
                      "potential (V)",
                      "potential (lower)",
                      "potential (upper)",
                      "SR × potential",
                      "scan rate",
                      "√scan rate",
                      "log scan rate",
                    ],
                    marker: {
                      color: [0.110, 0.042, 0.041, 0.040, 0.039, 0.038, 0.013, 0.001, 0.001, 0.000],
                      colorscale: [[0, "#1e3a5f"], [0.3, "#3b82f6"], [1, "#60a5fa"]],
                      showscale: false,
                      opacity: 0.9,
                    },
                    hovertemplate: "<b>%{y}</b><br>Permutation importance: %{x:.4f} RMSE increase<extra></extra>",
                    text: ["0.110", "0.042", "0.041", "0.040", "0.039", "0.038", "0.013", "", "", ""].map(v => v),
                    textposition: "outside",
                    textfont: { size: 9, color: "#94a3b8", family: "JetBrains Mono" },
                  }]}
                  layout={DarkLayout({
                    height: 320,
                    margin: { l: 120, r: 60, t: 8, b: 40 },
                    xaxis: {
                      title: { text: "RMSE increase on val set", font: { size: 10, color: "#94a3b8" } },
                      gridcolor: "rgba(255,255,255,0.04)",
                      tickfont: { size: 9, family: "JetBrains Mono" },
                      range: [0, 0.135],
                    },
                    yaxis: {
                      tickfont: { size: 10, color: "#c8cfd8" },
                      autorange: "reversed",
                    },
                    bargap: 0.35,
                  })}
                  config={{ ...PlotCfg, displayModeBar: false }}
                  style={{ width: "100%" }}
                  useResizeHandler
                />
              </CardContent>
            </Card>
          </Grid>

          {/* LightGBM Gain */}
          <Grid item xs={12} md={6}>
            <Card sx={{ background: "rgba(7,11,20,0.9)", border: "1px solid rgba(34,211,238,0.2)" }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="overline" sx={{ color: "#22d3ee", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 0.5, display: "block" }}>
                  LIGHTGBM — GAIN IMPORTANCE (Total Information Gain)
                </Typography>
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", display: "block", mb: 1.5 }}>
                  Total information gain contributed by each feature across all 278 trees
                </Typography>
                <Plot
                  data={[{
                    type: "bar",
                    orientation: "h",
                    x: [25100, 10000, 9500, 8500, 2000, 900, 800, 600, 80, 40],
                    y: [
                      "dir × potential",
                      "potential (V)",
                      "sweep position",
                      "potential (upper)",
                      "SR × potential",
                      "potential (lower)",
                      "sweep direction",
                      "scan rate",
                      "log scan rate",
                      "√scan rate",
                    ],
                    marker: {
                      color: [25100, 10000, 9500, 8500, 2000, 900, 800, 600, 80, 40],
                      colorscale: [[0, "#0c3045"], [0.3, "#0891b2"], [1, "#22d3ee"]],
                      showscale: false,
                      opacity: 0.9,
                    },
                    hovertemplate: "<b>%{y}</b><br>Gain: %{x:,.0f}<extra></extra>",
                  }]}
                  layout={DarkLayout({
                    height: 320,
                    margin: { l: 120, r: 30, t: 8, b: 40 },
                    xaxis: {
                      title: { text: "Total gain across all trees", font: { size: 10, color: "#94a3b8" } },
                      gridcolor: "rgba(255,255,255,0.04)",
                      tickfont: { size: 9, family: "JetBrains Mono" },
                    },
                    yaxis: {
                      tickfont: { size: 10, color: "#c8cfd8" },
                      autorange: "reversed",
                    },
                    bargap: 0.35,
                  })}
                  config={{ ...PlotCfg, displayModeBar: false }}
                  style={{ width: "100%" }}
                  useResizeHandler
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Feature explanation cards */}
        <Card sx={{ background: "rgba(15,25,35,0.85)", border: "1px solid rgba(245,158,11,0.15)" }}>
          <CardContent sx={{ p: 2.5 }}>
            <Typography variant="overline" sx={{ color: "#f59e0b", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 2, display: "block" }}>
              ELECTROCHEMICAL INTERPRETATION OF TOP FEATURES
            </Typography>
            <Grid container spacing={2}>
              {[
                {
                  feature: "dir × potential",
                  rank: "#1",
                  color: "#f59e0b",
                  echem: "direction_x_potential_norm — the interaction between sweep direction (±1) and normalised potential. This single feature encodes the CV hysteresis: the same voltage point has different currents depending on whether the sweep is going anodic→cathodic or cathodic→anodic. It is the primary driver of CV shape asymmetry in all tree models.",
                  importance: "Dominant (RF perm: 0.110, LGB gain: 25100)",
                },
                {
                  feature: "sweep position",
                  rank: "#2",
                  color: "#22d3ee",
                  echem: "sweep_position — normalised index (0→651) within the half-sweep. Encodes temporal progression of the electrochemical scan. Trees use this to localise predictions within the sweep without requiring sequence modelling — a key reason why tabular models can approximate RNNs.",
                  importance: "High (RF perm: 0.042)",
                },
                {
                  feature: "sweep direction",
                  rank: "#3",
                  color: "#a78bfa",
                  echem: "sweep_direction (+1 anodic, −1 cathodic) — binary indicator distinguishing the oxidation half-sweep from the reduction half-sweep. All models must learn different current responses for the same potential depending on this flag — it is the primary asymmetry signal.",
                  importance: "High (RF perm: 0.041)",
                },
                {
                  feature: "potential (V)",
                  rank: "#4",
                  color: "#10b981",
                  echem: "potential_V_norm — the normalised electrode potential at each point. Directly maps to the CV x-axis. Models use this with sweep_direction to build a 2D feature space (V, direction) that reconstructs the full CV loop topology.",
                  importance: "High (RF perm: 0.040, LGB gain: 10000)",
                },
                {
                  feature: "SR × potential",
                  rank: "#7",
                  color: "#f472b6",
                  echem: "sr_x_potential_norm — scan rate multiplied by potential. Captures scan rate–voltage interaction: the Randles–Ševčík equation (I ∝ ν·E) predicts that current scales differently at different potentials for different scan rates. This term is the key electrochemical coupling feature.",
                  importance: "Moderate (RF perm: 0.013)",
                },
                {
                  feature: "scan rate",
                  rank: "#8+",
                  color: "#94a3b8",
                  echem: "scan_rate_mVs_norm, log_scan_rate_norm, sqrt_scan_rate_norm — the raw scan rate and its transforms. Surprisingly low direct importance after accounting for the interaction terms (dir×potential, SR×potential), suggesting the scan rate influence is primarily captured through interactions rather than as an independent main effect.",
                  importance: "Low direct importance — captured via interactions",
                },
              ].map((item) => (
                <Grid item xs={12} md={6} key={item.feature}>
                  <Box sx={{ p: 1.75, borderRadius: 2, background: "rgba(0,0,0,0.2)", border: `1px solid ${alpha(item.color, 0.2)}`, height: "100%" }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                      <Chip label={item.rank} size="small" sx={{ fontSize: "0.62rem", height: 18, fontWeight: 800, backgroundColor: alpha(item.color, 0.15), color: item.color, border: `1px solid ${alpha(item.color, 0.3)}` }} />
                      <Typography variant="caption" sx={{ color: item.color, fontFamily: "JetBrains Mono", fontSize: "0.68rem", fontWeight: 700 }}>{item.feature}</Typography>
                    </Box>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem", lineHeight: 1.65, display: "block", mb: 0.75 }}>
                      {item.echem}
                    </Typography>
                    <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.6rem", fontFamily: "JetBrains Mono" }}>
                      {item.importance}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      </motion.div>
    </Box>
  );
}
