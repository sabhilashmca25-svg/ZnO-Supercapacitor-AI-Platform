import React, { useState } from "react";
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  alpha,
} from "@mui/material";
import {
  CheckCircleRounded,
  CancelRounded,
  MemoryRounded,
  TrendingUpRounded,
  RocketLaunchRounded,
  TimerRounded,
  BoltRounded,
  BiotechRounded,
  StorageRounded,
  EmojiObjectsRounded,
  BalanceRounded,
  SpeedRounded,
  PsychologyAltRounded,
} from "@mui/icons-material";
import { motion, AnimatePresence } from "framer-motion";
import SectionHeader from "../components/common/SectionHeader";
import ModelBadge from "../components/common/ModelBadge";
import { MODELS, MODEL_COLORS } from "../constants/models";
import { cardVariants, staggerContainer, cardHover, cardTap } from "../animations/variants";

const TIER_INFO: Record<string, { tier: 1 | 2; role: string; color: string }> = {
  rf:       { tier: 1, role: "Primary Deployment (High Accuracy)",   color: "#3b82f6" },
  lightgbm: { tier: 1, role: "Primary Deployment (Best Trade-off)",  color: "#22d3ee" },
  gru:      { tier: 1, role: "Primary Research (Extrapolation)",     color: "#a78bfa" },
  xgboost:  { tier: 2, role: "Research Comparison (Regularisation)", color: "#fb923c" },
  ann:      { tier: 2, role: "Baseline Comparison (MLP reference)",  color: "#facc15" },
  lstm:     { tier: 2, role: "Research Comparison (Long-range seq)", color: "#f472b6" },
};

// Source: research/exports/final_leaderboard.csv (verified ground-truth benchmarks)
const MODEL_PERF: Record<string, { valRMSE: number; valR2: number; matRMSE: number; matR2: number }> = {
  rf:       { valRMSE: 26.59, valR2: 0.9851, matRMSE: 33.65, matR2: 0.9707 },
  lightgbm: { valRMSE: 26.86, valR2: 0.9831, matRMSE: 35.18, matR2: 0.9678 },
  xgboost:  { valRMSE: 28.83, valR2: 0.9800, matRMSE: 36.64, matR2: 0.9668 },
  gru:      { valRMSE: 36.84, valR2: 0.9723, matRMSE: 34.52, matR2: 0.9751 },
  lstm:     { valRMSE: 36.51, valR2: 0.9717, matRMSE: 38.40, matR2: 0.9674 },
  ann:      { valRMSE: 49.93, valR2: 0.9475, matRMSE: 46.16, matR2: 0.9606 },
};

// Qualitative suitability scores (0–100). Realtime reflects per-curve CPU inference:
// XGBoost <1ms, LightGBM ~5ms, RF/ANN/GRU ~120ms, LSTM ~140ms.
const ECHEM_SUITABILITY: Record<string, { interpolation: number; extrapolation: number; realtime: number; newMaterial: number }> = {
  rf:       { interpolation: 98, extrapolation: 70, realtime: 48, newMaterial: 68 },
  lightgbm: { interpolation: 97, extrapolation: 69, realtime: 95, newMaterial: 67 },
  xgboost:  { interpolation: 94, extrapolation: 66, realtime: 97, newMaterial: 64 },
  gru:      { interpolation: 89, extrapolation: 96, realtime: 46, newMaterial: 95 },
  lstm:     { interpolation: 85, extrapolation: 91, realtime: 42, newMaterial: 90 },
  ann:      { interpolation: 72, extrapolation: 58, realtime: 48, newMaterial: 56 },
};

export default function ModelEncyclopedia() {
  const [selected, setSelected] = useState<string | null>(null);

  const selectedModel = MODELS.find((m) => m.id === selected);

  return (
    <Box sx={{ width: "100%", overflowX: "hidden" }}>
      <SectionHeader
        title="Model Encyclopedia"
        subtitle="Deep-dive into the 6 ML architectures trained on ZnO CV data — from ensemble trees to deep recurrent networks"
        accent="#a78bfa"
      />

      <Grid container spacing={3}>
        {/* ── Model cards ─────────────────────────────────────────────── */}
        <Grid item xs={12} lg={selected ? 4 : 12}>
          <motion.div variants={staggerContainer} initial="initial" animate="animate">
            <Grid container spacing={2.5}>
              {MODELS.map((model, i) => {
                const color = MODEL_COLORS[model.id];
                const isActive = selected === model.id;
                return (
                  <Grid item xs={12} sm={6} md={selected ? 12 : 4} lg={selected ? 12 : 4} key={model.id}>
                    <motion.div
                      variants={cardVariants}
                      whileHover={cardHover}
                      whileTap={cardTap}
                      layout
                    >
                      <Card
                        onClick={() => setSelected(isActive ? null : model.id)}
                        sx={{
                          cursor: "pointer",
                          background: isActive
                            ? `linear-gradient(135deg, ${alpha(color, 0.15)} 0%, ${alpha(color, 0.05)} 100%)`
                            : `linear-gradient(135deg, rgba(15,25,35,0.9) 0%, rgba(10,16,28,0.95) 100%)`,
                          border: `1px solid ${isActive ? alpha(color, 0.5) : alpha(color, 0.18)}`,
                          borderTop: `3px solid ${color}`,
                          transition: "all 0.25s ease",
                          boxShadow: isActive ? `0 0 24px ${alpha(color, 0.2)}` : "none",
                          position: "relative",
                          overflow: "hidden",
                        }}
                        tabIndex={0}
                        role="button"
                        aria-expanded={isActive}
                        onKeyDown={(e) => e.key === "Enter" && setSelected(isActive ? null : model.id)}
                      >
                        {/* Background glow */}
                        <Box
                          sx={{
                            position: "absolute",
                            top: -20,
                            right: -20,
                            width: 80,
                            height: 80,
                            borderRadius: "50%",
                            background: `radial-gradient(circle, ${alpha(color, 0.1)} 0%, transparent 70%)`,
                            pointerEvents: "none",
                          }}
                        />

                        <CardContent sx={{ p: 2.5, "&:last-child": { pb: 2.5 } }}>
                          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.5 }}>
                            <ModelBadge modelId={model.id} size="medium" selected={isActive} />
                            <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap", justifyContent: "flex-end" }}>
                              <Chip
                                label={model.type === "tree" ? "Tree" : model.type === "boost" ? "Boosted" : "Deep Learning"}
                                size="small"
                                sx={{ fontSize: "0.6rem", height: 20, backgroundColor: alpha(color, 0.1), color: alpha(color, 0.8), border: `1px solid ${alpha(color, 0.2)}`, "& .MuiChip-label": { px: 0.8 } }}
                              />
                              <Chip
                                icon={TIER_INFO[model.id]?.tier === 1
                                  ? <BoltRounded sx={{ fontSize: "10px !important" }} />
                                  : <TimerRounded sx={{ fontSize: "10px !important" }} />}
                                label={TIER_INFO[model.id]?.tier === 1 ? "Tier 1" : "Tier 2"}
                                size="small"
                                sx={{ fontSize: "0.6rem", height: 20, backgroundColor: TIER_INFO[model.id]?.tier === 1 ? "rgba(16,185,129,0.08)" : "rgba(251,191,36,0.08)", color: TIER_INFO[model.id]?.tier === 1 ? "#10b981" : "#fbbf24", border: `1px solid ${TIER_INFO[model.id]?.tier === 1 ? "rgba(16,185,129,0.2)" : "rgba(251,191,36,0.2)"}`, "& .MuiChip-label": { px: 0.8 } }}
                              />
                            </Box>
                          </Box>

                          <Typography variant="subtitle2" sx={{ color: "#f1f5f9", fontWeight: 700, mb: 0.5, fontSize: "0.9rem" }}>
                            {model.display}
                          </Typography>

                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.72rem", lineHeight: 1.5, display: "block", mb: 1.5 }}>
                            {model.description.slice(0, selected ? 160 : 100)}…
                          </Typography>

                          {/* Deploy score */}
                          <Box>
                            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem" }}>Deploy Score</Typography>
                              <Typography variant="caption" sx={{ color: color, fontWeight: 700, fontSize: "0.68rem" }}>{model.deployScore}/100</Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={model.deployScore}
                              sx={{
                                height: 4,
                                borderRadius: 2,
                                backgroundColor: "rgba(255,255,255,0.06)",
                                "& .MuiLinearProgress-bar": { backgroundColor: color, borderRadius: 2 },
                              }}
                            />
                          </Box>

                          <Box sx={{ display: "flex", gap: 1, mt: 1.5, flexWrap: "wrap" }}>
                            <Chip
                              icon={<MemoryRounded sx={{ fontSize: "10px !important" }} />}
                              label={model.size}
                              size="small"
                              sx={{ fontSize: "0.6rem", height: 18, backgroundColor: "rgba(255,255,255,0.05)", color: "rgba(255,255,255,0.4)", "& .MuiChip-label": { px: 0.8 } }}
                            />
                          </Box>
                        </CardContent>
                      </Card>
                    </motion.div>
                  </Grid>
                );
              })}
            </Grid>
          </motion.div>
        </Grid>

        {/* ── Detail panel ─────────────────────────────────────────────── */}
        <AnimatePresence>
          {selected && selectedModel && (
            <Grid item xs={12} lg={8}>
              <motion.div
                initial={{ opacity: 0, x: 24 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 24 }}
                transition={{ duration: 0.35 }}
                layout
              >
                <Card
                  sx={{
                    background: "rgba(15,25,35,0.9)",
                    border: `1px solid ${alpha(MODEL_COLORS[selectedModel.id], 0.3)}`,
                    borderTop: `3px solid ${MODEL_COLORS[selectedModel.id]}`,
                    height: "100%",
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
                      <Box
                        sx={{
                          width: 48,
                          height: 48,
                          borderRadius: "14px",
                          background: `linear-gradient(135deg, ${MODEL_COLORS[selectedModel.id]} 0%, ${alpha(MODEL_COLORS[selectedModel.id], 0.5)} 100%)`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "22px",
                          boxShadow: `0 0 20px ${alpha(MODEL_COLORS[selectedModel.id], 0.4)}`,
                        }}
                      >
                        {selectedModel.type === "tree" ? "🌲" : selectedModel.type === "boost" ? "⚡" : "🧠"}
                      </Box>
                      <Box>
                        <Typography variant="h5" sx={{ fontWeight: 800, color: "#f1f5f9" }}>
                          {selectedModel.display}
                        </Typography>
                        <Typography variant="caption" sx={{ color: MODEL_COLORS[selectedModel.id], fontSize: "0.72rem", fontFamily: "JetBrains Mono" }}>
                          {selectedModel.architecture}
                        </Typography>
                      </Box>
                    </Box>

                    <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.6)", lineHeight: 1.7, mb: 3 }}>
                      {selectedModel.description}
                    </Typography>

                    {/* Research Role */}
                    {TIER_INFO[selectedModel.id] && (
                      <Box sx={{ mb: 2.5, p: 1.5, borderRadius: 2, backgroundColor: alpha(TIER_INFO[selectedModel.id].tier === 1 ? "#10b981" : "#fbbf24", 0.06), border: `1px solid ${alpha(TIER_INFO[selectedModel.id].tier === 1 ? "#10b981" : "#fbbf24", 0.2)}` }}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                          {TIER_INFO[selectedModel.id].tier === 1
                            ? <BoltRounded sx={{ fontSize: 16, color: "#10b981" }} />
                            : <TimerRounded sx={{ fontSize: 16, color: "#fbbf24" }} />}
                          <Typography variant="caption" sx={{ color: TIER_INFO[selectedModel.id].tier === 1 ? "#10b981" : "#fbbf24", fontWeight: 700, fontSize: "0.72rem" }}>
                            Tier {TIER_INFO[selectedModel.id].tier} — {TIER_INFO[selectedModel.id].tier === 1 ? "Hot (loaded at startup)" : "Lazy (loaded on first request)"}
                          </Typography>
                        </Box>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.7rem" }}>
                          Research Role: {TIER_INFO[selectedModel.id].role}
                        </Typography>
                      </Box>
                    )}

                    <Divider sx={{ mb: 2.5 }} />

                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="overline" sx={{ color: "#10b981", fontSize: "0.62rem", letterSpacing: "0.1em", mb: 1.5, display: "block" }}>
                          Strengths
                        </Typography>
                        {selectedModel.strengths.map((s) => (
                          <Box key={s} sx={{ display: "flex", alignItems: "flex-start", gap: 1, mb: 1 }}>
                            <CheckCircleRounded sx={{ color: "#10b981", fontSize: 15, mt: 0.2, flexShrink: 0 }} />
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.78rem", lineHeight: 1.5 }}>
                              {s}
                            </Typography>
                          </Box>
                        ))}
                      </Grid>

                      <Grid item xs={12} sm={6}>
                        <Typography variant="overline" sx={{ color: "#ef4444", fontSize: "0.62rem", letterSpacing: "0.1em", mb: 1.5, display: "block" }}>
                          Limitations
                        </Typography>
                        {selectedModel.weaknesses.map((w) => (
                          <Box key={w} sx={{ display: "flex", alignItems: "flex-start", gap: 1, mb: 1 }}>
                            <CancelRounded sx={{ color: "#ef4444", fontSize: 15, mt: 0.2, flexShrink: 0 }} />
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.78rem", lineHeight: 1.5 }}>
                              {w}
                            </Typography>
                          </Box>
                        ))}
                      </Grid>
                    </Grid>

                    <Divider sx={{ my: 2.5 }} />

                    {/* Performance Profile */}
                    {MODEL_PERF[selectedModel.id] && (
                      <Box sx={{ mb: 2.5 }}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 1.5 }}>
                          <BiotechRounded sx={{ fontSize: 16, color: MODEL_COLORS[selectedModel.id] }} />
                          <Typography variant="overline" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.62rem", letterSpacing: "0.1em" }}>
                            Performance Profile
                          </Typography>
                        </Box>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", py: 0.5, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>Partition</TableCell>
                              <TableCell align="right" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", py: 0.5, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>RMSE (µA)</TableCell>
                              <TableCell align="right" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.65rem", py: 0.5, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>R²</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {[
                              { label: "Val (SR=30 mV/s)", rmse: MODEL_PERF[selectedModel.id].valRMSE, r2: MODEL_PERF[selectedModel.id].valR2, color: "#10b981" },
                              { label: "Test-MAT (NM4 cross-material)", rmse: MODEL_PERF[selectedModel.id].matRMSE, r2: MODEL_PERF[selectedModel.id].matR2, color: "#f472b6" },
                            ].map((row) => (
                              <TableRow key={row.label}>
                                <TableCell sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.72rem", py: 0.75, borderBottom: "1px solid rgba(255,255,255,0.04)" }}>{row.label}</TableCell>
                                <TableCell align="right" sx={{ color: row.color, fontFamily: "JetBrains Mono, monospace", fontSize: "0.78rem", py: 0.75, fontWeight: 600, borderBottom: "1px solid rgba(255,255,255,0.04)" }}>{row.rmse.toFixed(2)}</TableCell>
                                <TableCell align="right" sx={{ color: row.color, fontFamily: "JetBrains Mono, monospace", fontSize: "0.78rem", py: 0.75, fontWeight: 600, borderBottom: "1px solid rgba(255,255,255,0.04)" }}>{row.r2.toFixed(4)}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </Box>
                    )}

                    <Divider sx={{ mb: 2.5 }} />

                    {/* Electrochemical Suitability */}
                    {ECHEM_SUITABILITY[selectedModel.id] && (
                      <Box sx={{ mb: 2.5 }}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
                          <Typography variant="overline" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.62rem", letterSpacing: "0.1em" }}>
                            Electrochemical Suitability
                          </Typography>
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.58rem", fontStyle: "italic" }}>
                            (qualitative assessment)
                          </Typography>
                        </Box>
                        {[
                          { label: "Interpolation (known scan rates)", value: ECHEM_SUITABILITY[selectedModel.id].interpolation, color: "#10b981" },
                          { label: "Extrapolation (new scan rates)", value: ECHEM_SUITABILITY[selectedModel.id].extrapolation, color: "#f472b6" },
                          { label: "Real-time inference", value: ECHEM_SUITABILITY[selectedModel.id].realtime, color: "#00d4ff" },
                          { label: "New material generalisation", value: ECHEM_SUITABILITY[selectedModel.id].newMaterial, color: "#a78bfa" },
                        ].map((item) => (
                          <Box key={item.label} sx={{ mb: 1.25 }}>
                            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.68rem" }}>{item.label}</Typography>
                              <Typography variant="caption" sx={{ color: item.color, fontWeight: 700, fontSize: "0.68rem" }}>{item.value}%</Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={item.value}
                              sx={{ height: 4, borderRadius: 2, backgroundColor: "rgba(255,255,255,0.05)", "& .MuiLinearProgress-bar": { backgroundColor: item.color, borderRadius: 2 } }}
                            />
                          </Box>
                        ))}
                      </Box>
                    )}

                    <Divider sx={{ mb: 2.5 }} />

                    {/* Deploy score bar */}
                    <Box>
                      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                          <RocketLaunchRounded sx={{ fontSize: 16, color: MODEL_COLORS[selectedModel.id] }} />
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontWeight: 600, fontSize: "0.72rem" }}>
                            Deployment Suitability
                          </Typography>
                        </Box>
                        <Typography variant="caption" sx={{ color: MODEL_COLORS[selectedModel.id], fontWeight: 800, fontSize: "0.82rem" }}>
                          {selectedModel.deployScore} / 100
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={selectedModel.deployScore}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: "rgba(255,255,255,0.05)",
                          "& .MuiLinearProgress-bar": {
                            background: `linear-gradient(90deg, ${MODEL_COLORS[selectedModel.id]} 0%, ${alpha(MODEL_COLORS[selectedModel.id], 0.6)} 100%)`,
                            borderRadius: 4,
                            boxShadow: `0 0 8px ${alpha(MODEL_COLORS[selectedModel.id], 0.5)}`,
                          },
                        }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          )}
        </AnimatePresence>
      </Grid>

      {/* ══ WHY THIS RESEARCH MATTERS ════════════════════════════════ */}
      <Box sx={{ mt: 5 }}>
        <motion.div variants={cardVariants} initial="initial" animate="animate">
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}>
            <EmojiObjectsRounded sx={{ color: "#f59e0b", fontSize: 22 }} />
            <Typography variant="h6" sx={{ fontWeight: 700, color: "#f1f5f9" }}>Why This Research Matters</Typography>
            <Chip label="IMPACT & MOTIVATION" size="small"
              sx={{ fontSize: "0.58rem", height: 20, fontWeight: 700, backgroundColor: alpha("#f59e0b", 0.1), color: "#f59e0b", border: "1px solid rgba(245,158,11,0.3)", letterSpacing: "0.04em" }} />
          </Box>
        </motion.div>

        {/* Motivation banner */}
        <motion.div variants={cardVariants} initial="initial" animate="animate">
          <Card sx={{ mb: 3, background: "linear-gradient(135deg, rgba(245,158,11,0.06) 0%, rgba(124,58,237,0.06) 50%, rgba(0,212,255,0.04) 100%)", border: "1px solid rgba(245,158,11,0.2)", borderLeft: "4px solid #f59e0b" }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.8, mb: 1.5 }}>
                Physical cyclic voltammetry experiments take hours to days per electrode — requiring material synthesis, cell assembly, electrolyte preparation, and electrochemical characterisation. For a 4-material, 6-scan-rate study, this represents weeks of lab work.
              </Typography>
              <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.65)", lineHeight: 1.8 }}>
                Our 6-model ML pipeline compresses the prediction of a complete 651-point CV curve to <strong style={{ color: "#22d3ee" }}>under 1 millisecond</strong> (LightGBM / RF) or under 50ms (deep learning). This enables real-time electrode screening at scale — a critical bottleneck for next-generation energy storage research.
              </Typography>
            </CardContent>
          </Card>
        </motion.div>

        {/* Key architectural insights */}
        <motion.div variants={staggerContainer} initial="initial" animate="animate">
          <Grid container spacing={2.5} sx={{ mb: 3 }}>
            {[
              {
                Icon: BalanceRounded,
                color: "#22d3ee",
                title: "The Accuracy–Efficiency Trade-off",
                text: "The key finding across all 6 models is that LightGBM sits at the Pareto frontier of accuracy vs. deployment cost. Random Forest achieves marginally lower RMSE (26.59 vs 26.86 µA) but requires ~360× more storage (615 MB vs 1.7 MB). For any resource-constrained deployment, LightGBM is the clear choice.",
                stat: "~360× size advantage",
                statColor: "#22d3ee",
              },
              {
                Icon: PsychologyAltRounded,
                color: "#a78bfa",
                title: "Recurrent Models Learn CV Physics",
                text: "GRU's test-MAT R² (0.9751) leads tree models (RF: 0.9707, LGB: 0.9678, XGB: 0.9668), suggesting sequential architectures capture directional hysteresis — the electrochemical memory in the anodic→cathodic voltage sweep. Tree models approximate this well but cannot model it directly.",
                stat: "R² 0.9751 cross-material",
                statColor: "#a78bfa",
              },
              {
                Icon: SpeedRounded,
                color: "#10b981",
                title: "Feature Engineering Is the Foundation",
                text: "Despite architectural differences, all models rely on the same 10 engineered features. Log/sqrt scan rate transformations capture the Randles–Ševčík I∝ν^0.5 relationship. The sweep direction feature enables models to distinguish the anodic (forward) and cathodic (reverse) sweep — essential for correct CV morphology.",
                stat: "10 features drive 6 models",
                statColor: "#10b981",
              },
              {
                Icon: BiotechRounded,
                color: "#f472b6",
                title: "Implications for Materials Discovery",
                text: "The trained models can provide initial CV predictions for hypothetical ZnO composite formulations before synthesis. By comparing predicted peak currents, symmetry factors, and integral areas across material combinations, researchers can prioritise which composites to synthesise — reducing wasted experimental effort.",
                stat: "Virtual electrode screening",
                statColor: "#f472b6",
              },
            ].map(({ Icon, color, title, text, stat, statColor }) => (
              <Grid item xs={12} sm={6} key={title}>
                <motion.div variants={cardVariants}>
                  <Card sx={{ height: "100%", background: `linear-gradient(135deg, ${alpha(color, 0.07)} 0%, rgba(15,25,35,0.9) 100%)`, border: `1px solid ${alpha(color, 0.2)}`, borderTop: `3px solid ${color}` }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, mb: 1.5 }}>
                        <Box sx={{ width: 38, height: 38, borderRadius: "11px", background: alpha(color, 0.12), border: `1px solid ${alpha(color, 0.3)}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                          <Icon sx={{ fontSize: 20, color }} />
                        </Box>
                        <Typography variant="subtitle2" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.88rem", lineHeight: 1.3 }}>{title}</Typography>
                      </Box>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.73rem", lineHeight: 1.7, display: "block", mb: 1.5 }}>
                        {text}
                      </Typography>
                      <Box sx={{ px: 1.25, py: 0.6, borderRadius: 1.5, background: alpha(statColor, 0.08), border: `1px solid ${alpha(statColor, 0.2)}`, display: "inline-flex" }}>
                        <Typography variant="caption" sx={{ color: statColor, fontFamily: "JetBrains Mono", fontWeight: 700, fontSize: "0.67rem" }}>{stat}</Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </motion.div>

        {/* Model selection guide */}
        <motion.div variants={cardVariants} initial="initial" animate="animate">
          <Card sx={{ background: "rgba(15,25,35,0.85)", border: "1px solid rgba(0,212,255,0.12)" }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography variant="overline" sx={{ color: "#00d4ff", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 2, display: "block" }}>
                Model Selection Guide — When to Use Which
              </Typography>
              <Grid container spacing={2}>
                {[
                  { scenario: "Production deployment / edge device", pick: "LightGBM", reason: "1.7 MB, ~5ms, best accuracy:size ratio", color: "#22d3ee" },
                  { scenario: "Maximum accuracy, storage not constrained", pick: "Random Forest", reason: "Lowest val RMSE (26.59 µA), deterministic", color: "#3b82f6" },
                  { scenario: "Predicting unseen material compositions", pick: "Stacked GRU", reason: "Best test-MAT R² (0.9751), learns CV physics", color: "#a78bfa" },
                  { scenario: "Regulatory / interpretable output", pick: "XGBoost", reason: "Regularised trees, feature importance available", color: "#fb923c" },
                  { scenario: "Research baseline (MLP reference)", pick: "Dense ANN", reason: "Smallest model (176 KB), point-wise prediction", color: "#facc15" },
                  { scenario: "Sequence-aware temporal modelling", pick: "Stacked LSTM", reason: "Cell state retains long-range sweep context", color: "#f472b6" },
                ].map((row) => (
                  <Grid item xs={12} sm={6} key={row.scenario}>
                    <Box sx={{ p: 1.5, borderRadius: 1.5, background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.05)" }}>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", display: "block", mb: 0.5 }}>
                        {row.scenario}
                      </Typography>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <Chip label={row.pick} size="small"
                          sx={{ fontSize: "0.62rem", height: 18, fontWeight: 700, backgroundColor: alpha(row.color, 0.12), color: row.color, border: `1px solid ${alpha(row.color, 0.3)}` }} />
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.65rem" }}>{row.reason}</Typography>
                      </Box>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </motion.div>
      </Box>
    </Box>
  );
}
