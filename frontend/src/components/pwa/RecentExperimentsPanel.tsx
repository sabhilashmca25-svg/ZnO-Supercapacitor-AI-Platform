import React from "react";
import {
  Box, Card, CardContent, Typography, Chip, IconButton, Tooltip, alpha,
} from "@mui/material";
import {
  DeleteOutlineRounded, AssignmentRounded, InventoryRounded,
  CheckCircleRounded, ScheduleRounded, ScienceRounded,
} from "@mui/icons-material";
import { motion, AnimatePresence } from "framer-motion";
import { cardVariants } from "../../animations/variants";
import type { ExperimentRecord } from "../../hooks/useRecentExperiments";
import { MODEL_COLORS, MATERIALS } from "../../constants/models";
import { formatRelativeTime } from "../../utils/timeFormat";

interface RecentExperimentsPanelProps {
  records: ExperimentRecord[];
  cachedKeys: Set<string>;
  onRestore: (exp: ExperimentRecord) => void;
  onClear: () => void;
}

function MaterialBadge({ material }: { material: string }) {
  const mat = MATERIALS.find((m) => m.id === material);
  const color = mat?.color ?? "#94a3b8";
  return (
    <Chip
      label={material}
      size="small"
      sx={{
        height: 18,
        fontSize: "0.6rem",
        fontWeight: 700,
        letterSpacing: "0.04em",
        backgroundColor: alpha(color, 0.12),
        color,
        border: `1px solid ${alpha(color, 0.3)}`,
        "& .MuiChip-label": { px: 0.75 },
      }}
    />
  );
}

function ModelBadge({ model }: { model: string }) {
  const color = MODEL_COLORS[model] ?? "#94a3b8";
  return (
    <Chip
      label={model.toUpperCase()}
      size="small"
      sx={{
        height: 18,
        fontSize: "0.6rem",
        fontWeight: 700,
        fontFamily: "JetBrains Mono, monospace",
        backgroundColor: alpha(color, 0.1),
        color,
        border: `1px solid ${alpha(color, 0.25)}`,
        "& .MuiChip-label": { px: 0.75 },
      }}
    />
  );
}

function CacheStatusIcon({ isCached }: { isCached: boolean }) {
  return (
    <Tooltip title={isCached ? "Prediction cached — click to restore" : "Session ended — restore selections only"} arrow placement="top">
      <Box component="span">
        {isCached
          ? <CheckCircleRounded sx={{ fontSize: 12, color: "#10b981" }} />
          : <ScheduleRounded sx={{ fontSize: 12, color: "rgba(255,255,255,0.2)" }} />}
      </Box>
    </Tooltip>
  );
}

export default function RecentExperimentsPanel({
  records,
  cachedKeys,
  onRestore,
  onClear,
}: RecentExperimentsPanelProps) {
  if (records.length === 0) return null;

  return (
    <motion.div variants={cardVariants} style={{ marginTop: 16 }}>
      <Card sx={{ background: "rgba(13,19,33,0.8)", border: "1px solid rgba(139,92,246,0.18)", borderTop: "2px solid rgba(139,92,246,0.45)" }}>
        <CardContent sx={{ p: 2 }}>
          {/* Header */}
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.75 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <AssignmentRounded sx={{ fontSize: 15, color: "#a78bfa" }} />
              <Typography variant="caption" sx={{ color: "#a78bfa", fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.04em" }}>
                Recent Experiments
              </Typography>
              <Chip
                label={records.length}
                size="small"
                sx={{ height: 16, fontSize: "0.58rem", backgroundColor: "rgba(139,92,246,0.12)", color: "#a78bfa", "& .MuiChip-label": { px: 0.6 } }}
              />
            </Box>
            <Tooltip title="Clear all saved experiments">
              <IconButton size="small" onClick={onClear} sx={{ p: 0.25, "&:hover": { color: "#ef4444" } }}>
                <DeleteOutlineRounded sx={{ fontSize: 14, color: "rgba(255,255,255,0.2)" }} />
              </IconButton>
            </Tooltip>
          </Box>

          {/* Experiment cards */}
          <AnimatePresence>
            {records.map((exp) => {
              const modelColor = MODEL_COLORS[exp.model] ?? "#94a3b8";
              const cacheKey = `${exp.model}_${exp.material}_${exp.scanRate}`;
              const isCached = cachedKeys.has(cacheKey);

              return (
                <motion.div
                  key={exp.id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 8, height: 0 }}
                  transition={{ duration: 0.2 }}
                  whileHover={{ scale: 1.01, y: -1 }}
                  whileTap={{ scale: 0.99 }}
                  onClick={() => onRestore(exp)}
                  style={{ cursor: "pointer", marginBottom: 6 }}
                >
                  <Box
                    sx={{
                      p: 1.25,
                      borderRadius: 1.75,
                      border: `1px solid ${alpha(modelColor, 0.15)}`,
                      background: `linear-gradient(135deg, ${alpha(modelColor, 0.04)} 0%, rgba(13,19,33,0.6) 100%)`,
                      "&:hover": {
                        border: `1px solid ${alpha(modelColor, 0.3)}`,
                        background: `linear-gradient(135deg, ${alpha(modelColor, 0.08)} 0%, rgba(13,19,33,0.8) 100%)`,
                        boxShadow: `0 2px 12px ${alpha(modelColor, 0.1)}`,
                      },
                      transition: "all 0.15s ease",
                    }}
                  >
                    {/* Row 1: badges + cache status */}
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.75, flexWrap: "wrap" }}>
                      <MaterialBadge material={exp.material} />
                      <ModelBadge model={exp.model} />
                      <Chip
                        label={`${exp.scanRate} mV/s`}
                        size="small"
                        sx={{
                          height: 18, fontSize: "0.6rem", fontFamily: "JetBrains Mono",
                          backgroundColor: "rgba(0,212,255,0.07)", color: "#00d4ff",
                          border: "1px solid rgba(0,212,255,0.2)", "& .MuiChip-label": { px: 0.75 },
                        }}
                      />
                      {exp.isZeroShot && (
                        <Chip
                          label="Zero-Shot"
                          size="small"
                          sx={{
                            height: 18, fontSize: "0.58rem",
                            backgroundColor: "rgba(244,114,182,0.08)", color: "#f472b6",
                            border: "1px solid rgba(244,114,182,0.2)", "& .MuiChip-label": { px: 0.75 },
                          }}
                        />
                      )}
                      <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: 0.5 }}>
                        {exp.reportId && (
                          <Tooltip title="Report available" arrow>
                            <InventoryRounded sx={{ fontSize: 11, color: "#f59e0b" }} />
                          </Tooltip>
                        )}
                        <CacheStatusIcon isCached={isCached} />
                      </Box>
                    </Box>

                    {/* Row 2: confidence + metrics */}
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 0.5 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <ScienceRounded sx={{ fontSize: 10, color: exp.confidence.color }} />
                        <Typography sx={{ color: exp.confidence.color, fontSize: "0.62rem", fontWeight: 600 }}>
                          {exp.confidence.label}
                        </Typography>
                      </Box>
                      <Typography sx={{ color: "rgba(255,255,255,0.22)", fontSize: "0.58rem" }}>·</Typography>
                      <Typography sx={{ color: "#10b981", fontSize: "0.62rem", fontFamily: "JetBrains Mono" }}>
                        ↑ {exp.peakAnodic.toFixed(1)} µA
                      </Typography>
                      <Typography sx={{ color: "#f472b6", fontSize: "0.62rem", fontFamily: "JetBrains Mono" }}>
                        ↓ {Math.abs(exp.peakCathodic).toFixed(1)} µA
                      </Typography>
                    </Box>

                    {/* Row 3: timestamp */}
                    <Typography sx={{ color: "rgba(255,255,255,0.22)", fontSize: "0.6rem" }}>
                      {formatRelativeTime(exp.timestamp)}
                    </Typography>
                  </Box>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  );
}
