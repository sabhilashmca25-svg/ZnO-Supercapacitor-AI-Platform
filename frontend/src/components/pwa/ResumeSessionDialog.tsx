import React from "react";
import {
  Dialog, DialogContent, Box, Typography, Divider, alpha,
} from "@mui/material";
import {
  RestoreRounded, ScienceRounded, SpeedRounded, SmartToyRounded, AccessTimeRounded,
  FiberNewRounded,
} from "@mui/icons-material";
import { motion } from "framer-motion";
import GlowButton from "../common/GlowButton";
import type { LastSession } from "../../hooks/useLastSession";
import { MODEL_COLORS, MODELS, MATERIALS } from "../../constants/models";
import { formatRelativeTime } from "../../utils/timeFormat";

interface ResumeSessionDialogProps {
  session: LastSession;
  open: boolean;
  isCached: boolean;
  hasReport: boolean;
  onResume: () => void;
  onStartNew: () => void;
}

function InfoRow({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, py: 0.75 }}>
      <Box sx={{ display: "flex", alignItems: "center", color: "rgba(255,255,255,0.3)", flexShrink: 0, width: 18 }}>
        {icon}
      </Box>
      <Typography sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.74rem", minWidth: 72, flexShrink: 0 }}>
        {label}
      </Typography>
      <Typography sx={{ color: color ?? "#f1f5f9", fontSize: "0.78rem", fontWeight: 600 }}>
        {value}
      </Typography>
    </Box>
  );
}

export default function ResumeSessionDialog({
  session,
  open,
  isCached,
  hasReport,
  onResume,
  onStartNew,
}: ResumeSessionDialogProps) {
  const modelColor = MODEL_COLORS[session.model] ?? "#94a3b8";
  const modelDisplay = MODELS.find((m) => m.id === session.model)?.display ?? session.model.toUpperCase();
  const material = MATERIALS.find((m) => m.id === session.material);
  const materialDisplay = material?.label ?? session.material;
  const materialColor = material?.color ?? "#94a3b8";

  return (
    <Dialog
      open={open}
      maxWidth="xs"
      fullWidth
      disableEscapeKeyDown={false}
      onClose={onStartNew}
      PaperProps={{
        sx: {
          background: "rgba(8,14,26,0.97)",
          backdropFilter: "blur(28px)",
          border: "1px solid rgba(0,212,255,0.2)",
          borderTop: "2px solid rgba(0,212,255,0.5)",
          borderRadius: 3,
          boxShadow: "0 0 80px rgba(0,212,255,0.06), 0 32px 64px rgba(0,0,0,0.7)",
          overflow: "visible",
        },
      }}
      slotProps={{
        backdrop: {
          sx: { backgroundColor: "rgba(0,0,0,0.65)", backdropFilter: "blur(4px)" },
        },
      }}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 24 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 340, damping: 28 }}
      >
        <DialogContent sx={{ p: 3.5 }}>
          {/* Header */}
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", mb: 3 }}>
            <Box
              sx={{
                width: 52,
                height: 52,
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: alpha("#00d4ff", 0.08),
                border: "1px solid rgba(0,212,255,0.25)",
                boxShadow: "0 0 28px rgba(0,212,255,0.15)",
                mb: 2,
              }}
            >
              <RestoreRounded sx={{ color: "#00d4ff", fontSize: 24 }} />
            </Box>
            <Typography variant="h6" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "1rem", textAlign: "center" }}>
              Resume Previous Session?
            </Typography>
            <Typography sx={{ color: "rgba(255,255,255,0.38)", fontSize: "0.78rem", textAlign: "center", mt: 0.5, lineHeight: 1.5 }}>
              {isCached
                ? "Prediction results are ready to restore."
                : "Restore your selections and run prediction again."}
            </Typography>
          </Box>

          {/* Info rows */}
          <Box
            sx={{
              backgroundColor: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.07)",
              borderRadius: 2,
              px: 2,
              py: 0.5,
              mb: 3,
            }}
          >
            <InfoRow
              icon={<ScienceRounded sx={{ fontSize: 15 }} />}
              label="Material"
              value={materialDisplay}
              color={materialColor}
            />
            <Divider sx={{ borderColor: "rgba(255,255,255,0.05)" }} />
            <InfoRow
              icon={<SmartToyRounded sx={{ fontSize: 15 }} />}
              label="Model"
              value={modelDisplay}
              color={modelColor}
            />
            <Divider sx={{ borderColor: "rgba(255,255,255,0.05)" }} />
            <InfoRow
              icon={<SpeedRounded sx={{ fontSize: 15 }} />}
              label="Scan Rate"
              value={`${session.scanRate} mV/s`}
              color="#00d4ff"
            />
            <Divider sx={{ borderColor: "rgba(255,255,255,0.05)" }} />
            <InfoRow
              icon={<AccessTimeRounded sx={{ fontSize: 15 }} />}
              label="Last Used"
              value={formatRelativeTime(session.timestamp)}
            />
          </Box>

          {/* Restore capability notice */}
          {!isCached && (
            <Box
              sx={{
                display: "flex",
                gap: 1,
                alignItems: "flex-start",
                backgroundColor: "rgba(251,191,36,0.06)",
                border: "1px solid rgba(251,191,36,0.18)",
                borderRadius: 1.5,
                p: 1.25,
                mb: 2.5,
              }}
            >
              <AccessTimeRounded sx={{ fontSize: 14, color: "#fbbf24", mt: 0.1, flexShrink: 0 }} />
              <Typography sx={{ color: "rgba(251,191,36,0.75)", fontSize: "0.72rem", lineHeight: 1.5 }}>
                Prediction data is not in this session&apos;s cache — selections will be restored so you can re-run the prediction.
              </Typography>
            </Box>
          )}

          {isCached && hasReport && (
            <Box
              sx={{
                display: "flex",
                gap: 1,
                alignItems: "center",
                backgroundColor: "rgba(245,158,11,0.05)",
                border: "1px solid rgba(245,158,11,0.15)",
                borderRadius: 1.5,
                p: 1.25,
                mb: 2.5,
              }}
            >
              <FiberNewRounded sx={{ fontSize: 14, color: "#f59e0b", flexShrink: 0 }} />
              <Typography sx={{ color: "rgba(245,158,11,0.8)", fontSize: "0.72rem" }}>
                A saved report from this session is also available.
              </Typography>
            </Box>
          )}

          {/* Action buttons */}
          <Box sx={{ display: "flex", gap: 1.5 }}>
            <GlowButton
              variant="contained"
              fullWidth
              onClick={onResume}
              glowColor="#00d4ff"
              startIcon={<RestoreRounded sx={{ fontSize: "17px !important" }} />}
              sx={{
                background: "linear-gradient(135deg, #00d4ff 0%, rgba(0,180,220,0.85) 100%)",
                color: "#070b14",
                fontWeight: 700,
                fontSize: "0.82rem",
              }}
            >
              Resume
            </GlowButton>
            <GlowButton
              variant="outlined"
              fullWidth
              onClick={onStartNew}
              glow={false}
              startIcon={<FiberNewRounded sx={{ fontSize: "17px !important" }} />}
              sx={{
                borderColor: "rgba(255,255,255,0.12)",
                color: "rgba(255,255,255,0.4)",
                fontSize: "0.82rem",
                "&:hover": { borderColor: "rgba(255,255,255,0.25)", color: "rgba(255,255,255,0.6)" },
              }}
            >
              Start New
            </GlowButton>
          </Box>
        </DialogContent>
      </motion.div>
    </Dialog>
  );
}
