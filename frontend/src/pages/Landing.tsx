import React from "react";
import { Box, Typography, Button, Chip, alpha } from "@mui/material";
import { FlashOnRounded, ArrowForwardRounded } from "@mui/icons-material";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "../constants/routes";
import { useHealth } from "../hooks/useHealth";

const STATS = [
  { value: "6",      label: "ML Models" },
  { value: "97.51%", label: "Best Generalisation R²" },
  { value: "651",    label: "CV Points" },
  { value: "176 KB", label: "Smallest Model" },
];

export default function Landing() {
  const navigate = useNavigate();
  const { health } = useHealth(60_000);
  const isOnline = health?.status === "ok";

  return (
    <Box sx={{
      height: "100vh",
      background: "#070b14",
      overflow: "hidden",
      position: "relative",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
    }}>

      {/* ── Ambient glow ─────────────────────────────────────────────── */}
      <Box sx={{
        position: "absolute", inset: 0, pointerEvents: "none",
        backgroundImage: `
          radial-gradient(ellipse 65% 45% at 50% 0%, rgba(0,212,255,0.07) 0%, transparent 65%),
          radial-gradient(ellipse 40% 30% at 80% 80%, rgba(167,139,250,0.05) 0%, transparent 60%)
        `,
      }} />
      <Box sx={{
        position: "absolute", inset: 0, pointerEvents: "none",
        backgroundImage: `
          linear-gradient(rgba(0,212,255,0.02) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0,212,255,0.02) 1px, transparent 1px)
        `,
        backgroundSize: "64px 64px",
      }} />

      {/* ── Content ──────────────────────────────────────────────────── */}
      <Box sx={{
        position: "relative", zIndex: 1,
        maxWidth: 760, width: "100%",
        textAlign: "center",
        px: { xs: 3, sm: 4 },
        /* Push content slightly above dead-center for visual balance */
        mt: { xs: 0, md: "-48px" },
      }}>

        {/* Status pill */}
        <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.38 }}>
          <Chip
            label={isOnline ? `⚡ ${health?.models_loaded?.length ?? 6} Models Online` : "○ Backend Offline"}
            sx={{
              mb: 2.5, fontWeight: 700, fontSize: "0.66rem", letterSpacing: "0.06em",
              backgroundColor: isOnline ? "rgba(16,185,129,0.1)" : "rgba(239,68,68,0.1)",
              color: isOnline ? "#10b981" : "#ef4444",
              border: `1px solid ${isOnline ? "rgba(16,185,129,0.28)" : "rgba(239,68,68,0.28)"}`,
              height: 24, px: 0.5,
            }}
          />
        </motion.div>

        {/* Headline */}
        <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.07 }}>
          <Typography component="h1" sx={{
            fontSize: { xs: "2rem", sm: "2.8rem", md: "3.6rem" },
            fontWeight: 900,
            letterSpacing: "-0.04em",
            lineHeight: 1.06,
            mb: 1.5,
            background: "linear-gradient(135deg, #f1f5f9 20%, #00d4ff 55%, #a78bfa 90%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
          }}>
            AI-Powered Supercapacitor<br />Electrochemistry
          </Typography>
        </motion.div>

        {/* Subtitle */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, delay: 0.15 }}>
          <Typography sx={{
            fontSize: { xs: "0.88rem", md: "0.98rem" },
            color: "#64748b",
            lineHeight: 1.65,
            maxWidth: 520,
            mx: "auto",
            mb: 3,
          }}>
            Predict cyclic voltammetry curves for ZnO-based nanocomposite supercapacitors
            using six ML models — from gradient boosting to stacked GRUs.
          </Typography>
        </motion.div>

        {/* CTAs */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.23 }}>
          <Box sx={{ display: "flex", gap: 1.5, justifyContent: "center", flexWrap: "wrap", mb: 4 }}>
            <Button
              variant="contained" size="large"
              startIcon={<FlashOnRounded />}
              onClick={() => navigate(ROUTES.PREDICTION_STUDIO)}
              sx={{
                px: 3.5, py: 1.25, fontSize: "0.88rem", fontWeight: 700,
                borderRadius: "12px",
                background: "linear-gradient(135deg, #00d4ff 0%, #00a8cc 100%)",
                color: "#070b14",
                boxShadow: "0 0 22px rgba(0,212,255,0.28)",
                "&:hover": { boxShadow: "0 0 38px rgba(0,212,255,0.48)", transform: "translateY(-2px)" },
                transition: "all 0.2s ease",
              }}
            >
              Open Studio
            </Button>
            <Button
              variant="outlined" size="large"
              endIcon={<ArrowForwardRounded />}
              onClick={() => navigate(ROUTES.DASHBOARD)}
              sx={{
                px: 3.5, py: 1.25, fontSize: "0.88rem", fontWeight: 600,
                borderRadius: "12px",
                borderColor: "rgba(0,212,255,0.3)",
                color: "#00d4ff",
                "&:hover": { borderColor: "#00d4ff", backgroundColor: "rgba(0,212,255,0.06)", transform: "translateY(-2px)" },
                transition: "all 0.2s ease",
              }}
            >
              Dashboard
            </Button>
          </Box>
        </motion.div>

        {/* Stats strip */}
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.42, delay: 0.3 }}>
          <Box sx={{
            display: "flex",
            gap: { xs: 3, sm: 5 },
            justifyContent: "center",
            flexWrap: "wrap",
          }}>
            {STATS.map((s, i) => (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, delay: 0.32 + i * 0.06 }}
              >
                <Box sx={{ textAlign: "center" }}>
                  <Typography sx={{
                    fontSize: { xs: "1.5rem", sm: "1.7rem" },
                    fontWeight: 900,
                    fontFamily: "JetBrains Mono, monospace",
                    color: "#00d4ff",
                    lineHeight: 1,
                  }}>
                    {s.value}
                  </Typography>
                  <Typography sx={{
                    fontSize: "0.62rem",
                    color: "#475569",
                    letterSpacing: "0.07em",
                    mt: 0.4,
                    textTransform: "uppercase",
                  }}>
                    {s.label}
                  </Typography>
                </Box>
              </motion.div>
            ))}
          </Box>
        </motion.div>
      </Box>

      {/* ── Footer — anchored at viewport bottom ─────────────────────── */}
      <Box sx={{
        position: "absolute",
        bottom: 20,
        left: 0, right: 0,
        textAlign: "center",
        color: "#1e2d40",
        fontSize: "0.6rem",
        letterSpacing: "0.08em",
      }}>
        MCA 2ND SEMESTER PROJECT · ZnO ELECTROCHEMISTRY · ML RESEARCH PLATFORM
      </Box>
    </Box>
  );
}
