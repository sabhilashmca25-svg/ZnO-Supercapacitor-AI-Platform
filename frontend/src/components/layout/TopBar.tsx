import React from "react";
import {
  AppBar,
  Toolbar,
  Box,
  Typography,
  IconButton,
  Chip,
  Tooltip,
  alpha,
} from "@mui/material";
import {
  MenuRounded,
  OpenInNewRounded,
  FiberManualRecordRounded,
  GetAppRounded,
} from "@mui/icons-material";
import { useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { useHealth } from "../../hooks/useHealth";
import { usePWAInstall } from "../../hooks/usePWAInstall";
import { ROUTES } from "../../constants/routes";

const PAGE_TITLES: Record<string, string> = {
  [ROUTES.DASHBOARD]: "Dashboard",
  [ROUTES.PREDICTION_STUDIO]: "Prediction Studio",
  [ROUTES.MODEL_COMPARISON]: "Model Comparison",
  [ROUTES.RESEARCH_ANALYTICS]: "Research Analytics",
  [ROUTES.VALIDATION]: "Validation Analysis",
  [ROUTES.BENCHMARKS]: "Benchmark Results",
  [ROUTES.MODEL_ENCYCLOPEDIA]: "Model Encyclopedia",
  [ROUTES.ABOUT]: "About Research",
};

interface TopBarProps {
  onMenuClick?: () => void;
}

export default function TopBar({ onMenuClick }: TopBarProps) {
  const { pathname } = useLocation();
  const { health } = useHealth(60_000);

  const { canInstall, install } = usePWAInstall();

  const pageTitle = PAGE_TITLES[pathname] ?? "ZnO AI Platform";
  const isOnline = health?.status === "ok";
  const modelsLoaded = health?.models_loaded ?? [];

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        width: "100%",
        backgroundColor: alpha("#0d1321", 0.88),
        backdropFilter: "blur(12px)",
        borderBottom: "1px solid rgba(0,212,255,0.08)",
        backgroundImage: "none",
        zIndex: (theme) => theme.zIndex.drawer - 1,
      }}
    >
      <Toolbar sx={{ minHeight: "52px !important", px: { xs: 1.5, sm: 2.5 } }}>

        {/* ── Hamburger — visible only below md ──────────────────── */}
        <IconButton
          onClick={onMenuClick}
          edge="start"
          size="small"
          aria-label="open navigation"
          sx={{
            mr: 1.5,
            color: "rgba(255,255,255,0.6)",
            display: { md: "none" },
            padding: "8px", // 40px touch target
          }}
        >
          <MenuRounded fontSize="small" />
        </IconButton>

        {/* ── Page title ─────────────────────────────────────────── */}
        <Box
          component={motion.div}
          key={pathname}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.25 }}
          sx={{ flex: 1, minWidth: 0 }}
        >
          <Typography
            variant="subtitle1"
            sx={{
              fontWeight: 700,
              color: "#f1f5f9",
              fontSize: { xs: "0.88rem", sm: "0.95rem" },
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {pageTitle}
          </Typography>
        </Box>

        {/* ── Model badges — hidden on mobile (saves space) ─────── */}
        <Box sx={{ display: { xs: "none", lg: "flex" }, gap: 0.75, mr: 1.5, flexWrap: "nowrap" }}>
          {modelsLoaded.map((m) => (
            <Chip
              key={m}
              label={m.toUpperCase()}
              size="small"
              sx={{
                height: 20, fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.06em",
                backgroundColor: "rgba(0,212,255,0.1)", color: "#00d4ff",
                border: "1px solid rgba(0,212,255,0.25)",
                "& .MuiChip-label": { px: 0.8 },
              }}
            />
          ))}
        </Box>

        {/* ── API status pill ─────────────────────────────────────── */}
        <Tooltip title={isOnline ? "Backend API online" : "Backend API offline"} arrow>
          <Box
            sx={{
              display: "flex", alignItems: "center", gap: 0.7,
              px: { xs: 1, sm: 1.5 }, py: 0.5, borderRadius: "20px",
              backgroundColor: isOnline ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.08)",
              border: `1px solid ${isOnline ? "rgba(16,185,129,0.2)" : "rgba(239,68,68,0.2)"}`,
              cursor: "default",
              flexShrink: 0,
            }}
          >
            <Box
              component={motion.div}
              animate={isOnline ? { opacity: [1, 0.3, 1] } : {}}
              transition={{ duration: 2.5, repeat: Infinity }}
            >
              <FiberManualRecordRounded sx={{ fontSize: 8, color: isOnline ? "#10b981" : "#ef4444" }} />
            </Box>
            {/* Status text — hidden on xs to save horizontal space */}
            <Typography
              variant="caption"
              sx={{
                display: { xs: "none", sm: "block" },
                color: isOnline ? "#10b981" : "#ef4444",
                fontWeight: 700, fontSize: "0.65rem", letterSpacing: "0.07em",
              }}
            >
              {isOnline ? "ONLINE" : "OFFLINE"}
            </Typography>
          </Box>
        </Tooltip>

        {/* ── Install PWA button — only shown when installable ──────── */}
        {canInstall && (
          <Tooltip title="Install ZnO AI Platform as an app" arrow>
            <Chip
              icon={<GetAppRounded sx={{ fontSize: "13px !important" }} />}
              label="Install"
              size="small"
              onClick={install}
              sx={{
                ml: 0.75,
                height: 24,
                fontSize: "0.62rem",
                fontWeight: 700,
                letterSpacing: "0.05em",
                backgroundColor: "rgba(0,212,255,0.08)",
                color: "#00d4ff",
                border: "1px solid rgba(0,212,255,0.25)",
                cursor: "pointer",
                display: { xs: "none", sm: "flex" },
                "&:hover": {
                  backgroundColor: "rgba(0,212,255,0.14)",
                  borderColor: "rgba(0,212,255,0.45)",
                },
                "& .MuiChip-label": { px: 1 },
                "& .MuiChip-icon": { ml: 0.75 },
              }}
            />
          </Tooltip>
        )}

        {/* ── Swagger link — hidden on mobile ─────────────────────── */}
        <Tooltip title="Open Swagger API Docs" arrow>
          <IconButton
            component="a"
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener"
            size="small"
            sx={{
              ml: 0.75,
              color: "rgba(0,212,255,0.5)",
              display: { xs: "none", sm: "flex" },
              "&:hover": { color: "#00d4ff" },
            }}
          >
            <OpenInNewRounded sx={{ fontSize: 16 }} />
          </IconButton>
        </Tooltip>
      </Toolbar>
    </AppBar>
  );
}
