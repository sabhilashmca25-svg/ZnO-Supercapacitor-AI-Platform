import React from "react";
import {
  Drawer,
  Box,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Chip,
  alpha,
  Tooltip,
} from "@mui/material";
import {
  DashboardRounded,
  ScienceRounded,
  CompareArrowsRounded,
  BarChartRounded,
  AutoAwesomeRounded,
  InfoRounded,
  LeaderboardRounded,
  HomeRounded,
  FiberManualRecordRounded,
  VerifiedRounded,
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { ROUTES } from "../../constants/routes";
import { useHealth } from "../../hooks/useHealth";

export const SIDEBAR_WIDTH = 256;

const NAV = [
  { label: "Dashboard",          path: ROUTES.DASHBOARD,          icon: <DashboardRounded />,      section: "main" },
  { label: "Prediction Studio",  path: ROUTES.PREDICTION_STUDIO,  icon: <ScienceRounded />,        section: "main", highlight: true },
  { label: "Model Comparison",   path: ROUTES.MODEL_COMPARISON,   icon: <CompareArrowsRounded />,  section: "main" },
  { label: "Research Analytics", path: ROUTES.RESEARCH_ANALYTICS, icon: <BarChartRounded />,       section: "research" },
  { label: "Validation Analysis",path: ROUTES.VALIDATION,         icon: <VerifiedRounded />,       section: "research", highlight: true },
  { label: "Benchmarks",         path: ROUTES.BENCHMARKS,         icon: <LeaderboardRounded />,    section: "research" },
  { label: "Model Encyclopedia", path: ROUTES.MODEL_ENCYCLOPEDIA, icon: <AutoAwesomeRounded />,    section: "research" },
  { label: "About Research",     path: ROUTES.ABOUT,              icon: <InfoRounded />,           section: "info" },
];

interface SidebarProps {
  width?: number;
  /** Controls mobile temporary drawer visibility */
  mobileOpen?: boolean;
  /** Called when mobile drawer should close */
  onClose?: () => void;
}

export default function Sidebar({ width = SIDEBAR_WIDTH, mobileOpen = false, onClose }: SidebarProps) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { health } = useHealth(60_000);

  const mainNav     = NAV.filter((n) => n.section === "main");
  const researchNav = NAV.filter((n) => n.section === "research");
  const infoNav     = NAV.filter((n) => n.section === "info");

  const isOnline = health?.status === "ok";

  /** Navigate and close mobile drawer on route change */
  const handleNavigate = (path: string) => {
    navigate(path);
    onClose?.();
  };

  const drawerPaperSx = {
    width,
    boxSizing: "border-box" as const,
    backgroundColor: "#0d1321",
    borderRight: "1px solid rgba(0,212,255,0.08)",
    backgroundImage: "none",
    overflowX: "hidden",
  };

  /* ── Shared drawer content ──────────────────────────────────────── */
  const drawerContent = (
    <>
      {/* Logo */}
      <Box
        component={motion.div}
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        sx={{ px: 2.5, py: 2.5, cursor: "pointer", borderBottom: "1px solid rgba(0,212,255,0.08)" }}
        onClick={() => handleNavigate(ROUTES.LANDING)}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 0.5 }}>
          <Box sx={{
            width: 34, height: 34, borderRadius: "10px",
            background: "linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "16px", boxShadow: "0 0 16px rgba(0,212,255,0.35)", flexShrink: 0,
          }}>
            ⚡
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "#f1f5f9", fontSize: "0.78rem", lineHeight: 1.2, letterSpacing: "0.01em" }}>
              ZnO AI Platform
            </Typography>
            <Typography variant="caption" sx={{ color: "rgba(0,212,255,0.7)", fontSize: "0.6rem", letterSpacing: "0.08em" }}>
              ELECTROCHEMICAL · ML
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Nav sections */}
      <Box sx={{ flex: 1, overflow: "auto", py: 1 }}>
        <NavSection items={mainNav}     pathname={pathname} navigate={handleNavigate} />
        <SectionLabel label="Research" />
        <NavSection items={researchNav} pathname={pathname} navigate={handleNavigate} />
        <SectionLabel label="Info" />
        <NavSection items={infoNav}     pathname={pathname} navigate={handleNavigate} />
      </Box>

      {/* Footer: API status */}
      <Box sx={{ p: 2, borderTop: "1px solid rgba(0,212,255,0.08)" }}>
        <Box sx={{
          display: "flex", alignItems: "center", gap: 1,
          px: 1.5, py: 1, borderRadius: 2,
          backgroundColor: isOnline ? "rgba(16,185,129,0.07)" : "rgba(239,68,68,0.07)",
          border: `1px solid ${isOnline ? "rgba(16,185,129,0.2)" : "rgba(239,68,68,0.2)"}`,
        }}>
          <Box
            component={motion.div}
            animate={isOnline ? { opacity: [1, 0.4, 1] } : {}}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <FiberManualRecordRounded sx={{ fontSize: 10, color: isOnline ? "#10b981" : "#ef4444" }} />
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="caption" sx={{ color: isOnline ? "#10b981" : "#ef4444", fontWeight: 600, fontSize: "0.65rem", letterSpacing: "0.08em", display: "block" }}>
              {isOnline ? "API ONLINE" : "API OFFLINE"}
            </Typography>
            {isOnline && health?.models_loaded && (
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.6rem" }}>
                {health.models_loaded.length} models loaded
              </Typography>
            )}
          </Box>
        </Box>
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.6rem", display: "block", textAlign: "center", mt: 1.5 }}>
          v2.0.0 · ZnO Research Platform
        </Typography>
      </Box>
    </>
  );

  return (
    /**
     * Box reserves sidebar width in the flex layout ONLY on desktop (md+).
     * On mobile the Box collapses to 0 width; the temporary Drawer overlays.
     */
    <Box
      component="nav"
      sx={{ width: { md: width }, flexShrink: { md: 0 } }}
    >
      {/* ── Mobile: temporary overlay drawer ──────────────────────── */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onClose}
        ModalProps={{ keepMounted: true }} // Better mobile performance
        sx={{
          display: { xs: "block", md: "none" },
          "& .MuiDrawer-paper": drawerPaperSx,
        }}
      >
        {drawerContent}
      </Drawer>

      {/* ── Desktop: permanent always-visible drawer ───────────────── */}
      <Drawer
        variant="permanent"
        open
        sx={{
          display: { xs: "none", md: "block" },
          "& .MuiDrawer-paper": drawerPaperSx,
        }}
      >
        {drawerContent}
      </Drawer>
    </Box>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────

function SectionLabel({ label }: { label: string }) {
  return (
    <Typography variant="overline" sx={{ px: 3, py: 0.5, display: "block", color: "rgba(255,255,255,0.25)", fontSize: "0.6rem", letterSpacing: "0.12em" }}>
      {label}
    </Typography>
  );
}

function NavSection({
  items,
  pathname,
  navigate,
}: {
  items: typeof NAV;
  pathname: string;
  navigate: (p: string) => void;
}) {
  return (
    <List dense disablePadding>
      {items.map((item, i) => {
        const active = pathname === item.path;
        return (
          <motion.div
            key={item.path}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04, duration: 0.3 }}
          >
            <Tooltip title={item.label} placement="right" arrow disableHoverListener>
              <ListItemButton
                selected={active}
                onClick={() => navigate(item.path)}
                sx={{
                  mx: 1, borderRadius: "8px", mb: 0.25,
                  py: { xs: 1.1, md: 0.9 }, // larger tap target on mobile
                  px: 1.5,
                  position: "relative", overflow: "hidden",
                  ...(item.highlight && !active && {
                    border: "1px solid rgba(0,212,255,0.15)",
                    backgroundColor: "rgba(0,212,255,0.04)",
                  }),
                  "&.Mui-selected": {
                    backgroundColor: "rgba(0,212,255,0.1)",
                    borderLeft: "3px solid #00d4ff",
                    paddingLeft: "9px",
                    "&:hover": { backgroundColor: "rgba(0,212,255,0.14)" },
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 34, color: active ? "#00d4ff" : "rgba(255,255,255,0.45)", "& svg": { fontSize: 18 } }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{
                    fontSize: "0.82rem",
                    fontWeight: active ? 700 : 500,
                    color: active ? "#00d4ff" : "rgba(255,255,255,0.7)",
                    letterSpacing: "0.01em",
                    whiteSpace: "nowrap",
                  }}
                />
                {item.highlight && (
                  <Chip
                    label="RUN"
                    size="small"
                    sx={{
                      height: 16, fontSize: "0.55rem", fontWeight: 700, letterSpacing: "0.06em",
                      backgroundColor: active ? "transparent" : "rgba(0,212,255,0.12)",
                      color: "#00d4ff", border: "1px solid rgba(0,212,255,0.3)",
                      "& .MuiChip-label": { px: 0.7 },
                    }}
                  />
                )}
              </ListItemButton>
            </Tooltip>
          </motion.div>
        );
      })}
    </List>
  );
}
