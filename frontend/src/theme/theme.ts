import { createTheme, alpha } from "@mui/material/styles";

// ── Design tokens ──────────────────────────────────────────────────────────
const CYAN = "#00d4ff";
const VIOLET = "#a78bfa";
const GREEN = "#10b981";
const AMBER = "#f59e0b";
const RED = "#ef4444";
const PINK = "#f472b6";

const BG_DEEP = "#070b14";
const BG_SURFACE = "#0d1321";
const BG_PAPER = "#0f1923";
const BG_CARD = "#111c2d";

const BORDER_DEFAULT = "rgba(0, 212, 255, 0.10)";
const BORDER_HOVER = "rgba(0, 212, 255, 0.25)";

export const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: CYAN,
      light: "#33dcff",
      dark: "#00aacf",
      contrastText: "#070b14",
    },
    secondary: {
      main: VIOLET,
      light: "#9d61ff",
      dark: "#5b27b5",
      contrastText: "#ffffff",
    },
    success: { main: GREEN, light: "#34d399", dark: "#059669" },
    warning: { main: AMBER, light: "#fbbf24", dark: "#d97706" },
    error: { main: RED, light: "#f87171", dark: "#dc2626" },
    background: {
      default: BG_DEEP,
      paper: BG_PAPER,
    },
    text: {
      primary: "#f1f5f9",
      secondary: "#8892a4",
      disabled: "rgba(241, 245, 249, 0.3)",
    },
    divider: BORDER_DEFAULT,
    action: {
      hover: "rgba(0, 212, 255, 0.06)",
      selected: "rgba(0, 212, 255, 0.12)",
      focus: "rgba(0, 212, 255, 0.15)",
    },
  },

  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica Neue", Arial, sans-serif',
    h1: { fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1 },
    h2: { fontWeight: 800, letterSpacing: "-0.025em", lineHeight: 1.15 },
    h3: { fontWeight: 700, letterSpacing: "-0.02em", lineHeight: 1.2 },
    h4: { fontWeight: 700, letterSpacing: "-0.015em", lineHeight: 1.25 },
    h5: { fontWeight: 600, letterSpacing: "-0.01em" },
    h6: { fontWeight: 600, letterSpacing: "-0.005em" },
    subtitle1: { fontWeight: 500, letterSpacing: "0.01em" },
    subtitle2: { fontWeight: 500, letterSpacing: "0.02em", fontSize: "0.8rem" },
    body1: { lineHeight: 1.7, letterSpacing: "0.01em" },
    body2: { lineHeight: 1.65, letterSpacing: "0.01em" },
    caption: { letterSpacing: "0.06em", fontSize: "0.7rem" },
    overline: { letterSpacing: "0.12em", fontWeight: 600 },
    button: { fontWeight: 600, letterSpacing: "0.04em", textTransform: "none" },
  },

  shape: { borderRadius: 12 },

  shadows: [
    "none",
    `0 1px 4px rgba(0,0,0,0.4)`,
    `0 2px 8px rgba(0,0,0,0.5)`,
    `0 4px 16px rgba(0,0,0,0.5)`,
    `0 6px 24px rgba(0,0,0,0.6)`,
    `0 8px 32px rgba(0,0,0,0.6)`,
    `0 0 0 1px ${BORDER_DEFAULT}, 0 8px 32px rgba(0,0,0,0.6)`,
    `0 0 0 1px ${BORDER_HOVER}, 0 8px 32px rgba(0,0,0,0.7)`,
    `0 0 20px rgba(0,212,255,0.1), 0 8px 32px rgba(0,0,0,0.7)`,
    `0 0 30px rgba(0,212,255,0.15), 0 12px 48px rgba(0,0,0,0.8)`,
    `0 0 40px rgba(0,212,255,0.2), 0 16px 64px rgba(0,0,0,0.8)`,
    ...Array(14).fill("none"),
  ] as any,

  components: {
    // ── CssBaseline ───────────────────────────────────────────────────────
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: BG_DEEP,
          backgroundImage:
            "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 212, 255, 0.03) 0%, transparent 60%)",
        },
      },
    },

    // ── Card ──────────────────────────────────────────────────────────────
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          backgroundColor: BG_CARD,
          border: `1px solid ${BORDER_DEFAULT}`,
          backdropFilter: "blur(8px)",
          transition: "border-color 0.2s ease, box-shadow 0.2s ease",
          "&:hover": {
            borderColor: BORDER_HOVER,
            boxShadow: `0 0 24px rgba(0, 212, 255, 0.08)`,
          },
        },
      },
    },

    // ── Paper ─────────────────────────────────────────────────────────────
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          backgroundColor: BG_PAPER,
          border: `1px solid ${BORDER_DEFAULT}`,
        },
      },
    },

    // ── Button ────────────────────────────────────────────────────────────
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
          letterSpacing: "0.04em",
          textTransform: "none",
          transition: "all 0.2s ease",
        },
        containedPrimary: {
          background: `linear-gradient(135deg, ${CYAN} 0%, #00a8cc 100%)`,
          color: BG_DEEP,
          boxShadow: `0 0 16px rgba(0, 212, 255, 0.25)`,
          "&:hover": {
            background: `linear-gradient(135deg, #33dcff 0%, ${CYAN} 100%)`,
            boxShadow: `0 0 28px rgba(0, 212, 255, 0.45)`,
            transform: "translateY(-1px)",
          },
          "&:active": { transform: "translateY(0)" },
        },
        outlinedPrimary: {
          borderColor: alpha(CYAN, 0.4),
          color: CYAN,
          "&:hover": {
            borderColor: CYAN,
            backgroundColor: alpha(CYAN, 0.06),
          },
        },
      },
    },

    // ── Chip ──────────────────────────────────────────────────────────────
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          letterSpacing: "0.05em",
          fontSize: "0.7rem",
        },
        colorPrimary: {
          backgroundColor: alpha(CYAN, 0.12),
          color: CYAN,
          border: `1px solid ${alpha(CYAN, 0.3)}`,
        },
      },
    },

    // ── TextField ─────────────────────────────────────────────────────────
    MuiTextField: {
      defaultProps: { variant: "outlined" },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          "&:hover .MuiOutlinedInput-notchedOutline": {
            borderColor: alpha(CYAN, 0.4),
          },
          "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
            borderColor: CYAN,
          },
        },
        notchedOutline: {
          borderColor: BORDER_DEFAULT,
        },
      },
    },

    // ── Select ────────────────────────────────────────────────────────────
    MuiSelect: {
      styleOverrides: {
        icon: { color: alpha(CYAN, 0.6) },
      },
    },

    // ── Slider ────────────────────────────────────────────────────────────
    MuiSlider: {
      styleOverrides: {
        root: { color: CYAN },
        thumb: {
          boxShadow: `0 0 8px rgba(0, 212, 255, 0.5)`,
          "&:hover": { boxShadow: `0 0 16px rgba(0, 212, 255, 0.7)` },
        },
        track: {
          background: `linear-gradient(90deg, ${CYAN} 0%, ${VIOLET} 100%)`,
          border: "none",
        },
      },
    },

    // ── Table ─────────────────────────────────────────────────────────────
    MuiTableCell: {
      styleOverrides: {
        head: {
          backgroundColor: BG_SURFACE,
          color: CYAN,
          fontWeight: 600,
          letterSpacing: "0.05em",
          fontSize: "0.75rem",
          textTransform: "uppercase",
          borderBottom: `1px solid ${BORDER_DEFAULT}`,
        },
        body: {
          borderBottom: `1px solid rgba(255,255,255,0.04)`,
          fontSize: "0.875rem",
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          transition: "background-color 0.15s ease",
          "&:hover": { backgroundColor: alpha(CYAN, 0.03) },
        },
      },
    },

    // ── Tooltip ───────────────────────────────────────────────────────────
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: BG_CARD,
          border: `1px solid ${BORDER_DEFAULT}`,
          fontSize: "0.75rem",
          letterSpacing: "0.02em",
        },
        arrow: { color: BG_CARD },
      },
    },

    // ── Drawer ────────────────────────────────────────────────────────────
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: BG_SURFACE,
          borderRight: `1px solid ${BORDER_DEFAULT}`,
          backgroundImage: "none",
        },
      },
    },

    // ── LinearProgress ────────────────────────────────────────────────────
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          backgroundColor: alpha(CYAN, 0.1),
          borderRadius: 4,
        },
        bar: {
          background: `linear-gradient(90deg, ${CYAN} 0%, ${VIOLET} 100%)`,
          borderRadius: 4,
        },
      },
    },

    // ── Tabs ──────────────────────────────────────────────────────────────
    MuiTab: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          letterSpacing: "0.03em",
          textTransform: "none",
          minHeight: 48,
          "&.Mui-selected": { color: CYAN },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        indicator: {
          background: `linear-gradient(90deg, ${CYAN} 0%, ${VIOLET} 100%)`,
          height: 3,
          borderRadius: "3px 3px 0 0",
        },
      },
    },

    // ── Divider ───────────────────────────────────────────────────────────
    MuiDivider: {
      styleOverrides: {
        root: { borderColor: BORDER_DEFAULT },
      },
    },

    // ── Alert ─────────────────────────────────────────────────────────────
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          border: "1px solid",
          borderColor: "inherit",
        },
        standardError: {
          backgroundColor: alpha(RED, 0.08),
          borderColor: alpha(RED, 0.25),
        },
        standardSuccess: {
          backgroundColor: alpha(GREEN, 0.08),
          borderColor: alpha(GREEN, 0.25),
        },
        standardWarning: {
          backgroundColor: alpha(AMBER, 0.08),
          borderColor: alpha(AMBER, 0.25),
        },
      },
    },

    // ── IconButton ────────────────────────────────────────────────────────
    MuiIconButton: {
      styleOverrides: {
        root: {
          transition: "color 0.2s ease, background-color 0.2s ease",
          "&:hover": { backgroundColor: alpha(CYAN, 0.08) },
        },
      },
    },

    // ── AppBar ────────────────────────────────────────────────────────────
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: alpha(BG_SURFACE, 0.85),
          backdropFilter: "blur(12px)",
          borderBottom: `1px solid ${BORDER_DEFAULT}`,
          backgroundImage: "none",
          boxShadow: "none",
        },
      },
    },

    // ── ListItemButton ────────────────────────────────────────────────────
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          margin: "2px 8px",
          transition: "all 0.2s ease",
          "&:hover": {
            backgroundColor: alpha(CYAN, 0.06),
            "& .MuiListItemText-primary": { color: "#f1f5f9" },
          },
          "&.Mui-selected": {
            backgroundColor: alpha(CYAN, 0.10),
            borderLeft: `3px solid ${CYAN}`,
            paddingLeft: "13px",
            "&:hover": { backgroundColor: alpha(CYAN, 0.14) },
            "& .MuiListItemText-primary": { color: CYAN, fontWeight: 700 },
          },
        },
      },
    },

    // ── Skeleton ──────────────────────────────────────────────────────────
    MuiSkeleton: {
      styleOverrides: {
        root: {
          backgroundColor: alpha(CYAN, 0.06),
          "&::after": {
            background: `linear-gradient(90deg, transparent, ${alpha(CYAN, 0.06)}, transparent)`,
          },
        },
      },
    },
  },
});

// ── Design-system exports ──────────────────────────────────────────────────
export const colors = {
  cyan: CYAN,
  violet: VIOLET,
  green: GREEN,
  amber: AMBER,
  red: RED,
  pink: PINK,
  bgDeep: BG_DEEP,
  bgSurface: BG_SURFACE,
  bgPaper: BG_PAPER,
  bgCard: BG_CARD,
  borderDefault: BORDER_DEFAULT,
  borderHover: BORDER_HOVER,
};
