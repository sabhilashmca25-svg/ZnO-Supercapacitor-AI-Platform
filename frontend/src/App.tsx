import React, { Suspense, lazy } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider, CssBaseline, Box, CircularProgress } from "@mui/material";
import { theme } from "./theme/theme";
import AppShell from "./components/layout/AppShell";
import { ROUTES } from "./constants/routes";

// ── Eager import — Landing is the true first paint, must be instant ──────────
import Landing from "./pages/Landing";

// ── Lazy imports — all app pages code-split into separate chunks ──────────────
// Dashboard is lazy because it imports Plotly — keeping it eager would pull
// the entire Plotly library into the initial JS bundle unnecessarily.
// Each lazy() call creates its own JS chunk that is only downloaded when the
// user first navigates to that route, keeping the initial bundle minimal.
const Dashboard         = lazy(() => import("./pages/Dashboard"));
const PredictionStudio  = lazy(() => import("./pages/PredictionStudio"));
const ModelComparison   = lazy(() => import("./pages/ModelComparison"));
const ResearchAnalytics = lazy(() => import("./pages/ResearchAnalytics"));
const ValidationAnalysis= lazy(() => import("./pages/ValidationAnalysis"));
const ModelEncyclopedia = lazy(() => import("./pages/ModelEncyclopedia"));
const BenchmarkResults  = lazy(() => import("./pages/BenchmarkResults"));
const About             = lazy(() => import("./pages/About"));

// ── Page-level loading fallback ───────────────────────────────────────────────
function PageLoader() {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "60vh",
        width: "100%",
      }}
    >
      <CircularProgress size={36} sx={{ color: "#00d4ff" }} />
    </Box>
  );
}

/**
 * React Router v6 layout-route pattern.
 * AppShell is a layout route — it renders <Outlet /> for its children.
 * Heavy pages are code-split with React.lazy + Suspense.
 */
export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Routes>
        {/* Landing page — full screen, no sidebar */}
        <Route path={ROUTES.LANDING} element={<Landing />} />

        {/* All app pages — rendered inside AppShell (sidebar + topbar) */}
        <Route element={<AppShell />}>
          {/* All app pages are lazy — Plotly is ~3 MB; defer until needed */}
          <Route
            path={ROUTES.DASHBOARD}
            element={<Suspense fallback={<PageLoader />}><Dashboard /></Suspense>}
          />

          {/* Lazy — downloaded on first visit to each route */}
          <Route
            path={ROUTES.PREDICTION_STUDIO}
            element={<Suspense fallback={<PageLoader />}><PredictionStudio /></Suspense>}
          />
          <Route
            path={ROUTES.MODEL_COMPARISON}
            element={<Suspense fallback={<PageLoader />}><ModelComparison /></Suspense>}
          />
          <Route
            path={ROUTES.RESEARCH_ANALYTICS}
            element={<Suspense fallback={<PageLoader />}><ResearchAnalytics /></Suspense>}
          />
          <Route
            path={ROUTES.VALIDATION}
            element={<Suspense fallback={<PageLoader />}><ValidationAnalysis /></Suspense>}
          />
          <Route
            path={ROUTES.MODEL_ENCYCLOPEDIA}
            element={<Suspense fallback={<PageLoader />}><ModelEncyclopedia /></Suspense>}
          />
          <Route
            path={ROUTES.BENCHMARKS}
            element={<Suspense fallback={<PageLoader />}><BenchmarkResults /></Suspense>}
          />
          <Route
            path={ROUTES.ABOUT}
            element={<Suspense fallback={<PageLoader />}><About /></Suspense>}
          />
          {/* Unknown paths → dashboard */}
          <Route path="*" element={<Navigate to={ROUTES.DASHBOARD} replace />} />
        </Route>
      </Routes>
    </ThemeProvider>
  );
}
