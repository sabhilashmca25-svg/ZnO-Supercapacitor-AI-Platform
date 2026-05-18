import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import { Box, Chip, Skeleton, Typography, alpha } from "@mui/material";
import { motion } from "framer-motion";
import type { PredictionResponse, CompareResponse } from "../../types";
import { MODEL_COLORS } from "../../constants/models";

// ── Shared dark Plotly layout ──────────────────────────────────────────────
export const darkLayout = (title?: string): Partial<Plotly.Layout> => ({
  paper_bgcolor: "rgba(7,11,20,0)",
  plot_bgcolor: "rgba(13,19,33,0.6)",
  font: { family: "Inter, sans-serif", color: "#8892a4" },
  title: title
    ? { text: title, font: { size: 13, color: "#f1f5f9", family: "Inter, sans-serif" }, x: 0.02, xanchor: "left" }
    : undefined,
  xaxis: {
    title: { text: "Potential (V)", standoff: 12, font: { size: 12, color: "#94a3b8" } },
    gridcolor: "rgba(255,255,255,0.04)",
    zerolinecolor: "rgba(255,255,255,0.12)",
    zerolinewidth: 1.5,
    linecolor: "rgba(255,255,255,0.12)",
    tickfont: { size: 11, color: "#8892a4", family: "JetBrains Mono, monospace" },
    tickformat: ".2f",
    showgrid: true,
  },
  yaxis: {
    title: { text: "Current (µA)", standoff: 12, font: { size: 12, color: "#94a3b8" } },
    gridcolor: "rgba(255,255,255,0.04)",
    zerolinecolor: "rgba(0,212,255,0.25)",
    zerolinewidth: 1.5,
    linecolor: "rgba(255,255,255,0.12)",
    tickfont: { size: 11, color: "#8892a4", family: "JetBrains Mono, monospace" },
    tickformat: ".1f",
    showgrid: true,
  },
  legend: {
    bgcolor: "rgba(13,19,33,0.8)",
    bordercolor: "rgba(0,212,255,0.15)",
    borderwidth: 1,
    font: { size: 11, color: "#f1f5f9" },
    orientation: "h",
    x: 0,
    y: -0.18,
    traceorder: "normal",
  },
  hoverlabel: {
    bgcolor: "#111c2d",
    bordercolor: "#00d4ff",
    font: { size: 12, color: "#f1f5f9", family: "JetBrains Mono, monospace" },
  },
  margin: { l: 64, r: 24, t: title ? 48 : 24, b: 88 },
  hovermode: "x unified",
  autosize: true,
});

const plotConfig: Partial<Plotly.Config> = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  displaylogo: false,
  toImageButtonOptions: {
    format: "svg",
    filename: "cv_curve_zno_supercapacitor",
    width: 1200,
    height: 600,
    scale: 2,
  },
};

// ── Helper: find peak index ────────────────────────────────────────────────
function peakIdx(arr: number[], mode: "max" | "min"): number {
  let best = mode === "max" ? -Infinity : Infinity;
  let idx = 0;
  for (let i = 0; i < arr.length; i++) {
    if (mode === "max" ? arr[i] > best : arr[i] < best) { best = arr[i]; idx = i; }
  }
  return idx;
}

// ── Single-model CV plot ──────────────────────────────────────────────────

interface SingleCVPlotProps {
  result: PredictionResponse;
  /** Ghost overlay — previous prediction for visual comparison */
  previousResult?: PredictionResponse | null;
  /** Show ±RMSE prediction error band around the prediction curve (NOT a calibrated uncertainty interval) */
  showConfidenceBand?: boolean;
  /** RMSE value (µA) from benchmark evaluation — half-width of the ±RMSE error band */
  rmseUa?: number;
  /** Show peak current annotations (Ia, Ic) on the chart */
  showPeakAnnotations?: boolean;
  /** Experimental (measured) CV curve to overlay — shown as solid green */
  experimentalData?: { potential_V: number[]; actual_current_uA: number[] } | null;
  height?: number;
  loading?: boolean;
}

export function SingleCVPlot({
  result,
  previousResult = null,
  showConfidenceBand = true,
  rmseUa = 30,
  showPeakAnnotations = true,
  experimentalData = null,
  height = 420,
  loading = false,
}: SingleCVPlotProps) {
  const modelName = result.model_name ?? (result as any).model ?? "unknown";
  const color = MODEL_COLORS[modelName] ?? "#00d4ff";

  const { potential_V, predicted_current_uA } = result;

  // ── Traces ───────────────────────────────────────────────────────────────
  const traces = useMemo<Plotly.Data[]>(() => {
    const list: Plotly.Data[] = [];

    // 1. ±RMSE prediction error band (polygon, rendered first so it's behind everything)
    if (showConfidenceBand && rmseUa > 0 && potential_V.length > 0) {
      const upper = predicted_current_uA.map((v) => v + rmseUa);
      const lower = predicted_current_uA.map((v) => v - rmseUa);
      const rev_pot = [...potential_V].reverse();
      const rev_lower = [...lower].reverse();
      list.push({
        x: [...potential_V, ...rev_pot] as number[],
        y: [...upper, ...rev_lower] as number[],
        type: "scatter",
        mode: "none",
        fill: "toself",
        fillcolor: `${color}16`,
        name: `±${rmseUa.toFixed(0)} µA RMSE band`,
        hoverinfo: "skip",
        showlegend: true,
        legendrank: 99,
      });
    }

    // 2. Previous prediction ghost line
    if (previousResult && previousResult.potential_V.length > 0) {
      const prevColor = MODEL_COLORS[previousResult.model_name] ?? "#94a3b8";
      list.push({
        x: previousResult.potential_V,
        y: previousResult.predicted_current_uA,
        type: "scatter",
        mode: "lines",
        name: `${previousResult.model_name.toUpperCase()} (prev)`,
        line: { color: prevColor, width: 1.5, dash: "dash" },
        opacity: 0.38,
        hovertemplate:
          `<b>Prev: ${previousResult.model_name.toUpperCase()}</b><br>` +
          `V: %{x:.3f} V<br>I: %{y:.2f} µA<extra></extra>`,
        legendrank: 98,
      });
    }

    // 3. Experimental (measured) overlay — shown before prediction so prediction sits on top
    if (experimentalData && experimentalData.potential_V.length > 0) {
      list.push({
        x: experimentalData.potential_V,
        y: experimentalData.actual_current_uA,
        type: "scatter",
        mode: "lines",
        name: "Experimental (Measured)",
        line: { color: "#10b981", width: 2, dash: "solid" },
        opacity: 0.85,
        hovertemplate: "<b>Experimental</b><br>V: %{x:.3f} V<br>I: %{y:.2f} µA<extra></extra>",
        legendrank: 2,
      });
    }

    // 4. Main prediction curve
    list.push({
      x: potential_V,
      y: predicted_current_uA,
      type: "scatter",
      mode: "lines",
      name: `${modelName.toUpperCase()} (Predicted)`,
      line: { color, width: 2.5, shape: "spline", smoothing: 0.5, dash: experimentalData ? "dash" : "solid" },
      fill: experimentalData ? undefined : "tozeroy",
      fillcolor: experimentalData ? undefined : `${color}0d`,
      hovertemplate: "<b>V</b>: %{x:.3f} V<br><b>I</b>: %{y:.2f} µA<extra></extra>",
      legendrank: 1,
    });

    return list;
  }, [result, previousResult, showConfidenceBand, rmseUa, color, modelName, potential_V, predicted_current_uA, experimentalData]);

  // ── Peak annotations ─────────────────────────────────────────────────────
  const annotations = useMemo<Partial<Plotly.Annotations>[]>(() => {
    if (!showPeakAnnotations || potential_V.length === 0) return [];

    const iAnodic = peakIdx(predicted_current_uA, "max");
    const iCathodic = peakIdx(predicted_current_uA, "min");
    const Ia = predicted_current_uA[iAnodic];
    const Ic = predicted_current_uA[iCathodic];
    const Va = potential_V[iAnodic];
    const Vc = potential_V[iCathodic];

    const items: Partial<Plotly.Annotations>[] = [
      {
        x: Va, y: Ia,
        xref: "x", yref: "y",
        text: `<b>I<sub>a</sub> = ${Ia.toFixed(1)} µA</b>`,
        showarrow: true,
        arrowhead: 3, arrowsize: 0.9, arrowwidth: 1.5,
        arrowcolor: "#10b981",
        ax: 48, ay: -34,
        font: { size: 10, color: "#10b981", family: "JetBrains Mono" },
        bgcolor: "rgba(16,185,129,0.09)",
        bordercolor: "rgba(16,185,129,0.28)",
        borderwidth: 1, borderpad: 4,
      },
      {
        x: Vc, y: Ic,
        xref: "x", yref: "y",
        text: `<b>I<sub>c</sub> = ${Ic.toFixed(1)} µA</b>`,
        showarrow: true,
        arrowhead: 3, arrowsize: 0.9, arrowwidth: 1.5,
        arrowcolor: "#f472b6",
        ax: 48, ay: 34,
        font: { size: 10, color: "#f472b6", family: "JetBrains Mono" },
        bgcolor: "rgba(244,114,182,0.09)",
        bordercolor: "rgba(244,114,182,0.28)",
        borderwidth: 1, borderpad: 4,
      },
    ];

    // Only add annotations if peaks are reasonably separated
    if (Math.abs(Va - Vc) < 0.01) return [];
    return items;
  }, [result, showPeakAnnotations, potential_V, predicted_current_uA]);

  // ── Layout ────────────────────────────────────────────────────────────────
  const layout = useMemo(
    () => ({
      ...darkLayout(
        `${modelName.toUpperCase()} — ${result.material_id} @ ${result.scan_rate_mVs} mV/s`
      ),
      annotations,
      shapes: [
        {
          type: "line" as const,
          x0: Math.min(...potential_V),
          x1: Math.max(...potential_V),
          y0: 0, y1: 0,
          line: { color: "rgba(255,255,255,0.1)", width: 1, dash: "dot" as Plotly.Dash },
        },
      ],
    }),
    [result, annotations, potential_V]
  );

  if (loading) return <Skeleton variant="rounded" width="100%" height={height} />;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      style={{ width: "100%", height }}
    >
      <Plot
        data={traces}
        layout={{ ...layout, height }}
        config={plotConfig}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </motion.div>
  );
}

// ── Multi-model comparison CV plot ────────────────────────────────────────

interface MultiCVPlotProps {
  result: CompareResponse;
  height?: number;
}

export function MultiCVPlot({ result, height = 480 }: MultiCVPlotProps) {
  const traces = useMemo<Plotly.Data[]>(() => {
    return Object.entries(result.predictions).map(([modelId, pred]) => {
      const color = MODEL_COLORS[modelId] ?? "#94a3b8";
      return {
        x: result.potential_V,
        y: pred.predicted_current_uA,
        type: "scatter" as const,
        mode: "lines" as const,
        name: modelId.toUpperCase(),
        line: { color, width: 2, shape: "spline" as const, smoothing: 0.5 },
        hovertemplate:
          `<b>${modelId.toUpperCase()}</b><br>V: %{x:.3f} V<br>I: %{y:.2f} µA<extra></extra>`,
      };
    });
  }, [result]);

  const layout = useMemo(
    () => darkLayout(`Multi-Model — ${result.material_id} @ ${result.scan_rate_mVs} mV/s`),
    [result]
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      style={{ width: "100%", height }}
    >
      <Plot
        data={traces}
        layout={{ ...layout, height }}
        config={plotConfig}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </motion.div>
  );
}

export default SingleCVPlot;
