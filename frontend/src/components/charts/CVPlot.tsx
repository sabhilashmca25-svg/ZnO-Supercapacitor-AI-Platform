import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import { Skeleton } from "@mui/material";
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
    // Spike line: marks the hovered voltage on the axis
    showspikes: true,
    spikemode: "across",
    spikethickness: 1,
    spikecolor: "rgba(0,212,255,0.35)",
    spikedash: "dot",
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
    bgcolor: "rgba(10,16,32,0.92)",
    bordercolor: "#00d4ff",
    font: { size: 11, color: "#f1f5f9", family: "JetBrains Mono, monospace" },
    align: "left",
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

// ── Dual-branch hover utilities ───────────────────────────────────────────
/**
 * Split a full CV loop into its two sweep branches.
 * The forward (anodic) branch runs from the most-negative potential up to the
 * turning point; the reverse (cathodic) branch runs back down.
 */
export function splitCVBranches(
  potential_V: number[],
  current_uA: number[]
): {
  fwdPot: number[];  fwdCurr: number[];
  revPot: number[];  revCurr: number[];
} {
  let split = potential_V.length;
  for (let i = 1; i < potential_V.length; i++) {
    if (potential_V[i] < potential_V[i - 1]) { split = i; break; }
  }
  return {
    fwdPot:  potential_V.slice(0, split),
    fwdCurr: current_uA.slice(0, split),
    revPot:  potential_V.slice(split),
    revCurr: current_uA.slice(split),
  };
}

/**
 * For each forward-sweep voltage, find the nearest matching reverse-sweep current
 * via linear voltage proximity search.
 */
function pairReverse(
  fwdPot: number[],
  revPot: number[],
  revCurr: number[]
): number[] {
  return fwdPot.map(v => {
    let best = 0, bestDist = Math.abs(revPot[0] - v);
    for (let j = 1; j < revPot.length; j++) {
      const d = Math.abs(revPot[j] - v);
      if (d < bestDist) { bestDist = d; best = j; }
    }
    return revCurr[best];
  });
}

/**
 * For each forward-sweep voltage, find the nearest current in an arbitrary
 * (potentially different-length) array with its own potential axis.
 */
function pairToAxis(
  queryPot: number[],
  targetPot: number[],
  targetCurr: number[]
): number[] {
  return queryPot.map(v => {
    let best = 0, bestDist = Math.abs(targetPot[0] - v);
    for (let j = 1; j < targetPot.length; j++) {
      const d = Math.abs(targetPot[j] - v);
      if (d < bestDist) { bestDist = d; best = j; }
    }
    return targetCurr[best];
  });
}

// ── Exported for use in ValidationAnalysis ────────────────────────────────
export { pairReverse, pairToAxis };

// Kept for legacy callers
export function buildRevCustomdata(
  revPot: number[], revCurr: number[],
  fwdPot: number[], fwdCurr: number[]
): number[][] {
  return revPot.map((v, i) => {
    let best = 0, bestDist = Math.abs(fwdPot[0] - v);
    for (let j = 1; j < fwdPot.length; j++) {
      const d = Math.abs(fwdPot[j] - v);
      if (d < bestDist) { bestDist = d; best = j; }
    }
    return [Math.abs(fwdCurr[best] - revCurr[i])];
  });
}

// ── Master hover trace ─────────────────────────────────────────────────────
//
// A single invisible wide-hit trace that is the ONLY hover-active trace in
// SingleCVPlot / ValidationAnalysis charts.  All visual traces are set to
// hoverinfo: "skip" so they never produce tooltips.
//
// The tooltip is compact, vertically grouped, and shows both branches:
//
//   -0.140 V
//   ────────────
//   GRU
//   ↑ Fwd :  +325.3 µA
//   ↓ Rev :  -153.9 µA
//   ΔI    :   479.2 µA
//
// When experimentalData is supplied a second section appears:
//
//   ────────────
//   EXPERIMENTAL
//   ↑ Fwd :  +409.1 µA
//   ↓ Rev :  -158.3 µA
//   ΔI    :   567.4 µA
//
export function buildMasterHoverTrace(
  fwdPot:  number[],
  fwdCurr: number[],
  revPot:  number[],
  revCurr: number[],
  modelName: string,
  experimentalData?: { potential_V: number[]; actual_current_uA: number[] } | null
): Plotly.Data {
  const pairedRev = pairReverse(fwdPot, revPot, revCurr);
  const dIPred    = fwdCurr.map((f, i) => Math.abs(f - pairedRev[i]));
  const SEP       = "────────────";

  if (!experimentalData || experimentalData.potential_V.length === 0) {
    return {
      x: fwdPot,
      y: fwdCurr,
      type: "scatter",
      mode: "lines",
      line: { color: "rgba(0,0,0,0)", width: 10 },   // invisible, wide hit area
      showlegend: false,
      customdata: fwdCurr.map((_, i) => [pairedRev[i], dIPred[i]]) as number[][],
      hovertemplate:
        `<b>%{x:.3f} V</b><br>${SEP}<br>` +
        `<b>${modelName.toUpperCase()}</b><br>` +
        `↑ Fwd : <b>%{y:+.1f}</b> µA<br>` +
        `↓ Rev : <b>%{customdata[0]:+.1f}</b> µA<br>` +
        `ΔI     : <b>%{customdata[1]:.1f}</b> µA` +
        `<extra></extra>`,
    };
  }

  // With experimental overlay
  const eS        = splitCVBranches(experimentalData.potential_V, experimentalData.actual_current_uA);
  const expFwd    = pairToAxis(fwdPot, eS.fwdPot, eS.fwdCurr);
  const expRev    = pairToAxis(fwdPot, eS.revPot, eS.revCurr);
  const dIExp     = expFwd.map((f, i) => Math.abs(f - expRev[i]));

  return {
    x: fwdPot,
    y: fwdCurr,
    type: "scatter",
    mode: "lines",
    line: { color: "rgba(0,0,0,0)", width: 10 },
    showlegend: false,
    customdata: fwdCurr.map((_, i) => [
      pairedRev[i], dIPred[i],
      expFwd[i], expRev[i], dIExp[i],
    ]) as number[][],
    hovertemplate:
      `<b>%{x:.3f} V</b><br>${SEP}<br>` +
      `<b>PREDICTED  (${modelName.toUpperCase()})</b><br>` +
      `↑ Fwd : <b>%{y:+.1f}</b> µA<br>` +
      `↓ Rev : <b>%{customdata[0]:+.1f}</b> µA<br>` +
      `ΔI     : <b>%{customdata[1]:.1f}</b> µA<br>` +
      `${SEP}<br>` +
      `<b>EXPERIMENTAL</b><br>` +
      `↑ Fwd : <b>%{customdata[2]:+.1f}</b> µA<br>` +
      `↓ Rev : <b>%{customdata[3]:+.1f}</b> µA<br>` +
      `ΔI     : <b>%{customdata[4]:.1f}</b> µA` +
      `<extra></extra>`,
  };
}

// ── Single-model CV plot ──────────────────────────────────────────────────

interface SingleCVPlotProps {
  result: PredictionResponse;
  previousResult?: PredictionResponse | null;
  showConfidenceBand?: boolean;
  rmseUa?: number;
  showPeakAnnotations?: boolean;
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
  const color     = MODEL_COLORS[modelName] ?? "#00d4ff";
  const { potential_V, predicted_current_uA } = result;

  // ── Traces ───────────────────────────────────────────────────────────────
  const traces = useMemo<Plotly.Data[]>(() => {
    const list: Plotly.Data[] = [];

    // All visual traces use hoverinfo: "skip" — the master trace is the sole
    // hover source.  This prevents Plotly showing duplicate / wide tooltips.

    // 1. ±RMSE prediction error band (polygon, no hover)
    if (showConfidenceBand && rmseUa > 0 && potential_V.length > 0) {
      const upper    = predicted_current_uA.map(v => v + rmseUa);
      const lower    = predicted_current_uA.map(v => v - rmseUa);
      const rev_pot  = [...potential_V].reverse();
      const rev_lower = [...lower].reverse();
      list.push({
        x: [...potential_V, ...rev_pot] as number[],
        y: [...upper, ...rev_lower] as number[],
        type: "scatter", mode: "none",
        fill: "toself", fillcolor: `${color}16`,
        name: `±${rmseUa.toFixed(0)} µA RMSE band`,
        hoverinfo: "skip",
        showlegend: true, legendrank: 99,
      });
    }

    // 2. Previous prediction ghost (visual only, no hover)
    if (previousResult && previousResult.potential_V.length > 0) {
      const prevColor = MODEL_COLORS[previousResult.model_name] ?? "#94a3b8";
      list.push({
        x: previousResult.potential_V,
        y: previousResult.predicted_current_uA,
        type: "scatter", mode: "lines",
        name: `${previousResult.model_name.toUpperCase()} (prev)`,
        line: { color: prevColor, width: 1.5, dash: "dash" },
        opacity: 0.38,
        hoverinfo: "skip",
        showlegend: true, legendrank: 98,
      });
    }

    // 3. Experimental overlay — split into forward / reverse (visual only)
    if (experimentalData && experimentalData.potential_V.length > 0) {
      const { fwdPot: eFP, fwdCurr: eFC, revPot: eRP, revCurr: eRC } =
        splitCVBranches(experimentalData.potential_V, experimentalData.actual_current_uA);
      list.push({
        x: eFP, y: eFC,
        type: "scatter", mode: "lines",
        name: "Experimental (Measured)",
        line: { color: "#10b981", width: 2, dash: "solid" },
        opacity: 0.85, hoverinfo: "skip",
        showlegend: false, legendrank: 2,
      });
      list.push({
        x: eRP, y: eRC,
        type: "scatter", mode: "lines",
        name: "Experimental (Measured)",
        line: { color: "#10b981", width: 2, dash: "solid" },
        opacity: 0.85, hoverinfo: "skip",
        showlegend: true, legendrank: 2,
      });
    }

    // 4. Main prediction curve — split into forward / reverse (visual only)
    const { fwdPot, fwdCurr, revPot, revCurr } =
      splitCVBranches(potential_V, predicted_current_uA);
    const lineStyle = experimentalData ? "dash" : "solid";
    const predLabel = `${modelName.toUpperCase()} (Predicted)`;

    list.push({
      x: fwdPot, y: fwdCurr,
      type: "scatter", mode: "lines",
      name: predLabel,
      line: { color, width: 2.5, shape: "spline", smoothing: 0.5, dash: lineStyle },
      fill: experimentalData ? undefined : "tozeroy",
      fillcolor: experimentalData ? undefined : `${color}0d`,
      hoverinfo: "skip",
      showlegend: false, legendrank: 1,
    });
    list.push({
      x: revPot, y: revCurr,
      type: "scatter", mode: "lines",
      name: predLabel,
      line: { color, width: 2.5, shape: "spline", smoothing: 0.5, dash: lineStyle },
      hoverinfo: "skip",
      showlegend: true, legendrank: 1,
    });

    // 5. Master hover trace — invisible, carries all dual-branch data
    //    Must be last so it sits on top of the hit-test stack.
    list.push(buildMasterHoverTrace(
      fwdPot, fwdCurr, revPot, revCurr,
      modelName, experimentalData
    ));

    return list;
  }, [result, previousResult, showConfidenceBand, rmseUa, color, modelName,
      potential_V, predicted_current_uA, experimentalData]);

  // ── Peak annotations — unchanged (computed from full curve) ──────────────
  const annotations = useMemo<Partial<Plotly.Annotations>[]>(() => {
    if (!showPeakAnnotations || potential_V.length === 0) return [];

    const iAnodic   = peakIdx(predicted_current_uA, "max");
    const iCathodic = peakIdx(predicted_current_uA, "min");
    const Ia = predicted_current_uA[iAnodic];
    const Ic = predicted_current_uA[iCathodic];
    const Va = potential_V[iAnodic];
    const Vc = potential_V[iCathodic];
    if (Math.abs(Va - Vc) < 0.01) return [];

    return [
      {
        x: Va, y: Ia, xref: "x", yref: "y",
        text: `<b>I<sub>a</sub> = ${Ia.toFixed(1)} µA</b>`,
        showarrow: true, arrowhead: 3, arrowsize: 0.9, arrowwidth: 1.5,
        arrowcolor: "#10b981", ax: 48, ay: -34,
        font: { size: 10, color: "#10b981", family: "JetBrains Mono" },
        bgcolor: "rgba(16,185,129,0.09)", bordercolor: "rgba(16,185,129,0.28)",
        borderwidth: 1, borderpad: 4,
      },
      {
        x: Vc, y: Ic, xref: "x", yref: "y",
        text: `<b>I<sub>c</sub> = ${Ic.toFixed(1)} µA</b>`,
        showarrow: true, arrowhead: 3, arrowsize: 0.9, arrowwidth: 1.5,
        arrowcolor: "#f472b6", ax: 48, ay: 34,
        font: { size: 10, color: "#f472b6", family: "JetBrains Mono" },
        bgcolor: "rgba(244,114,182,0.09)", bordercolor: "rgba(244,114,182,0.28)",
        borderwidth: 1, borderpad: 4,
      },
    ];
  }, [result, showPeakAnnotations, potential_V, predicted_current_uA]);

  // ── Layout — "closest" hovermode so the master trace owns the tooltip ─────
  const layout = useMemo(() => ({
    ...darkLayout(
      `${modelName.toUpperCase()} — ${result.material_id} @ ${result.scan_rate_mVs} mV/s`
    ),
    // "closest" shows the master trace tooltip anchored near the cursor rather
    // than the wide "x unified" box that spans all traces at once.
    hovermode: "closest" as const,
    annotations,
    shapes: [
      {
        type: "line" as const,
        x0: Math.min(...potential_V), x1: Math.max(...potential_V),
        y0: 0, y1: 0,
        line: { color: "rgba(255,255,255,0.1)", width: 1, dash: "dot" as Plotly.Dash },
      },
    ],
  }), [result, annotations, potential_V, modelName]);

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
//
// Per-model forward trace carries [revCurrent, ΔI] in customdata and shows
// a compact 3-line tooltip.  Reverse trace is visual-only (hoverinfo:"skip").
// hovermode "closest" shows one model at a time — far less cluttered than
// "x unified" across 6 simultaneous models.

interface MultiCVPlotProps {
  result: CompareResponse;
  height?: number;
}

export function MultiCVPlot({ result, height = 480 }: MultiCVPlotProps) {
  const traces = useMemo<Plotly.Data[]>(() => {
    return Object.entries(result.predictions).flatMap(([modelId, pred]) => {
      const color = MODEL_COLORS[modelId] ?? "#94a3b8";
      const { fwdPot, fwdCurr, revPot, revCurr } =
        splitCVBranches(result.potential_V, pred.predicted_current_uA);
      const pairedRev = pairReverse(fwdPot, revPot, revCurr);
      const fwdCD = fwdCurr.map((f, i) => [pairedRev[i], Math.abs(f - pairedRev[i])]);
      const SEP   = "────────────";

      return [
        // Forward: compact dual-branch tooltip (3 lines + voltage header)
        {
          x: fwdPot, y: fwdCurr,
          type: "scatter" as const, mode: "lines" as const,
          name: modelId.toUpperCase(),
          line: { color, width: 2, shape: "spline" as const, smoothing: 0.5 },
          customdata: fwdCD,
          hovertemplate:
            `<b>%{x:.3f} V</b><br>${SEP}<br>` +
            `<b>${modelId.toUpperCase()}</b><br>` +
            `↑ Fwd : <b>%{y:+.1f}</b> µA<br>` +
            `↓ Rev : <b>%{customdata[0]:+.1f}</b> µA<br>` +
            `ΔI     : <b>%{customdata[1]:.1f}</b> µA` +
            `<extra></extra>`,
          showlegend: false,
        },
        // Reverse: visual only
        {
          x: revPot, y: revCurr,
          type: "scatter" as const, mode: "lines" as const,
          name: modelId.toUpperCase(),
          line: { color, width: 2, shape: "spline" as const, smoothing: 0.5 },
          hoverinfo: "skip" as const,
          showlegend: true,
        },
      ];
    });
  }, [result]);

  const layout = useMemo(() => ({
    ...darkLayout(`Multi-Model — ${result.material_id} @ ${result.scan_rate_mVs} mV/s`),
    hovermode: "closest" as const,
  }), [result]);

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
