import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import { motion } from "framer-motion";
import type { LeaderboardEntry, ComparisonRow } from "../../types";
import { MODEL_COLORS } from "../../constants/models";

/* ── Shared dark layout ──────────────────────────────────────────────────── */
const baseLayout: Partial<Plotly.Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(13,19,33,0.5)",
  font: { family: "Inter, sans-serif", color: "#8892a4", size: 11 },
  autosize: true,
  hoverlabel: {
    bgcolor: "#111c2d",
    bordercolor: "#00d4ff",
    font: { size: 12, color: "#f1f5f9", family: "JetBrains Mono, monospace" },
  },
};

const baseConfig: Partial<Plotly.Config> = {
  responsive: true,
  displayModeBar: false,
  displaylogo: false,
};

/* ── RMSE horizontal bar chart ───────────────────────────────────────────── */

interface RMSEChartProps {
  leaderboard: LeaderboardEntry[];
  height?: number;
}

export function RMSEBarChart({ leaderboard, height = 280 }: RMSEChartProps) {
  const sorted = useMemo(
    () => [...leaderboard].sort((a, b) => b.rmse_val_uA - a.rmse_val_uA),
    [leaderboard]
  );

  const data: Plotly.Data[] = [
    {
      type: "bar",
      orientation: "h",
      y: sorted.map((r) => r.model_display),
      x: sorted.map((r) => r.rmse_val_uA),
      marker: {
        color: sorted.map((r) => MODEL_COLORS[r.model_id] ?? "#94a3b8"),
        opacity: 0.85,
        line: { color: sorted.map((r) => MODEL_COLORS[r.model_id] ?? "#94a3b8"), width: 1 },
      },
      text: sorted.map((r) => `${r.rmse_val_uA.toFixed(1)} µA`),
      textposition: "outside",
      textfont: { size: 10, color: "#8892a4", family: "JetBrains Mono" },
      cliponaxis: false,
      hovertemplate: "<b>%{y}</b><br>RMSE: %{x:.2f} µA<extra></extra>",
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    ...baseLayout,
    height,
    margin: { l: 110, r: 70, t: 12, b: 45 },
    xaxis: {
      title: { text: "RMSE Val (µA)", font: { size: 11, color: "#94a3b8" } },
      gridcolor: "rgba(255,255,255,0.04)",
      tickfont: { size: 10, color: "#8892a4", family: "JetBrains Mono" },
      zeroline: false,
      automargin: true,
    },
    yaxis: {
      tickfont: { size: 11, color: "#f1f5f9" },
      gridcolor: "rgba(255,255,255,0)",
      automargin: true,
    },
  };

  return (
    <motion.div
      initial={{ opacity: 0, scaleY: 0.92 }}
      animate={{ opacity: 1, scaleY: 1 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      style={{ width: "100%", transformOrigin: "bottom" }}
    >
      <Plot data={data} layout={layout} config={baseConfig} style={{ width: "100%", height }} useResizeHandler />
    </motion.div>
  );
}

/* ── R² grouped bar chart (all partitions) ───────────────────────────────
 *
 * IMPORTANT: These MUST match the partition field values returned by the
 * API's /leaderboard/comparison endpoint exactly.  Display labels are
 * kept separately so the lookup never fails.
 */

const PARTITION_KEYS  = ["train", "val", "test_SR", "test_MAT"] as const;
const PARTITION_LABELS = ["Train", "Val (SR=30)", "Test-SR", "Test-MAT (NM4)"] as const;

interface R2ChartProps {
  rows: ComparisonRow[];
  height?: number;
}

export function R2GroupedChart({ rows, height = 320 }: R2ChartProps) {
  // Derive model list preserving display order when possible
  const models = useMemo(() => {
    const seen = new Set<string>();
    const order: string[] = [];
    for (const r of rows) {
      if (!seen.has(r.model_id)) { seen.add(r.model_id); order.push(r.model_id); }
    }
    return order;
  }, [rows]);

  const traces: Plotly.Data[] = useMemo(() => models.map((modelId) => {
    const modelRows = rows.filter((r) => r.model_id === modelId);

    // FIX: match against actual API partition keys, NOT display strings
    const r2Values = PARTITION_KEYS.map((p) => {
      const row = modelRows.find((r) => r.partition === p);
      return row?.r2 ?? null;   // null renders a gap rather than a zero bar
    });

    return {
      type: "bar" as const,
      name: modelId.toUpperCase(),
      x: [...PARTITION_LABELS],  // display labels for x-axis ticks
      y: r2Values,
      marker: { color: MODEL_COLORS[modelId] ?? "#94a3b8", opacity: 0.82 },
      hovertemplate: `<b>${modelId.toUpperCase()}</b><br>%{x}<br>R²: %{y:.4f}<extra></extra>`,
    };
  }), [rows, models]);

  const layout: Partial<Plotly.Layout> = {
    ...baseLayout,
    height,
    barmode: "group",
    bargap: 0.15,
    bargroupgap: 0.06,
    margin: { l: 60, r: 20, t: 52, b: 50 },
    xaxis: {
      tickfont: { size: 11, color: "#f1f5f9" },
      gridcolor: "rgba(255,255,255,0)",
      automargin: true,
    },
    yaxis: {
      title: { text: "R² Score", font: { size: 11, color: "#94a3b8" } },
      gridcolor: "rgba(255,255,255,0.05)",
      tickfont: { size: 10, color: "#8892a4", family: "JetBrains Mono" },
      range: [0.88, 1.002],   // wider than 0.9–1.0 so bars are not cut off
      automargin: true,
    },
    legend: {
      orientation: "h",
      x: 0,
      y: 1.14,
      bgcolor: "rgba(0,0,0,0)",
      font: { size: 10, color: "#f1f5f9" },
    },
  };

  return (
    <motion.div
      initial={{ opacity: 0, scaleY: 0.92 }}
      animate={{ opacity: 1, scaleY: 1 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      style={{ width: "100%", transformOrigin: "bottom" }}
    >
      <Plot data={traces} layout={layout} config={baseConfig} style={{ width: "100%", height }} useResizeHandler />
    </motion.div>
  );
}

/* ── Model size vs RMSE scatter ──────────────────────────────────────────── */

interface SizeVsRMSEProps {
  leaderboard: LeaderboardEntry[];
  height?: number;
}

export function SizeVsRMSEScatter({ leaderboard, height = 300 }: SizeVsRMSEProps) {
  // Stagger text positions to avoid label collisions
  const textPositions = useMemo(() => leaderboard.map((r) => {
    if (r.model_id === "rf") return "top left";
    if (r.model_id === "lightgbm") return "bottom right";
    if (r.model_id === "ann") return "top right";
    return "top center";
  }), [leaderboard]);

  const data: Plotly.Data[] = [
    {
      type: "scatter",
      mode: "markers+text" as Plotly.ScatterData["mode"],
      x: leaderboard.map((r) => r.size_MB),
      y: leaderboard.map((r) => r.rmse_val_uA),
      text: leaderboard.map((r) => r.model_display),
      textposition: textPositions as any,
      textfont: { size: 11, color: "#f1f5f9" },
      marker: {
        color: leaderboard.map((r) => MODEL_COLORS[r.model_id] ?? "#94a3b8"),
        size: 13,
        opacity: 0.9,
        line: { color: "rgba(255,255,255,0.25)", width: 1.5 },
      },
      hovertemplate: "<b>%{text}</b><br>Size: %{x:.2f} MB<br>RMSE: %{y:.2f} µA<extra></extra>",
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    ...baseLayout,
    height,
    margin: { l: 60, r: 24, t: 20, b: 55 },
    xaxis: {
      title: { text: "Model Size (MB)", font: { size: 11, color: "#94a3b8" } },
      type: "log",
      gridcolor: "rgba(255,255,255,0.04)",
      tickfont: { size: 10, color: "#8892a4", family: "JetBrains Mono" },
      automargin: true,
    },
    yaxis: {
      title: { text: "RMSE Val (µA)", font: { size: 11, color: "#94a3b8" } },
      gridcolor: "rgba(255,255,255,0.04)",
      tickfont: { size: 10, color: "#8892a4", family: "JetBrains Mono" },
      automargin: true,
    },
  };

  return (
    <motion.div
      initial={{ opacity: 0, scaleY: 0.92 }}
      animate={{ opacity: 1, scaleY: 1 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      style={{ width: "100%", transformOrigin: "bottom" }}
    >
      <Plot data={data} layout={layout} config={baseConfig} style={{ width: "100%", height }} useResizeHandler />
    </motion.div>
  );
}
