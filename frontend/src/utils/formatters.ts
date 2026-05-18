// ── Number formatters ──────────────────────────────────────────────────────

export const fmtUa = (v: number | null | undefined, decimals = 2): string => {
  if (v == null) return "—";
  return `${v.toFixed(decimals)} µA`;
};

export const fmtR2 = (v: number | null | undefined): string => {
  if (v == null) return "—";
  return v.toFixed(4);
};

export const fmtPct = (v: number | null | undefined): string => {
  if (v == null) return "—";
  return `${(v * 100).toFixed(2)}%`;
};

export const fmtMB = (v: number | null | undefined): string => {
  if (v == null) return "—";
  if (v < 1) return `${(v * 1000).toFixed(0)} KB`;
  return `${v.toFixed(1)} MB`;
};

export const fmtMs = (v: number | null | undefined): string => {
  if (v == null) return "—";
  if (v < 1) return `<1 ms`;
  return `${v.toFixed(1)} ms`;
};

export const fmtV = (v: number): string => `${v.toFixed(3)} V`;

export const fmtScanRate = (v: number): string => `${v} mV/s`;

// ── Colour helpers ────────────────────────────────────────────────────────

export const rmseColor = (rmse: number): string => {
  if (rmse <= 28) return "#22d3ee";  // cyan — excellent
  if (rmse <= 40) return "#f59e0b";  // amber — good
  return "#ef4444";                   // red — poor
};

export const r2Color = (r2: number): string => {
  if (r2 >= 0.97) return "#10b981";   // green — excellent
  if (r2 >= 0.95) return "#22d3ee";   // cyan — good
  if (r2 >= 0.90) return "#f59e0b";   // amber — ok
  return "#ef4444";                    // red — poor
};

// ── Partition labels ──────────────────────────────────────────────────────

export const shortPartition = (label: string): string => {
  if (label.includes("train")) return "Train";
  if (label.includes("val")) return "Val";
  if (label.includes("SR")) return "Test-SR";
  if (label.includes("MAT") || label.includes("NM4")) return "Test-MAT";
  return label;
};
