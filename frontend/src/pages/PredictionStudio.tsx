import React, { useState, useCallback, useMemo, useEffect, useRef } from "react";
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Alert,
  Chip,
  alpha,
  Tooltip,
  TextField,
  InputAdornment,
  Divider,
  IconButton,
  CircularProgress,
} from "@mui/material";
import {
  FlashOnRounded,
  MemoryRounded,
  TrendingUpRounded,
  InfoOutlined,
  TimerRounded,
  TableChartRounded,
  DownloadRounded,
  HistoryRounded,
  ScienceRounded,
  DeleteOutlineRounded,
  VerifiedRounded,
  CompareArrowsRounded,
  BiotechRounded,
  TrendingFlatRounded,
  ArrowUpwardRounded,
  ArrowDownwardRounded,
  AssignmentRounded,
} from "@mui/icons-material";
import { motion, AnimatePresence } from "framer-motion";
import { usePrediction } from "../hooks/usePrediction";
import { SingleCVPlot } from "../components/charts/CVPlot";
import { getExperimentalCurve } from "../api/endpoints";
import type { ExperimentalCurveResponse } from "../types";
import GlowButton from "../components/common/GlowButton";
import SectionHeader from "../components/common/SectionHeader";
import LoadingOverlay from "../components/common/LoadingOverlay";
import ModelBadge from "../components/common/ModelBadge";
import { MODELS, MATERIALS, MODEL_COLORS } from "../constants/models";
import { fmtUa } from "../utils/formatters";
import { cardVariants, staggerContainer } from "../animations/variants";
import GlossaryTooltip from "../components/common/GlossaryTooltip";
import ExperimentReport from "../components/report/ExperimentReport";
import type { PredictionResponse } from "../types";
import { useRecentExperiments } from "../hooks/useRecentExperiments";
import { useLastSession } from "../hooks/useLastSession";
import { useAppDispatch } from "../hooks/useAppDispatch";
import { setLatestKey } from "../store/predictionSlice";
import RecentExperimentsPanel from "../components/pwa/RecentExperimentsPanel";
import ResumeSessionDialog from "../components/pwa/ResumeSessionDialog";
import type { ExperimentRecord } from "../hooks/useRecentExperiments";

const SCAN_RATE_MARKS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((v) => ({
  value: v,
  label: v % 20 === 0 || v === 10 ? `${v}` : "",
}));
const VALID_SR = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
const LAZY_MODELS = new Set(["xgboost", "ann", "lstm"]);

// ── Model benchmark RMSE (µA) from offline evaluation ─────────────────────
const MODEL_RMSE_UA: Record<string, number> = {
  rf: 26.59,
  lightgbm: 26.86,
  xgboost: 28.83,
  gru: 36.84,
  lstm: 36.51,
  ann: 49.93,
};

function nearestSR(val: number): number {
  const r = Math.round(val);
  if (VALID_SR.includes(r)) return r;
  return VALID_SR.reduce((a, b) => (Math.abs(b - val) < Math.abs(a - val) ? b : a));
}

// ── Confidence tier classification ─────────────────────────────────────────
function getConfidenceTier(
  modelId: string,
  materialId: string,
  scanRateMvs: number
): { label: string; color: string; desc: string } {
  const rmse = MODEL_RMSE_UA[modelId] ?? 40;
  const isZeroShot = materialId === "NM4";
  const isEdgeSR = scanRateMvs >= 90;

  if (isZeroShot) {
    return {
      label: "Cross-Material Transfer",
      color: "#f472b6",
      desc: "NM4 was fully excluded from training — model generalises CV topology learned from NM1–3; test-MAT R² = 0.9751 (GRU) validates cross-material transfer capability",
    };
  }
  if (isEdgeSR) {
    return {
      label: "Upper Training Boundary",
      color: "#f59e0b",
      desc: `SR=90/100 mV/s are training scan rates — in-distribution, but edge-of-range predictions may carry slight asymmetric bias (~${(rmse * 1.15).toFixed(0)} µA est.)`,
    };
  }
  if (rmse < 30) {
    return {
      label: "Interpolation — High Accuracy",
      color: "#10b981",
      desc: `Val RMSE ${rmse.toFixed(2)} µA — within-distribution interpolation regime; well-validated on held-out SR=30 partition`,
    };
  }
  if (rmse < 45) {
    return {
      label: "Validated Regime",
      color: "#22d3ee",
      desc: `Val RMSE ${rmse.toFixed(2)} µA — within training distribution; suitable for quantitative trend analysis`,
    };
  }
  return {
    label: "Elevated Uncertainty",
    color: "#f59e0b",
    desc: `Val RMSE ${rmse.toFixed(2)} µA — higher prediction spread; interpret as qualitative trends only; not recommended for quantitative analysis`,
  };
}

// ── Advanced electrochemical metrics (computed client-side) ────────────────
function computeAdvancedMetrics(result: PredictionResponse) {
  const { potential_V, predicted_current_uA, statistics, scan_rate_mVs } = result;
  const { peak_anodic_uA, peak_cathodic_uA, integral_area } = statistics;

  const vMin = Math.min(...potential_V);
  const vMax = Math.max(...potential_V);
  const voltageWindow = vMax - vMin;

  const symFactor =
    Math.abs(peak_cathodic_uA) > 0
      ? Math.abs(peak_anodic_uA) / Math.abs(peak_cathodic_uA)
      : null;

  const sr_Vs = scan_rate_mVs * 1e-3;
  // C = ∮I dV / (2·ν·ΔV) — factor of 2 because integral_area covers the full CV loop (both sweeps)
  const chargeStorageIndex =
    voltageWindow > 0 && sr_Vs > 0
      ? integral_area / (2 * sr_Vs * voltageWindow)
      : null;

  const energyProxy =
    chargeStorageIndex != null ? 0.5 * chargeStorageIndex * voltageWindow * voltageWindow : null;

  const asymmetryPct =
    symFactor != null ? Math.abs(symFactor - 1) * 100 : null;

  const maxI = Math.max(...predicted_current_uA);
  const minI = Math.min(...predicted_current_uA);
  const idxMax = predicted_current_uA.indexOf(maxI);
  const idxMin = predicted_current_uA.indexOf(minI);
  const peakSep =
    idxMax >= 0 && idxMin >= 0
      ? Math.abs(potential_V[idxMax] - potential_V[idxMin])
      : null;

  return { symFactor, chargeStorageIndex, energyProxy, asymmetryPct, peakSep, voltageWindow };
}

// ── Scientific interpretation engine ──────────────────────────────────────
function generateInterpretation(
  result: PredictionResponse,
  adv: ReturnType<typeof computeAdvancedMetrics>,
  previousResult: PredictionResponse | null
): string[] {
  const lines: string[] = [];
  const model = result.model_name.toUpperCase();
  const sr = result.scan_rate_mVs;
  const mat = result.material_id;
  const { peak_anodic_uA, integral_area } = result.statistics;
  const { symFactor, chargeStorageIndex, peakSep } = adv;

  // 1. CV morphology
  const isSymmetric = symFactor != null && Math.abs(symFactor - 1) < 0.08;
  const isIdealCapacitor = isSymmetric && (peakSep ?? 1) < 0.05;
  if (isIdealCapacitor) {
    lines.push(
      `${model} predicts a near-rectangular CV profile for ${mat} at ${sr} mV/s — characteristic of ideal EDLC behaviour with surface-confined charge storage and no diffusion-limited Faradaic contribution.`
    );
  } else if (isSymmetric) {
    lines.push(
      `${model} predicts a symmetric CV envelope for ${mat} at ${sr} mV/s. Symmetry factor ${symFactor!.toFixed(3)} ≈ 1.000 confirms predominantly capacitive charge storage; minor peak offset (${(peakSep ?? 0).toFixed(3)} V) may indicate shallow redox activity.`
    );
  } else {
    lines.push(
      `${model} predicts a ${(symFactor ?? 1) > 1.2 ? "anodic-dominant" : "cathodic-dominant"} CV profile for ${mat} at ${sr} mV/s (symmetry factor ${symFactor?.toFixed(3) ?? "N/A"}), suggesting a mixed capacitive–Faradaic mechanism or ion diffusion limitation in the bulk electrode.`
    );
  }

  // 2. Scan-rate dependency
  if (sr <= 20) {
    lines.push(
      `At ${sr} mV/s (low scan rate), ions have sufficient time to penetrate all accessible active sites — the integral area ${integral_area.toFixed(1)} µA·V reflects near-equilibrium, maximum utilised capacitance.`
    );
  } else if (sr <= 50) {
    lines.push(
      `Moderate scan rate (${sr} mV/s) balances power delivery and ion access depth. The ${peak_anodic_uA.toFixed(1)} µA anodic peak reflects partial diffusion limitation consistent with the Randles–Ševčík I∝ν^0.5 relationship.`
    );
  } else {
    lines.push(
      `At ${sr} mV/s (high scan rate), ion diffusion depth is curtailed — CV area compression to ${integral_area.toFixed(1)} µA·V is consistent with IR drop and the ν^0.5 peak-current scaling. True specific capacitance should be evaluated at ≤20 mV/s.`
    );
  }

  // 3. Charge storage context
  if (chargeStorageIndex != null) {
    lines.push(
      `Estimated Charge Storage Index: ${chargeStorageIndex.toFixed(1)} µF (C = ∮I dV / [2·ν·ΔV]). ` +
      (chargeStorageIndex > 200
        ? "High charge storage — composite electrode likely exhibiting pseudocapacitive enhancement."
        : chargeStorageIndex > 80
        ? "Moderate charge storage — typical for pure ZnO nanostructures in Na₂SO₄ electrolyte."
        : "Low charge storage — may indicate high scan rate compression or material with limited active surface.")
    );
  }

  // 4. Material-specific electrochemistry
  const materialInsights: Record<string, string> = {
    NM1: `NM1 (ZnO baseline) — the predicted CV establishes the pure-ZnO reference. Charge storage arises from ZnO surface interactions with Na₂SO₄ electrolyte ions. Any difference in composites (NM2–4) is benchmarked against this prediction.`,
    NM2: `NM2 (experimental sample) — shows a CV profile distinct from NM1. The change in integral area vs. NM1 reflects differences in charge storage behaviour, surface area, or electrode composition. Specific composition is not defined in the source dataset.`,
    NM3: `NM3 (experimental sample) — shows CV behaviour different from NM1 and NM2. Any peaks or asymmetry relative to NM1 may indicate different charge storage mechanisms. Specific composition is not defined in the source dataset.`,
    NM4: `NM4 (zero-shot experimental sample) — completely unseen during training. The model predicts CV behaviour based purely on patterns learned from NM1–3. This is the hardest generalisation test: any morphological differences vs. NM1–3 probe the model's transfer capability. Specific composition is not defined in the source dataset.`,
  };
  if (materialInsights[mat]) lines.push(materialInsights[mat]);

  // 5. Δ vs previous (if different combination)
  if (
    previousResult &&
    (previousResult.model_name !== result.model_name ||
      previousResult.material_id !== result.material_id ||
      previousResult.scan_rate_mVs !== result.scan_rate_mVs)
  ) {
    const Δanodic = peak_anodic_uA - previousResult.statistics.peak_anodic_uA;
    const Δintegral = integral_area - previousResult.statistics.integral_area;
    const prevLabel = `${previousResult.model_name.toUpperCase()} / ${previousResult.material_id} @ ${previousResult.scan_rate_mVs} mV/s`;
    const magnitude = Math.abs(Δanodic);
    const verdict =
      magnitude < 5
        ? "Electrochemically similar — difference within model RMSE range."
        : magnitude < 20
        ? "Moderate shift — parameter change has measurable electrochemical impact."
        : "Large shift — significant change in electrode response predicted.";
    lines.push(
      `vs. previous (${prevLabel}): anodic peak ${Δanodic >= 0 ? "↑" : "↓"}${Math.abs(Δanodic).toFixed(1)} µA, integral area ${Δintegral >= 0 ? "+" : ""}${Δintegral.toFixed(1)} µA·V. ${verdict}`
    );
  }

  return lines;
}

// ── CSV export ─────────────────────────────────────────────────────────────
function exportCSV(result: PredictionResponse) {
  const header = "Potential_V,Current_uA";
  const rows = result.potential_V.map((v, i) =>
    `${v.toFixed(6)},${result.predicted_current_uA[i].toFixed(4)}`
  );
  const csv = [header, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `cv_${result.model_name}_${result.material_id}_${result.scan_rate_mVs}mVs.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Summary export ─────────────────────────────────────────────────────────
function exportSummary(result: PredictionResponse, adv: ReturnType<typeof computeAdvancedMetrics>) {
  const lines = [
    "ZnO Supercapacitor AI Platform — Prediction Summary",
    "=".repeat(52),
    "",
    `Model        : ${result.model_name.toUpperCase()}`,
    `Material     : ${result.material_id}`,
    `Scan Rate    : ${result.scan_rate_mVs} mV/s`,
    `Data Points  : ${result.n_points}`,
    "",
    "── Basic Statistics ──",
    `Peak Anodic Current         : ${result.statistics.peak_anodic_uA.toFixed(2)} µA`,
    `Peak Cathodic Current       : ${result.statistics.peak_cathodic_uA.toFixed(2)} µA`,
    `Current Range               : ${result.statistics.current_range_uA.toFixed(2)} µA`,
    `Integral Area               : ${result.statistics.integral_area.toFixed(2)} µA·V`,
    "",
    "── Advanced Electrochemical Metrics ──",
    `Symmetry Factor             : ${adv.symFactor?.toFixed(4) ?? "N/A"} (ideal = 1.000)`,
    `Charge Storage Index        : ${adv.chargeStorageIndex?.toFixed(2) ?? "N/A"} µF`,
    `Energy Storage Proxy        : ${adv.energyProxy?.toFixed(4) ?? "N/A"} µJ`,
    `CV Asymmetry                : ${adv.asymmetryPct?.toFixed(1) ?? "N/A"} %`,
    `Peak Separation             : ${adv.peakSep?.toFixed(3) ?? "N/A"} V`,
    `Voltage Window              : ${adv.voltageWindow.toFixed(3)} V`,
    "",
    `Model Benchmark RMSE        : ${(MODEL_RMSE_UA[result.model_name] ?? 0).toFixed(2)} µA`,
    `Generated                   : ${new Date().toISOString()}`,
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `summary_${result.model_name}_${result.material_id}_${result.scan_rate_mVs}mVs.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Delta indicator helper ─────────────────────────────────────────────────
function DeltaCell({ delta, unit }: { delta: number; unit: string }) {
  const abs = Math.abs(delta);
  const color =
    abs < 3 ? "rgba(255,255,255,0.35)" : delta > 0 ? "#10b981" : "#f472b6";
  const Icon =
    abs < 3 ? TrendingFlatRounded : delta > 0 ? ArrowUpwardRounded : ArrowDownwardRounded;
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
      <Icon sx={{ fontSize: 12, color }} />
      <Typography sx={{ color, fontFamily: "JetBrains Mono", fontSize: "0.72rem", fontWeight: 700 }}>
        {delta >= 0 ? "+" : ""}{delta.toFixed(1)}{unit}
      </Typography>
    </Box>
  );
}

// ═══════════════════════════════════════════════════════════════════════════

export default function PredictionStudio() {
  const [modelId, setModelId] = useState("lightgbm");
  const [materialId, setMaterialId] = useState("NM1");
  const [scanRate, setScanRate] = useState<number>(30);
  const [scanRateText, setScanRateText] = useState("30");
  const [previousResult, setPreviousResult] = useState<PredictionResponse | null>(null);
  // Experimental overlay state
  const [showExperimental, setShowExperimental] = useState(false);
  const [experimentalData, setExperimentalData] = useState<ExperimentalCurveResponse | null>(null);
  const [experimentalLoading, setExperimentalLoading] = useState(false);
  const { predict, clear, results, latestResult, loading, error } = usePrediction();
  const dispatch = useAppDispatch();

  // ── Experiment Report state ────────────────────────────────────────────
  const [reportOpen, setReportOpen] = useState(false);
  const [reportGeneratedAt, setReportGeneratedAt] = useState(() => new Date());
  const [predTimeMs, setPredTimeMs] = useState<number | null>(null);
  const predStartRef = useRef<number | null>(null);
  const prevLoadingRef = useRef(false);

  // ── Persistent PWA hooks ───────────────────────────────────────────────
  const { records: recentExps, save: saveExperiment, attachReportId, clear: clearExperiments } = useRecentExperiments();
  const { lastSession, save: saveSession } = useLastSession();
  const currentExpIdRef = useRef<string | null>(null);
  const [resumeOpen, setResumeOpen] = useState(() => !!lastSession);
  const [restoreMessage, setRestoreMessage] = useState<string | null>(null);

  const snappedSR = useMemo(() => nearestSR(scanRate), [scanRate]);
  const isSnapped = snappedSR !== Math.round(scanRate * 10) / 10;
  const isLazyModel = LAZY_MODELS.has(modelId);
  const modelColor = MODEL_COLORS[modelId] ?? "#00d4ff";
  const modelConfig = MODELS.find((m) => m.id === modelId);

  // History (most recent 5, reversed)
  const historyEntries = useMemo(
    () => Object.values(results).slice(-5).reverse(),
    [results]
  );

  // Keys currently in Redux cache — used by RecentExperimentsPanel to show restore status
  const cachedKeys = useMemo(() => new Set(Object.keys(results)), [results]);

  // Advanced metrics for current result
  const advMetrics = useMemo(
    () => (latestResult ? computeAdvancedMetrics(latestResult) : null),
    [latestResult]
  );

  // Confidence tier for current result
  const confidenceTier = useMemo(
    () =>
      latestResult
        ? getConfidenceTier(latestResult.model_name, latestResult.material_id, latestResult.scan_rate_mVs)
        : null,
    [latestResult]
  );

  // Track prediction wall-clock time (loading false→true = start, true→false = end)
  // Also persist session + experiment record on each successful prediction
  useEffect(() => {
    if (loading && !prevLoadingRef.current) {
      predStartRef.current = Date.now();
    } else if (!loading && prevLoadingRef.current && latestResult) {
      setPredTimeMs(Math.round(Date.now() - (predStartRef.current ?? Date.now())));
      setReportGeneratedAt(new Date());

      // Persist experiment record for Feature 4 (Recent Experiments)
      const tier = getConfidenceTier(latestResult.model_name, latestResult.material_id, latestResult.scan_rate_mVs);
      const expId = saveExperiment({
        model: latestResult.model_name,
        material: latestResult.material_id,
        scanRate: latestResult.scan_rate_mVs,
        confidence: { label: tier.label, color: tier.color },
        peakAnodic: latestResult.statistics.peak_anodic_uA,
        peakCathodic: latestResult.statistics.peak_cathodic_uA,
        maxCurrent: latestResult.statistics.peak_anodic_uA,
        minCurrent: latestResult.statistics.peak_cathodic_uA,
        rmse: MODEL_RMSE_UA[latestResult.model_name] ?? 0,
        isZeroShot: latestResult.material_id === "NM4",
      });
      currentExpIdRef.current = expId;

      // Persist last session for Feature 5 (Resume Last Session)
      saveSession({
        model: latestResult.model_name,
        material: latestResult.material_id,
        scanRate: latestResult.scan_rate_mVs,
        lastExperimentId: expId,
        timestamp: new Date().toISOString(),
      });
    }
    prevLoadingRef.current = loading;
  }, [loading, latestResult, saveSession, saveExperiment]);

  // Scientific interpretation paragraphs
  const interpretation = useMemo(
    () =>
      latestResult && advMetrics
        ? generateInterpretation(latestResult, advMetrics, previousResult)
        : [],
    [latestResult, advMetrics, previousResult]
  );

  // Delta vs previous (only if different prediction key)
  const deltaVsPrev = useMemo(() => {
    if (!latestResult || !previousResult) return null;
    if (
      latestResult.model_name === previousResult.model_name &&
      latestResult.material_id === previousResult.material_id &&
      latestResult.scan_rate_mVs === previousResult.scan_rate_mVs
    )
      return null;
    return {
      anodic: latestResult.statistics.peak_anodic_uA - previousResult.statistics.peak_anodic_uA,
      cathodic: latestResult.statistics.peak_cathodic_uA - previousResult.statistics.peak_cathodic_uA,
      integral: latestResult.statistics.integral_area - previousResult.statistics.integral_area,
      range: latestResult.statistics.current_range_uA - previousResult.statistics.current_range_uA,
    };
  }, [latestResult, previousResult]);

  // RMSE for the current result's model (used for confidence band width)
  const resultRmseUa = latestResult ? (MODEL_RMSE_UA[latestResult.model_name] ?? 35) : 35;

  const handleSliderChange = useCallback((_: Event, v: number | number[]) => {
    const val = v as number;
    setScanRate(val);
    setScanRateText(Number.isInteger(val) ? String(val) : val.toFixed(1));
  }, []);

  const handleTextChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setScanRateText(raw);
    const parsed = parseFloat(raw);
    if (!isNaN(parsed) && parsed >= 10 && parsed <= 100) {
      setScanRate(Math.round(parsed * 10) / 10);
    }
  }, []);

  const handleTextBlur = useCallback(() => {
    const parsed = parseFloat(scanRateText);
    if (isNaN(parsed) || parsed < 10 || parsed > 100) {
      setScanRate(30);
      setScanRateText("30");
    } else {
      const clamped = Math.min(100, Math.max(10, Math.round(parsed * 10) / 10));
      setScanRate(clamped);
      setScanRateText(Number.isInteger(clamped) ? String(clamped) : clamped.toFixed(1));
    }
  }, [scanRateText]);

  const handleResume = useCallback(() => {
    if (!lastSession) return;
    setModelId(lastSession.model);
    setMaterialId(lastSession.material);
    setScanRate(lastSession.scanRate);
    setScanRateText(String(lastSession.scanRate));
    const cacheKey = `${lastSession.model}_${lastSession.material}_${lastSession.scanRate}`;
    if (cacheKey in results) {
      dispatch(setLatestKey(cacheKey));
    }
    setResumeOpen(false);
  }, [lastSession, results, dispatch]);

  const handleRestoreExperiment = useCallback((exp: ExperimentRecord) => {
    setModelId(exp.model);
    setMaterialId(exp.material);
    setScanRate(exp.scanRate);
    setScanRateText(String(exp.scanRate));
    const cacheKey = `${exp.model}_${exp.material}_${exp.scanRate}`;
    if (cacheKey in results) {
      dispatch(setLatestKey(cacheKey));
      if (exp.reportId) {
        setReportOpen(true);
      }
    } else {
      setRestoreMessage("This experiment's prediction data is no longer available in this session. Selections have been restored — run the prediction again to see results.");
    }
  }, [results, dispatch]);

  const handlePredict = useCallback(() => {
    // Save current result as "previous" before issuing the new prediction
    setPreviousResult(latestResult ?? null);
    // Clear experimental overlay when running new prediction (different params)
    if (latestResult && (
      latestResult.material_id !== materialId ||
      latestResult.scan_rate_mVs !== snappedSR
    )) {
      setExperimentalData(null);
      setShowExperimental(false);
    }
    predict({ model_name: modelId, material_id: materialId, scan_rate_mVs: snappedSR });
  }, [predict, modelId, materialId, snappedSR, latestResult]);

  const handleToggleExperimental = useCallback(async () => {
    if (!latestResult) return;
    if (showExperimental) {
      setShowExperimental(false);
      return;
    }
    // If we already have data for this material+SR, just show it
    if (
      experimentalData &&
      experimentalData.material_id === latestResult.material_id &&
      experimentalData.scan_rate_mVs === latestResult.scan_rate_mVs
    ) {
      setShowExperimental(true);
      return;
    }
    // Fetch experimental data — capture current params to guard against concurrent calls
    const fetchMat = latestResult.material_id;
    const fetchSR = latestResult.scan_rate_mVs;
    setExperimentalLoading(true);
    try {
      const data = await getExperimentalCurve(fetchMat, fetchSR);
      // Only commit if the result still matches the prediction that triggered the fetch
      if (
        latestResult &&
        latestResult.material_id === fetchMat &&
        latestResult.scan_rate_mVs === fetchSR
      ) {
        setExperimentalData(data);
        setShowExperimental(true);
      }
    } catch {
      // Silently ignore — experimental data unavailable
    } finally {
      setExperimentalLoading(false);
    }
  }, [latestResult, showExperimental, experimentalData]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => { if (e.key === "Enter") handlePredict(); },
    [handlePredict]
  );

  return (
    <Box sx={{ width: "100%", overflowX: "hidden" }}>
      <SectionHeader
        title="Prediction Studio"
        subtitle="Select a material, scan rate, and ML model — run real-time CV curve prediction with full electrochemical analysis and scientific interpretation"
        accent="#00d4ff"
      />

      {/* ── Feature 5: Resume Last Session dialog ───────────────────────── */}
      {resumeOpen && lastSession && (
        <ResumeSessionDialog
          session={lastSession}
          open={resumeOpen}
          isCached={`${lastSession.model}_${lastSession.material}_${lastSession.scanRate}` in results}
          hasReport={recentExps.find((e) => e.id === lastSession.lastExperimentId)?.reportId !== undefined}
          onResume={handleResume}
          onStartNew={() => setResumeOpen(false)}
        />
      )}

      {/* ── Restore-unavailable snackbar ─────────────────────────────────── */}
      <AnimatePresence>
        {restoreMessage && (
          <motion.div
            key="restore-msg"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.22 }}
            style={{ marginBottom: 16 }}
          >
            <Alert
              severity="info"
              onClose={() => setRestoreMessage(null)}
              sx={{
                background: "rgba(0,212,255,0.06)",
                border: "1px solid rgba(0,212,255,0.2)",
                color: "rgba(255,255,255,0.7)",
                fontSize: "0.78rem",
                "& .MuiAlert-icon": { color: "#00d4ff" },
              }}
            >
              {restoreMessage}
            </Alert>
          </motion.div>
        )}
      </AnimatePresence>

      <Grid container spacing={{ xs: 2, md: 3 }}>
        {/* ── Left panel: controls ───────────────────────────────────── */}
        <Grid item xs={12} lg={4}>
          <motion.div variants={staggerContainer} initial="initial" animate="animate">

            {/* Model selector */}
            <motion.div variants={cardVariants}>
              <Card sx={{ mb: 2.5, background: "rgba(15,25,35,0.8)", border: `1px solid ${alpha(modelColor, 0.25)}`, borderTop: `3px solid ${modelColor}`, transition: "border-color 0.3s ease" }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography variant="overline" sx={{ color: modelColor, fontSize: "0.65rem", letterSpacing: "0.1em", mb: 1.5, display: "block" }}>
                    ML Model
                  </Typography>
                  <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                    <Select value={modelId} onChange={(e) => setModelId(e.target.value)} onKeyDown={handleKeyDown} sx={{ fontWeight: 600 }}>
                      {MODELS.map((m) => (
                        <MenuItem key={m.id} value={m.id}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                            <Box sx={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: MODEL_COLORS[m.id], boxShadow: `0 0 6px ${MODEL_COLORS[m.id]}80`, flexShrink: 0 }} />
                            <span>{m.display}</span>
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", ml: "auto", fontSize: "0.65rem" }}>
                              {m.size}
                            </Typography>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  {modelConfig && (
                    <AnimatePresence mode="wait">
                      <motion.div key={modelId} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.72rem", lineHeight: 1.5, display: "block" }}>
                          {modelConfig.description.slice(0, 120)}…
                        </Typography>
                        <Box sx={{ display: "flex", gap: 1, mt: 1.5, flexWrap: "wrap" }}>
                          <Chip icon={<MemoryRounded sx={{ fontSize: "12px !important" }} />} label={modelConfig.size} size="small"
                            sx={{ fontSize: "0.6rem", backgroundColor: alpha(modelColor, 0.1), color: modelColor, border: `1px solid ${alpha(modelColor, 0.25)}`, height: 20 }} />
                          <Chip icon={<TrendingUpRounded sx={{ fontSize: "12px !important" }} />} label={`Deploy ${modelConfig.deployScore}/100`} size="small"
                            sx={{ fontSize: "0.6rem", backgroundColor: "rgba(16,185,129,0.08)", color: "#10b981", border: "1px solid rgba(16,185,129,0.2)", height: 20 }} />
                          <Chip
                            label={`RMSE ${(MODEL_RMSE_UA[modelId] ?? 0).toFixed(1)} µA`} size="small"
                            sx={{ fontSize: "0.6rem", backgroundColor: "rgba(0,212,255,0.07)", color: "#00d4ff", border: "1px solid rgba(0,212,255,0.2)", height: 20 }}
                          />
                          {isLazyModel && (
                            <Chip icon={<TimerRounded sx={{ fontSize: "12px !important" }} />} label="Tier-2 · cold start ~15s" size="small"
                              sx={{ fontSize: "0.6rem", backgroundColor: "rgba(251,191,36,0.08)", color: "#fbbf24", border: "1px solid rgba(251,191,36,0.2)", height: 20 }} />
                          )}
                        </Box>
                      </motion.div>
                    </AnimatePresence>
                  )}
                </CardContent>
              </Card>
            </motion.div>

            {/* Material + Scan rate */}
            <motion.div variants={cardVariants}>
              <Card sx={{ mb: 2.5, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.12)" }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography variant="overline" sx={{ color: "#00d4ff", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 1.5, display: "block" }}>
                    Experiment Parameters
                  </Typography>
                  <FormControl fullWidth size="small" sx={{ mb: 2.5 }}>
                    <InputLabel sx={{ fontSize: "0.82rem" }}>Material</InputLabel>
                    <Select value={materialId} label="Material" onChange={(e) => setMaterialId(e.target.value)} onKeyDown={handleKeyDown}>
                      {MATERIALS.map((m) => (
                        <MenuItem key={m.id} value={m.id}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                            <Box sx={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: m.color, flexShrink: 0 }} />
                            <span>{m.label}</span>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <Box>
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.72rem" }}>Scan Rate</Typography>
                      <TextField
                        value={scanRateText} onChange={handleTextChange} onBlur={handleTextBlur}
                        onKeyDown={(e) => { if (e.key === "Enter") { handleTextBlur(); handlePredict(); } }}
                        size="small" variant="outlined"
                        inputProps={{ style: { fontFamily: "JetBrains Mono, monospace", fontSize: "0.8rem", color: "#00d4ff", fontWeight: 700, padding: "4px 6px", textAlign: "right", width: 44 } }}
                        InputProps={{
                          endAdornment: <InputAdornment position="end"><Typography sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.65rem" }}>mV/s</Typography></InputAdornment>,
                          sx: { "& fieldset": { borderColor: "rgba(0,212,255,0.25)" }, "&:hover fieldset": { borderColor: "rgba(0,212,255,0.5) !important" }, "&.Mui-focused fieldset": { borderColor: "#00d4ff !important" }, backgroundColor: "rgba(0,212,255,0.04)", borderRadius: 1 },
                        }}
                        sx={{ width: 110 }}
                      />
                    </Box>
                    <Slider
                      value={scanRate} onChange={handleSliderChange}
                      min={10} max={100} step={0.5} marks={SCAN_RATE_MARKS}
                      valueLabelDisplay="auto" valueLabelFormat={(v) => `${v} mV/s`}
                      sx={{
                        /* Larger thumb for mobile touch accuracy */
                        "& .MuiSlider-thumb": { width: { xs: 20, md: 16 }, height: { xs: 20, md: 16 } },
                        "& .MuiSlider-track": { height: { xs: 5, md: 4 } },
                        "& .MuiSlider-rail":  { height: { xs: 5, md: 4 } },
                        "& .MuiSlider-valueLabel": { fontSize: "0.65rem", backgroundColor: "rgba(0,212,255,0.15)", border: "1px solid rgba(0,212,255,0.3)", backdropFilter: "blur(4px)" },
                        "& .MuiSlider-markLabel": { fontSize: "0.58rem", color: "rgba(255,255,255,0.3)" },
                      }}
                    />
                    {isSnapped && (
                      <Typography variant="caption" sx={{ color: "rgba(251,191,36,0.8)", fontSize: "0.62rem", display: "block", mt: 0.25 }}>
                        → snapped to {snappedSR} mV/s (nearest training rate)
                      </Typography>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </motion.div>

            {/* Predict button */}
            <motion.div variants={cardVariants}>
              <GlowButton
                variant="contained" color="primary" fullWidth size="large"
                onClick={handlePredict} loading={loading} disabled={loading}
                startIcon={<FlashOnRounded />} glowColor={modelColor}
                sx={{ py: 1.5, fontSize: "0.9rem", background: `linear-gradient(135deg, ${modelColor} 0%, ${alpha(modelColor, 0.7)} 100%)`, color: "#070b14" }}
              >
                Run Prediction
              </GlowButton>
              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", display: "block", textAlign: "center", mt: 1, fontSize: "0.65rem" }}>
                Press Enter or click · predicts 651 CV points
              </Typography>
            </motion.div>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} style={{ marginTop: 16 }}>
                  <Alert severity="error" sx={{ fontSize: "0.8rem" }}>{error}</Alert>
                </motion.div>
              )}
            </AnimatePresence>

            {/* ── Recent prediction history ─────────────────────────── */}
            {historyEntries.length > 0 && (
              <motion.div variants={cardVariants} style={{ marginTop: 20 }}>
                <Card sx={{ background: "rgba(15,25,35,0.7)", border: "1px solid rgba(255,255,255,0.07)" }}>
                  <CardContent sx={{ p: 2 }}>
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.5 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <HistoryRounded sx={{ fontSize: 15, color: "rgba(255,255,255,0.3)" }} />
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.68rem", fontWeight: 600 }}>
                          Session History
                        </Typography>
                      </Box>
                      <Tooltip title="Clear history">
                        <IconButton size="small" onClick={clear} sx={{ p: 0.25 }}>
                          <DeleteOutlineRounded sx={{ fontSize: 14, color: "rgba(255,255,255,0.2)" }} />
                        </IconButton>
                      </Tooltip>
                    </Box>
                    {historyEntries.map((r, i) => {
                      const color = MODEL_COLORS[r.model_name] ?? "#94a3b8";
                      return (
                        <Box
                          key={`${r.model_name}_${r.material_id}_${r.scan_rate_mVs}`}
                          onClick={() => {
                            setModelId(r.model_name);
                            setMaterialId(r.material_id);
                            setScanRate(r.scan_rate_mVs);
                            setScanRateText(String(r.scan_rate_mVs));
                          }}
                          sx={{ display: "flex", alignItems: "center", gap: 1, py: 0.7, px: 1, borderRadius: 1.5, mb: 0.5, cursor: "pointer", background: i === 0 ? alpha(color, 0.07) : "transparent", border: `1px solid ${i === 0 ? alpha(color, 0.15) : "transparent"}`, "&:hover": { background: alpha(color, 0.06) } }}
                        >
                          <Box sx={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: color, flexShrink: 0 }} />
                          <Typography variant="caption" sx={{ color: i === 0 ? color : "rgba(255,255,255,0.45)", fontFamily: "JetBrains Mono", fontSize: "0.65rem", flex: 1 }}>
                            {r.model_name.toUpperCase()}
                          </Typography>
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem" }}>
                            {r.material_id} · {r.scan_rate_mVs} mV/s
                          </Typography>
                        </Box>
                      );
                    })}
                  </CardContent>
                </Card>
              </motion.div>
            )}
            {/* ── Persistent recent experiments (Feature 4) ────────── */}
            <RecentExperimentsPanel
              records={recentExps}
              cachedKeys={cachedKeys}
              onRestore={handleRestoreExperiment}
              onClear={clearExperiments}
            />
          </motion.div>
        </Grid>

        {/* ── Right panel: chart + stats ─────────────────────────────── */}
        <Grid item xs={12} lg={8}>
          <AnimatePresence mode="wait">
            {loading && (
              <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Card sx={{ background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.12)", mb: 2.5 }}>
                  <CardContent sx={{ p: 3 }}>
                    <LoadingOverlay
                      message={`Running ${modelId.toUpperCase()} inference…`}
                      subMessage={isLazyModel ? "Tier-2 model — loading TensorFlow graph on first request (~15s)" : "Predicting 651 CV points across the voltage sweep"}
                    />
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {!loading && latestResult && advMetrics && confidenceTier && (
              <motion.div
                key={`result-${latestResult.model_name}-${latestResult.material_id}-${latestResult.scan_rate_mVs}`}
                initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }}
              >
                {/* ── CV Plot ────────────────────────────────────────── */}
                <Card sx={{ mb: 2.5, background: "rgba(7,11,20,0.9)", border: "1px solid rgba(0,212,255,0.12)" }}>
                  <CardContent sx={{ p: 2 }}>
                    {/* Header row */}
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5, flexWrap: "wrap" }}>
                      <ModelBadge modelId={latestResult.model_name} />
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.72rem" }}>
                        {latestResult.material_id} · {latestResult.scan_rate_mVs} mV/s · {latestResult.n_points} points
                      </Typography>
                      {/* Experimental overlay toggle */}
                      <Tooltip title={showExperimental ? "Hide experimental overlay" : "Overlay real experimental CV data"} arrow placement="top">
                        <Chip
                          icon={experimentalLoading
                            ? <CircularProgress size={10} sx={{ color: "inherit" }} />
                            : <BiotechRounded sx={{ fontSize: "11px !important" }} />}
                          label={showExperimental ? "Exp. ON" : "Show Experimental"}
                          size="small"
                          onClick={handleToggleExperimental}
                          sx={{
                            fontSize: "0.6rem", height: 20, cursor: "pointer",
                            backgroundColor: showExperimental ? "rgba(16,185,129,0.15)" : "rgba(255,255,255,0.05)",
                            color: showExperimental ? "#10b981" : "rgba(255,255,255,0.45)",
                            border: `1px solid ${showExperimental ? "rgba(16,185,129,0.35)" : "rgba(255,255,255,0.12)"}`,
                            "&:hover": { backgroundColor: "rgba(16,185,129,0.1)" },
                          }}
                        />
                      </Tooltip>
                      {/* Confidence chip */}
                      <Box sx={{ ml: "auto" }}>
                        <Tooltip title={confidenceTier.desc} arrow placement="top">
                          <Chip
                            icon={<VerifiedRounded sx={{ fontSize: "11px !important" }} />}
                            label={confidenceTier.label}
                            size="small"
                            sx={{
                              fontSize: "0.6rem",
                              height: 20,
                              backgroundColor: alpha(confidenceTier.color, 0.12),
                              color: confidenceTier.color,
                              border: `1px solid ${alpha(confidenceTier.color, 0.3)}`,
                              cursor: "help",
                            }}
                          />
                        </Tooltip>
                      </Box>
                    </Box>

                    {/* Previous prediction ghost label */}
                    {previousResult &&
                      (previousResult.model_name !== latestResult.model_name ||
                        previousResult.material_id !== latestResult.material_id ||
                        previousResult.scan_rate_mVs !== latestResult.scan_rate_mVs) && (
                        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 1, px: 0.5 }}>
                          <Box sx={{ width: 18, height: 2, borderBottom: "2px dashed rgba(148,163,184,0.5)", flexShrink: 0 }} />
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.62rem" }}>
                            ghost: {previousResult.model_name.toUpperCase()} / {previousResult.material_id} @ {previousResult.scan_rate_mVs} mV/s
                          </Typography>
                        </Box>
                      )}

                    <SingleCVPlot
                      result={latestResult}
                      previousResult={previousResult}
                      showConfidenceBand
                      rmseUa={resultRmseUa}
                      showPeakAnnotations
                      experimentalData={
                        showExperimental && experimentalData
                          ? { potential_V: experimentalData.potential_V, actual_current_uA: experimentalData.actual_current_uA }
                          : null
                      }
                      height={window.innerWidth < 600 ? 300 : 400}
                    />
                  </CardContent>
                </Card>

                {/* ── Δ vs Previous panel ────────────────────────────── */}
                <AnimatePresence>
                  {deltaVsPrev && previousResult && (
                    <motion.div
                      key="delta-panel"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.35 }}
                      style={{ overflow: "hidden", marginBottom: 20 }}
                    >
                      <Card sx={{ background: "rgba(15,25,35,0.75)", border: "1px solid rgba(99,102,241,0.25)" }}>
                        <CardContent sx={{ p: 2 }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
                            <CompareArrowsRounded sx={{ fontSize: 15, color: "#6366f1" }} />
                            <Typography variant="overline" sx={{ color: "#6366f1", fontSize: "0.62rem", letterSpacing: "0.1em" }}>
                              Δ vs Previous Prediction
                            </Typography>
                            <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.6rem", ml: "auto" }}>
                              {previousResult.model_name.toUpperCase()} / {previousResult.material_id} @ {previousResult.scan_rate_mVs} mV/s
                            </Typography>
                          </Box>
                          <Grid container spacing={1.5}>
                            {[
                              { label: "Peak Anodic", delta: deltaVsPrev.anodic, unit: " µA", color: "#10b981" },
                              { label: "Peak Cathodic", delta: deltaVsPrev.cathodic, unit: " µA", color: "#f472b6" },
                              { label: "Integral Area", delta: deltaVsPrev.integral, unit: " µA·V", color: "#a78bfa" },
                              { label: "Current Range", delta: deltaVsPrev.range, unit: " µA", color: "#00d4ff" },
                            ].map((d) => (
                              <Grid item xs={6} sm={3} key={d.label}>
                                <Box sx={{ p: 1.25, borderRadius: 1.5, backgroundColor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.05)" }}>
                                  <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", display: "block", mb: 0.5 }}>{d.label}</Typography>
                                  <DeltaCell delta={d.delta} unit={d.unit} />
                                </Box>
                              </Grid>
                            ))}
                          </Grid>
                          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.18)", fontSize: "0.6rem", mt: 1.25, display: "block" }}>
                            Dashed ghost line on chart shows previous CV curve for direct visual comparison
                          </Typography>
                        </CardContent>
                      </Card>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* ── Basic Statistics ───────────────────────────────── */}
                <Card sx={{ mb: 2.5, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(0,212,255,0.12)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Typography variant="overline" sx={{ color: "#00d4ff", fontSize: "0.65rem", letterSpacing: "0.1em", mb: 2, display: "block" }}>
                      Electrochemical Statistics
                    </Typography>
                    <Grid container spacing={2}>
                      {[
                        { label: "Peak Anodic Current", value: fmtUa(latestResult.statistics.peak_anodic_uA), color: "#10b981", tip: "Maximum positive current (oxidation peak) — higher = greater anodic charge capacity" },
                        { label: "Peak Cathodic Current", value: fmtUa(latestResult.statistics.peak_cathodic_uA), color: "#f472b6", tip: "Maximum negative current (reduction peak) — should mirror anodic peak for ideal capacitive behaviour" },
                        { label: "Current Range", value: fmtUa(latestResult.statistics.current_range_uA), color: "#00d4ff", tip: "Peak-to-peak current span — wider range indicates higher power capability" },
                        { label: "Integral Area", value: `${latestResult.statistics.integral_area.toFixed(1)} µA·V`, color: "#a78bfa", tip: "Area enclosed by CV loop ∝ charge stored (Q). Larger area = higher energy storage capacity" },
                      ].map((stat) => (
                        <Grid item xs={6} sm={3} key={stat.label}>
                          <Box sx={{ p: 1.5, borderRadius: 2, backgroundColor: alpha(stat.color, 0.06), border: `1px solid ${alpha(stat.color, 0.2)}`, textAlign: "center" }}>
                            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.5, mb: 0.5 }}>
                              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.65rem" }}>{stat.label}</Typography>
                              <Tooltip title={stat.tip} arrow>
                                <InfoOutlined sx={{ fontSize: 12, color: "rgba(255,255,255,0.2)", cursor: "help" }} />
                              </Tooltip>
                            </Box>
                            <Typography variant="h6" sx={{ color: stat.color, fontWeight: 700, fontFamily: "JetBrains Mono, monospace", fontSize: "0.95rem" }}>
                              {stat.value}
                            </Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </CardContent>
                </Card>

                {/* ── Advanced Electrochemical Analysis ─────────────── */}
                <Card sx={{ mb: 2.5, background: "rgba(15,25,35,0.8)", border: "1px solid rgba(139,92,246,0.2)" }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                      <ScienceRounded sx={{ fontSize: 16, color: "#a78bfa" }} />
                      <Typography variant="overline" sx={{ color: "#a78bfa", fontSize: "0.65rem", letterSpacing: "0.1em" }}>
                        Advanced Electrochemical Analysis
                      </Typography>
                    </Box>
                    <Grid container spacing={2} sx={{ mb: 2 }}>
                      {[
                        {
                          label: "Symmetry Factor",
                          value: advMetrics.symFactor != null ? advMetrics.symFactor.toFixed(4) : "—",
                          unit: "",
                          color: Math.abs((advMetrics.symFactor ?? 1) - 1) < 0.05 ? "#10b981" : Math.abs((advMetrics.symFactor ?? 1) - 1) < 0.15 ? "#f59e0b" : "#ef4444",
                          tip: "Ratio |I_anodic| / |I_cathodic|. Ideal = 1.000 for a perfectly reversible, symmetric capacitive electrode. Deviation indicates Faradaic asymmetry or diffusion limitations.",
                          badge: advMetrics.symFactor != null
                            ? (Math.abs(advMetrics.symFactor - 1) < 0.05 ? "Excellent" : Math.abs(advMetrics.symFactor - 1) < 0.15 ? "Good" : "Asymmetric")
                            : null,
                        },
                        {
                          label: "Charge Storage Index",
                          value: advMetrics.chargeStorageIndex != null ? advMetrics.chargeStorageIndex.toFixed(1) : "—",
                          unit: "µF",
                          color: "#22d3ee",
                          tip: "Estimated relative capacitance C = ∮I dV / (2·ν·ΔV), CV method — factor of 2 accounts for full-loop integration over both sweep directions. Treat as a relative indicator — absolute value requires electrode mass/area normalisation.",
                          badge: null,
                        },
                        {
                          label: "Energy Storage Proxy",
                          value: advMetrics.energyProxy != null ? advMetrics.energyProxy.toFixed(3) : "—",
                          unit: "µJ",
                          color: "#f59e0b",
                          tip: "Estimated stored energy E = ½ · C · ΔV². Derived from Charge Storage Index and voltage window. Higher scan rates compress this value due to IR drop.",
                          badge: null,
                        },
                        {
                          label: "CV Asymmetry",
                          value: advMetrics.asymmetryPct != null ? `${advMetrics.asymmetryPct.toFixed(1)}%` : "—",
                          unit: "",
                          color: (advMetrics.asymmetryPct ?? 0) < 5 ? "#10b981" : (advMetrics.asymmetryPct ?? 0) < 15 ? "#f59e0b" : "#ef4444",
                          tip: "Percentage deviation from perfect anodic/cathodic symmetry. Low = ideal capacitive behaviour. High = Faradaic contribution or diffusion limitation.",
                          badge: null,
                        },
                        {
                          label: "Peak Separation",
                          value: advMetrics.peakSep != null ? `${advMetrics.peakSep.toFixed(3)} V` : "—",
                          unit: "",
                          color: "#a78bfa",
                          tip: "Potential difference between anodic and cathodic peak positions. Small separation → surface-confined (ideal capacitor). Large separation → diffusion-controlled Faradaic process.",
                          badge: null,
                        },
                        {
                          label: "Voltage Window",
                          value: `${advMetrics.voltageWindow.toFixed(3)} V`,
                          unit: "",
                          color: "#94a3b8",
                          tip: "Total potential range of the CV sweep. Wider window = more energy stored (E ∝ ΔV²) but also greater risk of electrolyte decomposition at extreme potentials.",
                          badge: null,
                        },
                      ].map((m) => (
                        <Grid item xs={6} sm={4} key={m.label}>
                          <Box sx={{ p: 1.5, borderRadius: 2, backgroundColor: alpha(m.color, 0.06), border: `1px solid ${alpha(m.color, 0.18)}`, textAlign: "center" }}>
                            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.5, mb: 0.5 }}>
                              <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.38)", fontSize: "0.63rem" }}>{m.label}</Typography>
                              <Tooltip title={m.tip} arrow placement="top">
                                <InfoOutlined sx={{ fontSize: 11, color: "rgba(255,255,255,0.18)", cursor: "help" }} />
                              </Tooltip>
                            </Box>
                            <Typography variant="h6" sx={{ color: m.color, fontWeight: 700, fontFamily: "JetBrains Mono", fontSize: "0.9rem" }}>
                              {m.value}
                              {m.unit && <Typography component="span" sx={{ fontSize: "0.62rem", ml: 0.3, opacity: 0.7 }}>{m.unit}</Typography>}
                            </Typography>
                            {m.badge && (
                              <Chip label={m.badge} size="small" sx={{ mt: 0.5, fontSize: "0.55rem", height: 16, backgroundColor: alpha(m.color, 0.12), color: m.color }} />
                            )}
                          </Box>
                        </Grid>
                      ))}
                    </Grid>

                    {/* ── Scientific Interpretation Engine ──────────── */}
                    <Box sx={{ p: 2, borderRadius: 2, background: "rgba(0,0,0,0.25)", border: "1px solid rgba(167,139,250,0.12)", mb: 2 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 1.25 }}>
                        <ScienceRounded sx={{ fontSize: 13, color: "#a78bfa" }} />
                        <Typography variant="caption" sx={{ color: "#a78bfa", fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.06em" }}>
                          Scientific Electrochemical Interpretation
                        </Typography>
                        <Chip
                          label={`${latestResult.model_name.toUpperCase()} · ${latestResult.material_id} · ${latestResult.scan_rate_mVs} mV/s`}
                          size="small"
                          sx={{ fontSize: "0.55rem", height: 16, backgroundColor: alpha(MODEL_COLORS[latestResult.model_name] ?? "#a78bfa", 0.1), color: MODEL_COLORS[latestResult.model_name] ?? "#a78bfa", ml: "auto" }}
                        />
                      </Box>
                      <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                        {interpretation.map((para, idx) => (
                          <Typography
                            key={idx}
                            variant="caption"
                            sx={{
                              color: idx === 0 ? "rgba(255,255,255,0.65)" : "rgba(255,255,255,0.42)",
                              fontSize: "0.72rem",
                              lineHeight: 1.7,
                              display: "block",
                              borderLeft: (idx === interpretation.length - 1 && deltaVsPrev) ? "2px solid rgba(99,102,241,0.4)" : "none",
                              pl: (idx === interpretation.length - 1 && deltaVsPrev) ? 1.25 : 0,
                            }}
                          >
                            {para}
                          </Typography>
                        ))}
                      </Box>
                      {/* Key Terms glossary row */}
                      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mt: 1.5, pt: 1.25, borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.6rem", mr: 0.5 }}>Hover terms:</Typography>
                        {[
                          { key: "cv", label: "CV" },
                          { key: "scan_rate", label: "Scan Rate" },
                          { key: "symmetry_factor", label: "Symmetry Factor" },
                          { key: "charge_storage_index", label: "Charge Storage Index" },
                          { key: "faradaic", label: "Faradaic" },
                          { key: "edlc", label: "EDLC" },
                          { key: "integral_area", label: "Integral Area" },
                          { key: "zero_shot", label: "Zero-Shot" },
                        ].map((g) => (
                          <GlossaryTooltip key={g.key} termKey={g.key}>
                            <Typography
                              component="span"
                              variant="caption"
                              sx={{ color: "rgba(0,212,255,0.6)", fontSize: "0.6rem", cursor: "help" }}
                            >
                              {g.label}
                            </Typography>
                          </GlossaryTooltip>
                        ))}
                      </Box>
                    </Box>

                    <Divider sx={{ mb: 2, borderColor: "rgba(255,255,255,0.06)" }} />

                    {/* Export buttons */}
                    <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap", alignItems: "center" }}>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.65rem", mr: 0.5 }}>Export:</Typography>
                      <GlowButton
                        variant="outlined" size="small"
                        startIcon={<TableChartRounded sx={{ fontSize: 14 }} />}
                        onClick={() => exportCSV(latestResult)}
                        glowColor="#10b981"
                        sx={{ fontSize: "0.72rem", py: 0.5, px: 1.5, borderColor: "rgba(16,185,129,0.3)", color: "#10b981", "&:hover": { borderColor: "#10b981" } }}
                      >
                        CSV Data
                      </GlowButton>
                      <GlowButton
                        variant="outlined" size="small"
                        startIcon={<DownloadRounded sx={{ fontSize: 14 }} />}
                        onClick={() => exportSummary(latestResult, advMetrics)}
                        glowColor="#a78bfa"
                        sx={{ fontSize: "0.72rem", py: 0.5, px: 1.5, borderColor: "rgba(139,92,246,0.3)", color: "#a78bfa", "&:hover": { borderColor: "#a78bfa" } }}
                      >
                        Summary TXT
                      </GlowButton>
                      <GlowButton
                        variant="outlined" size="small"
                        startIcon={<AssignmentRounded sx={{ fontSize: 14 }} />}
                        onClick={() => {
                          setReportOpen(true);
                          if (currentExpIdRef.current) {
                            const rptId = `rpt_${Date.now().toString(36)}`;
                            attachReportId(currentExpIdRef.current, rptId);
                          }
                        }}
                        glowColor="#f59e0b"
                        sx={{ fontSize: "0.72rem", py: 0.5, px: 1.5, borderColor: "rgba(245,158,11,0.35)", color: "#f59e0b", "&:hover": { borderColor: "#f59e0b", background: "rgba(245,158,11,0.06)" } }}
                      >
                        Experiment Report
                      </GlowButton>
                      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.62rem" }}>
                        · PNG: use camera icon in chart toolbar
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {!loading && !latestResult && (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Card sx={{ background: "rgba(15,25,35,0.6)", border: "1px dashed rgba(0,212,255,0.15)", borderRadius: 3, minHeight: 480, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Box sx={{ textAlign: "center", p: 4 }}>
                    <Box component={motion.div} animate={{ y: [0, -10, 0], opacity: [0.5, 1, 0.5] }} transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }} sx={{ mb: 2, fontSize: 48 }}>
                      ⚡
                    </Box>
                    <Typography variant="h6" sx={{ color: "rgba(255,255,255,0.5)", fontWeight: 600, mb: 1 }}>Ready for Prediction</Typography>
                    <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.25)", maxWidth: 320, mx: "auto", lineHeight: 1.7 }}>
                      Configure the parameters on the left and click{" "}
                      <span style={{ color: "#00d4ff" }}>Run Prediction</span> to generate a CV curve with full electrochemical analysis
                    </Typography>
                    <Box sx={{ mt: 3, display: "flex", gap: 2, justifyContent: "center", flexWrap: "wrap" }}>
                      {[
                        { model: "lightgbm", mat: "NM1", sr: 30, label: "Quick Try: LightGBM" },
                        { model: "gru", mat: "NM2", sr: 50, label: "Quick Try: GRU" },
                      ].map((p) => (
                        <Chip
                          key={p.label} label={p.label} size="small" clickable
                          onClick={() => {
                            setModelId(p.model); setMaterialId(p.mat); setScanRate(p.sr); setScanRateText(String(p.sr));
                            predict({ model_name: p.model, material_id: p.mat, scan_rate_mVs: p.sr });
                          }}
                          sx={{ fontSize: "0.68rem", backgroundColor: alpha(MODEL_COLORS[p.model], 0.1), color: MODEL_COLORS[p.model], border: `1px solid ${alpha(MODEL_COLORS[p.model], 0.3)}`, cursor: "pointer" }}
                        />
                      ))}
                    </Box>
                  </Box>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </Grid>
      </Grid>

      {/* ── Experiment Report dialog ──────────────────────────────── */}
      {latestResult && advMetrics && confidenceTier && modelConfig && (
        <ExperimentReport
          open={reportOpen}
          onClose={() => setReportOpen(false)}
          result={latestResult}
          advMetrics={advMetrics}
          confidenceTier={confidenceTier}
          interpretation={interpretation}
          modelConfig={modelConfig}
          predictionMs={predTimeMs}
          generatedAt={reportGeneratedAt}
        />
      )}
    </Box>
  );
}
