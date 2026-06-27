import React, { useEffect, useCallback, useMemo } from "react";
import {
  Dialog, DialogContent,
  Box, Typography, Table, TableHead, TableBody, TableRow, TableCell,
  Grid, Chip, Divider, IconButton, Tooltip, alpha, CircularProgress,
} from "@mui/material";
import {
  CloseRounded, PrintRounded, DownloadRounded, ScienceRounded,
  AssignmentRounded, BarChartRounded, MemoryRounded, BiotechRounded,
  InfoOutlined, AutoAwesomeRounded, VerifiedRounded, DescriptionRounded,
  SummarizeRounded, MenuBookRounded, InsightsRounded, InfoRounded,
  FormatListNumberedRounded, CheckCircleRounded,
} from "@mui/icons-material";
import { SingleCVPlot } from "../charts/CVPlot";
import { useLeaderboard } from "../../hooks/useLeaderboard";
import { MODEL_COLORS } from "../../constants/models";
import { fmtUa, fmtMs, rmseColor, r2Color } from "../../utils/formatters";
import type { PredictionResponse, ModelConfig, LeaderboardEntry, ComparisonRow } from "../../types";
import GlowButton from "../common/GlowButton";

// ── Types ──────────────────────────────────────────────────────────────────

export interface ReportAdvMetrics {
  symFactor: number | null;
  chargeStorageIndex: number | null;
  energyProxy: number | null;
  asymmetryPct: number | null;
  peakSep: number | null;
  voltageWindow: number;
}

export interface ReportConfidenceTier {
  label: string;
  color: string;
  desc: string;
}

export interface ExperimentReportProps {
  open: boolean;
  onClose: () => void;
  result: PredictionResponse;
  advMetrics: ReportAdvMetrics;
  confidenceTier: ReportConfidenceTier;
  interpretation: string[];
  modelConfig: ModelConfig;
  predictionMs: number | null;
  generatedAt: Date;
}

// ── Constants ──────────────────────────────────────────────────────────────

const MODEL_RMSE_LOOKUP: Record<string, number> = {
  rf: 26.59,
  lightgbm: 26.86,
  xgboost: 28.83,
  gru: 36.84,
  lstm: 36.51,
  ann: 49.93,
};

const INFERENCE_TIER: Record<string, string> = {
  rf: "Tier-1 (hot — always loaded)",
  lightgbm: "Tier-1 (hot — always loaded)",
  gru: "Tier-1 (hot — always loaded)",
  xgboost: "Tier-2 (lazy — loaded on demand)",
  ann: "Tier-2 (lazy — loaded on demand)",
  lstm: "Tier-2 (lazy — loaded on demand)",
};

const MATERIAL_LABEL: Record<string, string> = {
  NM1: "NM1 — ZnO Baseline",
  NM2: "NM2 — Experimental Sample",
  NM3: "NM3 — Experimental Sample",
  NM4: "NM4 — Experimental Sample (Zero-Shot)",
};

const PARTITION_DISPLAY: Record<string, string> = {
  train: "Training (in-sample)",
  val: "Validation (SR = 30 mV/s)",
  test_SR: "Test-SR (SR = 50 mV/s)",
  test_MAT: "Test-MAT (NM4 — unseen material)",
};

// ── ID generator ───────────────────────────────────────────────────────────

function makeExperimentId(result: PredictionResponse, ts: Date): string {
  const d = ts.toISOString().slice(0, 10).replace(/-/g, "");
  const t = ts.toISOString().slice(11, 19).replace(/:/g, "");
  const m = result.model_name.slice(0, 3).toUpperCase();
  const sr = String(result.scan_rate_mVs).padStart(3, "0");
  return `EXP-${d}${t}-${m}-${result.material_id}-SR${sr}`;
}

// ── Executive Summary (~200 words, fully dynamic) ─────────────────────────

function generateExecutiveSummary(
  result: PredictionResponse,
  adv: ReportAdvMetrics,
  modelConfig: ModelConfig,
  lbEntry: LeaderboardEntry | null,
  confidenceTier: ReportConfidenceTier,
  predictionMs: number | null,
): string {
  const matLabel = MATERIAL_LABEL[result.material_id] ?? result.material_id;
  const rmse = lbEntry?.rmse_val_uA ?? MODEL_RMSE_LOOKUP[result.model_name];
  const r2 = lbEntry?.r2_val ?? null;

  const zeroShotNote = result.material_id === "NM4"
    ? " NM4 was entirely withheld from training, constituting a zero-shot cross-material generalisation test."
    : " The material was present in the training dataset, placing this prediction in the validated interpolation regime.";

  const srNote = result.scan_rate_mVs <= 20
    ? "At this low scan rate, near-equilibrium ion access is achieved, maximising the reliability of capacitance estimates."
    : result.scan_rate_mVs >= 70
    ? "At this high scan rate, IR drop effects compress the apparent CV area; absolute capacitance evaluation requires measurements at ≤20 mV/s."
    : "This intermediate scan rate provides a balance between kinetic and thermodynamic electrochemical information.";

  const symNote = adv.symFactor != null
    ? (Math.abs(adv.symFactor - 1) < 0.1
        ? `The predicted profile exhibits near-ideal capacitive symmetry (symmetry factor = ${adv.symFactor.toFixed(3)}), with peak anodic current of ${result.statistics.peak_anodic_uA.toFixed(2)} µA and peak cathodic current of ${result.statistics.peak_cathodic_uA.toFixed(2)} µA.`
        : `The predicted profile exhibits mixed capacitive–Faradaic behaviour (symmetry factor = ${adv.symFactor.toFixed(3)}), with peak anodic and cathodic currents of ${result.statistics.peak_anodic_uA.toFixed(2)} µA and ${result.statistics.peak_cathodic_uA.toFixed(2)} µA respectively.`)
    : `The predicted profile has peak anodic current of ${result.statistics.peak_anodic_uA.toFixed(2)} µA and peak cathodic current of ${result.statistics.peak_cathodic_uA.toFixed(2)} µA.`;

  const mlNote = rmse != null
    ? `The ${modelConfig.shortName} model (${modelConfig.type}-type) achieves validation RMSE of ${rmse.toFixed(2)} µA` +
      (r2 != null ? ` and R² of ${r2.toFixed(4)}` : "") +
      `, indicating ${rmse < 30 ? "strong" : rmse < 45 ? "moderate" : "acceptable"} predictive accuracy.`
    : `The ${modelConfig.shortName} model produced the prediction with ${confidenceTier.label} confidence.`;

  const confNote = confidenceTier.label === "Cross-Material Transfer"
    ? "Predictions should be treated as indicative trends pending physical experimental validation."
    : "Results fall within the validated operational envelope and are suitable for preliminary electrochemical screening and comparative material analysis.";

  return (
    `This report presents an AI-predicted cyclic voltammetry analysis for ${matLabel} at a scan rate of ` +
    `${result.scan_rate_mVs} mV/s, generated by the ZnO Supercapacitor AI Platform using the ` +
    `${modelConfig.display} model.${zeroShotNote} A total of ${result.n_points} prediction points were generated ` +
    `spanning the complete voltage sweep window of ${adv.voltageWindow.toFixed(3)} V, reconstructing the full ` +
    `CV loop for quantitative electrochemical analysis. ${srNote} ${symNote} ` +
    `The enclosed CV integral area of ${result.statistics.integral_area.toFixed(2)} µA·V provides a relative ` +
    `measure of charge stored per cycle` +
    (adv.chargeStorageIndex != null ? `, with a Charge Storage Index of ${adv.chargeStorageIndex.toFixed(1)} µF` : "") +
    `. ${mlNote} ${confNote}` +
    (predictionMs != null ? ` Total prediction time: ${predictionMs} ms.` : "")
  );
}

// ── Scientific Background (dynamic based on model type and scan rate) ──────

function generateScientificBackground(result: PredictionResponse, modelConfig: ModelConfig): string {
  const srContext = result.scan_rate_mVs <= 20
    ? "low scan rates (≤20 mV/s) provide near-equilibrium ion access, enabling maximum charge storage utilisation and reliable capacitance determination"
    : result.scan_rate_mVs >= 70
    ? "high scan rates (≥70 mV/s) reveal kinetic limitations including ion diffusion bottlenecks and resistive IR drop effects that compress the apparent CV area"
    : "intermediate scan rates balance kinetic and thermodynamic contributions, providing information on both capacitive storage and rate-dependent polarisation";

  const modelContext = modelConfig.type === "deep"
    ? `Deep learning architectures such as ${modelConfig.display} exploit the sequential nature of the voltage sweep to capture directional hysteresis and peak asymmetry patterns. The model processes the 651-point CV sweep as a time-series sequence, enabling it to learn the directional current response that distinguishes the anodic forward sweep from the cathodic reverse sweep.`
    : `Gradient-boosted and ensemble tree models such as ${modelConfig.display} extract physics-informed CV features — including scan rate, normalised potential position within the voltage window, and sweep direction — to reconstruct the current response without explicit sequential modelling. This feature engineering approach encodes the electrochemical physics into tabular inputs, enabling efficient cross-material generalisation.`;

  return (
    `Zinc oxide (ZnO) is a wide-bandgap transition metal oxide semiconductor that exhibits pseudocapacitive ` +
    `charge storage when employed as a thin-film electrode in aqueous electrolyte systems. Energy is stored ` +
    `through a combination of electrochemical double-layer capacitance (EDLC) at the electrode–electrolyte ` +
    `interface and reversible surface-confined Faradaic reactions involving Zn²⁺/OH⁻ species. Cyclic ` +
    `voltammetry (CV) is the primary characterisation technique, sweeping the applied potential between two ` +
    `vertex values and recording the resulting current response. The shape, area, and peak positions of the ` +
    `CV loop quantify capacitance, reversibility, and charge transfer kinetics. In this dataset, measurements ` +
    `were performed in 1 M Na₂SO₄ aqueous electrolyte across four ZnO nanocomposite material groups (NM1–NM4) ` +
    `at scan rates of 10–100 mV/s. In general, ${srContext}. Machine learning models trained on ` +
    `physics-informed features derived from the voltage sweep enable rapid prediction of the complete CV curve ` +
    `for any material–scan-rate combination within (and beyond) the training distribution, substantially ` +
    `reducing the need for time-consuming physical measurements during preliminary material screening. ` +
    `${modelContext}`
  );
}

// ── Result Interpretation (adaptive numbered observations) ─────────────────

function generateResultInterpretation(
  result: PredictionResponse,
  adv: ReportAdvMetrics,
  lbEntry: LeaderboardEntry | null,
  confidenceTier: ReportConfidenceTier,
): string[] {
  const observations: string[] = [];
  const rmse = lbEntry?.rmse_val_uA ?? MODEL_RMSE_LOOKUP[result.model_name];
  const r2 = lbEntry?.r2_val;
  const r2Mat = lbEntry?.r2_test_mat;

  if (rmse != null) {
    if (rmse < 30) {
      observations.push(
        `Validation RMSE of ${rmse.toFixed(2)} µA is in the low-error regime (< 30 µA), indicating close agreement between predicted and experimental CV profiles. Quantitative peak current values are reliable for preliminary capacitance estimation.`
      );
    } else if (rmse < 45) {
      observations.push(
        `Validation RMSE of ${rmse.toFixed(2)} µA indicates moderate predictive accuracy. Qualitative CV shape and peak positions are reliable; absolute peak current magnitudes carry an uncertainty of approximately ±${(rmse * 1.2).toFixed(0)} µA.`
      );
    } else {
      observations.push(
        `Validation RMSE of ${rmse.toFixed(2)} µA reflects higher uncertainty, typical of models lacking sequential voltage sweep modelling. Trend analysis and relative comparisons remain valid, but absolute values should be treated as approximate.`
      );
    }
  }

  if (r2 != null) {
    if (r2 > 0.97) {
      observations.push(
        `Validation R² of ${r2.toFixed(4)} demonstrates strong predictive capability — over ${(r2 * 100).toFixed(1)}% of CV current variance is explained by the model, confirming that physics-informed feature engineering effectively encodes the electrochemical relationships.`
      );
    } else if (r2 > 0.92) {
      observations.push(
        `Validation R² of ${r2.toFixed(4)} indicates good predictive performance. The model captures dominant CV characteristics including peak positions and loop shape; residual variance is attributable to cycle-to-cycle experimental noise.`
      );
    }
  }

  if (r2Mat != null) {
    const quality = r2Mat > 0.97 ? "exceptional" : r2Mat > 0.95 ? "strong" : "moderate";
    observations.push(
      `Cross-material R² (test-MAT partition) of ${r2Mat.toFixed(4)} demonstrates ${quality} generalisation to unseen material group NM4, confirming that electrochemical feature representations transfer effectively across ZnO nanocomposite variants.`
    );
  }

  if (adv.symFactor != null) {
    const sf = adv.symFactor;
    if (Math.abs(sf - 1) < 0.05) {
      observations.push(
        `Symmetry factor of ${sf.toFixed(4)} (≈1.0) confirms ideal EDLC-like capacitive behaviour with minimal Faradaic contribution. Anodic and cathodic charge integrals are nearly equal, indicating highly reversible charge storage.`
      );
    } else if (Math.abs(sf - 1) < 0.15) {
      observations.push(
        `Symmetry factor of ${sf.toFixed(4)} (within ±0.15 of unity) indicates predominantly capacitive charge storage with a minor Faradaic component. Slight asymmetry may reflect limited diffusion-controlled processes at ${result.scan_rate_mVs} mV/s.`
      );
    } else {
      observations.push(
        `Symmetry factor of ${sf.toFixed(4)} deviates from unity, indicating mixed capacitive–Faradaic charge storage. The anodic half-cycle dominates in charge integration, suggesting more pronounced oxidation peaks relative to reduction counterparts.`
      );
    }
  }

  if (adv.chargeStorageIndex != null) {
    const csi = adv.chargeStorageIndex;
    if (csi > 200) {
      observations.push(
        `Charge Storage Index of ${csi.toFixed(1)} µF indicates high relative charge accumulation, exceeding typical ZnO baseline values. This suggests enhanced surface area or composite effects contributing to pseudocapacitive enhancement.`
      );
    } else if (csi > 80) {
      observations.push(
        `Charge Storage Index of ${csi.toFixed(1)} µF is consistent with moderate charge storage typical of ZnO nanostructures in aqueous Na₂SO₄ electrolyte at this scan rate.`
      );
    } else {
      observations.push(
        `Charge Storage Index of ${csi.toFixed(1)} µF is in the lower range, likely due to kinetic limitations at ${result.scan_rate_mVs} mV/s. Evaluation at ≤20 mV/s is recommended for accurate specific capacitance determination.`
      );
    }
  }

  if (adv.peakSep != null) {
    const ps = adv.peakSep;
    observations.push(
      ps < 0.05
        ? `Peak separation of ${ps.toFixed(3)} V is minimal, consistent with surface-confined, highly reversible charge transfer and quasi-ideal EDLC behaviour.`
        : `Peak separation of ${ps.toFixed(3)} V suggests a diffusion-controlled Faradaic contribution alongside capacitive storage, typical of pseudocapacitive ZnO electrodes.`
    );
  }

  observations.push(
    confidenceTier.label === "Cross-Material Transfer"
      ? `Prediction confidence tier: '${confidenceTier.label}' — ${confidenceTier.desc} Physical experimental validation is strongly recommended before using these values for device design.`
      : `Prediction confidence tier: '${confidenceTier.label}' — ${confidenceTier.desc} These predictions are suitable for preliminary screening and comparative material analysis.`
  );

  return observations;
}

// ── Experiment Objective ───────────────────────────────────────────────────

function generateObjective(result: PredictionResponse, modelConfig: ModelConfig): string {
  const matMap: Record<string, string> = {
    NM1: "the ZnO baseline material group (NM1)",
    NM2: "experimental material group NM2",
    NM3: "experimental material group NM3",
    NM4: "experimental zero-shot material group NM4 — a material entirely excluded from the training dataset",
  };
  const matDesc = matMap[result.material_id] ?? `material group ${result.material_id}`;
  const zeroShotNote = result.material_id === "NM4"
    ? " This constitutes a zero-shot cross-material generalisation test: NM4 was withheld from all training, validation, and hyperparameter-tuning splits, making this prediction a direct assessment of the model’s ability to transfer electrochemical knowledge learned from NM1–NM3."
    : "";
  return (
    `The objective of this experiment is to predict the cyclic voltammetry (CV) behaviour of ` +
    `${matDesc} at a scan rate of ${result.scan_rate_mVs} mV/s using the ` +
    `${modelConfig.display} (${modelConfig.shortName}) machine learning model, ` +
    `and to analyse its electrochemical charge storage characteristics through the ` +
    `ZnO Supercapacitor AI Platform.${zeroShotNote} ` +
    `The prediction generates ${result.n_points} data points spanning the full voltage sweep window, ` +
    `enabling reconstruction of the complete CV loop for quantitative electrochemical analysis.`
  );
}

// ── Scientific Inference ───────────────────────────────────────────────────

function generateScientificInference(
  result: PredictionResponse,
  adv: ReportAdvMetrics,
  lbEntry: LeaderboardEntry | null,
): string {
  const { peak_anodic_uA, integral_area } = result.statistics;
  const { symFactor, chargeStorageIndex, peakSep } = adv;

  const shapeDesc = symFactor != null && Math.abs(symFactor - 1) < 0.1
    ? "a predominantly capacitive, near-symmetric CV profile"
    : "a CV profile exhibiting mixed capacitive–Faradaic charge storage";

  const peakDesc = peakSep != null
    ? (peakSep < 0.05
        ? `The minimal anodic–cathodic peak separation (${peakSep.toFixed(3)} V) is indicative of surface-confined, highly reversible charge transfer.`
        : `The anodic–cathodic peak separation of ${peakSep.toFixed(3)} V suggests moderate diffusion-controlled Faradaic contribution alongside capacitive storage.`)
    : "";

  const csDesc = chargeStorageIndex != null
    ? (chargeStorageIndex > 200
        ? "The enclosed CV integral area indicates high relative charge storage capacity, suggestive of pseudocapacitive enhancement beyond pure EDLC behaviour."
        : chargeStorageIndex > 80
        ? "The enclosed CV integral area is consistent with moderate charge storage, typical of ZnO nanostructures in aqueous Na₂SO₄ electrolyte."
        : "CV area compression at this scan rate limits apparent charge storage; true specific capacitance should be evaluated at lower scan rates (≤20 mV/s).")
    : `The integral area of ${integral_area.toFixed(1)} µA·V provides a relative measure of the charge stored per CV cycle.`;

  const rmse = lbEntry?.rmse_val_uA ?? MODEL_RMSE_LOOKUP[result.model_name];
  const r2 = lbEntry?.r2_test_mat;
  const mlDesc = r2 != null
    ? `The ${result.model_name.toUpperCase()} model demonstrates strong predictive performance with validation RMSE of ${rmse?.toFixed(2) ?? "—"} µA and cross-material R² of ${r2.toFixed(4)}, confirming reliable CV prediction across experimental conditions.`
    : `The ${result.model_name.toUpperCase()} model demonstrates validated predictive performance with RMSE of ${rmse?.toFixed(2) ?? "—"} µA.`;

  return (
    `The predicted cyclic voltammogram for ${result.material_id} at ${result.scan_rate_mVs} mV/s exhibits ` +
    `${shapeDesc}, characterised by an anodic peak current of ${peak_anodic_uA.toFixed(2)} µA ` +
    `and a symmetry factor of ${symFactor?.toFixed(3) ?? "N/A"}. ` +
    `${peakDesc} ${csDesc} ${mlDesc}`
  );
}

// ── Experimental Conclusion ────────────────────────────────────────────────

function generateConclusion(
  result: PredictionResponse,
  adv: ReportAdvMetrics,
  modelConfig: ModelConfig,
  lbEntry: LeaderboardEntry | null,
  confidenceTier: ReportConfidenceTier,
): string {
  const { chargeStorageIndex, symFactor, peakSep } = adv;
  const isZeroShot = result.material_id === "NM4";
  const isHighSR = result.scan_rate_mVs >= 70;
  const isLowSR = result.scan_rate_mVs <= 20;
  const r2Mat = lbEntry?.r2_test_mat;
  const rmseVal = lbEntry?.rmse_val_uA ?? MODEL_RMSE_LOOKUP[result.model_name];

  const p1 = `This experiment employed the ${modelConfig.display} (${modelConfig.shortName}) model to predict the cyclic voltammogram of material group ${result.material_id} at a scan rate of ${result.scan_rate_mVs} mV/s, generating ${result.n_points} data points across the complete voltage sweep window.`;

  const p2 = isZeroShot
    ? `Material group NM4 was entirely withheld from the training dataset, making this a zero-shot cross-material generalisation test. The model predicts CV behaviour based solely on electrochemical patterns learned from NM1–NM3. Cross-material R² of ${r2Mat != null ? r2Mat.toFixed(4) : "—"} on the NM4 test partition demonstrates the robustness of the physics-informed feature engineering.`
    : `Material group ${result.material_id} was present in the training data${result.material_id === "NM1" ? " as the ZnO baseline reference" : ""}, placing this prediction in the interpolation regime where model performance is thoroughly validated. Validation RMSE of ${rmseVal?.toFixed(2) ?? "—"} µA confirms strong predictive accuracy within the training distribution.`;

  const symNote = symFactor != null
    ? ` Symmetry factor ${symFactor.toFixed(3)} ${Math.abs(symFactor - 1) < 0.08 ? "confirms predominantly ideal capacitive behaviour" : "indicates a mixed capacitive–Faradaic mechanism"}.`
    : "";
  const peakNote = peakSep != null
    ? ` Peak separation of ${peakSep.toFixed(3)} V ${peakSep < 0.05 ? "is consistent with surface-confined reversible charge transfer" : "suggests diffusion-controlled kinetics"}.`
    : "";
  const csNote = chargeStorageIndex != null
    ? ` The estimated Charge Storage Index of ${chargeStorageIndex.toFixed(1)} µF places this electrode in the ${chargeStorageIndex > 200 ? "high" : chargeStorageIndex > 80 ? "moderate" : "lower"} range for ZnO-based supercapacitor materials.`
    : "";
  const srNote = isHighSR
    ? ` At ${result.scan_rate_mVs} mV/s (high scan rate), IR drop compresses the CV area; absolute capacitance evaluation requires measurements at ≤20 mV/s.`
    : isLowSR
    ? ` The low scan rate of ${result.scan_rate_mVs} mV/s provides near-equilibrium ion access, maximising the reliability of capacitance estimates.`
    : "";
  const p3 = `The predicted CV profile provides a reliable basis for electrochemical assessment.${symNote}${peakNote}${csNote}${srNote}`;

  const confNote = confidenceTier.label === "Cross-Material Transfer"
    ? "Predictions should be treated as indicative trends pending physical experimental validation of this unseen material."
    : confidenceTier.label.includes("High Accuracy")
    ? "The high-accuracy interpolation regime provides strong confidence in the quantitative validity of these predictions."
    : "Predictions fall within the validated operational envelope; quantitative trend analysis is supported.";
  const p4 = `Model deployment footprint of ${modelConfig.size} ensures suitability for resource-constrained environments. ${confNote} Further experimental measurement is recommended for electrode-specific absolute capacitance normalisation and electrolyte characterisation.`;

  return [p1, p2, p3, p4].join("\n\n");
}

// ── SVG chart for static export ────────────────────────────────────────────

function buildSVGChart(
  potential_V: number[],
  current_uA: number[],
  modelColor: string,
): string {
  if (potential_V.length === 0) return "";
  const step = Math.max(1, Math.ceil(potential_V.length / 220));
  const pV = potential_V.filter((_, i) => i % step === 0);
  const cI = current_uA.filter((_, i) => i % step === 0);

  const W = 640, H = 280;
  const pad = { t: 22, r: 18, b: 48, l: 62 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  const xMin = Math.min(...pV), xMax = Math.max(...pV);
  const yMin = Math.min(...cI), yMax = Math.max(...cI);
  const xR = xMax - xMin || 1, yR = yMax - yMin || 1;

  const sx = (v: number) => pad.l + ((v - xMin) / xR) * pw;
  const sy = (i: number) => pad.t + ((yMax - i) / yR) * ph;

  const d = `M ${pV.map((v, i) => `${sx(v).toFixed(1)},${sy(cI[i]).toFixed(1)}`).join(" L ")}`;
  const z = sy(0).toFixed(1);

  const xT = [0, 1, 2, 3, 4].map(i => xMin + (xR * i) / 4);
  const yT = [0, 1, 2, 3, 4].map(i => yMin + (yR * i) / 4);

  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
  <rect width="${W}" height="${H}" fill="#f8f9fa" rx="3"/>
  ${yT.map(v => `<line x1="${pad.l}" y1="${sy(v).toFixed(1)}" x2="${pad.l + pw}" y2="${sy(v).toFixed(1)}" stroke="#e0e0e0" stroke-width="0.7"/>`).join("\n  ")}
  ${xT.map(v => `<line x1="${sx(v).toFixed(1)}" y1="${pad.t}" x2="${sx(v).toFixed(1)}" y2="${pad.t + ph}" stroke="#e0e0e0" stroke-width="0.7"/>`).join("\n  ")}
  <line x1="${pad.l}" y1="${z}" x2="${pad.l + pw}" y2="${z}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>
  <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + ph}" stroke="#333" stroke-width="1.5"/>
  <line x1="${pad.l}" y1="${pad.t + ph}" x2="${pad.l + pw}" y2="${pad.t + ph}" stroke="#333" stroke-width="1.5"/>
  <path d="${d}" fill="none" stroke="${modelColor}" stroke-width="1.8" stroke-linejoin="round"/>
  ${yT.map(v => `<text x="${pad.l - 5}" y="${sy(v).toFixed(1)}" text-anchor="end" dominant-baseline="middle" font-family="monospace" font-size="8.5" fill="#555">${v.toFixed(0)}</text>`).join("\n  ")}
  ${xT.map(v => `<text x="${sx(v).toFixed(1)}" y="${pad.t + ph + 15}" text-anchor="middle" font-family="monospace" font-size="8.5" fill="#555">${v.toFixed(2)}</text>`).join("\n  ")}
  <text x="${pad.l + pw / 2}" y="${H - 5}" text-anchor="middle" font-family="Georgia,serif" font-size="10" fill="#333">Potential (V)</text>
  <text x="12" y="${pad.t + ph / 2}" text-anchor="middle" font-family="Georgia,serif" font-size="10" fill="#333" transform="rotate(-90,12,${pad.t + ph / 2})">Current (µA)</text>
</svg>`;
}

// ── Print / Download HTML (all 18 sections) ────────────────────────────────

function buildPrintHTML(
  result: PredictionResponse,
  adv: ReportAdvMetrics,
  confidenceTier: ReportConfidenceTier,
  interpretation: string[],
  modelConfig: ModelConfig,
  lbEntry: LeaderboardEntry | null,
  comparison: ComparisonRow[],
  predictionMs: number | null,
  generatedAt: Date,
  experimentId: string,
): string {
  const modelColor = MODEL_COLORS[result.model_name] ?? "#00d4ff";
  const maxI = Math.max(...result.predicted_current_uA);
  const minI = Math.min(...result.predicted_current_uA);
  const avgI = result.predicted_current_uA.reduce((a, b) => a + b, 0) / result.predicted_current_uA.length;
  const potMin = Math.min(...result.potential_V);
  const potMax = Math.max(...result.potential_V);

  const modelRows = comparison
    .filter(r => r.model_id === result.model_name)
    .sort((a, b) => {
      const order = ["train", "val", "test_SR", "test_MAT"];
      return order.indexOf(a.partition) - order.indexOf(b.partition);
    });

  const svgChart = buildSVGChart(result.potential_V, result.predicted_current_uA, modelColor);
  const execSummary = generateExecutiveSummary(result, adv, modelConfig, lbEntry, confidenceTier, predictionMs);
  const objectiveText = generateObjective(result, modelConfig);
  const sciBg = generateScientificBackground(result, modelConfig);
  const inferenceText = generateScientificInference(result, adv, lbEntry);
  const conclusionText = generateConclusion(result, adv, modelConfig, lbEntry, confidenceTier);
  const resultObservations = generateResultInterpretation(result, adv, lbEntry, confidenceTier);

  const mlNarrative = (() => {
    const rmse = lbEntry?.rmse_val_uA ?? MODEL_RMSE_LOOKUP[result.model_name];
    const r2 = lbEntry?.r2_test_mat;
    const typeDesc = modelConfig.type === "deep"
      ? "a deep learning recurrent architecture capable of capturing sequential voltage sweep patterns"
      : "a gradient-boosted ensemble that models the CV feature space without sequential inductive bias";
    return (
      `<p>The ${modelConfig.display} was selected as ${typeDesc}. ${modelConfig.description}</p>` +
      `<p>Validation RMSE: ${rmse?.toFixed(2) ?? "—"} µA on held-out SR=30 mV/s data` +
      (r2 != null ? `. Cross-material R²: ${r2.toFixed(4)} on fully-withheld NM4 partition.` : ".") +
      ` Strengths: ${modelConfig.strengths.join("; ")}. Limitations: ${modelConfig.weaknesses.join("; ")}.</p>` +
      `<p>Architecture: ${modelConfig.architecture}. Size: ${modelConfig.size}. Deploy score: ${modelConfig.deployScore}/100.</p>`
    );
  })();

  const mlRows = modelRows.length > 0
    ? modelRows.map(r => `<tr><td>${PARTITION_DISPLAY[r.partition] ?? r.partition}</td><td class="mono">${r.rmse_uA.toFixed(2)} µA</td><td class="mono">${r.r2.toFixed(4)}</td><td class="mono">${r.mae_uA != null ? r.mae_uA.toFixed(2) + " µA" : "—"}</td></tr>`).join("")
    : lbEntry ? [
        { p: "val", rmse: lbEntry.rmse_val_uA, r2: lbEntry.r2_val },
        { p: "test_SR", rmse: lbEntry.rmse_test_sr_uA, r2: lbEntry.r2_test_sr },
        { p: "test_MAT", rmse: lbEntry.rmse_test_mat_uA, r2: lbEntry.r2_test_mat },
      ].map(r => `<tr><td>${PARTITION_DISPLAY[r.p]}</td><td class="mono">${r.rmse.toFixed(2)} µA</td><td class="mono">${r.r2.toFixed(4)}</td><td class="mono">—</td></tr>`).join("")
    : `<tr><td colspan="4" style="text-align:center;color:#888;">Metrics unavailable</td></tr>`;

  const configRows = [
    ["Experiment ID", experimentId],
    ["Material Group", MATERIAL_LABEL[result.material_id] ?? result.material_id],
    ["Scan Rate", `${result.scan_rate_mVs} mV/s`],
    ["Potential Range", `${potMin.toFixed(3)} V to ${potMax.toFixed(3)} V`],
    ["Voltage Window", `${adv.voltageWindow.toFixed(3)} V`],
    ["Prediction Model", `${modelConfig.display} (${modelConfig.shortName})`],
    ["Dataset Used", "ZnO Nanocomposite CV Dataset — NM1–NM4"],
    ["Training Scenario", result.material_id === "NM4" ? "Cross-material extrapolation (zero-shot)" : "Interpolation — within training distribution"],
    ["Material Status", result.material_id === "NM4" ? "Zero-shot — excluded from training" : "Known material — present in training splits"],
    ["Prediction Time", predictionMs != null ? `${predictionMs} ms` : "—"],
    ["Benchmark Inference", lbEntry?.inference_ms != null ? `${lbEntry.inference_ms.toFixed(1)} ms` : "—"],
    ["Points Generated", `${result.n_points} (full CV sweep)`],
  ];

  const inputRows = [
    ["Material Group", MATERIAL_LABEL[result.material_id] ?? result.material_id],
    ["Scan Rate", `${result.scan_rate_mVs} mV/s`],
    ["Potential Window", `${potMin.toFixed(3)} V to ${potMax.toFixed(3)} V`],
    ["Model", `${modelConfig.display} (${modelConfig.shortName})`],
    ["Dataset", "ZnO Nanocomposite Supercapacitor CV Dataset"],
    ["Feature Set", "Physics-informed (scan_rate, norm_potential, sweep_direction, ...)"],
    ["Prediction Type", result.material_id === "NM4" ? "Cross-material zero-shot" : "Interpolation"],
    ["Electrolyte", "1 M Na₂SO₄ (aqueous)"],
    ["Electrode Type", "ZnO nanocomposite thin film"],
    ["Inference Tier", INFERENCE_TIER[result.model_name] ?? "Tier-1"],
  ];

  const resultMetricRows = [
    ["Maximum Current", `${maxI.toFixed(4)} µA`],
    ["Minimum Current", `${minI.toFixed(4)} µA`],
    ["Average Current", `${avgI.toFixed(4)} µA`],
    ["Peak Anodic Current", `${result.statistics.peak_anodic_uA.toFixed(4)} µA`],
    ["Peak Cathodic Current", `${result.statistics.peak_cathodic_uA.toFixed(4)} µA`],
    ["Current Range", `${result.statistics.current_range_uA.toFixed(4)} µA`],
    ["Integral Area (∮I dV)", `${result.statistics.integral_area.toFixed(4)} µA·V`],
    ["Charge Storage Index", adv.chargeStorageIndex != null ? `${adv.chargeStorageIndex.toFixed(2)} µF` : "—"],
    ["Peak Separation", adv.peakSep != null ? `${adv.peakSep.toFixed(4)} V` : "—"],
    ["Symmetry Factor", adv.symFactor != null ? adv.symFactor.toFixed(5) : "—"],
    ["Energy Storage Proxy", adv.energyProxy != null ? `${adv.energyProxy.toFixed(5)} µJ` : "—"],
    ["CV Asymmetry", adv.asymmetryPct != null ? `${adv.asymmetryPct.toFixed(2)} %` : "—"],
  ];

  const figCaptions = [
    ["Table 1", `Experiment Configuration (§04) — Specification of ${experimentId} including material, scan rate, model, dataset, and timing parameters.`],
    ["Table 2", `Input Parameters (§06) — Full input spec for ${result.material_id} at ${result.scan_rate_mVs} mV/s including electrode type and electrolyte.`],
    ["Figure 1", `Predicted CV Curve (§07) — Cyclic voltammogram for ${result.material_id} at ${result.scan_rate_mVs} mV/s using ${modelConfig.display} (${result.n_points} data points, ${potMin.toFixed(3)} V to ${potMax.toFixed(3)} V).`],
    ["Table 3", `Electrochemical Results (§08) — Quantitative metrics from the predicted CV: peak currents, integral area, symmetry factor, charge storage index.`],
    ["Table 4", `ML Performance (§10) — Validation, test-SR, and test-MAT RMSE, R², and MAE for ${modelConfig.display}.`],
    ["Table 5", `Report Metadata (§17) — Platform version, model architecture, dataset version, and generation timestamp for ${experimentId}.`],
  ];

  const css = `
    * { box-sizing: border-box; margin: 0; padding: 0; }
    @page { margin: 1.8cm 2cm; size: A4; }
    body { font-family: Georgia, 'Times New Roman', serif; color: #1a1a2e; background: #fff; font-size: 10.5pt; line-height: 1.65; }
    .cover { text-align: center; padding: 5rem 1rem 4rem; border-bottom: 3px double #1a1a2e; margin-bottom: 2.5rem; page-break-after: always; }
    .cover-platform { font-size: 11pt; color: #555; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.8rem; }
    .cover-title { font-size: 21pt; font-weight: bold; color: #1a1a2e; letter-spacing: 0.02em; line-height: 1.3; margin-bottom: 0.5rem; }
    .cover-subtitle { font-size: 11pt; color: #666; font-style: italic; margin-bottom: 2rem; }
    .cover-rule { border: none; border-top: 1.5px solid #1a1a2e; width: 40%; margin: 1.5rem auto; }
    .cover-id { font-family: 'Courier New', monospace; font-size: 10pt; font-weight: bold; border: 1px solid #bbb; display: inline-block; padding: 5px 16px; border-radius: 3px; letter-spacing: 0.05em; margin-bottom: 2rem; }
    .cover-meta { display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin-top: 1.5rem; }
    .cover-item-label { color: #888; text-transform: uppercase; letter-spacing: 0.07em; font-size: 7.5pt; }
    .cover-item-value { font-family: 'Courier New', monospace; font-weight: bold; color: #1a1a2e; font-size: 10pt; }
    .cover-generated { font-size: 8.5pt; color: #888; margin-top: 2.5rem; font-style: italic; }
    .sec { margin-bottom: 2.2rem; page-break-inside: avoid; }
    .sec-hdr { display: flex; align-items: center; gap: 0.6rem; border-bottom: 1.5px solid #1a1a2e; padding-bottom: 0.35rem; margin-bottom: 0.9rem; }
    .sec-num { font-family: 'Courier New', monospace; font-size: 8pt; font-weight: bold; background: #1a1a2e; color: #fff; padding: 2px 7px; border-radius: 3px; letter-spacing: 0.06em; }
    .sec-title { font-size: 11.5pt; font-weight: bold; text-transform: uppercase; letter-spacing: 0.09em; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem 2.5rem; }
    .kv { display: flex; justify-content: space-between; padding: 0.28rem 0; border-bottom: 1px dotted #d5d5d5; }
    .kv-l { color: #555; font-size: 9pt; }
    .kv-v { font-family: 'Courier New', monospace; font-size: 9.5pt; font-weight: bold; color: #1a1a2e; text-align: right; }
    table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 9pt; }
    th { background: #f0f0f0; font-weight: bold; text-align: left; padding: 0.4rem 0.65rem; border: 1px solid #bbb; font-size: 8.5pt; text-transform: uppercase; letter-spacing: 0.05em; }
    td { padding: 0.38rem 0.65rem; border: 1px solid #ddd; }
    tr:nth-child(even) td { background: #fafafa; }
    .mono { font-family: 'Courier New', monospace; font-weight: bold; }
    .mgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-top: 0.5rem; }
    .mc { border: 1px solid #d5d5d5; padding: 0.5rem 0.65rem; border-radius: 4px; }
    .mc-l { font-size: 7.5pt; color: #666; text-transform: uppercase; letter-spacing: 0.07em; }
    .mc-v { font-family: 'Courier New', monospace; font-size: 10pt; font-weight: bold; color: #1a1a2e; margin-top: 0.1rem; }
    p { margin-bottom: 0.8rem; text-align: justify; }
    .exec-box { background: #f7f7f7; border-left: 4px solid #1a1a2e; padding: 1rem 1.2rem; border-radius: 0 4px 4px 0; font-style: italic; line-height: 1.8; }
    .sci-box { background: #f9f9f9; border: 1px solid #e5e5e5; padding: 1rem 1.2rem; border-radius: 4px; line-height: 1.8; }
    .interp { background: #fafbff; border-left: 3px solid #1a1a2e; padding: 0.7rem 0.9rem; margin-bottom: 0.55rem; border-radius: 0 4px 4px 0; }
    .interp p { margin: 0; font-size: 9.5pt; color: #222; }
    .obs-list { padding: 0; list-style: none; }
    .obs-item { display: flex; gap: 0.6rem; padding: 0.55rem 0.8rem; margin-bottom: 0.45rem; border-left: 3px solid #555; background: #f8f8f8; border-radius: 0 3px 3px 0; font-size: 9.5pt; }
    .obs-num { font-family: 'Courier New', monospace; font-weight: bold; color: #1a1a2e; flex-shrink: 0; min-width: 1.8rem; }
    .chart-wrap { text-align: center; margin: 1rem 0; }
    .chart-cap { font-size: 8.5pt; color: #555; font-style: italic; margin-top: 0.5rem; text-align: center; }
    .notes-box { background: #fffbf0; border: 1.5px solid #d4a017; padding: 0.8rem 1.1rem; border-radius: 4px; }
    .notes-box p { margin-bottom: 0.4rem; font-size: 9.5pt; }
    .notes-box p:last-child { margin-bottom: 0; }
    .notes-title { font-weight: bold; text-transform: uppercase; letter-spacing: 0.07em; font-size: 9pt; margin-bottom: 0.5rem; color: #a05c00; }
    .status-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; }
    .status-item { display: flex; align-items: center; gap: 0.5rem; padding: 0.45rem 0.7rem; border: 1px solid #ccc; border-radius: 3px; }
    .status-dot { width: 9px; height: 9px; border-radius: 50%; background: #16a34a; flex-shrink: 0; }
    .status-label { font-size: 8.5pt; color: #222; }
    .meta { background: #f5f5f5; border: 1px solid #ddd; padding: 0.7rem 1rem; border-radius: 4px; }
    .meta-row { display: flex; gap: 2rem; flex-wrap: wrap; }
    .meta-item { font-size: 8pt; color: #555; margin-bottom: 0.25rem; }
    .meta-item strong { color: #1a1a2e; }
    .foot { text-align: center; border-top: 1px solid #ccc; padding-top: 0.8rem; margin-top: 2rem; font-size: 8pt; color: #888; }
    @media print { .no-print { display: none !important; } }
    @media screen { body { max-width: 900px; margin: 2rem auto; padding: 0 2rem; } }
  `;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Experiment Report — ${experimentId}</title>
  <style>${css}</style>
</head>
<body>

  <!-- §01 COVER PAGE -->
  <div class="cover">
    <div class="cover-platform">⚡ ZnO Supercapacitor AI Platform</div>
    <div class="cover-title">AI-Predicted Cyclic Voltammetry<br/>Analysis Report</div>
    <div class="cover-subtitle">Electrochemical Machine Learning Research</div>
    <hr class="cover-rule"/>
    <div class="cover-id">${experimentId}</div>
    <div class="cover-meta">
      <div><div class="cover-item-label">Material Group</div><div class="cover-item-value">${result.material_id}</div></div>
      <div><div class="cover-item-label">Scan Rate</div><div class="cover-item-value">${result.scan_rate_mVs} mV/s</div></div>
      <div><div class="cover-item-label">ML Model</div><div class="cover-item-value">${modelConfig.shortName}</div></div>
      <div><div class="cover-item-label">Points</div><div class="cover-item-value">${result.n_points}</div></div>
    </div>
    <div class="cover-generated">Generated: ${generatedAt.toLocaleString("en-IN", { dateStyle: "long", timeStyle: "medium" })}</div>
  </div>

  <!-- §02 EXECUTIVE SUMMARY -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">02</span><span class="sec-title">Executive Summary</span></div>
    <div class="exec-box"><p>${execSummary}</p></div>
  </div>

  <!-- §03 EXPERIMENT OBJECTIVE -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">03</span><span class="sec-title">Experiment Objective</span></div>
    <p>${objectiveText}</p>
  </div>

  <!-- §04 EXPERIMENT CONFIGURATION -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">04</span><span class="sec-title">Experiment Configuration</span></div>
    <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>
      ${configRows.map(([l, v]) => `<tr><td>${l}</td><td class="mono">${v}</td></tr>`).join("")}
    </tbody></table>
  </div>

  <!-- §05 SCIENTIFIC BACKGROUND -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">05</span><span class="sec-title">Scientific Background</span></div>
    <div class="sci-box"><p>${sciBg}</p></div>
  </div>

  <!-- §06 INPUT PARAMETERS -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">06</span><span class="sec-title">Input Parameters</span></div>
    <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>
      ${inputRows.map(([l, v]) => `<tr><td>${l}</td><td class="mono">${v}</td></tr>`).join("")}
    </tbody></table>
  </div>

  <!-- §07 PREDICTED CV CURVE -->
  <div class="sec" style="page-break-before: always;">
    <div class="sec-hdr"><span class="sec-num">07</span><span class="sec-title">Predicted CV Curve</span></div>
    <div class="chart-wrap">
      ${svgChart}
      <div class="chart-cap"><strong>Figure 1.</strong> Predicted cyclic voltammogram for ${result.material_id} at ${result.scan_rate_mVs} mV/s using ${modelConfig.display}. ${result.n_points} prediction points spanning ${potMin.toFixed(3)} V to ${potMax.toFixed(3)} V. Dashed line = zero-current axis.</div>
    </div>
  </div>

  <!-- §08 ELECTROCHEMICAL RESULT SUMMARY -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">08</span><span class="sec-title">Electrochemical Result Summary</span></div>
    <div class="mgrid">
      ${resultMetricRows.map(([l, v]) => `<div class="mc"><div class="mc-l">${l}</div><div class="mc-v">${v}</div></div>`).join("")}
    </div>
  </div>

  <!-- §09 ELECTROCHEMICAL INTERPRETATION -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">09</span><span class="sec-title">Electrochemical Interpretation</span></div>
    ${interpretation.map(p => `<div class="interp"><p>${p}</p></div>`).join("\n    ")}
  </div>

  <!-- §10 MACHINE LEARNING PERFORMANCE -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">10</span><span class="sec-title">Machine Learning Performance</span></div>
    ${mlNarrative}
    <table style="margin-top:1rem;">
      <thead><tr><th>Evaluation Partition</th><th>RMSE (µA)</th><th>R²</th><th>MAE (µA)</th></tr></thead>
      <tbody>${mlRows}</tbody>
    </table>
  </div>

  <!-- §11 RESULT INTERPRETATION -->
  <div class="sec" style="page-break-before: always;">
    <div class="sec-hdr"><span class="sec-num">11</span><span class="sec-title">Result Interpretation</span></div>
    <ol class="obs-list">
      ${resultObservations.map((obs, i) => `<li class="obs-item"><span class="obs-num">${String(i + 1).padStart(2, "0")}.</span><span>${obs}</span></li>`).join("\n      ")}
    </ol>
  </div>

  <!-- §12 SCIENTIFIC INFERENCE -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">12</span><span class="sec-title">Scientific Inference</span></div>
    <p>${inferenceText}</p>
  </div>

  <!-- §13 EXPERIMENTAL CONCLUSION -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">13</span><span class="sec-title">Experimental Conclusion</span></div>
    ${conclusionText.split("\n\n").map(p => `<p>${p}</p>`).join("\n    ")}
  </div>

  <!-- §14 RESEARCH NOTES -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">14</span><span class="sec-title">Research Notes</span></div>
    <div class="notes-box">
      <div class="notes-title">⚠ Important Disclaimer</div>
      <p>This report was generated automatically by an AI/ML prediction system. All numerical values are model predictions, not experimentally measured data.</p>
      <p>Experimental laboratory validation is strongly recommended before using these results for device design, publication, or material procurement decisions.</p>
      <p>This platform is suitable for preliminary electrochemical screening and comparative material analysis. Model predictions carry inherent uncertainty quantified by RMSE values in §10.</p>
      <p>Prediction confidence: <strong>${confidenceTier.label}</strong> — ${confidenceTier.desc}</p>
    </div>
  </div>

  <!-- §15 FIGURE AND TABLE CAPTIONS -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">15</span><span class="sec-title">Figure and Table Captions</span></div>
    <table><thead><tr><th style="width:90px;">Reference</th><th>Caption</th></tr></thead><tbody>
      ${figCaptions.map(([ref, cap]) => `<tr><td class="mono">${ref}</td><td style="font-style:italic;">${cap}</td></tr>`).join("\n      ")}
    </tbody></table>
  </div>

  <!-- §16 EXPERIMENT STATUS -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">16</span><span class="sec-title">Experiment Status</span></div>
    <div class="status-grid">
      <div class="status-item"><div class="status-dot"></div><span class="status-label">Prediction Completed</span></div>
      <div class="status-item"><div class="status-dot"></div><span class="status-label">Model Loaded</span></div>
      <div class="status-item"><div class="status-dot"></div><span class="status-label">${result.n_points} Points Generated</span></div>
      <div class="status-item"><div class="status-dot"></div><span class="status-label">Inference Successful</span></div>
      <div class="status-item"><div class="status-dot"></div><span class="status-label">Report Generated</span></div>
      <div class="status-item"><div class="status-dot"></div><span class="status-label">Overall Status: Complete</span></div>
    </div>
  </div>

  <!-- §17 REPORT METADATA -->
  <div class="sec">
    <div class="sec-hdr"><span class="sec-num">17</span><span class="sec-title">Report Metadata</span></div>
    <div class="meta">
      <div class="meta-row">
        <div class="meta-item"><strong>Platform Version:</strong> v2.0.0</div>
        <div class="meta-item"><strong>Report Version:</strong> v18s-2.0</div>
        <div class="meta-item"><strong>Dataset Version:</strong> ZnO-CV-v1.0</div>
        <div class="meta-item"><strong>Model Version:</strong> Production</div>
        <div class="meta-item"><strong>Model:</strong> ${modelConfig.display} — ${modelConfig.architecture.slice(0, 55)}…</div>
        <div class="meta-item"><strong>Generated:</strong> ${generatedAt.toISOString()}</div>
        <div class="meta-item"><strong>Experiment ID:</strong> ${experimentId}</div>
        <div class="meta-item"><strong>Application:</strong> ZnO Supercapacitor AI Platform</div>
      </div>
    </div>
  </div>

  <div class="foot">ZnO Supercapacitor AI Platform · Electrochemical Machine Learning Research · ${experimentId} · Generated ${generatedAt.toLocaleDateString("en-IN")}</div>
</body>
</html>`;
}

// ── Sub-components ─────────────────────────────────────────────────────────

function SecHeader({ num, title, icon }: { num: string; title: string; icon: React.ReactNode }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2, pb: 1, borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
      <Box sx={{ width: 26, height: 26, borderRadius: 1, background: "rgba(0,212,255,0.12)", border: "1px solid rgba(0,212,255,0.25)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
        <Typography sx={{ fontFamily: "JetBrains Mono", fontSize: "0.58rem", fontWeight: 800, color: "#00d4ff" }}>{num}</Typography>
      </Box>
      <Box sx={{ "& svg": { fontSize: 15, color: "rgba(0,212,255,0.6)" }, mr: 0.25 }}>{icon}</Box>
      <Typography variant="overline" sx={{ color: "#00d4ff", fontSize: "0.68rem", fontWeight: 700, letterSpacing: "0.1em" }}>
        {title}
      </Typography>
    </Box>
  );
}

function KVRow({ label, value, mono = true }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", py: 0.6, borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
      <Typography sx={{ color: "rgba(255,255,255,0.38)", fontSize: "0.72rem" }}>{label}</Typography>
      <Typography sx={{ fontFamily: mono ? "JetBrains Mono" : "inherit", fontSize: "0.78rem", fontWeight: 600, color: "rgba(255,255,255,0.85)", textAlign: "right", maxWidth: "55%" }}>
        {value}
      </Typography>
    </Box>
  );
}

function MetricTile({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <Box sx={{ p: 1.5, borderRadius: 1.5, background: alpha(color, 0.05), border: `1px solid ${alpha(color, 0.18)}` }}>
      <Typography sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.4 }}>{label}</Typography>
      <Typography sx={{ color, fontFamily: "JetBrains Mono", fontSize: "0.86rem", fontWeight: 700 }}>{value}</Typography>
    </Box>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export default function ExperimentReport({
  open, onClose,
  result, advMetrics, confidenceTier, interpretation,
  modelConfig, predictionMs, generatedAt,
}: ExperimentReportProps) {

  const { leaderboard, comparison, loading: lbLoading } = useLeaderboard();

  const lbEntry = useMemo<LeaderboardEntry | null>(
    () => leaderboard.find(e => e.model_id === result.model_name) ?? null,
    [leaderboard, result.model_name]
  );

  const modelColor = MODEL_COLORS[result.model_name] ?? "#00d4ff";
  const experimentId = useMemo(() => makeExperimentId(result, generatedAt), [result, generatedAt]);

  const maxI = useMemo(() => Math.max(...result.predicted_current_uA), [result]);
  const minI = useMemo(() => Math.min(...result.predicted_current_uA), [result]);
  const avgI = useMemo(
    () => result.predicted_current_uA.reduce((a, b) => a + b, 0) / result.predicted_current_uA.length,
    [result]
  );
  const potMin = useMemo(() => Math.min(...result.potential_V), [result]);
  const potMax = useMemo(() => Math.max(...result.potential_V), [result]);

  const execSummary = useMemo(
    () => generateExecutiveSummary(result, advMetrics, modelConfig, lbEntry, confidenceTier, predictionMs),
    [result, advMetrics, modelConfig, lbEntry, confidenceTier, predictionMs]
  );
  const sciBg = useMemo(() => generateScientificBackground(result, modelConfig), [result, modelConfig]);
  const objectiveText = useMemo(() => generateObjective(result, modelConfig), [result, modelConfig]);
  const inferenceText = useMemo(() => generateScientificInference(result, advMetrics, lbEntry), [result, advMetrics, lbEntry]);
  const conclusionText = useMemo(() => generateConclusion(result, advMetrics, modelConfig, lbEntry, confidenceTier), [result, advMetrics, modelConfig, lbEntry, confidenceTier]);
  const resultObservations = useMemo(
    () => generateResultInterpretation(result, advMetrics, lbEntry, confidenceTier),
    [result, advMetrics, lbEntry, confidenceTier]
  );

  const modelPartitions = useMemo(
    () => comparison
      .filter(r => r.model_id === result.model_name)
      .sort((a, b) => {
        const ord = ["train", "val", "test_SR", "test_MAT"];
        return ord.indexOf(a.partition) - ord.indexOf(b.partition);
      }),
    [comparison, result.model_name]
  );

  useEffect(() => {
    if (!open) return;
    const el = document.createElement("style");
    el.id = "zno-report-print-css";
    el.textContent = `@media print {
      body > *:not(.MuiModal-root) { display: none !important; }
      .MuiModal-root .MuiDialog-paper { box-shadow: none !important; }
      .report-no-print { display: none !important; }
      .report-print-area { page-break-inside: avoid; }
    }`;
    document.head.appendChild(el);
    return () => { document.getElementById("zno-report-print-css")?.remove(); };
  }, [open]);

  const handlePrint = useCallback(() => {
    const html = buildPrintHTML(
      result, advMetrics, confidenceTier, interpretation,
      modelConfig, lbEntry, comparison, predictionMs, generatedAt, experimentId
    );
    const win = window.open("", "_blank", "width=960,height=760");
    if (!win) return;
    win.document.write(html);
    win.document.close();
    win.focus();
    setTimeout(() => win.print(), 700);
  }, [result, advMetrics, confidenceTier, interpretation, modelConfig, lbEntry, comparison, predictionMs, generatedAt, experimentId]);

  const handleDownload = useCallback(() => {
    const html = buildPrintHTML(
      result, advMetrics, confidenceTier, interpretation,
      modelConfig, lbEntry, comparison, predictionMs, generatedAt, experimentId
    );
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `experiment_report_${experimentId}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result, advMetrics, confidenceTier, interpretation, modelConfig, lbEntry, comparison, predictionMs, generatedAt, experimentId]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullScreen
      PaperProps={{
        sx: {
          bgcolor: "#070b14",
          backgroundImage: "none",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        },
      }}
    >
      {/* ── Toolbar ─────────────────────────────────────────────────── */}
      <Box
        className="report-no-print"
        sx={{
          display: "flex", alignItems: "center", gap: 1.5,
          px: 2.5, py: 1.5,
          borderBottom: "1px solid rgba(0,212,255,0.1)",
          background: "rgba(7,11,20,0.96)",
          flexShrink: 0,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <DescriptionRounded sx={{ fontSize: 17, color: "#00d4ff" }} />
          <Typography variant="subtitle2" sx={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.84rem" }}>
            Experiment Report
          </Typography>
          <Chip
            label={experimentId}
            size="small"
            sx={{ fontSize: "0.58rem", height: 17, fontFamily: "JetBrains Mono", bgcolor: "rgba(0,212,255,0.07)", color: "rgba(0,212,255,0.65)", border: "1px solid rgba(0,212,255,0.18)" }}
          />
        </Box>
        <Box sx={{ ml: "auto", display: "flex", gap: 1 }}>
          <GlowButton
            variant="outlined" size="small"
            startIcon={<PrintRounded sx={{ fontSize: 13 }} />}
            onClick={handlePrint}
            glowColor="#10b981"
            sx={{ fontSize: "0.7rem", py: 0.4, px: 1.25, borderColor: "rgba(16,185,129,0.3)", color: "#10b981" }}
          >
            Print / PDF
          </GlowButton>
          <GlowButton
            variant="outlined" size="small"
            startIcon={<DownloadRounded sx={{ fontSize: 13 }} />}
            onClick={handleDownload}
            glowColor="#a78bfa"
            sx={{ fontSize: "0.7rem", py: 0.4, px: 1.25, borderColor: "rgba(167,139,250,0.3)", color: "#a78bfa" }}
          >
            Download HTML
          </GlowButton>
          <Tooltip title="Close report">
            <IconButton size="small" onClick={onClose} sx={{ color: "rgba(255,255,255,0.4)", "&:hover": { color: "#ef4444" } }}>
              <CloseRounded sx={{ fontSize: 17 }} />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* ── Report body ──────────────────────────────────────────────── */}
      <DialogContent sx={{ p: 0, overflow: "auto", flex: 1 }}>
        <Box sx={{ maxWidth: 980, mx: "auto", px: { xs: 2, sm: 3.5 }, py: 3.5 }}>

          {/* ── §01 COVER PAGE ────────────────────────────────────────── */}
          <Box
            className="report-print-area"
            sx={{
              textAlign: "center",
              pb: 4, mb: 5,
              borderBottom: "1px solid rgba(0,212,255,0.12)",
              background: "linear-gradient(180deg, rgba(0,212,255,0.03) 0%, transparent 100%)",
              borderRadius: 2,
              pt: 4,
              px: 3,
            }}
          >
            <Typography variant="overline" sx={{ color: "rgba(0,212,255,0.5)", fontSize: "0.62rem", letterSpacing: "0.18em", display: "block", mb: 1.5 }}>
              ⚡ ZnO Supercapacitor AI Platform
            </Typography>
            <Typography sx={{ fontSize: { xs: "1.4rem", sm: "1.75rem" }, fontWeight: 800, color: "#f1f5f9", letterSpacing: "0.02em", lineHeight: 1.25, mb: 0.5 }}>
              AI-Predicted Cyclic Voltammetry
            </Typography>
            <Typography sx={{ fontSize: { xs: "1.4rem", sm: "1.75rem" }, fontWeight: 800, color: "#f1f5f9", letterSpacing: "0.02em", lineHeight: 1.25, mb: 1.5 }}>
              Analysis Report
            </Typography>
            <Typography sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.8rem", fontStyle: "italic", mb: 3 }}>
              Electrochemical Machine Learning Research
            </Typography>

            <Box sx={{ display: "flex", justifyContent: "center", gap: 1, flexWrap: "wrap", mb: 3 }}>
              <Chip label={experimentId} size="small" sx={{ fontFamily: "JetBrains Mono", fontSize: "0.6rem", bgcolor: "rgba(0,212,255,0.07)", color: "#00d4ff", border: "1px solid rgba(0,212,255,0.2)", height: 22 }} />
              <Chip
                icon={<Box component="span" sx={{ width: 7, height: 7, borderRadius: "50%", bgcolor: modelColor, ml: "6px !important" }} />}
                label={modelConfig.display}
                size="small"
                sx={{ fontSize: "0.6rem", bgcolor: alpha(modelColor, 0.07), color: modelColor, border: `1px solid ${alpha(modelColor, 0.2)}`, height: 22 }}
              />
              <Chip label={`${result.material_id} · ${result.scan_rate_mVs} mV/s`} size="small" sx={{ fontSize: "0.6rem", bgcolor: "rgba(167,139,250,0.07)", color: "#a78bfa", border: "1px solid rgba(167,139,250,0.2)", height: 22 }} />
              <Chip label={confidenceTier.label} size="small" sx={{ fontSize: "0.6rem", bgcolor: alpha(confidenceTier.color, 0.07), color: confidenceTier.color, border: `1px solid ${alpha(confidenceTier.color, 0.2)}`, height: 22 }} />
            </Box>

            <Box sx={{ display: "flex", justifyContent: "center", gap: 4, flexWrap: "wrap" }}>
              {[
                ["Material", MATERIAL_LABEL[result.material_id] ?? result.material_id],
                ["Scan Rate", `${result.scan_rate_mVs} mV/s`],
                ["Points", `${result.n_points}`],
                ["Generated", generatedAt.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" })],
              ].map(([l, v]) => (
                <Box key={l} sx={{ textAlign: "center" }}>
                  <Typography sx={{ color: "rgba(255,255,255,0.25)", fontSize: "0.58rem", textTransform: "uppercase", letterSpacing: "0.1em", mb: 0.3 }}>{l}</Typography>
                  <Typography sx={{ fontFamily: "JetBrains Mono", fontSize: "0.78rem", fontWeight: 600, color: "rgba(255,255,255,0.7)" }}>{v}</Typography>
                </Box>
              ))}
            </Box>
          </Box>

          {/* ── §02 EXECUTIVE SUMMARY ─────────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="02" title="Executive Summary" icon={<SummarizeRounded />} />
            <Box sx={{ p: 2.5, borderRadius: 2, background: "rgba(0,212,255,0.025)", border: "1px solid rgba(0,212,255,0.1)", borderLeft: "3px solid rgba(0,212,255,0.4)" }}>
              <Typography sx={{ color: "rgba(255,255,255,0.7)", fontSize: "0.84rem", lineHeight: 1.85, fontStyle: "italic" }}>
                {execSummary}
              </Typography>
            </Box>
          </Box>

          {/* ── §03 EXPERIMENT OBJECTIVE ──────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="03" title="Experiment Objective" icon={<ScienceRounded />} />
            <Box sx={{ p: 2.5, borderRadius: 2, background: "rgba(0,212,255,0.02)", border: "1px solid rgba(0,212,255,0.08)" }}>
              <Typography sx={{ color: "rgba(255,255,255,0.68)", fontSize: "0.84rem", lineHeight: 1.8 }}>
                {objectiveText}
              </Typography>
            </Box>
          </Box>

          {/* ── §04 EXPERIMENT CONFIGURATION ──────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="04" title="Experiment Configuration" icon={<AssignmentRounded />} />
            <Table size="small" sx={{ "& td, & th": { borderColor: "rgba(255,255,255,0.07)", fontSize: "0.78rem" }, "& th": { color: "rgba(255,255,255,0.38)", fontSize: "0.61rem", textTransform: "uppercase", letterSpacing: "0.07em", py: 1.1, bgcolor: "rgba(0,0,0,0.25)" } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ width: "42%" }}>Parameter</TableCell>
                  <TableCell>Value</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {[
                  ["Experiment ID", experimentId],
                  ["Material Group", MATERIAL_LABEL[result.material_id] ?? result.material_id],
                  ["Scan Rate", `${result.scan_rate_mVs} mV/s`],
                  ["Potential Range", `${potMin.toFixed(3)} V to ${potMax.toFixed(3)} V`],
                  ["Voltage Window", `${advMetrics.voltageWindow.toFixed(3)} V`],
                  ["Prediction Model", `${modelConfig.display} (${modelConfig.shortName})`],
                  ["Dataset Used", "ZnO Nanocomposite CV Dataset — NM1–NM4"],
                  ["Training Scenario", result.material_id === "NM4" ? "Cross-material extrapolation (zero-shot)" : "Interpolation — within training distribution"],
                  ["Material Status", result.material_id === "NM4" ? "Zero-shot — excluded from training" : "Known material — present in training splits"],
                  ["Prediction Time", predictionMs != null ? `${predictionMs} ms (wall clock)` : "—"],
                  ["Benchmark Inference", lbEntry?.inference_ms != null ? `${lbEntry.inference_ms.toFixed(1)} ms` : "—"],
                  ["Points Generated", `${result.n_points} (full CV sweep)`],
                ].map(([l, v]) => (
                  <TableRow key={l} sx={{ "&:last-child td": { border: 0 } }}>
                    <TableCell sx={{ color: "rgba(255,255,255,0.42)" }}>{l}</TableCell>
                    <TableCell sx={{ fontFamily: "JetBrains Mono", fontWeight: 600, color: "rgba(255,255,255,0.82)" }}>{v}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>

          {/* ── §05 SCIENTIFIC BACKGROUND ─────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="05" title="Scientific Background" icon={<MenuBookRounded />} />
            <Box sx={{ p: 2.5, borderRadius: 2, background: "rgba(167,139,250,0.025)", border: "1px solid rgba(167,139,250,0.1)" }}>
              <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.83rem", lineHeight: 1.85 }}>
                {sciBg}
              </Typography>
            </Box>
          </Box>

          {/* ── §06 INPUT PARAMETERS ──────────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="06" title="Input Parameters" icon={<MemoryRounded />} />
            <Table size="small" sx={{ "& td, & th": { borderColor: "rgba(255,255,255,0.07)", fontSize: "0.78rem" }, "& th": { color: "rgba(255,255,255,0.38)", fontSize: "0.61rem", textTransform: "uppercase", letterSpacing: "0.07em", py: 1.1, bgcolor: "rgba(0,0,0,0.25)" } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ width: "42%" }}>Parameter</TableCell>
                  <TableCell>Value</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {[
                  ["Material Group", MATERIAL_LABEL[result.material_id] ?? result.material_id],
                  ["Scan Rate", `${result.scan_rate_mVs} mV/s`],
                  ["Potential Window", `${potMin.toFixed(3)} V to ${potMax.toFixed(3)} V`],
                  ["Model", `${modelConfig.display} (${modelConfig.shortName})`],
                  ["Dataset", "ZnO Nanocomposite Supercapacitor CV Dataset"],
                  ["Feature Set", "Physics-informed (scan_rate, norm_potential, sweep_direction, ...)"],
                  ["Prediction Type", result.material_id === "NM4" ? "Cross-material zero-shot" : "Interpolation"],
                  ["Inference Mode", INFERENCE_TIER[result.model_name] ?? "Tier-1"],
                  ["Electrolyte (Dataset)", "1 M Na₂SO₄ (aqueous)"],
                  ["Electrode Type (Dataset)", "ZnO nanocomposite thin film"],
                ].map(([l, v]) => (
                  <TableRow key={l} sx={{ "&:last-child td": { border: 0 } }}>
                    <TableCell sx={{ color: "rgba(255,255,255,0.42)" }}>{l}</TableCell>
                    <TableCell sx={{ fontFamily: "JetBrains Mono", fontWeight: 600, color: "rgba(255,255,255,0.82)" }}>{v}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>

          {/* ── §07 PREDICTED CV CURVE ────────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="07" title="Predicted CV Curve" icon={<BiotechRounded />} />
            <Box sx={{ borderRadius: 2, overflow: "hidden", border: "1px solid rgba(0,212,255,0.1)", background: "rgba(7,11,20,0.9)" }}>
              <SingleCVPlot
                result={result}
                previousResult={null}
                showConfidenceBand={false}
                rmseUa={0}
                showPeakAnnotations={true}
                experimentalData={null}
                height={420}
              />
            </Box>
            <Typography variant="caption" sx={{ display: "block", textAlign: "center", color: "rgba(255,255,255,0.28)", mt: 1, fontSize: "0.67rem", fontStyle: "italic" }}>
              Figure 1. Predicted cyclic voltammogram — {result.material_id} at {result.scan_rate_mVs} mV/s using {modelConfig.display} ({result.n_points} data points, {potMin.toFixed(3)} V to {potMax.toFixed(3)} V). Toolbar: zoom, pan, PNG/SVG export.
            </Typography>
          </Box>

          {/* ── §08 ELECTROCHEMICAL RESULT SUMMARY ───────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="08" title="Electrochemical Result Summary" icon={<BarChartRounded />} />
            <Grid container spacing={1.5}>
              {[
                { l: "Maximum Current", v: `${maxI.toFixed(2)} µA`, c: "#10b981" },
                { l: "Minimum Current", v: `${minI.toFixed(2)} µA`, c: "#f472b6" },
                { l: "Average Current", v: `${avgI.toFixed(2)} µA`, c: "#94a3b8" },
                { l: "Peak Anodic Current", v: fmtUa(result.statistics.peak_anodic_uA), c: "#10b981" },
                { l: "Peak Cathodic Current", v: fmtUa(result.statistics.peak_cathodic_uA), c: "#f472b6" },
                { l: "Current Range", v: fmtUa(result.statistics.current_range_uA), c: "#00d4ff" },
                { l: "Integral Area", v: `${result.statistics.integral_area.toFixed(2)} µA·V`, c: "#a78bfa" },
                { l: "Charge Storage Index", v: advMetrics.chargeStorageIndex != null ? `${advMetrics.chargeStorageIndex.toFixed(1)} µF` : "—", c: "#22d3ee" },
                { l: "Peak Separation", v: advMetrics.peakSep != null ? `${advMetrics.peakSep.toFixed(3)} V` : "—", c: "#a78bfa" },
                { l: "Symmetry Factor", v: advMetrics.symFactor?.toFixed(4) ?? "—", c: advMetrics.symFactor != null && Math.abs(advMetrics.symFactor - 1) < 0.08 ? "#10b981" : "#f59e0b" },
                { l: "Energy Proxy", v: advMetrics.energyProxy != null ? `${advMetrics.energyProxy.toFixed(3)} µJ` : "—", c: "#f59e0b" },
                { l: "CV Asymmetry", v: advMetrics.asymmetryPct != null ? `${advMetrics.asymmetryPct.toFixed(1)} %` : "—", c: advMetrics.asymmetryPct != null && advMetrics.asymmetryPct < 8 ? "#10b981" : "#f59e0b" },
              ].map(({ l, v, c }) => (
                <Grid item xs={6} sm={4} md={3} key={l}>
                  <MetricTile label={l} value={v} color={c} />
                </Grid>
              ))}
            </Grid>
          </Box>

          {/* ── §09 ELECTROCHEMICAL INTERPRETATION ───────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="09" title="Electrochemical Interpretation" icon={<ScienceRounded />} />
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.25 }}>
              {interpretation.map((para, idx) => (
                <Box
                  key={idx}
                  sx={{
                    p: 2, borderRadius: 1.5,
                    background: idx === 0 ? "rgba(0,212,255,0.03)" : "rgba(255,255,255,0.015)",
                    borderLeft: `3px solid ${idx === 0 ? "rgba(0,212,255,0.35)" : "rgba(255,255,255,0.07)"}`,
                  }}
                >
                  <Typography sx={{ color: idx === 0 ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.48)", fontSize: "0.81rem", lineHeight: 1.8 }}>
                    {para}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>

          {/* ── §10 MACHINE LEARNING PERFORMANCE ─────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="10" title="Machine Learning Performance" icon={<AutoAwesomeRounded />} />
            {lbLoading ? (
              <Box sx={{ py: 3, textAlign: "center" }}>
                <CircularProgress size={22} sx={{ color: "rgba(255,255,255,0.3)" }} />
                <Typography variant="caption" sx={{ display: "block", mt: 1, color: "rgba(255,255,255,0.28)", fontSize: "0.7rem" }}>Loading benchmark metrics…</Typography>
              </Box>
            ) : (
              <>
                <Box sx={{ p: 2, mb: 2.5, borderRadius: 2, background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                  <Typography sx={{ color: "rgba(255,255,255,0.28)", fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: "0.08em", mb: 0.75 }}>Why {modelConfig.shortName}?</Typography>
                  <Typography sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.8rem", lineHeight: 1.75 }}>{modelConfig.description}</Typography>
                </Box>
                <Grid container spacing={1.5} sx={{ mb: 2.5 }}>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ p: 1.75, borderRadius: 1.5, background: "rgba(16,185,129,0.04)", border: "1px solid rgba(16,185,129,0.12)", height: "100%" }}>
                      <Typography sx={{ color: "#10b981", fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", mb: 1 }}>Strengths</Typography>
                      {modelConfig.strengths.map(s => (
                        <Box key={s} sx={{ display: "flex", gap: 0.75, mb: 0.5 }}>
                          <Typography sx={{ color: "#10b981", fontSize: "0.7rem", flexShrink: 0 }}>+</Typography>
                          <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.75rem" }}>{s}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ p: 1.75, borderRadius: 1.5, background: "rgba(245,158,11,0.04)", border: "1px solid rgba(245,158,11,0.12)", height: "100%" }}>
                      <Typography sx={{ color: "#f59e0b", fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", mb: 1 }}>Limitations</Typography>
                      {modelConfig.weaknesses.map(w => (
                        <Box key={w} sx={{ display: "flex", gap: 0.75, mb: 0.5 }}>
                          <Typography sx={{ color: "#f59e0b", fontSize: "0.7rem", flexShrink: 0 }}>−</Typography>
                          <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.75rem" }}>{w}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Grid>
                </Grid>
                <Table size="small" sx={{ "& td, & th": { borderColor: "rgba(255,255,255,0.07)", fontSize: "0.76rem" }, "& th": { color: "rgba(255,255,255,0.36)", fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: "0.06em", bgcolor: "rgba(0,0,0,0.2)", py: 1.1 } }}>
                  <TableHead>
                    <TableRow>
                      <TableCell>Evaluation Partition</TableCell>
                      <TableCell align="right">RMSE (µA)</TableCell>
                      <TableCell align="right">R²</TableCell>
                      <TableCell align="right">MAE (µA)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {modelPartitions.length > 0
                      ? modelPartitions.map(r => (
                        <TableRow key={r.partition} sx={{ "&:last-child td": { border: 0 } }}>
                          <TableCell sx={{ color: "rgba(255,255,255,0.5)" }}>{PARTITION_DISPLAY[r.partition] ?? r.partition}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: "JetBrains Mono", fontWeight: 700, color: rmseColor(r.rmse_uA) }}>{r.rmse_uA.toFixed(2)}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: "JetBrains Mono", fontWeight: 700, color: r2Color(r.r2) }}>{r.r2.toFixed(4)}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: "JetBrains Mono", color: "rgba(255,255,255,0.38)" }}>{r.mae_uA != null ? r.mae_uA.toFixed(2) : "—"}</TableCell>
                        </TableRow>
                      ))
                      : lbEntry && [
                          { p: "val", rmse: lbEntry.rmse_val_uA, r2: lbEntry.r2_val },
                          { p: "test_SR", rmse: lbEntry.rmse_test_sr_uA, r2: lbEntry.r2_test_sr },
                          { p: "test_MAT", rmse: lbEntry.rmse_test_mat_uA, r2: lbEntry.r2_test_mat },
                        ].map(r => (
                          <TableRow key={r.p} sx={{ "&:last-child td": { border: 0 } }}>
                            <TableCell sx={{ color: "rgba(255,255,255,0.5)" }}>{PARTITION_DISPLAY[r.p]}</TableCell>
                            <TableCell align="right" sx={{ fontFamily: "JetBrains Mono", fontWeight: 700, color: rmseColor(r.rmse) }}>{r.rmse.toFixed(2)}</TableCell>
                            <TableCell align="right" sx={{ fontFamily: "JetBrains Mono", fontWeight: 700, color: r2Color(r.r2) }}>{r.r2.toFixed(4)}</TableCell>
                            <TableCell align="right" sx={{ color: "rgba(255,255,255,0.28)" }}>—</TableCell>
                          </TableRow>
                        ))
                    }
                  </TableBody>
                </Table>
                {lbEntry && (
                  <Box sx={{ display: "flex", gap: 1, mt: 1.5, flexWrap: "wrap" }}>
                    {[
                      { l: "Rank", v: `#${lbEntry.rank}` },
                      { l: "Model Size", v: modelConfig.size },
                      { l: "Deploy Score", v: `${modelConfig.deployScore}/100` },
                      { l: "Type", v: modelConfig.type.toUpperCase() },
                      { l: "Benchmark Inference", v: lbEntry.inference_ms != null ? fmtMs(lbEntry.inference_ms) : "—" },
                    ].map(({ l, v }) => (
                      <Chip key={l} label={`${l}: ${v}`} size="small"
                        sx={{ fontSize: "0.6rem", bgcolor: "rgba(255,255,255,0.035)", color: "rgba(255,255,255,0.42)", border: "1px solid rgba(255,255,255,0.08)" }} />
                    ))}
                  </Box>
                )}
              </>
            )}
          </Box>

          {/* ── §11 RESULT INTERPRETATION ────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="11" title="Result Interpretation" icon={<InsightsRounded />} />
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {resultObservations.map((obs, idx) => (
                <Box
                  key={idx}
                  sx={{
                    display: "flex", gap: 1.5, alignItems: "flex-start",
                    p: 1.75, borderRadius: 1.5,
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid rgba(255,255,255,0.055)",
                    borderLeft: "3px solid rgba(167,139,250,0.3)",
                  }}
                >
                  <Typography sx={{ fontFamily: "JetBrains Mono", fontSize: "0.63rem", fontWeight: 800, color: "rgba(167,139,250,0.65)", flexShrink: 0, mt: 0.1, minWidth: "1.6rem" }}>
                    {String(idx + 1).padStart(2, "0")}.
                  </Typography>
                  <Typography sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.8rem", lineHeight: 1.8 }}>
                    {obs}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>

          {/* ── §12 SCIENTIFIC INFERENCE ──────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="12" title="Scientific Inference" icon={<InfoOutlined />} />
            <Box sx={{ p: 2.5, borderRadius: 2, background: "rgba(167,139,250,0.03)", border: "1px solid rgba(167,139,250,0.12)" }}>
              <Typography sx={{ color: "rgba(255,255,255,0.68)", fontSize: "0.84rem", lineHeight: 1.85 }}>
                {inferenceText}
              </Typography>
            </Box>
          </Box>

          {/* ── §13 EXPERIMENTAL CONCLUSION ──────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="13" title="Experimental Conclusion" icon={<VerifiedRounded />} />
            <Box sx={{ p: 2.5, borderRadius: 2, background: "rgba(16,185,129,0.025)", border: "1px solid rgba(16,185,129,0.12)" }}>
              {conclusionText.split("\n\n").map((para, i, arr) => (
                <Typography key={i} sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.82rem", lineHeight: 1.85, mb: i < arr.length - 1 ? 1.5 : 0 }}>
                  {para}
                </Typography>
              ))}
            </Box>
          </Box>

          {/* ── §14 RESEARCH NOTES ────────────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="14" title="Research Notes" icon={<InfoRounded />} />
            <Box sx={{ p: 2.25, borderRadius: 2, background: "rgba(245,158,11,0.04)", border: "1.5px solid rgba(245,158,11,0.18)" }}>
              <Typography sx={{ color: "#f59e0b", fontSize: "0.63rem", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", mb: 1.25 }}>
                ⚠ Important Disclaimer
              </Typography>
              {[
                "This report was generated automatically by an AI/ML prediction system. All numerical values are model predictions, not experimentally measured data.",
                "Experimental laboratory validation is strongly recommended before using these results for device design, publication, or material procurement decisions.",
                "This platform is suitable for preliminary electrochemical screening and comparative material analysis. Model predictions carry inherent uncertainty quantified by RMSE values in §10.",
                `Prediction confidence: ${confidenceTier.label} — ${confidenceTier.desc}`,
              ].map((note, i) => (
                <Typography key={i} sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.79rem", lineHeight: 1.75, mb: i < 3 ? 0.9 : 0 }}>
                  {note}
                </Typography>
              ))}
            </Box>
          </Box>

          {/* ── §15 FIGURE AND TABLE CAPTIONS ────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="15" title="Figure and Table Captions" icon={<FormatListNumberedRounded />} />
            <Table size="small" sx={{ "& td, & th": { borderColor: "rgba(255,255,255,0.07)", fontSize: "0.76rem" }, "& th": { color: "rgba(255,255,255,0.36)", fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: "0.06em", bgcolor: "rgba(0,0,0,0.2)", py: 1.1 } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ width: 88 }}>Reference</TableCell>
                  <TableCell>Caption</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {[
                  ["Table 1", `Experiment Configuration (§04) — Specification of ${experimentId} including material, scan rate, model, dataset, and timing.`],
                  ["Table 2", `Input Parameters (§06) — Full input spec for ${result.material_id} at ${result.scan_rate_mVs} mV/s including electrode type and electrolyte.`],
                  ["Figure 1", `Predicted CV Curve (§07) — Cyclic voltammogram for ${result.material_id} at ${result.scan_rate_mVs} mV/s using ${modelConfig.display} (${result.n_points} data points).`],
                  ["Table 3", `Electrochemical Results (§08) — Quantitative metrics from the predicted CV: peak currents, integral area, symmetry factor, charge storage index.`],
                  ["Table 4", `ML Performance (§10) — Validation, test-SR, and test-MAT RMSE, R², and MAE for ${modelConfig.display}.`],
                  ["Table 5", `Report Metadata (§17) — Platform version, model architecture, dataset version, and generation timestamp for ${experimentId}.`],
                ].map(([ref, cap]) => (
                  <TableRow key={ref} sx={{ "&:last-child td": { border: 0 } }}>
                    <TableCell sx={{ fontFamily: "JetBrains Mono", fontWeight: 700, color: "#00d4ff", fontSize: "0.72rem" }}>{ref}</TableCell>
                    <TableCell sx={{ color: "rgba(255,255,255,0.48)", fontStyle: "italic" }}>{cap}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>

          {/* ── §16 EXPERIMENT STATUS ─────────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="16" title="Experiment Status" icon={<CheckCircleRounded />} />
            <Grid container spacing={1.25}>
              {[
                "Prediction Completed",
                "Model Loaded",
                `${result.n_points} Points Generated`,
                "Inference Successful",
                "Report Generated",
                "Overall Status: Complete",
              ].map((label) => (
                <Grid item xs={12} sm={6} md={4} key={label}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, p: 1.5, borderRadius: 1.5, background: "rgba(16,185,129,0.04)", border: "1px solid rgba(16,185,129,0.14)" }}>
                    <CheckCircleRounded sx={{ fontSize: 15, color: "#10b981", flexShrink: 0 }} />
                    <Typography sx={{ color: "rgba(255,255,255,0.68)", fontSize: "0.77rem", fontFamily: "JetBrains Mono" }}>{label}</Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>

          {/* ── §17 REPORT METADATA ───────────────────────────────────── */}
          <Box sx={{ mb: 4.5 }} className="report-print-area">
            <SecHeader num="17" title="Report Metadata" icon={<DescriptionRounded />} />
            <Box sx={{ p: 2, borderRadius: 2, background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <Grid container spacing={1.5}>
                {[
                  ["Platform Version", "v2.0.0"],
                  ["Report Version", "v18s-2.0"],
                  ["Dataset Version", "ZnO-CV-v1.0"],
                  ["Model Version", "Production"],
                  ["Generation Timestamp", generatedAt.toISOString()],
                  ["Experiment ID", experimentId],
                  ["Application", "ZnO Supercapacitor AI Platform"],
                  ["Architecture", `${modelConfig.architecture.slice(0, 52)}…`],
                ].map(([l, v]) => (
                  <Grid item xs={12} sm={6} key={l}>
                    <KVRow label={l} value={v} />
                  </Grid>
                ))}
              </Grid>
            </Box>
          </Box>

          {/* ── §18 EXPORT OPTIONS ────────────────────────────────────── */}
          <Box sx={{ mb: 3 }} className="report-print-area">
            <SecHeader num="18" title="Export Options" icon={<DownloadRounded />} />
            <Box sx={{ p: 2.5, borderRadius: 2, background: "rgba(0,0,0,0.15)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <Typography sx={{ color: "rgba(255,255,255,0.32)", fontSize: "0.73rem", mb: 2 }}>
                All exports preserve the complete 18-section report including tables, the CV chart, figure captions, electrochemical metrics, and ML performance data.
              </Typography>
              <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
                <GlowButton variant="outlined" startIcon={<PrintRounded sx={{ fontSize: 15 }} />} onClick={handlePrint} glowColor="#10b981" sx={{ borderColor: "rgba(16,185,129,0.3)", color: "#10b981", fontSize: "0.78rem" }}>
                  Print Report
                </GlowButton>
                <GlowButton variant="outlined" startIcon={<PrintRounded sx={{ fontSize: 15 }} />} onClick={handlePrint} glowColor="#00d4ff" sx={{ borderColor: "rgba(0,212,255,0.3)", color: "#00d4ff", fontSize: "0.78rem" }}>
                  Export PDF
                </GlowButton>
                <GlowButton variant="outlined" startIcon={<DownloadRounded sx={{ fontSize: 15 }} />} onClick={handleDownload} glowColor="#a78bfa" sx={{ borderColor: "rgba(167,139,250,0.3)", color: "#a78bfa", fontSize: "0.78rem" }}>
                  Download HTML
                </GlowButton>
              </Box>
              <Typography sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.65rem", mt: 1.5, fontStyle: "italic" }}>
                Print / Export PDF: Opens browser print dialog — select "Save as PDF" for PDF export. Download HTML: Saves standalone report with embedded SVG chart.
              </Typography>
            </Box>
          </Box>

          <Divider sx={{ borderColor: "rgba(255,255,255,0.05)", mt: 4, mb: 2 }} />
          <Typography variant="caption" sx={{ display: "block", textAlign: "center", color: "rgba(255,255,255,0.13)", fontSize: "0.6rem", pb: 2 }}>
            ZnO Supercapacitor AI Platform · Electrochemical Machine Learning Research · {experimentId}
          </Typography>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
