/**
 * Electrochemistry & ML glossary — used by GlossaryTooltip component.
 * Each entry has a short term, a one-line definition, and an optional
 * extended explanation for deeper context.
 */

export interface GlossaryEntry {
  term: string;
  short: string;
  extended?: string;
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  cv: {
    term: "Cyclic Voltammetry (CV)",
    short: "An electrochemical technique that sweeps electrode potential back and forth to measure current response, revealing redox reactions and charge storage capacity.",
    extended: "The voltage is swept from a lower limit to an upper limit (anodic sweep) and back (cathodic sweep) at a fixed scan rate. The resulting I–V curve (CV curve) encodes the electrode's electrochemical behaviour.",
  },
  scan_rate: {
    term: "Scan Rate (mV/s)",
    short: "Speed of the voltage sweep. Lower rates allow more ion diffusion (higher capacitance). Higher rates reveal rate capability but reduce effective capacitance.",
    extended: "According to the Randles–Ševčík equation, for diffusion-controlled processes I_peak ∝ ν^0.5. For surface-confined (capacitive) processes, I ∝ ν. The scan rate dependence distinguishes Faradaic from EDLC behaviour.",
  },
  anodic: {
    term: "Anodic (Oxidation)",
    short: "The forward sweep where the electrode potential increases. Anodic current is positive — electrons flow from electrode to electrolyte (oxidation).",
    extended: "In an anodic sweep, cations (e.g. Na⁺ from Na₂SO₄ electrolyte) adsorb onto the electrode surface and electrons are extracted. The anodic peak current (I_a) marks the maximum oxidation rate.",
  },
  cathodic: {
    term: "Cathodic (Reduction)",
    short: "The reverse sweep where potential decreases. Cathodic current is negative — electrons flow from electrolyte to electrode (reduction).",
    extended: "In a cathodic sweep, cations desorb and electrons are returned to the electrode. The cathodic peak current (I_c) marks the maximum reduction rate. Ideal capacitors have |I_a| = |I_c|.",
  },
  edlc: {
    term: "EDLC (Electric Double-Layer Capacitor)",
    short: "A supercapacitor that stores charge electrostatically at the electrode–electrolyte interface without Faradaic reactions. Gives rectangular CV curves.",
    extended: "Charge is stored by physical adsorption of ions. No charge transfer occurs, so EDLC CVs are nearly rectangular with rapid, reversible response. ZnO electrodes often exhibit a mix of EDLC and pseudocapacitive behaviour.",
  },
  faradaic: {
    term: "Faradaic Process",
    short: "Electrochemical process involving actual charge transfer (redox reactions). Produces distinct current peaks in CV curves — e.g., Mn²⁺↔Mn⁴⁺ or Co²⁺↔Co³⁺.",
    extended: "Faradaic contributions increase apparent capacitance (pseudocapacitance) but also cause CV asymmetry and scan rate sensitivity. Excessive Faradaic processes reduce cycle stability.",
  },
  symmetry_factor: {
    term: "Symmetry Factor",
    short: "Ratio |I_anodic| / |I_cathodic|. Ideal = 1.000 for a perfectly reversible capacitive electrode.",
    extended: "Deviations from 1.0 indicate Faradaic contributions, diffusion limitations, or irreversible reactions. Values > 1 mean more charge stored during oxidation; < 1 means more during reduction.",
  },
  capacitance: {
    term: "Specific Capacitance",
    short: "Charge stored per unit electrode area (mF/cm²) or mass (F/g). Derived from CV as C = ∫I dV / (2·ν·ΔV).",
    extended: "Higher capacitance = more energy stored at the same voltage. The experimental ZnO-based electrode materials in this study show varying capacitance levels, influenced by their specific surface area and any pseudocapacitive contributions.",
  },
  charge_storage_index: {
    term: "Charge Storage Index (CSI)",
    short: "Estimated relative capacitance C = ∮I dV / (2·ν·ΔV). A normalised proxy for energy storage — not absolute capacitance without electrode geometry.",
    extended: "The CSI allows comparing relative energy storage capacity across scan rates and materials. Since electrode area/mass is not normalised here, the absolute value is a platform-specific relative indicator.",
  },
  rmse: {
    term: "RMSE (Root Mean Square Error)",
    short: "Average prediction error in µA. Lower is better. RMSE = √(mean((I_pred − I_true)²)).",
    extended: "RMSE penalises large errors more than MAE. In CV prediction, 26–50 µA RMSE on 651-point sweeps with peak currents of ±200–600 µA represents <15% relative error — considered excellent for electrochemical ML.",
  },
  r2: {
    term: "R² (Coefficient of Determination)",
    short: "Fraction of variance in the measured current explained by the model. R² = 1.0 is perfect; R² > 0.96 is excellent for CV prediction.",
    extended: "R² = 1 − (SS_res / SS_tot). For cross-material extrapolation (Test-MAT), R² ≥ 0.96 indicates the model has learned transferable CV physics rather than just memorising training materials.",
  },
  zero_shot: {
    term: "Zero-Shot Extrapolation",
    short: "Predicting behaviour of NM4 without any NM4 training examples — purely from patterns learned on NM1–NM3.",
    extended: "Zero-shot is the hardest generalisation test. GRU achieves R² = 0.9751 on NM4 despite never seeing it — demonstrating that recurrent models learn CV topology from the sweep sequence rather than material-specific fingerprints.",
  },
  integral_area: {
    term: "Integral Area (µA·V)",
    short: "Area enclosed by the CV loop ∝ total charge stored per cycle (Q). Larger area = higher energy storage.",
    extended: "Integral area = ∮ I dV = |∫_anodic I dV + ∫_cathodic I dV| — trapezoidal integration over voltage axis (not array indices). Proportional to specific capacitance via C = ∮I dV / (2·ν·ΔV). Scan rate compression reduces integral area at high rates.",
  },
  peak_separation: {
    term: "Peak Separation (ΔV)",
    short: "Voltage difference between anodic and cathodic peak positions. Small ΔV → surface-confined reaction (capacitor). Large ΔV → diffusion-controlled Faradaic process.",
    extended: "For ideal EDLC: ΔV → 0 (no distinct peaks). For battery-type materials: ΔV ~ 0.1–0.5 V. The Nicholson method uses ΔV to estimate rate constants for quasi-reversible Faradaic reactions.",
  },
  nanocomposite: {
    term: "ZnO Nanocomposite",
    short: "One of the four experimental ZnO-based electrode material groups (NM1–NM4) used in this study. Each group exhibits a distinct cyclic voltammetry profile.",
    extended: "NM1 is the ZnO baseline electrode. NM2, NM3, and NM4 are additional experimental material groups with distinct CV profiles. Their specific chemical compositions are not defined in the source dataset — they are identified only by their dataset labels (NM2, NM3, NM4).",
  },
};
