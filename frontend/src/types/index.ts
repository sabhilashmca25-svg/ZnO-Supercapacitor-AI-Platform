// ── Core API Response Types ─────────────────────────────────────────────

export interface PredictionRequest {
  model_name: string;
  material_id: string;
  scan_rate_mVs: number;
}

export interface PredictionStats {
  peak_anodic_uA: number;
  peak_cathodic_uA: number;
  current_range_uA: number;
  integral_area: number;
}

export interface PredictionResponse {
  model_name: string;
  material_id: string;
  scan_rate_mVs: number;
  n_points: number;
  potential_V: number[];
  predicted_current_uA: number[];
  statistics: PredictionStats;
}

export interface CompareRequest {
  models: string[];
  material_id: string;
  scan_rate_mVs: number;
}

export interface ModelPrediction {
  predicted_current_uA: number[];
  peak_anodic_uA: number;
  peak_cathodic_uA: number;
  current_range_uA: number;
  integral_area: number;
}

export interface CompareResponse {
  material_id: string;
  scan_rate_mVs: number;
  n_points: number;
  potential_V: number[];
  predictions: Record<string, ModelPrediction>;
  models_failed: string[];
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: string[];
  scalers_ok: boolean;
  environment: string;
}

export interface RootResponse {
  name: string;
  version: string;
  environment: string;
  status: string;
  models_loaded: string[];
  docs: Record<string, string>;
}

// ── Benchmark / Leaderboard Types ────────────────────────────────────────

export interface LeaderboardEntry {
  rank: number;
  model_id: string;
  model_display: string;
  rmse_val_uA: number;
  r2_val: number;
  rmse_test_sr_uA: number;
  r2_test_sr: number;
  rmse_test_mat_uA: number;
  r2_test_mat: number;
  size_MB: number;
  inference_ms: number | null;
}

export interface ComparisonRow {
  model_id: string;
  model_display: string;
  partition: string;
  rmse_uA: number;
  r2: number;
  mae_uA: number | null;
}

export interface BenchmarkSummary {
  generated: string;
  project: string;
  key_findings: {
    best_by_val_rmse: string;
    best_by_testmat_r2: string;
    best_for_deployment: string;
    worst_model: string;
  };
  model_size_MB: Record<string, number>;
  train_time_minutes: Record<string, number>;
  complexity: Record<string, number>;
  metrics?: Record<string, Record<string, { RMSE_uA: number; MAE_uA: number; MaxErr_uA: number; R2: number }>>;
  // legacy optional fields
  val_rmse_range?: [number, number];
  test_mat_r2_range?: [number, number];
  n_models?: number;
  n_partitions?: number;
}

export interface TrainingHistory {
  model_id: string;
  loss: number[];
  mae: number[];
  val_loss: number[];
  val_mae: number[];
  learning_rate: number[];
  epochs_ran: number;
  best_epoch: number;
  best_epoch_1indexed: number;
}

// ── Metrics Types ─────────────────────────────────────────────────────────

export interface MetricPartition {
  partition: string;
  rmse_uA: number;
  r2: number;
  mae_uA: number | null;
}

export interface ModelMetrics {
  model_id: string;
  model_display: string;
  metrics: MetricPartition[];
}

// ── Model Info Types ──────────────────────────────────────────────────────

export interface ModelInfo {
  model_id: string;
  model_name: string;
  loaded: boolean;
  size_mb: number | null;
  description?: string;
}

// ── UI / App State Types ───────────────────────────────────────────────────

export interface PredictionState {
  results: Record<string, PredictionResponse>;
  /** Key of the most-recently EXECUTED prediction (not just most-recently inserted).
   *  Needed because re-running an existing key doesn't change Object.values() order. */
  latestKey: string | null;
  compareResult: CompareResponse | null;
  loading: boolean;
  comparing: boolean;
  error: string | null;
}

export interface UIState {
  sidebarOpen: boolean;
  theme: "dark";
}

// ── Model Config (frontend metadata) ────────────────────────────────────

export interface ModelConfig {
  id: string;
  display: string;
  shortName: string;
  color: string;
  type: "tree" | "boost" | "deep";
  description: string;
  architecture: string;
  strengths: string[];
  weaknesses: string[];
  deployScore: number;
  size: string;
}

// ── Experimental & Validation Types ──────────────────────────────────────

export interface ExperimentalCurveResponse {
  material_id: string;
  scan_rate_mVs: number;
  n_points: number;
  potential_V: number[];
  actual_current_uA: number[];
  split: string;
  cycle_id: number;
}

export interface ValidationRequest {
  model_name: string;
  material_id: string;
  scan_rate_mVs: number;
}

export interface ValidationMetrics {
  rmse_uA: number;
  mae_uA: number;
  max_error_uA: number;
  r2: number;
}

export interface ValidationCompareResponse {
  model_name: string;
  material_id: string;
  scan_rate_mVs: number;
  split: string;
  experimental_potential_V: number[];
  experimental_current_uA: number[];
  predicted_potential_V: number[];
  predicted_current_uA: number[];
  residual_potential_V: number[];
  residual_uA: number[];
  abs_residual_uA: number[];
  metrics: ValidationMetrics;
}

export interface PerGroupRow {
  nm_id: string;
  scan_rate_mVs: number;
  rmse_uA: number;
  r2: number;
  partition: string;
}

export interface PerGroupResponse {
  model_name: string;
  rows: PerGroupRow[];
}
