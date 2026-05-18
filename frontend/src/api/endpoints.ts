import client from "./client";
import type {
  HealthResponse,
  RootResponse,
  PredictionRequest,
  PredictionResponse,
  CompareRequest,
  CompareResponse,
  LeaderboardEntry,
  ComparisonRow,
  BenchmarkSummary,
  ModelMetrics,
  ModelInfo,
  ExperimentalCurveResponse,
  ValidationRequest,
  ValidationCompareResponse,
  PerGroupResponse,
} from "../types";

// ── System ────────────────────────────────────────────────────────────────

export const getRoot = () =>
  client.get<RootResponse>("/api/v1/health").then((r) => r.data);

export const getHealth = () =>
  client.get<HealthResponse>("/api/v1/health").then((r) => r.data);

export const getModels = () =>
  client.get<{ models: ModelInfo[] }>("/api/v1/models").then((r) => r.data);

// ── Prediction ────────────────────────────────────────────────────────────

export const predict = (payload: PredictionRequest) =>
  client.post<PredictionResponse>("/api/v1/predict", payload).then((r) => r.data);

export const compare = (payload: CompareRequest) =>
  client.post<CompareResponse>("/api/v1/compare", payload).then((r) => r.data);

// ── Analytics / Benchmarks ────────────────────────────────────────────────

export const getLeaderboard = () =>
  client.get<LeaderboardEntry[]>("/api/v1/benchmarks/leaderboard").then((r) => r.data);

export const getComparison = () =>
  client.get<ComparisonRow[]>("/api/v1/benchmarks/comparison").then((r) => r.data);

export const getBenchmarkSummary = () =>
  client.get<BenchmarkSummary>("/api/v1/benchmarks/summary").then((r) => r.data);

// ── Metrics ───────────────────────────────────────────────────────────────

export const getAllMetrics = () =>
  client.get<ModelMetrics[]>("/api/v1/metrics").then((r) => r.data);

export const getModelMetrics = (modelId: string) =>
  client.get<ModelMetrics>(`/api/v1/metrics/${modelId}`).then((r) => r.data);

// ── Model management ──────────────────────────────────────────────────────

export const getModelsStatus = () =>
  client.get("/api/v1/models/status").then((r) => r.data);

export const loadModel = (modelId: string) =>
  client.post(`/api/v1/models/${modelId}/load`).then((r) => r.data);

// ── Training History ──────────────────────────────────────────────────────

export const getAllTrainingHistories = () =>
  client.get("/api/v1/training-history").then((r) => r.data);

export const getTrainingHistory = (modelId: string) =>
  client.get(`/api/v1/training-history/${modelId}`).then((r) => r.data);

// ── Experimental Data ─────────────────────────────────────────────────────

export const getExperimentalCurve = (materialId: string, scanRate: number) =>
  client
    .get<ExperimentalCurveResponse>(`/api/v1/experimental/${materialId}/${scanRate}`)
    .then((r) => r.data);

// ── Validation ────────────────────────────────────────────────────────────

export const compareValidation = (payload: ValidationRequest) =>
  client
    .post<ValidationCompareResponse>("/api/v1/validation/compare", payload)
    .then((r) => r.data);

export const getPerGroupMetrics = (modelId: string) =>
  client
    .get<PerGroupResponse>(`/api/v1/validation/per-group/${modelId}`)
    .then((r) => r.data);
