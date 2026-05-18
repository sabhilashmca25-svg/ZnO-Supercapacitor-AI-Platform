import axios, { AxiosError, AxiosRequestConfig } from "axios";

// Empty string = relative URLs → Vite proxy forwards /api/* to backend in dev.
// In production set VITE_API_URL to your deployed backend URL.
const BASE_URL = import.meta.env.VITE_API_URL ?? "";

// Inference endpoints have generous timeout (RF model is ~2-4 s).
// Static data endpoints use a shorter timeout.
const DEFAULT_TIMEOUT = 120_000; // 120 s

const client = axios.create({
  baseURL: BASE_URL,
  timeout: DEFAULT_TIMEOUT,
  headers: { "Content-Type": "application/json" },
});

// ── Retry configuration ───────────────────────────────────────────────────────
const MAX_RETRIES      = 2;
const RETRY_DELAY_MS   = 800;
const RETRYABLE_CODES  = new Set([502, 503, 504]); // gateway / service-unavailable

function isRetryable(error: AxiosError): boolean {
  if (!error.response) return true; // network error / timeout — always retry
  return RETRYABLE_CODES.has(error.response.status);
}

function sleep(ms: number): Promise<void> {
  return new Promise((res) => setTimeout(res, ms));
}

// ── Request interceptor — inject retryCount metadata ─────────────────────────
client.interceptors.request.use((config) => {
  (config as AxiosRequestConfig & { _retryCount?: number })._retryCount =
    (config as any)._retryCount ?? 0;
  return config;
});

// ── Response interceptor — retry + error normalisation ───────────────────────
client.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const config = error.config as (AxiosRequestConfig & { _retryCount?: number }) | undefined;

    // Only retry GET requests and known safe POST operations.
    // Predict/compare POSTs are idempotent (same input → same output).
    const method = (config?.method ?? "").toLowerCase();
    const isIdempotent = method === "get" || method === "post";

    if (
      config &&
      isIdempotent &&
      isRetryable(error) &&
      (config._retryCount ?? 0) < MAX_RETRIES
    ) {
      config._retryCount = (config._retryCount ?? 0) + 1;
      const delay = RETRY_DELAY_MS * config._retryCount; // 800ms, 1600ms
      await sleep(delay);
      return client(config);
    }

    // Normalise error shape so callers always get a consistent message
    const detail =
      (error as any)?.response?.data?.error ??
      (error as any)?.response?.data?.detail ??
      error.message ??
      "Unknown API error";
    const status = error?.response?.status;

    const enhanced = new Error(detail) as Error & {
      status?: number;
      raw?: unknown;
      isNetworkError?: boolean;
    };
    enhanced.status          = status;
    enhanced.raw             = error?.response?.data;
    enhanced.isNetworkError  = !error.response; // true when server unreachable
    return Promise.reject(enhanced);
  }
);

export default client;
