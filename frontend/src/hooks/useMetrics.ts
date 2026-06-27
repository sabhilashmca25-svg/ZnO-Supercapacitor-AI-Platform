import { useState, useEffect } from "react";
import { getAllMetrics } from "../api/endpoints";
import type { ModelMetrics } from "../types";

const MODEL_DISPLAY: Record<string, string> = {
  rf: "Random Forest",
  lightgbm: "LightGBM",
  xgboost: "XGBoost",
  gru: "Stacked GRU",
  lstm: "Stacked LSTM",
  ann: "Dense ANN",
};

export function useMetrics() {
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
  const [featureImportance, setFeatureImportance] = useState<Record<string, Record<string, number>>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    getAllMetrics()
      .then((data: any) => {
        if (!mounted) return;

        // Normalize dict keyed by model_id to array of ModelMetrics
        const raw = Array.isArray(data) ? Object.fromEntries(data.map((d: any) => [d.model_id, d])) : data;
        const arr: ModelMetrics[] = Object.entries(raw).map(([model_id, m]: [string, any]) => ({
          model_id,
          model_display: m.model_display ?? MODEL_DISPLAY[model_id] ?? model_id.toUpperCase(),
          metrics: (m.metrics ?? []).map((p: any) => ({
            partition: p.partition ?? "",
            rmse_uA: Number(p.rmse_uA ?? p.RMSE_uA ?? 0),
            r2: Number(p.r2 ?? p.R2 ?? 0),
            mae_uA: p.mae_uA != null ? Number(p.mae_uA) : p.MAE_uA != null ? Number(p.MAE_uA) : null,
          })),
        }));

        // Extract feature importance for tree models
        const fi: Record<string, Record<string, number>> = {};
        Object.entries(raw).forEach(([model_id, m]: [string, any]) => {
          const imp = m.feature_importance;
          if (imp && typeof imp === "object" && Object.keys(imp).length > 0) {
            fi[model_id] = imp;
          }
        });

        setMetrics(arr);
        setFeatureImportance(fi);
      })
      .catch((err) => {
        if (!mounted) return;
        setError(err.message ?? "Failed to load metrics");
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, []);

  return { metrics, featureImportance, loading, error };
}
