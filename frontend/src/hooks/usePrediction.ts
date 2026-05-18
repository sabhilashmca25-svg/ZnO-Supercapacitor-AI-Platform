import { useAppDispatch, useAppSelector } from "./useAppDispatch";
import { runPrediction, runComparison, clearResults, clearComparison } from "../store/predictionSlice";
import type { PredictionRequest, CompareRequest } from "../types";

export function usePrediction() {
  const dispatch = useAppDispatch();
  const { results, latestKey, loading, error } = useAppSelector((s) => s.prediction);

  const predict = (req: PredictionRequest) => dispatch(runPrediction(req));
  const clear = () => dispatch(clearResults());

  // Use latestKey so re-running a previous combo always shows the freshly-run result
  // (Object.values().at(-1) fails here because it returns last INSERTED, not last RUN)
  const latestResult = latestKey ? (results[latestKey] ?? null) : null;

  return { predict, clear, results, latestResult, loading, error };
}

export function useComparison() {
  const dispatch = useAppDispatch();
  const { compareResult, comparing, error } = useAppSelector((s) => s.prediction);

  const compare = (req: CompareRequest) => dispatch(runComparison(req));
  const clear = () => dispatch(clearComparison());

  return { compare, clear, compareResult, comparing, error };
}
