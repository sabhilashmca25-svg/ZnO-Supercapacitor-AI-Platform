import { useState, useEffect, useCallback } from "react";
import { getHealth } from "../api/endpoints";
import type { HealthResponse } from "../types";

export function useHealth(pollInterval = 30_000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealth(data);
      setError(null);
    } catch {
      setError("Backend unreachable");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetch();
    if (pollInterval > 0) {
      const id = setInterval(fetch, pollInterval);
      return () => clearInterval(id);
    }
  }, [fetch, pollInterval]);

  return { health, loading, error, refetch: fetch };
}
