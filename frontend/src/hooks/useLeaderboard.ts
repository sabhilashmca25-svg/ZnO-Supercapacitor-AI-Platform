import { useState, useEffect } from "react";
import { getLeaderboard, getBenchmarkSummary, getComparison } from "../api/endpoints";
import type { LeaderboardEntry, BenchmarkSummary, ComparisonRow } from "../types";

export function useLeaderboard() {
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [summary, setSummary] = useState<BenchmarkSummary | null>(null);
  const [comparison, setComparison] = useState<ComparisonRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);

    Promise.all([getLeaderboard(), getBenchmarkSummary(), getComparison()])
      .then(([lb, sm, cmp]) => {
        if (!mounted) return;
        setLeaderboard(lb);
        setSummary(sm);
        setComparison(cmp);
        setError(null);
      })
      .catch((err) => {
        if (!mounted) return;
        setError(err.message ?? "Failed to load benchmark data");
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => { mounted = false; };
  }, []);

  return { leaderboard, summary, comparison, loading, error };
}
