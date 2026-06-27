import { useState, useCallback } from "react";

// ── Schema ────────────────────────────────────────────────────────────────────

export interface ExperimentConfidence {
  label: string;
  color: string;
}

export interface ExperimentRecord {
  id: string;
  timestamp: string;
  material: string;
  model: string;
  scanRate: number;
  confidence: ExperimentConfidence;
  peakAnodic: number;
  peakCathodic: number;
  maxCurrent: number;
  minCurrent: number;
  rmse: number;
  isZeroShot: boolean;
  reportId?: string;
}

// ── Storage ───────────────────────────────────────────────────────────────────

const STORAGE_KEY = "recentExperiments";
const MAX_RECORDS = 10;

function readFromStorage(): ExperimentRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      localStorage.removeItem(STORAGE_KEY);
      return [];
    }
    return parsed as ExperimentRecord[];
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return [];
  }
}

function writeToStorage(records: ExperimentRecord[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Ignore storage quota errors silently
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useRecentExperiments() {
  const [records, setRecords] = useState<ExperimentRecord[]>(readFromStorage);

  const save = useCallback((entry: Omit<ExperimentRecord, "id" | "timestamp">) => {
    const id = `exp_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const record: ExperimentRecord = { ...entry, id, timestamp: new Date().toISOString() };
    setRecords((prev) => {
      const next = [record, ...prev].slice(0, MAX_RECORDS);
      writeToStorage(next);
      return next;
    });
    return id;
  }, []);

  const attachReportId = useCallback((experimentId: string, reportId: string) => {
    setRecords((prev) => {
      const next = prev.map((r) => r.id === experimentId ? { ...r, reportId } : r);
      writeToStorage(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setRecords([]);
  }, []);

  return { records, save, attachReportId, clear };
}
