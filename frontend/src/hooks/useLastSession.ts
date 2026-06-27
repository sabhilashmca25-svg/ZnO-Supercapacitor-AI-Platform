import { useState, useCallback } from "react";

// ── Schema ────────────────────────────────────────────────────────────────────

export interface LastSession {
  model: string;
  material: string;
  scanRate: number;
  lastExperimentId: string;
  timestamp: string;
}

// ── Storage ───────────────────────────────────────────────────────────────────

const STORAGE_KEY = "lastSession";

function readFromStorage(): LastSession | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as LastSession;
    if (!parsed.model || !parsed.material || !parsed.scanRate) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return parsed;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

function writeToStorage(session: LastSession): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  } catch {
    // Ignore storage quota errors silently
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useLastSession() {
  const [lastSession] = useState<LastSession | null>(readFromStorage);

  const save = useCallback((session: LastSession) => {
    writeToStorage(session);
  }, []);

  const clear = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return { lastSession, save, clear };
}
