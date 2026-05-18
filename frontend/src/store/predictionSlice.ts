import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { predict as apiPredict, compare as apiCompare } from "../api/endpoints";
import type { PredictionState, PredictionRequest, CompareRequest } from "../types";

// ── Async thunks ──────────────────────────────────────────────────────────

export const runPrediction = createAsyncThunk(
  "prediction/run",
  async (payload: PredictionRequest, { rejectWithValue }) => {
    try {
      return await apiPredict(payload);
    } catch (err: any) {
      return rejectWithValue(err.message ?? "Prediction failed");
    }
  }
);

export const runComparison = createAsyncThunk(
  "prediction/compare",
  async (payload: CompareRequest, { rejectWithValue }) => {
    try {
      return await apiCompare(payload);
    } catch (err: any) {
      return rejectWithValue(err.message ?? "Comparison failed");
    }
  }
);

// ── Slice ─────────────────────────────────────────────────────────────────

const initialState: PredictionState = {
  results: {},
  latestKey: null,
  compareResult: null,
  loading: false,
  comparing: false,
  error: null,
};

const predictionSlice = createSlice({
  name: "prediction",
  initialState,
  reducers: {
    clearResults: (state) => {
      state.results = {};
      state.latestKey = null;
      state.error = null;
    },
    clearComparison: (state) => {
      state.compareResult = null;
      state.error = null;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // ── Single prediction ────────────────────────────────────────────────
    builder
      .addCase(runPrediction.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(runPrediction.fulfilled, (state, action) => {
        state.loading = false;
        const key = `${action.payload.model_name}_${action.payload.material_id}_${action.payload.scan_rate_mVs}`;
        state.results[key] = action.payload;
        // Always track which key was most recently RUN (not just last inserted)
        state.latestKey = key;
      })
      .addCase(runPrediction.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // ── Comparison ───────────────────────────────────────────────────────
    builder
      .addCase(runComparison.pending, (state) => {
        state.comparing = true;
        state.error = null;
      })
      .addCase(runComparison.fulfilled, (state, action) => {
        state.comparing = false;
        state.compareResult = action.payload;
      })
      .addCase(runComparison.rejected, (state, action) => {
        state.comparing = false;
        state.error = action.payload as string;
      });
  },
});

export const { clearResults, clearComparison, clearError } = predictionSlice.actions;
export default predictionSlice.reducer;
