import { configureStore } from "@reduxjs/toolkit";
import predictionReducer from "./predictionSlice";
import uiReducer from "./uiSlice";

export const store = configureStore({
  reducer: {
    prediction: predictionReducer,
    ui: uiReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
