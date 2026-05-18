import React from "react";
import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { store } from "./store";
import "./index.css";

// ── Version stamp — if this appears in console, NEW code is loading ────────
console.log("%c✅ ZnO Platform v2 — NEW CODE LOADED", "color: lime; font-size: 16px; font-weight: bold");

// ── Dev-only: evict stale service workers so code changes always take effect ──
// In PRODUCTION this block is skipped — the SW provides offline caching.
if (import.meta.env.DEV && "serviceWorker" in navigator) {
  navigator.serviceWorker.getRegistrations().then((registrations) => {
    registrations.forEach((reg) => reg.unregister());
  });
  if ("caches" in window) {
    caches.keys().then((keys) => keys.forEach((key) => caches.delete(key)));
  }
}

// NOTE: StrictMode is intentionally omitted — React 18 StrictMode double-invokes
// effects in dev mode which breaks Framer Motion's animation state machine,
// causing motion components that start at opacity:0 to stay permanently invisible.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <Provider store={store}>
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <App />
    </BrowserRouter>
  </Provider>
);
