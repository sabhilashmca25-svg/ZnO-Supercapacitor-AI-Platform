import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import path from "path";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "prompt",
      devOptions: { enabled: true, type: "module" },
      /* manifest.json is hand-crafted in public/ — no auto-generation needed */
      manifest: false,
      workbox: {
        // Bundle exceeds Workbox's 2 MiB default — raise limit so the main
        // JS chunks are pre-cached by the service worker.
        maximumFileSizeToCacheInBytes: 8 * 1024 * 1024,
        globPatterns: ["**/*.{js,css,html,ico,png,svg,woff2}"],
        runtimeCaching: [
          {
            urlPattern: /\/api\/v1\//,
            handler: "NetworkFirst",
            options: {
              cacheName: "api-cache",
              networkTimeoutSeconds: 10,
              expiration: { maxEntries: 50, maxAgeSeconds: 300 },
            },
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    port: 5173,
    host: true,   // expose on 0.0.0.0 so phones on same WiFi can reach it
    // Prevent browsers from caching dev-mode modules — ensures every hard
    // refresh always loads the latest source files from Vite.
    headers: {
      "Cache-Control": "no-store",
    },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    // Raise the warning threshold — our Plotly-heavy bundle is intentionally large
    chunkSizeWarningLimit: 2048,
    rollupOptions: {
      output: {
        // Manual chunks: separate vendor libraries so they can be cached
        // independently and not re-downloaded when app code changes.
        manualChunks(id: string) {
          const nid = id.replace(/\\/g, "/");
          // Plotly is very large (~3 MB) — isolate so it only loads when charts render
          if (nid.includes("/plotly") || nid.includes("/react-plotly")) {
            return "vendor-plotly";
          }
          // All other node_modules in one chunk — avoids React chunk ordering race
          // where libraries using useSyncExternalStore load before React itself
          if (nid.includes("/node_modules/")) {
            return "vendor";
          }
        },
      },
    },
  },
});
