import React, { useState, useEffect } from "react";
import { Box, Typography, alpha } from "@mui/material";
import {
  WifiOffRounded,
  CheckCircleRounded,
  CancelRounded,
} from "@mui/icons-material";
import { motion, AnimatePresence } from "framer-motion";

const WORKS_OFFLINE = [
  "About Research page",
  "Model Encyclopedia",
  "Previously cached pages",
  "Saved experiment reports",
];

const NEEDS_NETWORK = [
  "CV predictions (backend required)",
  "New experiment reports",
  "Live leaderboard data",
];

export default function OfflineBanner() {
  const [offline, setOffline] = useState(!navigator.onLine);

  useEffect(() => {
    const goOffline = () => setOffline(true);
    const goOnline = () => setOffline(false);
    window.addEventListener("offline", goOffline);
    window.addEventListener("online", goOnline);
    return () => {
      window.removeEventListener("offline", goOffline);
      window.removeEventListener("online", goOnline);
    };
  }, []);

  return (
    <AnimatePresence>
      {offline && (
        <motion.div
          key="offline-banner"
          initial={{ opacity: 0, y: 80 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 80 }}
          transition={{ type: "spring", stiffness: 280, damping: 30 }}
          style={{
            position: "fixed",
            bottom: 24,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 2000,
            width: "min(440px, calc(100vw - 32px))",
          }}
        >
          <Box
            sx={{
              background: "rgba(10,16,28,0.96)",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(251,191,36,0.3)",
              borderRadius: 3,
              boxShadow: "0 0 40px rgba(0,0,0,0.6), 0 0 20px rgba(251,191,36,0.05)",
              p: 2.5,
            }}
          >
            {/* Header */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, mb: 2 }}>
              <Box
                sx={{
                  width: 32,
                  height: 32,
                  borderRadius: "50%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: alpha("#fbbf24", 0.12),
                  border: "1px solid rgba(251,191,36,0.3)",
                  flexShrink: 0,
                }}
              >
                <WifiOffRounded sx={{ fontSize: 16, color: "#fbbf24" }} />
              </Box>
              <Box>
                <Typography sx={{ color: "#fbbf24", fontWeight: 700, fontSize: "0.88rem", lineHeight: 1.2 }}>
                  No Internet Connection
                </Typography>
                <Typography sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.7rem" }}>
                  Some features are available offline
                </Typography>
              </Box>
            </Box>

            {/* Status grid */}
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1.5 }}>
              {/* What works */}
              <Box>
                <Typography sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", fontWeight: 700, letterSpacing: "0.08em", mb: 0.75, textTransform: "uppercase" }}>
                  Available offline
                </Typography>
                {WORKS_OFFLINE.map((item) => (
                  <Box key={item} sx={{ display: "flex", alignItems: "flex-start", gap: 0.75, mb: 0.5 }}>
                    <CheckCircleRounded sx={{ fontSize: 12, color: "#10b981", mt: "2px", flexShrink: 0 }} />
                    <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.68rem", lineHeight: 1.4 }}>
                      {item}
                    </Typography>
                  </Box>
                ))}
              </Box>

              {/* What doesn't work */}
              <Box>
                <Typography sx={{ color: "rgba(255,255,255,0.3)", fontSize: "0.62rem", fontWeight: 700, letterSpacing: "0.08em", mb: 0.75, textTransform: "uppercase" }}>
                  Requires network
                </Typography>
                {NEEDS_NETWORK.map((item) => (
                  <Box key={item} sx={{ display: "flex", alignItems: "flex-start", gap: 0.75, mb: 0.5 }}>
                    <CancelRounded sx={{ fontSize: 12, color: "rgba(239,68,68,0.7)", mt: "2px", flexShrink: 0 }} />
                    <Typography sx={{ color: "rgba(255,255,255,0.35)", fontSize: "0.68rem", lineHeight: 1.4 }}>
                      {item}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Box>

            {/* Footer note */}
            <Typography sx={{ color: "rgba(255,255,255,0.2)", fontSize: "0.62rem", mt: 1.75, borderTop: "1px solid rgba(255,255,255,0.06)", pt: 1.25 }}>
              This notice will disappear automatically when connectivity returns.
            </Typography>
          </Box>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
