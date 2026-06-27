import React from "react";
import {
  Dialog, DialogContent, Box, Typography, alpha,
} from "@mui/material";
import { SystemUpdateRounded } from "@mui/icons-material";
import { motion, AnimatePresence } from "framer-motion";
import { useRegisterSW } from "virtual:pwa-register/react";
import GlowButton from "../common/GlowButton";

export default function UpdateDialog() {
  const {
    needRefresh: [needRefresh, setNeedRefresh],
    updateServiceWorker,
  } = useRegisterSW({
    onRegistered(registration) {
      // Poll for updates every hour in production
      if (registration) {
        setInterval(() => registration.update(), 60 * 60 * 1000);
      }
    },
  });

  return (
    <AnimatePresence>
      {needRefresh && (
        <Dialog
          open
          maxWidth="xs"
          fullWidth
          onClose={() => setNeedRefresh(false)}
          PaperProps={{
            sx: {
              background: "rgba(10,16,28,0.97)",
              backdropFilter: "blur(24px)",
              border: "1px solid rgba(0,212,255,0.25)",
              borderRadius: 3,
              boxShadow: "0 0 60px rgba(0,212,255,0.08), 0 24px 48px rgba(0,0,0,0.6)",
              overflow: "visible",
            },
          }}
          slotProps={{
            backdrop: { sx: { backgroundColor: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" } },
          }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.88, y: 32 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 16 }}
            transition={{ type: "spring", stiffness: 320, damping: 28 }}
          >
            <DialogContent sx={{ p: 3.5 }}>
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: 2.5 }}>

                {/* Glowing icon */}
                <Box
                  sx={{
                    width: 56,
                    height: 56,
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    backgroundColor: alpha("#00d4ff", 0.1),
                    border: "1px solid rgba(0,212,255,0.3)",
                    boxShadow: "0 0 24px rgba(0,212,255,0.2)",
                  }}
                >
                  <SystemUpdateRounded sx={{ color: "#00d4ff", fontSize: 26 }} />
                </Box>

                <Box>
                  <Typography
                    variant="h6"
                    sx={{ color: "#f1f5f9", fontWeight: 700, mb: 0.75, fontSize: "1.05rem", letterSpacing: "-0.01em" }}
                  >
                    New Version Available
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.82rem", lineHeight: 1.6 }}
                  >
                    A newer version of the ZnO AI Platform is available.
                  </Typography>
                </Box>

                <Box sx={{ display: "flex", gap: 1.5, width: "100%", pt: 0.5 }}>
                  <GlowButton
                    variant="contained"
                    fullWidth
                    onClick={() => updateServiceWorker(true)}
                    glowColor="#00d4ff"
                    sx={{
                      background: "linear-gradient(135deg, #00d4ff 0%, rgba(0,180,220,0.85) 100%)",
                      color: "#070b14",
                      fontWeight: 700,
                      fontSize: "0.85rem",
                    }}
                  >
                    Update Now
                  </GlowButton>
                  <GlowButton
                    variant="outlined"
                    fullWidth
                    onClick={() => setNeedRefresh(false)}
                    glow={false}
                    sx={{
                      borderColor: "rgba(255,255,255,0.15)",
                      color: "rgba(255,255,255,0.45)",
                      "&:hover": { borderColor: "rgba(255,255,255,0.3)", color: "rgba(255,255,255,0.7)" },
                    }}
                  >
                    Later
                  </GlowButton>
                </Box>
              </Box>
            </DialogContent>
          </motion.div>
        </Dialog>
      )}
    </AnimatePresence>
  );
}
