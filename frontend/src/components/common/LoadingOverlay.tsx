import React from "react";
import { Box, Typography } from "@mui/material";
import { motion } from "framer-motion";

interface LoadingOverlayProps {
  message?: string;
  subMessage?: string;
  fullScreen?: boolean;
}

export default function LoadingOverlay({
  message = "Running inference…",
  subMessage,
  fullScreen = false,
}: LoadingOverlayProps) {
  return (
    <Box
      component={motion.div}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 2.5,
        ...(fullScreen
          ? { position: "fixed", inset: 0, zIndex: 9999, backgroundColor: "rgba(7,11,20,0.85)", backdropFilter: "blur(8px)" }
          : { minHeight: 200, width: "100%" }),
      }}
    >
      {/* Animated rings */}
      <Box sx={{ position: "relative", width: 60, height: 60 }}>
        {[0, 1, 2].map((i) => (
          <Box
            key={i}
            component={motion.div}
            animate={{ rotate: 360 }}
            transition={{
              duration: 1.4 + i * 0.3,
              ease: "linear",
              repeat: Infinity,
              direction: i % 2 === 0 ? "normal" : "reverse",
            }}
            sx={{
              position: "absolute",
              borderRadius: "50%",
              border: "2px solid transparent",
              borderTopColor: i === 0 ? "#00d4ff" : i === 1 ? "#7c3aed" : "#10b981",
              inset: i * 8,
              opacity: 1 - i * 0.2,
            }}
          />
        ))}
        {/* Center dot */}
        <Box
          component={motion.div}
          animate={{ scale: [1, 1.3, 1], opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          sx={{
            position: "absolute",
            inset: "50%",
            transform: "translate(-50%, -50%)",
            width: 8,
            height: 8,
            borderRadius: "50%",
            backgroundColor: "#00d4ff",
            boxShadow: "0 0 10px rgba(0,212,255,0.8)",
          }}
        />
      </Box>

      <Box sx={{ textAlign: "center" }}>
        <Typography
          variant="body2"
          sx={{ color: "#00d4ff", fontWeight: 600, letterSpacing: "0.05em" }}
        >
          {message}
        </Typography>
        {subMessage && (
          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.35)", display: "block", mt: 0.5 }}>
            {subMessage}
          </Typography>
        )}
      </Box>
    </Box>
  );
}
