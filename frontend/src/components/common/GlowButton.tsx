import React from "react";
import { Button, CircularProgress, type ButtonProps } from "@mui/material";
import { motion } from "framer-motion";
import { alpha } from "@mui/material/styles";

interface GlowButtonProps extends ButtonProps {
  loading?: boolean;
  glow?: boolean;
  glowColor?: string;
}

export default function GlowButton({
  children,
  loading = false,
  glow = true,
  glowColor = "#00d4ff",
  disabled,
  sx,
  ...rest
}: GlowButtonProps) {
  return (
    <Button
      component={motion.button}
      whileHover={!disabled && !loading ? { scale: 1.02, y: -1 } : undefined}
      whileTap={!disabled && !loading ? { scale: 0.98 } : undefined}
      disabled={disabled || loading}
      startIcon={loading ? <CircularProgress size={16} sx={{ color: "inherit" }} /> : rest.startIcon}
      sx={{
        position: "relative",
        fontWeight: 700,
        letterSpacing: "0.05em",
        fontSize: "0.85rem",
        px: 3,
        py: 1.1,
        borderRadius: "10px",
        transition: "all 0.2s ease",
        ...(glow && !disabled && !loading && {
          boxShadow: `0 0 16px ${alpha(glowColor, 0.25)}`,
          "&:hover": {
            boxShadow: `0 0 28px ${alpha(glowColor, 0.5)}, 0 0 60px ${alpha(glowColor, 0.15)}`,
          },
        }),
        ...sx,
      }}
      {...rest}
    >
      {loading ? "Processing…" : children}
    </Button>
  );
}
