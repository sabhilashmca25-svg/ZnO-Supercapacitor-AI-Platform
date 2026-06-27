import React from "react";
import { Chip, alpha } from "@mui/material";
import { motion } from "framer-motion";
import { MODEL_COLORS } from "../../constants/models";

interface ModelBadgeProps {
  modelId: string;
  size?: "small" | "medium";
  selected?: boolean;
  onClick?: () => void;
}

const MODEL_SHORT: Record<string, string> = {
  rf: "RF",
  lightgbm: "LGB",
  xgboost: "XGB",
  gru: "GRU",
  lstm: "LSTM",
  ann: "ANN",
};

export default function ModelBadge({
  modelId,
  size = "small",
  selected = false,
  onClick,
}: ModelBadgeProps) {
  const color = MODEL_COLORS[modelId] ?? "#94a3b8";
  const label = MODEL_SHORT[modelId] ?? modelId.toUpperCase();

  return (
    <Chip
      component={motion.div}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.97 }}
      label={label}
      size={size}
      onClick={onClick}
      clickable={!!onClick}
      sx={{
        fontWeight: 700,
        letterSpacing: "0.06em",
        fontSize: size === "small" ? "0.65rem" : "0.75rem",
        height: size === "small" ? 22 : 28,
        backgroundColor: selected ? alpha(color, 0.2) : alpha(color, 0.08),
        color: color,
        border: `1px solid ${selected ? alpha(color, 0.5) : alpha(color, 0.25)}`,
        boxShadow: selected ? `0 0 10px ${alpha(color, 0.3)}` : "none",
        transition: "all 0.2s ease",
        cursor: onClick ? "pointer" : "default",
        "& .MuiChip-label": { px: size === "small" ? 0.8 : 1.2 },
        "&:hover": onClick
          ? {
              backgroundColor: alpha(color, 0.18),
              borderColor: alpha(color, 0.5),
              boxShadow: `0 0 12px ${alpha(color, 0.25)}`,
            }
          : {},
      }}
    />
  );
}
