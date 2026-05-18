import React, { useEffect, useRef, useState } from "react";
import { Box, Card, CardContent, Typography, Skeleton } from "@mui/material";
import { motion } from "framer-motion";
import { alpha } from "@mui/material/styles";
import { cardVariants, cardHover, cardTap } from "../../animations/variants";

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  subtext?: string;
  icon?: React.ReactNode;
  color?: string;
  loading?: boolean;
  animate?: boolean;
  delay?: number;
}

export default function MetricCard({
  label,
  value,
  unit,
  subtext,
  icon,
  color = "#00d4ff",
  loading = false,
  animate = true,
  delay = 0,
}: MetricCardProps) {
  return (
    <motion.div
      variants={animate ? cardVariants : undefined}
      initial={animate ? "initial" : undefined}
      animate={animate ? "animate" : undefined}
      whileHover={cardHover}
      whileTap={cardTap}
      transition={{ delay }}
      style={{ height: "100%" }}
    >
      <Card
        sx={{
          height: "100%",
          background: `linear-gradient(135deg, rgba(15,25,35,0.9) 0%, rgba(10,16,28,0.95) 100%)`,
          border: `1px solid ${alpha(color, 0.2)}`,
          borderRadius: 3,
          position: "relative",
          overflow: "hidden",
          "&::before": {
            content: '""',
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "3px",
            background: `linear-gradient(90deg, ${color} 0%, ${alpha(color, 0.3)} 100%)`,
          },
        }}
      >
        {/* Background glow */}
        <Box
          sx={{
            position: "absolute",
            top: -30,
            right: -30,
            width: 120,
            height: 120,
            borderRadius: "50%",
            background: `radial-gradient(circle, ${alpha(color, 0.08)} 0%, transparent 70%)`,
            pointerEvents: "none",
          }}
        />

        <CardContent sx={{ p: 2.5, "&:last-child": { pb: 2.5 }, position: "relative" }}>
          {/* Header row */}
          <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", mb: 1.5 }}>
            <Typography
              variant="caption"
              sx={{
                color: "rgba(255,255,255,0.5)",
                fontWeight: 600,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                fontSize: "0.65rem",
              }}
            >
              {label}
            </Typography>
            {icon && (
              <Box sx={{ color: alpha(color, 0.7), "& svg": { fontSize: 20 } }}>
                {icon}
              </Box>
            )}
          </Box>

          {/* Value */}
          {loading ? (
            <Skeleton variant="text" width="60%" height={40} />
          ) : (
            <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.5 }}>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 800,
                  color: color,
                  lineHeight: 1,
                  fontFamily: '"JetBrains Mono", monospace',
                  letterSpacing: "-0.02em",
                  fontSize: { xs: "1.5rem", sm: "1.75rem" },
                }}
              >
                {value}
              </Typography>
              {unit && (
                <Typography
                  variant="caption"
                  sx={{ color: alpha(color, 0.7), fontWeight: 600, fontSize: "0.7rem" }}
                >
                  {unit}
                </Typography>
              )}
            </Box>
          )}

          {/* Subtext */}
          {subtext && !loading && (
            <Typography
              variant="caption"
              sx={{ color: "rgba(255,255,255,0.35)", mt: 0.75, display: "block", fontSize: "0.7rem" }}
            >
              {subtext}
            </Typography>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
