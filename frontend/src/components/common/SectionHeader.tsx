import React from "react";
import { Box, Typography, Divider } from "@mui/material";
import { motion } from "framer-motion";
import { fadeUp } from "../../animations/variants";

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  accent?: string;
}

export default function SectionHeader({
  title,
  subtitle,
  action,
  accent = "#00d4ff",
}: SectionHeaderProps) {
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}>
      <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", mb: { xs: 2, md: 3 } }}>
        <Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: subtitle ? 0.5 : 0 }}>
            <Box
              sx={{
                width: 4,
                height: 22,
                borderRadius: 2,
                background: `linear-gradient(180deg, ${accent} 0%, rgba(124,58,237,0.6) 100%)`,
                boxShadow: `0 0 8px ${accent}60`,
              }}
            />
            <Typography
              variant="h5"
              sx={{
                fontWeight: 700,
                color: "#f1f5f9",
                fontSize: { xs: "1.1rem", sm: "1.25rem" },
              }}
            >
              {title}
            </Typography>
          </Box>
          {subtitle && (
            <Typography
              variant="body2"
              sx={{
                color: "rgba(255,255,255,0.45)",
                ml: "20px",
                fontSize: { xs: "0.74rem", sm: "0.82rem" },
                lineHeight: 1.55,
              }}
            >
              {subtitle}
            </Typography>
          )}
        </Box>
        {action && <Box>{action}</Box>}
      </Box>
    </motion.div>
  );
}
