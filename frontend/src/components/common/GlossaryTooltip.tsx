import React from "react";
import { Box, Tooltip, Typography } from "@mui/material";
import { GLOSSARY } from "../../constants/glossary";

interface GlossaryTooltipProps {
  /** Key in the GLOSSARY constant (e.g. "cv", "scan_rate", "rmse") */
  termKey: string;
  /** The text to display. If omitted, renders the glossary term label. */
  children?: React.ReactNode;
  /** Additional sx styles for the underline wrapper */
  sx?: object;
}

/**
 * Wraps any text with a dashed underline that shows an electrochemistry
 * glossary definition on hover. Uses the GLOSSARY constant as the source.
 *
 * Usage:
 *   <GlossaryTooltip termKey="symmetry_factor">Symmetry Factor</GlossaryTooltip>
 */
export default function GlossaryTooltip({ termKey, children, sx }: GlossaryTooltipProps) {
  const entry = GLOSSARY[termKey];
  if (!entry) {
    return <>{children}</>;
  }

  const tooltipContent = (
    <Box sx={{ maxWidth: 280, p: 0.25 }}>
      <Typography
        variant="caption"
        sx={{ color: "#00d4ff", fontWeight: 700, fontSize: "0.72rem", display: "block", mb: 0.75 }}
      >
        {entry.term}
      </Typography>
      <Typography
        variant="caption"
        sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.68rem", lineHeight: 1.6, display: "block" }}
      >
        {entry.short}
      </Typography>
      {entry.extended && (
        <Typography
          variant="caption"
          sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.62rem", lineHeight: 1.55, display: "block", mt: 0.75, borderTop: "1px solid rgba(255,255,255,0.08)", pt: 0.75 }}
        >
          {entry.extended}
        </Typography>
      )}
    </Box>
  );

  return (
    <Tooltip
      title={tooltipContent}
      arrow
      placement="top"
      componentsProps={{
        tooltip: {
          sx: {
            backgroundColor: "#0d1b2e",
            border: "1px solid rgba(0,212,255,0.25)",
            boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
            borderRadius: 2,
            px: 1.5,
            py: 1,
            maxWidth: 300,
          },
        },
        arrow: {
          sx: { color: "#0d1b2e" },
        },
      }}
    >
      <Box
        component="span"
        sx={{
          borderBottom: "1px dashed rgba(0,212,255,0.45)",
          cursor: "help",
          "&:hover": { borderBottomColor: "#00d4ff" },
          transition: "border-bottom-color 0.15s",
          ...sx,
        }}
      >
        {children ?? entry.term}
      </Box>
    </Tooltip>
  );
}
