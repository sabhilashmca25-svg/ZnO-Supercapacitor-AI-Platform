import React, { Component, ErrorInfo, ReactNode } from "react";
import { Box, Typography, Button } from "@mui/material";
import ErrorOutlineRounded from "@mui/icons-material/ErrorOutlineRounded";

interface Props {
  children: ReactNode;
  /** Optional label shown in the error card — e.g. "Training History Chart" */
  label?: string;
}

interface State {
  hasError: boolean;
  message?: string;
}

/**
 * Class-based error boundary that catches uncaught render errors in any
 * child component tree (e.g. a Plotly chart crash) and shows a friendly
 * fallback instead of a white blank screen.
 *
 * Usage:
 *   <ErrorBoundary label="CV Chart">
 *     <Plot ... />
 *   </ErrorBoundary>
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error?.message ?? "Unknown render error" };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", this.props.label ?? "Component", error, info);
  }

  handleRetry = () => {
    this.setState({ hasError: false, message: undefined });
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 1.5,
            py: 4,
            px: 2,
            borderRadius: 2,
            border: "1px solid rgba(255,80,80,0.2)",
            background: "rgba(255,50,50,0.05)",
            minHeight: 120,
            textAlign: "center",
          }}
        >
          <ErrorOutlineRounded sx={{ fontSize: 32, color: "rgba(255,100,100,0.7)" }} />
          <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.82rem" }}>
            {this.props.label ? `${this.props.label} failed to render` : "Component failed to render"}
          </Typography>
          {this.state.message && (
            <Typography
              sx={{
                color: "rgba(255,255,255,0.25)",
                fontSize: "0.72rem",
                fontFamily: "monospace",
                maxWidth: 400,
                wordBreak: "break-word",
              }}
            >
              {this.state.message}
            </Typography>
          )}
          <Button
            size="small"
            variant="outlined"
            onClick={this.handleRetry}
            sx={{
              mt: 1,
              borderColor: "rgba(255,255,255,0.15)",
              color: "rgba(255,255,255,0.5)",
              fontSize: "0.75rem",
              textTransform: "none",
              "&:hover": { borderColor: "#00d4ff", color: "#00d4ff" },
            }}
          >
            Retry
          </Button>
        </Box>
      );
    }

    return this.props.children;
  }
}
