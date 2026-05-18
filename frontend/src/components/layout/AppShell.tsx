import React, { useState } from "react";
import { Outlet } from "react-router-dom";
import { Box } from "@mui/material";
import Sidebar, { SIDEBAR_WIDTH } from "./Sidebar";
import TopBar from "./TopBar";
import OfflineBanner from "../common/OfflineBanner";

export default function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <Box sx={{ display: "flex", minHeight: "100vh", backgroundColor: "#070b14" }}>
      {/* Sidebar — permanent on desktop, temporary overlay on mobile */}
      <Sidebar
        mobileOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
      />

      {/* Main area — fills remaining width; on mobile takes full width */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          minHeight: "100vh",
          minWidth: 0,
          /* Ensure content doesn't overflow on narrow viewports */
          overflow: "hidden",
        }}
      >
        <TopBar onMenuClick={() => setMobileOpen(true)} />
        <Box
          sx={{
            flex: 1,
            p: { xs: 1.5, sm: 2.5, md: 3.5 },
            overflowY: "auto",
            overflowX: "hidden",
            /* Prevent layout from expanding beyond a comfortable reading width
               on ultra-wide displays (≥1920px). Content is still centred. */
            "& > *": {
              maxWidth: 1700,
              mx: "auto",
            },
          }}
        >
          {/* React Router v6 Outlet — renders the matched child route */}
          <Outlet />
        </Box>
      </Box>
      {/* Global offline indicator — shown whenever network is unavailable */}
      <OfflineBanner />
    </Box>
  );
}
