import React, { useState, useEffect } from "react";
import { Snackbar, Alert } from "@mui/material";
import WifiOffRounded from "@mui/icons-material/WifiOffRounded";

/**
 * Listens for browser online/offline events and shows a persistent banner
 * when the network is unavailable. Auto-dismisses when connectivity returns.
 */
export default function OfflineBanner() {
  const [offline, setOffline] = useState(!navigator.onLine);

  useEffect(() => {
    const goOffline = () => setOffline(true);
    const goOnline  = () => setOffline(false);
    window.addEventListener("offline", goOffline);
    window.addEventListener("online",  goOnline);
    return () => {
      window.removeEventListener("offline", goOffline);
      window.removeEventListener("online",  goOnline);
    };
  }, []);

  return (
    <Snackbar
      open={offline}
      anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
    >
      <Alert
        icon={<WifiOffRounded fontSize="small" />}
        severity="warning"
        variant="filled"
        sx={{ fontWeight: 600 }}
      >
        No internet connection — predictions unavailable until reconnected
      </Alert>
    </Snackbar>
  );
}
