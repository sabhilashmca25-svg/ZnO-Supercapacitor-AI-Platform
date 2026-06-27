import React, { useEffect, useRef, useState } from "react";
import { Typography, type TypographyProps } from "@mui/material";
import { useInView } from "framer-motion";

interface AnimatedCounterProps extends TypographyProps {
  target: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  startOnView?: boolean;
}

export default function AnimatedCounter({
  target,
  duration = 1800,
  decimals = 0,
  prefix = "",
  suffix = "",
  startOnView = true,
  ...typographyProps
}: AnimatedCounterProps) {
  const [current, setCurrent] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });
  const hasStarted = useRef(false);
  const prevTarget = useRef(target);

  useEffect(() => {
    // Reset animation when target changes to a meaningful value
    if (target !== prevTarget.current) {
      prevTarget.current = target;
      if (target !== 0) hasStarted.current = false;
    }
    if (startOnView && !isInView) return;
    if (hasStarted.current) return;
    hasStarted.current = true;

    setCurrent(0);
    const startTime = performance.now();
    const intervalId = setInterval(() => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setCurrent(target * eased);
      if (progress >= 1) clearInterval(intervalId);
    }, 16);
    return () => clearInterval(intervalId);
  }, [isInView, startOnView, target, duration]);

  return (
    <Typography component="span" ref={ref} {...typographyProps}>
      {prefix}
      {current.toFixed(decimals)}
      {suffix}
    </Typography>
  );
}
