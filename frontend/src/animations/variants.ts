import type { Variants, Transition } from "framer-motion";

// ── Shared transitions ─────────────────────────────────────────────────────

export const smooth: Transition = {
  type: "spring",
  stiffness: 260,
  damping: 30,
};

export const easeOut: Transition = {
  duration: 0.4,
  ease: [0.16, 1, 0.3, 1],
};

export const snappy: Transition = {
  duration: 0.2,
  ease: [0.4, 0, 0.2, 1],
};

// ── Page-level transitions ─────────────────────────────────────────────────

export const pageVariants: Variants = {
  initial: { opacity: 0, y: 12 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.35, ease: [0.16, 1, 0.3, 1] },
  },
  exit: {
    opacity: 0,
    y: -6,
    transition: { duration: 0.18, ease: "easeIn" },
  },
};

// ── Fade in from below ────────────────────────────────────────────────────

export const fadeUp: Variants = {
  initial: { opacity: 0, y: 24 },
  animate: {
    opacity: 1,
    y: 0,
    transition: easeOut,
  },
  exit: { opacity: 0, y: -12, transition: snappy },
};

export const fadeIn: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.35 } },
  exit: { opacity: 0, transition: { duration: 0.2 } },
};

// ── Container stagger children ────────────────────────────────────────────

export const staggerContainer: Variants = {
  initial: {},
  animate: {
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

export const staggerFast: Variants = {
  initial: {},
  animate: {
    transition: { staggerChildren: 0.05, delayChildren: 0.05 },
  },
};

// ── Card / tile entrance ──────────────────────────────────────────────────

export const cardVariants: Variants = {
  initial: { opacity: 0, y: 20, scale: 0.97 },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      duration: 0.4,
      ease: [0.16, 1, 0.3, 1],
    },
  },
};

export const cardHover = {
  scale: 1.015,
  y: -3,
  transition: { duration: 0.2, ease: "easeOut" },
};

export const cardTap = {
  scale: 0.98,
  transition: { duration: 0.1 },
};

// ── Sidebar item ──────────────────────────────────────────────────────────

export const sidebarItem: Variants = {
  initial: { opacity: 0, x: -16 },
  animate: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.3, ease: "easeOut" },
  },
};

// ── Scale pop (for badges, chips) ─────────────────────────────────────────

export const scalePop: Variants = {
  initial: { opacity: 0, scale: 0.7 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: { type: "spring", stiffness: 400, damping: 22 },
  },
};

// ── Slide in from right ───────────────────────────────────────────────────

export const slideRight: Variants = {
  initial: { opacity: 0, x: 40 },
  animate: {
    opacity: 1,
    x: 0,
    transition: easeOut,
  },
  exit: { opacity: 0, x: 20, transition: snappy },
};

// ── Graph reveal ──────────────────────────────────────────────────────────

export const graphReveal: Variants = {
  initial: { opacity: 0, scaleY: 0.85, transformOrigin: "bottom" },
  animate: {
    opacity: 1,
    scaleY: 1,
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] },
  },
};

// ── Glow pulse (for live indicators) ─────────────────────────────────────

export const glowPulse = {
  animate: {
    boxShadow: [
      "0 0 4px rgba(0,212,255,0.2)",
      "0 0 16px rgba(0,212,255,0.5)",
      "0 0 4px rgba(0,212,255,0.2)",
    ],
    transition: {
      duration: 2,
      ease: "easeInOut",
      repeat: Infinity,
    },
  },
};

// ── Number counter (uses this for framer-motion animate attribute) ─────────

export const counterVariants: Variants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

// ── Hero float ───────────────────────────────────────────────────────────

export const heroFloat = {
  animate: {
    y: [0, -12, 0],
    transition: { duration: 4, ease: "easeInOut", repeat: Infinity },
  },
};

// ── Loading dots ─────────────────────────────────────────────────────────

export const loadingDot: Variants = {
  initial: { y: 0, opacity: 0.4 },
  animate: {
    y: [-6, 0],
    opacity: [0.4, 1, 0.4],
    transition: { duration: 0.7, ease: "easeInOut", repeat: Infinity },
  },
};

// ── Chart animations — GPU-accelerated clip-path reveals ──────────────────
// Line/scatter: draw curve left-to-right (clip right edge, reveal left→right)
export const lineDrawVariants: Variants = {
  initial: { opacity: 0, clipPath: "inset(0 100% 0 0)" },
  animate: {
    opacity: 1,
    clipPath: "inset(0 0% 0 0)",
    transition: { duration: 0.9, ease: [0.16, 1, 0.3, 1] },
  },
};

// Horizontal bar chart: bars grow left-to-right
export const hBarGrowVariants: Variants = {
  initial: { opacity: 0, clipPath: "inset(0 100% 0 0)" },
  animate: {
    opacity: 1,
    clipPath: "inset(0 0% 0 0)",
    transition: { duration: 0.75, ease: [0.16, 1, 0.3, 1] },
  },
};

// Vertical bar chart: bars grow from base upward
export const vBarGrowVariants: Variants = {
  initial: { opacity: 0, clipPath: "inset(100% 0 0 0)" },
  animate: {
    opacity: 1,
    clipPath: "inset(0% 0 0 0)",
    transition: { duration: 0.75, ease: [0.16, 1, 0.3, 1] },
  },
};

// Heatmap: fade in
export const heatmapFadeVariants: Variants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: { duration: 0.5, ease: "easeOut" },
  },
};
