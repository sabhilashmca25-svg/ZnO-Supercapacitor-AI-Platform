#!/usr/bin/env python3
"""
compress_rf.py — Compress the RF model from ~615 MB to ~150 MB.

Usage:
    python scripts/compress_rf.py

The script:
  1. Loads the existing rf_baseline.joblib (uncompressed)
  2. Re-saves with joblib compress=3 (zlib level 3 — good size/speed balance)
  3. Prints before/after sizes and compression ratio

Run this ONCE after training. The backend loads the compressed model
transparently via joblib — no code changes needed.

Expected results:
  Before: ~615 MB (uncompressed pickle)
  After:  ~140-180 MB (zlib-3 compressed)
  Ratio:  ~3.5-4x compression
"""
from pathlib import Path
import joblib
import sys
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PLATFORM_ROOT = SCRIPT_DIR.parent
MODEL_PATH   = PLATFORM_ROOT / "models" / "rf" / "rf_baseline.joblib"
BACKUP_PATH  = MODEL_PATH.with_suffix(".joblib.backup")

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

size_before_mb = MODEL_PATH.stat().st_size / 1e6
print(f"RF model path : {MODEL_PATH}")
print(f"Size before   : {size_before_mb:.1f} MB")
print()

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading model... (this may take 30-60 s for 615 MB)")
t0 = time.perf_counter()
model = joblib.load(MODEL_PATH)
load_time = time.perf_counter() - t0
print(f"  Loaded in {load_time:.1f}s")
print()

# ── Backup original ───────────────────────────────────────────────────────────
if not BACKUP_PATH.exists():
    import shutil
    print(f"Backing up original to {BACKUP_PATH.name}")
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    print(f"  Backup saved ({BACKUP_PATH.stat().st_size / 1e6:.1f} MB)")
    print()
else:
    print(f"Backup already exists at {BACKUP_PATH.name} — skipping backup.")
    print()

# ── Re-save with compression ──────────────────────────────────────────────────
# compress=3 → zlib level 3:  good compression ratio, fast decompression
# compress=9 → maximum ratio but very slow decompression (not recommended)
COMPRESS_LEVEL = 3
print(f"Re-saving with compress={COMPRESS_LEVEL}...")
t1 = time.perf_counter()
joblib.dump(model, MODEL_PATH, compress=COMPRESS_LEVEL)
save_time = time.perf_counter() - t1
print(f"  Saved in {save_time:.1f}s")

# ── Report ────────────────────────────────────────────────────────────────────
size_after_mb = MODEL_PATH.stat().st_size / 1e6
ratio = size_before_mb / size_after_mb
saving_mb = size_before_mb - size_after_mb

print()
print("=" * 50)
print(f"Before : {size_before_mb:7.1f} MB")
print(f"After  : {size_after_mb:7.1f} MB")
print(f"Saved  : {saving_mb:7.1f} MB  ({ratio:.1f}x compression)")
print("=" * 50)
print()
print("Done. The backend loads the compressed model transparently.")
print(f"  Backup retained at: {BACKUP_PATH.name}")
print(f"  Delete backup when satisfied: del {BACKUP_PATH}")
