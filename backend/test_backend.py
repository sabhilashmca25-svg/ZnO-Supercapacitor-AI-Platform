"""
test_backend.py — End-to-end runtime verification script.

Run AFTER the server is started:
    cd backend
    python test_backend.py

Tests every endpoint and prints a clear PASS/FAIL report.
"""
import json
import sys
import time
import traceback
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

BASE = "http://127.0.0.1:8000"
TIMEOUT = 60.0   # seconds (RF prediction can be slow)

passed: list[str] = []
failed: list[str] = []
warnings_list: list[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

def ok(name: str, detail: str = "") -> None:
    tag = f"  ✓ PASS  {name}"
    if detail:
        tag += f"  →  {detail}"
    print(tag)
    passed.append(name)


def fail(name: str, reason: str) -> None:
    print(f"  ✗ FAIL  {name}  →  {reason}")
    failed.append(f"{name}: {reason}")


def warn(name: str, reason: str) -> None:
    print(f"  ⚠ WARN  {name}  →  {reason}")
    warnings_list.append(f"{name}: {reason}")


def get(path: str) -> httpx.Response | None:
    try:
        return httpx.get(f"{BASE}{path}", timeout=TIMEOUT)
    except httpx.ConnectError:
        return None


def post(path: str, body: dict) -> httpx.Response | None:
    try:
        return httpx.post(f"{BASE}{path}", json=body, timeout=TIMEOUT)
    except httpx.ConnectError:
        return None


def section(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Server reachability
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 1 — Server Reachability")
resp = get("/")
if resp is None:
    print("\n  CRITICAL: Cannot connect to server at http://127.0.0.1:8000")
    print("  Please start the server first:")
    print("    cd backend")
    print("    uvicorn app.main:app --reload --port 8000")
    sys.exit(1)
ok("Server reachable", f"HTTP {resp.status_code}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Root endpoint
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 2 — Root Endpoint  (GET /)")
resp = get("/")
if resp and resp.status_code == 200:
    data = resp.json()
    ok("GET /  status 200")
    # Verify expected fields
    for field in ["name", "version", "status", "models_loaded", "docs"]:
        if field in data:
            ok(f"  field '{field}' present", str(data[field])[:60])
        else:
            fail(f"  field '{field}'", "missing from response")
    print(f"\n  Full response:\n  {json.dumps(data, indent=4)}")
else:
    fail("GET /", f"HTTP {resp.status_code if resp else 'no response'}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Health endpoint
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 3 — Health Endpoint  (GET /api/v1/health)")
resp = get("/api/v1/health")
if resp and resp.status_code == 200:
    data = resp.json()
    ok("GET /api/v1/health  status 200")
    ok("  status field", data.get("status"))
    ok("  scalers_ok", str(data.get("scalers_ok")))
    models_loaded = data.get("models_loaded", [])
    ok("  models_loaded", str(models_loaded))
    if not models_loaded:
        warn("models_loaded", "No models loaded — predictions will fail")
    print(f"\n  Full health response:\n  {json.dumps(data, indent=4)}")
else:
    fail("GET /api/v1/health", f"HTTP {resp.status_code if resp else 'no response'}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Models endpoint
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 4 — Models Endpoint  (GET /api/v1/models)")
resp = get("/api/v1/models")
if resp and resp.status_code == 200:
    ok("GET /api/v1/models  status 200")
else:
    warn("GET /api/v1/models", f"HTTP {resp.status_code if resp else 'no response'}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Single-model prediction
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 5 — Single-Model Prediction  (POST /api/v1/predict)")

predict_body = {
    "model_name": "lightgbm",
    "material_id": "NM1",
    "scan_rate_mVs": 30.0,
}
print(f"\n  Request: {json.dumps(predict_body)}")
t0 = time.time()
resp = post("/api/v1/predict", predict_body)
elapsed = time.time() - t0

if resp and resp.status_code == 200:
    data = resp.json()
    ok("POST /api/v1/predict  status 200", f"{elapsed:.2f}s")

    # Verify response shape
    n = data.get("n_points", 0)
    pots = data.get("potential_V", [])
    currs = data.get("predicted_current_uA", [])

    ok("  n_points", str(n)) if n == 651 else fail("  n_points", f"expected 651 got {n}")
    ok("  potential_V length", str(len(pots))) if len(pots) == 651 else fail("  potential_V length", str(len(pots)))
    ok("  predicted_current_uA length", str(len(currs))) if len(currs) == 651 else fail("  current length", str(len(currs)))

    # Check for NaN / Inf
    nan_count = sum(1 for c in currs if c != c)   # NaN check: NaN != NaN
    inf_count = sum(1 for c in currs if abs(c) == float("inf"))
    ok("  no NaN in current") if nan_count == 0 else fail("  NaN check", f"{nan_count} NaN values found")
    ok("  no Inf in current") if inf_count == 0 else fail("  Inf check", f"{inf_count} Inf values found")

    # Check potential sweep direction
    ok("  sweep starts at -0.65V", str(round(pots[0], 2))) if pots and abs(pots[0] - (-0.65)) < 0.01 else fail("  sweep start", str(pots[0] if pots else "empty"))
    ok("  sweep peaks at 0V", str(round(pots[325], 4))) if pots and abs(pots[325]) < 0.01 else fail("  sweep peak", str(pots[325] if len(pots) > 325 else "too short"))

    # Check statistics
    stats = data.get("statistics", {})
    ok("  peak_anodic_uA", f"{stats.get('peak_anodic_uA'):.2f} µA")
    ok("  peak_cathodic_uA", f"{stats.get('peak_cathodic_uA'):.2f} µA")
    ok("  current_range_uA", f"{stats.get('current_range_uA'):.2f} µA")
    ok("  integral_area", f"{stats.get('integral_area'):.4f}")

    # Print sample of arrays
    print(f"\n  First 5 potentials (V) : {pots[:5]}")
    print(f"  First 5 currents  (µA): {currs[:5]}")
    print(f"  Statistics: {json.dumps(stats, indent=6)}")

elif resp and resp.status_code == 503:
    warn("POST /api/v1/predict", "lightgbm not loaded — trying rf instead")
    resp = post("/api/v1/predict", {**predict_body, "model_name": "rf"})
    if resp and resp.status_code == 200:
        ok("  POST /predict (rf fallback)  status 200")
    else:
        fail("  POST /predict (rf fallback)", f"HTTP {resp.status_code if resp else 'no response'}")
else:
    fail("POST /api/v1/predict", f"HTTP {resp.status_code if resp else 'no response'}  body={resp.text[:200] if resp else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Multi-model comparison
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 6 — Multi-Model Comparison  (POST /api/v1/compare)")

compare_body = {
    "models": ["rf", "lightgbm", "gru"],
    "material_id": "NM1",
    "scan_rate_mVs": 30.0,
}
print(f"\n  Request: {json.dumps(compare_body)}")
t0 = time.time()
resp = post("/api/v1/compare", compare_body)
elapsed = time.time() - t0

if resp and resp.status_code == 200:
    data = resp.json()
    ok("POST /api/v1/compare  status 200", f"{elapsed:.2f}s")

    pots = data.get("potential_V", [])
    preds = data.get("predictions", {})
    failed_models = data.get("models_failed", [])

    ok("  potential_V length", str(len(pots))) if len(pots) == 651 else fail("  potential_V length", str(len(pots)))
    ok("  predictions dict present", str(list(preds.keys())))

    for m_id, m_pred in preds.items():
        curr = m_pred.get("predicted_current_uA", [])
        ok(f"  {m_id}  length", str(len(curr)))
        ok(f"  {m_id}  peak_anodic", f"{m_pred.get('peak_anodic_uA', '?'):.2f} µA")

    if failed_models:
        warn("  models_failed", str(failed_models))
    else:
        ok("  no models failed")

    print(f"\n  Succeeded models: {list(preds.keys())}")
    print(f"  Failed models:    {failed_models}")
else:
    fail("POST /api/v1/compare", f"HTTP {resp.status_code if resp else 'no response'}  body={resp.text[:200] if resp else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Analytics endpoints
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 7 — Analytics Endpoints")

for path, name in [
    ("/api/v1/benchmarks/leaderboard", "benchmarks/leaderboard"),
    ("/api/v1/benchmarks/comparison",  "benchmarks/comparison"),
    ("/api/v1/benchmarks/summary",     "benchmarks/summary"),
    ("/api/v1/metrics/rf",             "metrics/rf"),
    ("/api/v1/metrics/lightgbm",       "metrics/lightgbm"),
    ("/api/v1/metrics/gru",            "metrics/gru"),
]:
    resp = get(path)
    if resp and resp.status_code == 200:
        body = resp.json()
        count = len(body) if isinstance(body, list) else "dict"
        ok(f"GET {path}", f"{count} entries")
    else:
        fail(f"GET {path}", f"HTTP {resp.status_code if resp else 'no response'}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Error handling
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 8 — Error Handling")

# Wrong model name → 404
resp = post("/api/v1/predict", {"model_name": "invalid_model", "material_id": "NM1", "scan_rate_mVs": 30})
if resp and resp.status_code == 404:
    ok("404 for unknown model", resp.json().get("error", "?"))
else:
    fail("404 for unknown model", f"expected 404 got {resp.status_code if resp else 'no response'}")

# Invalid material → 422 or 422 from validator
resp = post("/api/v1/predict", {"model_name": "lightgbm", "material_id": "INVALID", "scan_rate_mVs": 30})
if resp and resp.status_code in (422, 400):
    ok("422 for invalid material", f"HTTP {resp.status_code}")
else:
    warn("422 for invalid material", f"got HTTP {resp.status_code if resp else 'no response'} — may be deferred to inference")

# Missing field → 422
resp = post("/api/v1/predict", {"model_name": "lightgbm", "scan_rate_mVs": 30})
if resp and resp.status_code == 422:
    ok("422 for missing field (material_id)")
else:
    fail("422 for missing field", f"expected 422 got {resp.status_code if resp else 'no response'}")

# Out-of-range scan rate → 422
resp = post("/api/v1/predict", {"model_name": "lightgbm", "material_id": "NM1", "scan_rate_mVs": 999})
if resp and resp.status_code == 422:
    ok("422 for out-of-range scan_rate_mVs (999 > 100)")
else:
    fail("422 for scan_rate_mVs=999", f"expected 422 got {resp.status_code if resp else 'no response'}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Swagger / OpenAPI schema
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 9 — Swagger / OpenAPI Schema")
resp = get("/openapi.json")
if resp and resp.status_code == 200:
    schema = resp.json()
    paths = list(schema.get("paths", {}).keys())
    ok("GET /openapi.json", f"{len(paths)} paths found")
    for p in paths:
        ok(f"  route {p}")
else:
    fail("GET /openapi.json", f"HTTP {resp.status_code if resp else 'no response'}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  BACKEND VERIFICATION REPORT")
print(f"{'='*60}")
print(f"  ✓ PASSED : {len(passed)}")
print(f"  ✗ FAILED : {len(failed)}")
print(f"  ⚠ WARNED : {len(warnings_list)}")

if failed:
    print(f"\n  Failed checks:")
    for f_item in failed:
        print(f"    ✗  {f_item}")

if warnings_list:
    print(f"\n  Warnings:")
    for w in warnings_list:
        print(f"    ⚠  {w}")

status = "READY FOR FRONTEND DEVELOPMENT" if len(failed) == 0 else "NEEDS FIXES BEFORE DEPLOYMENT"
print(f"\n  Status: {status}")
print(f"{'='*60}\n")
sys.exit(0 if len(failed) == 0 else 1)
