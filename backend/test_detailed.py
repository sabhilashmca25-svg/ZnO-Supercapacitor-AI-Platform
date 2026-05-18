"""Detailed API validation — run while server is live."""
import json
import sys
import httpx

BASE = "http://127.0.0.1:8000"

print("=== DETAILED API VALIDATION ===\n")

# ── RF Prediction (NM2, 50 mV/s) ─────────────────────────────────────────────
print("--- RF Prediction (NM2, 50 mV/s) ---")
r = httpx.post(f"{BASE}/api/v1/predict",
               json={"model_name": "rf", "material_id": "NM2", "scan_rate_mVs": 50},
               timeout=120)
d = r.json()
V = d["potential_V"]
I = d["predicted_current_uA"]
print(f"HTTP {r.status_code}")
print(f"n_points: {d['n_points']}")
print(f"Potential: [{V[0]:.4f}, ..., {V[325]:.6f}(peak), ..., {V[-1]:.4f}]")
print(f"Current range: [{min(I):.2f}, {max(I):.2f}] uA")
print(f"Statistics: {json.dumps(d['statistics'], indent=2)}")
assert d["n_points"] == 651
assert abs(V[325]) < 0.001, f"Peak should be 0V, got {V[325]}"
assert all(isinstance(x, float) for x in V[:5]), "Potential values must be Python floats"
print("  Shape / type assertions PASSED\n")

# ── LightGBM Prediction (NM4, 100 mV/s) ──────────────────────────────────────
print("--- LightGBM Prediction (NM4, 100 mV/s) ---")
r = httpx.post(f"{BASE}/api/v1/predict",
               json={"model_name": "lightgbm", "material_id": "NM4", "scan_rate_mVs": 100},
               timeout=60)
d = r.json()
I = d["predicted_current_uA"]
print(f"HTTP {r.status_code}")
print(f"Current range: [{min(I):.2f}, {max(I):.2f}] uA")
print(f"Stats: {json.dumps(d['statistics'])}\n")

# ── Benchmark Leaderboard ─────────────────────────────────────────────────────
print("--- Benchmark Leaderboard ---")
r = httpx.get(f"{BASE}/api/v1/benchmarks/leaderboard", timeout=30)
rows = r.json()
print(f"HTTP {r.status_code}, {len(rows)} entries")
for row in rows[:6]:
    print(f"  #{row['rank']} {row['model_display']:<10}  "
          f"RMSE_val={row['rmse_val_uA']:.2f} uA  "
          f"R2_mat={row['r2_test_mat']:.4f}  "
          f"size={row['size_MB']} MB")
print()

# ── Benchmark Summary ─────────────────────────────────────────────────────────
print("--- Benchmark Summary ---")
r = httpx.get(f"{BASE}/api/v1/benchmarks/summary", timeout=30)
d = r.json()
print(f"HTTP {r.status_code}")
print("Key findings:", json.dumps(d.get("key_findings"), indent=2))
print()

# ── Error Handling ────────────────────────────────────────────────────────────
print("--- Error Handling ---")
r = httpx.post(f"{BASE}/api/v1/predict",
               json={"model_name": "lightgbm", "material_id": "WRONG", "scan_rate_mVs": 30},
               timeout=10)
err = r.json()
print(f"Invalid material  HTTP {r.status_code}: error={err.get('error','?')}")
assert r.status_code == 422

r = httpx.post(f"{BASE}/api/v1/predict",
               json={"model_name": "banana", "material_id": "NM1", "scan_rate_mVs": 30},
               timeout=10)
err = r.json()
print(f"Unknown model     HTTP {r.status_code}: error={err.get('error','?')}")
assert r.status_code == 404
assert "valid_models" in err

r = httpx.post(f"{BASE}/api/v1/predict",
               json={"model_name": "xgboost", "material_id": "NM1", "scan_rate_mVs": 30},
               timeout=10)
err = r.json()
print(f"Not loaded model  HTTP {r.status_code}: error={err.get('error','?')}")
assert r.status_code == 503
assert "hint" in err
print()

# ── JSON Serialization Safety ─────────────────────────────────────────────────
print("--- JSON Serialization Safety (NumPy-free check) ---")
r = httpx.post(f"{BASE}/api/v1/predict",
               json={"model_name": "rf", "material_id": "NM1", "scan_rate_mVs": 20},
               timeout=120)
d = r.json()
V = d["potential_V"]
I = d["predicted_current_uA"]
print(f"potential_V[0] type: {type(V[0]).__name__}  value: {V[0]}")
print(f"current[0] type:     {type(I[0]).__name__}  value: {I[0]}")
all_float = all(type(x).__name__ == "float" for x in V[:10] + I[:10])
print(f"All native Python floats (JSON-safe): {all_float}")
assert all_float, "Response contains non-float values (numpy serialization risk)"
print()

# ── Metrics endpoint coverage ─────────────────────────────────────────────────
print("--- Metrics Coverage (all 6 models) ---")
for model_id in ["rf", "lightgbm", "xgboost", "gru", "lstm", "ann"]:
    r = httpx.get(f"{BASE}/api/v1/metrics/{model_id}", timeout=10)
    if r.status_code == 200:
        d = r.json()
        partitions = [m["partition"] for m in d.get("metrics", [])]
        print(f"  {model_id:<10}  {len(partitions)} partitions: {partitions}")
    else:
        print(f"  {model_id:<10}  HTTP {r.status_code}")

print()
print("=== ALL DETAILED TESTS PASSED ===")
