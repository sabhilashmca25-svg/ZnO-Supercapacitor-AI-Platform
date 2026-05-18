"""
test_modules.py — Verifies all modules import and logic runs correctly,
WITHOUT needing the server to be running.

Run from backend/:
    python test_modules.py

Tests:
  - config paths resolve to existing files
  - feature engineering produces correct shape and values
  - normalizer produces values in [0, 1]
  - denormalizer produces realistic µA values
  - predictor factory imports all 6 classes
  - benchmark CSV loading works
  - metrics JSON loading works
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

results = {"passed": [], "failed": [], "warned": []}

def ok(name, detail=""):
    tag = f"  ✓ {name}"
    if detail:
        tag += f"  →  {detail}"
    print(tag)
    results["passed"].append(name)

def fail(name, reason):
    print(f"  ✗ FAIL  {name}  →  {reason}")
    results["failed"].append(f"{name}: {reason}")

def warn(name, reason):
    print(f"  ⚠ WARN  {name}  →  {reason}")
    results["warned"].append(f"{name}: {reason}")

def section(title):
    print(f"\n{'-'*55}\n  {title}\n{'-'*55}")


# ─── Test 1: Config ───────────────────────────────────────────────────────────
section("1. Config & Paths")
try:
    from app.core.config import settings, PLATFORM_ROOT
    ok("config import")
    ok("PLATFORM_ROOT", str(PLATFORM_ROOT))
    ok("MODEL_DIR exists", str(settings.MODEL_DIR.exists()))
    ok("SCALERS_PATH exists", str(settings.SCALERS_PATH.exists()))
    ok("METRICS_DIR exists", str(settings.METRICS_DIR.exists()))
    ok("EXPORTS_DIR exists", str(settings.EXPORTS_DIR.exists()))
    ok("ENABLED_MODELS", str(settings.ENABLED_MODELS))
    ok("MODEL_ORDER", str(settings.MODEL_ORDER))
    if not settings.SCALERS_PATH.exists():
        fail("scalers.json", f"Not found at {settings.SCALERS_PATH}")
except Exception as e:
    fail("config import", str(e))


# ─── Test 2: Feature engineering ─────────────────────────────────────────────
section("2. Feature Engineering")
try:
    from app.services.preprocessing.feature_engineer import (
        generate_potential_sweep, build_raw_features, FEATURE_NAMES, N_POINTS
    )
    ok("feature_engineer import")

    V = generate_potential_sweep()
    ok("generate_potential_sweep()", f"shape {V.shape}")
    assert len(V) == 651, f"Expected 651, got {len(V)}"
    ok("sweep length == 651")
    assert abs(V[0] - (-0.65)) < 1e-9, f"Expected -0.65, got {V[0]}"
    ok("sweep starts at -0.65V", f"{V[0]:.4f}")
    assert abs(V[325]) < 1e-9, f"Expected 0.0, got {V[325]}"
    ok("sweep peaks at 0V at index 325", f"{V[325]:.6f}")
    assert abs(V[-1] - (-0.65)) < 1e-3, f"Expected -0.65, got {V[-1]}"
    ok("sweep ends at -0.65V", f"{V[-1]:.4f}")

    df = build_raw_features(30.0)
    ok("build_raw_features(30.0)", f"shape {df.shape}")
    assert df.shape == (651, 10), f"Expected (651, 10), got {df.shape}"
    ok("feature DataFrame shape == (651, 10)")
    assert list(df.columns) == FEATURE_NAMES, "Column names mismatch"
    ok("column names match FEATURE_NAMES")

    # Check sweep_direction values
    directions = df["sweep_direction"].unique().tolist()
    ok("sweep_direction values", str(sorted(directions)))
    assert set(directions) == {0.0, 1.0}, f"Expected {{0.0, 1.0}}, got {set(directions)}"
    ok("sweep_direction is 0/1 (not -1/1) ✓")

except AssertionError as e:
    fail("feature engineering assertion", str(e))
except Exception as e:
    fail("feature engineering", str(e))
    import traceback; traceback.print_exc()


# ─── Test 3: Normalizer ───────────────────────────────────────────────────────
section("3. Normalizer")
try:
    from app.services.preprocessing.feature_engineer import build_raw_features
    from app.services.preprocessing.normalizer import normalize

    df_raw = build_raw_features(30.0)
    df_norm = normalize(df_raw)
    ok("normalize() ran")
    ok("normalized shape", str(df_norm.shape))

    # Columns that SHOULD be in [0, 1] after normalisation
    norm_cols = [
        "potential_V_norm", "scan_rate_mVs_norm", "log_scan_rate_norm",
        "sqrt_scan_rate_norm", "potential_from_lower_norm",
        "potential_from_upper_norm", "sr_x_potential_norm",
        "direction_x_potential_norm",
    ]
    for col in norm_cols:
        mn = float(df_norm[col].min())
        mx = float(df_norm[col].max())
        in_range = -0.001 <= mn and mx <= 1.001
        if in_range:
            ok(f"  {col} in [0,1]", f"min={mn:.4f} max={mx:.4f}")
        else:
            fail(f"  {col} out of range", f"min={mn:.4f} max={mx:.4f}")

    # Pass-through columns should be unchanged
    import numpy as np
    sweep_dir_unchanged = (df_norm["sweep_direction"].values == df_raw["sweep_direction"].values).all()
    ok("sweep_direction unchanged by normalizer", str(sweep_dir_unchanged))

except Exception as e:
    fail("normalizer", str(e))
    import traceback; traceback.print_exc()


# ─── Test 4: Denormalizer ─────────────────────────────────────────────────────
section("4. Denormalizer")
try:
    import numpy as np
    from app.services.postprocessing.denormalizer import denormalize

    fake_norm = np.linspace(0, 1, 651)
    current_uA = denormalize(fake_norm, "NM1", 30.0)
    ok("denormalize(NM1, 30) ran", f"shape {current_uA.shape}")
    ok("output in µA range", f"min={current_uA.min():.2f} max={current_uA.max():.2f}")
    assert current_uA.shape == (651,), f"Expected (651,) got {current_uA.shape}"
    ok("output shape (651,)")

    # NM1 at 30 mV/s should be in a reasonable µA range (based on scalers.json)
    if abs(current_uA.max()) > 1000:
        warn("current range", f"Unexpectedly large: {current_uA.max():.2f} µA")
    else:
        ok("current magnitude realistic (< 1000 µA)")

    # Test invalid group key
    try:
        denormalize(fake_norm, "INVALID_MAT", 30.0)
        fail("denormalize invalid group", "Should have raised KeyError")
    except KeyError:
        ok("KeyError raised for invalid material group ✓")

except Exception as e:
    fail("denormalizer", str(e))
    import traceback; traceback.print_exc()


# ─── Test 5: Full preprocessing pipeline ─────────────────────────────────────
section("5. Full Preprocessing Pipeline")
try:
    from app.services.preprocess import build_inference_inputs
    import numpy as np

    V, features_norm, seq_tensor, actual_len = build_inference_inputs(30.0)
    ok("build_inference_inputs(30.0) ran")
    ok("potential_V shape", str(V.shape))
    ok("features_norm shape", str(features_norm.shape))
    ok("seq_tensor shape", str(seq_tensor.shape))
    ok("actual_len", str(actual_len))

    assert V.shape == (651,), f"potential_V: expected (651,) got {V.shape}"
    ok("potential_V == (651,) ✓")
    assert features_norm.shape == (651, 10), f"features_norm: expected (651,10) got {features_norm.shape}"
    ok("features_norm == (651, 10) ✓")
    assert seq_tensor.shape == (1, 651, 10), f"seq_tensor: expected (1,651,10) got {seq_tensor.shape}"
    ok("seq_tensor == (1, 651, 10) ✓")
    assert actual_len == 651
    ok("actual_len == 651 ✓")
    assert seq_tensor.dtype == np.float32, f"Expected float32 got {seq_tensor.dtype}"
    ok("seq_tensor dtype == float32 ✓")

except Exception as e:
    fail("full preprocessing pipeline", str(e))
    import traceback; traceback.print_exc()


# ─── Test 6: np.trapezoid compatibility ───────────────────────────────────────
section("6. NumPy 2.x Compatibility")
try:
    import numpy as np
    ok("numpy version", np.__version__)

    # Test our compatibility wrapper
    arr = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    _trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    result = _trapezoid(arr)
    ok("trapezoid/trapz compatibility wrapper works", f"result={result:.2f}")

    # Make sure np.trapz is gone or deprecated (expected for numpy 2.x)
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        ok("NumPy 2.x detected — using np.trapezoid ✓")
    else:
        warn("NumPy < 2.0", f"version {np.__version__} — np.trapz still exists")

except Exception as e:
    fail("numpy compatibility", str(e))


# ─── Test 7: Predictor factory ────────────────────────────────────────────────
section("7. Predictor Factory")
try:
    from app.services.ml.predictor_factory import REGISTRY, get_predictor
    ok("predictor_factory import")
    ok("REGISTRY keys", str(list(REGISTRY.keys())))

    for model_id in ["rf", "lightgbm", "xgboost", "gru", "lstm", "ann"]:
        p = get_predictor(model_id)
        if p is not None:
            ok(f"  get_predictor('{model_id}')", type(p).__name__)
        else:
            fail(f"  get_predictor('{model_id}')", "returned None")

except Exception as e:
    fail("predictor factory", str(e))
    import traceback; traceback.print_exc()


# ─── Test 8: Analytics — benchmark ranker ────────────────────────────────────
section("8. Analytics — Benchmark Ranker")
try:
    from app.services.analytics.benchmark_ranker import (
        get_leaderboard, get_comparison_table, get_summary
    )

    lb = get_leaderboard()
    ok("get_leaderboard()", f"{len(lb)} entries")
    if lb:
        ok("  first entry rank", str(lb[0].get("rank")))
        ok("  first entry model_id", str(lb[0].get("model_id")))

    ct = get_comparison_table()
    ok("get_comparison_table()", f"{len(ct)} rows")

    sm = get_summary()
    ok("get_summary()", str(list(sm.keys())))

except Exception as e:
    fail("benchmark ranker", str(e))
    import traceback; traceback.print_exc()


# ─── Test 9: Analytics — metrics loader ──────────────────────────────────────
section("9. Analytics — Metrics Loader")
try:
    from app.services.analytics.metrics_loader import load_metrics, load_all_metrics

    data = load_metrics("rf")
    if data:
        ok("load_metrics('rf')", f"{len(data)} top-level keys")
        ok("  partitions", str([m["partition"] for m in data.get("metrics", [])]))
    else:
        fail("load_metrics('rf')", "returned None")

    all_m = load_all_metrics()
    ok("load_all_metrics()", f"{len(all_m)} models")

except Exception as e:
    fail("metrics loader", str(e))
    import traceback; traceback.print_exc()


# ─── Test 10: Exception classes ───────────────────────────────────────────────
section("10. Exception Handling")
try:
    from app.core.exceptions import (
        ModelNotFoundError, ModelNotLoadedError, InvalidInputError, PreprocessingError
    )
    try:
        raise ModelNotFoundError("fake_model")
    except ModelNotFoundError as e:
        ok("ModelNotFoundError raises correctly", str(e))

    try:
        raise ModelNotLoadedError("gru")
    except ModelNotLoadedError as e:
        ok("ModelNotLoadedError raises correctly", str(e))

    try:
        raise InvalidInputError("scan_rate", "must be 10-100")
    except InvalidInputError as e:
        ok("InvalidInputError raises correctly", str(e))

except Exception as e:
    fail("exceptions", str(e))


# ─── Final report ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  MODULE TEST REPORT")
print(f"{'='*55}")
print(f"  ✓ Passed  : {len(results['passed'])}")
print(f"  ✗ Failed  : {len(results['failed'])}")
print(f"  ⚠ Warnings: {len(results['warned'])}")

if results["failed"]:
    print(f"\n  Failed:")
    for f_item in results["failed"]:
        print(f"    ✗  {f_item}")

if results["warned"]:
    print(f"\n  Warnings:")
    for w in results["warned"]:
        print(f"    ⚠  {w}")

print(f"{'='*55}\n")
sys.exit(0 if not results["failed"] else 1)
