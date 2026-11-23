"""Empirical scalability & memory benchmark across 1,000 -> 250,000 employees."""

from __future__ import annotations

import json
import time
import tracemalloc
from datetime import datetime, timezone

import pandas as pd

from src.business.reskilling_roi import (
    build_default_headcount,
    compute_workforce_diagnostics,
    params_from_assumptions,
)
from src.business.scenario_simulator import (
    compare_strategies,
    run_monte_carlo_sensitivity,
)
from src.etl.common import PROJECT_ROOT, load_business_assumptions


def run_benchmark() -> dict:
    assumptions = load_business_assumptions()
    exp_df = pd.read_csv(PROJECT_ROOT / "data/probes/app_snapshots/ilo_exposure_isco1.csv")
    exp_df["isco1"] = exp_df["isco1"].astype(str)
    salary = {
        str(k): float(v["monthly_gross_vnd"])
        for k, v in assumptions["monthly_salary_vnd_by_isco1"].items()
    }
    params = params_from_assumptions(assumptions, scenario_key="base")

    # Warmup JIT / NumPy allocations once before timing
    hc_warm = build_default_headcount(1000)
    compare_strategies(hc_warm, salary, exp_df, params, locked_isco1=["1"], targeted_isco1=["2", "3", "4"])

    scales = [1_000, 10_000, 50_000, 250_000]
    results = []
    for n_emp in scales:
        hc = build_default_headcount(n_emp)
        tracemalloc.start()
        t0 = time.perf_counter()
        compare_strategies(hc, salary, exp_df, params, locked_isco1=["1"], targeted_isco1=["2", "3", "4"], run_monte_carlo=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append(
            {
                "headcount": n_emp,
                "mc_trials": 1000,
                "latency_ms": round(elapsed_ms, 2),
                "peak_memory_mb": round(peak_bytes / (1024.0 * 1024.0), 3),
            }
        )

    # Stress-test 10,000 vectorized Monte Carlo trials at 50,000-employee enterprise scale
    diag_50k = compute_workforce_diagnostics(build_default_headcount(50_000), salary, exp_df, params, locked_isco1=["1"])
    tracemalloc.start()
    t0 = time.perf_counter()
    run_monte_carlo_sensitivity(diag_50k, params, targeted_isco1=["2", "3", "4"], n_sims=10_000, seed=42)
    stress_ms = (time.perf_counter() - t0) * 1000.0
    _, stress_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    out = {
        "benchmarked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "scales": results,
        "mc_10k_trials_stress": {
            "headcount": 50_000,
            "mc_trials": 10_000,
            "latency_ms": round(stress_ms, 2),
            "peak_memory_mb": round(stress_peak / (1024.0 * 1024.0), 3),
        },
    }
    out_path = PROJECT_ROOT / "data/probes/scalability_benchmark.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Benchmark] Saved {out_path}")
    for r in results:
        print(f"  - {r['headcount']:>7,} employees (1k MC): {r['latency_ms']:6.2f} ms | Peak RAM = {r['peak_memory_mb']:.2f} MB")
    print(f"  - Stress 50,000 employees (10k MC): {stress_ms:6.2f} ms | Peak RAM = {out['mc_10k_trials_stress']['peak_memory_mb']:.2f} MB")
    return out


if __name__ == "__main__":
    run_benchmark()
