"""Compare cold and warm-cache base-forecast runs on the synthetic demo."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_forecasting.base_forecaster import SeasonalBlendForecaster
from energy_forecasting.chronos_adapter import Chronos2Forecaster
from energy_forecasting.data import make_synthetic_market
from energy_forecasting.experiment import run_experiment


def run_once(cfg, frame, forecaster, cache_dir, results_dir, task):
    t0 = time.perf_counter()
    metrics, efficiency, _ = run_experiment(
        frame,
        forecaster=forecaster,
        train_days=cfg["train_days"],
        test_days=cfg["test_days"],
        step_days=cfg["step_days"],
        max_windows=cfg["max_windows"],
        cache_dir=cache_dir,
        results_dir=results_dir,
        random_state=cfg["seed"],
        target_col="price_day_ahead" if task == "day_ahead" else "price_real_time",
        market_task=task,
    )
    return time.perf_counter() - t0, metrics, efficiency


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.json")
    p.add_argument("--base-model", choices=["seasonal", "chronos2"], default="seasonal")
    p.add_argument("--task", choices=["day_ahead", "real_time"], default="day_ahead")
    a = p.parse_args()

    cfg = json.loads((ROOT / a.config).read_text())
    frame = make_synthetic_market(cfg["n_days"], cfg["seed"])
    cache_dir = ROOT / cfg["cache_dir"]
    results_dir = ROOT / "results" / "cache_benchmark"
    shutil.rmtree(cache_dir, ignore_errors=True)

    make_model = (lambda: SeasonalBlendForecaster()) if a.base_model == "seasonal" else (lambda: Chronos2Forecaster())
    cold_s, _, cold_eff = run_once(cfg, frame, make_model(), cache_dir, results_dir / "cold", a.task)
    warm_s, _, warm_eff = run_once(cfg, frame, make_model(), cache_dir, results_dir / "warm", a.task)

    summary = pd.DataFrame([
        {"run": "cold_cache", "wall_s": cold_s, "base_model": a.base_model, "task": a.task},
        {"run": "warm_cache", "wall_s": warm_s, "base_model": a.base_model, "task": a.task},
    ])
    summary["speedup_vs_cold"] = cold_s / summary["wall_s"]
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "summary.csv", index=False)
    cold_eff.to_csv(results_dir / "cold_stages.csv", index=False)
    warm_eff.to_csv(results_dir / "warm_stages.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSynthetic benchmark only; results depend on hardware and cache state.")


if __name__ == "__main__":
    main()
