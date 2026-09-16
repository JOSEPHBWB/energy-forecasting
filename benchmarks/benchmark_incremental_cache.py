from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_forecasting.base_forecaster import SeasonalBlendForecaster
from energy_forecasting.data import make_synthetic_market
from energy_forecasting.forecast_engine import ForecastEngine, ForecastRequest, PredictionStore
from energy_forecasting.runtime_monitor import measure_runtime


def build_jobs(frame, forecaster, n_jobs=8, horizon=24):
    jobs = []
    first_origin = 30 * 24
    for i in range(n_jobs):
        origin = first_origin + i * horizon
        history = frame.iloc[:origin].copy()
        history["price"] = history["price_real_time"]
        future = frame.iloc[origin:origin + horizon].copy()
        future["price"] = future["price_real_time"]
        req = ForecastRequest.from_frames(
            series_id="synthetic_rt", history=history, future=future, model_name=forecaster.name
        )
        jobs.append((req, history, future))
    return jobs


def run_scenario(name, engine, jobs, records):
    with measure_runtime(name, records):
        result = engine.run(jobs)
    records[-1].update({
        "requests": len(jobs),
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "cache_hit_rate": result.hit_rate,
        "model_calls": result.model_calls,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--cache-dir", default="results/incremental_cache")
    args = p.parse_args()

    cache_dir = ROOT / args.cache_dir
    shutil.rmtree(cache_dir, ignore_errors=True)
    store = PredictionStore(cache_dir)
    model = SeasonalBlendForecaster()
    engine = ForecastEngine(model, store)
    frame = make_synthetic_market(n_days=60, seed=11)
    jobs = build_jobs(frame, model, args.jobs, args.horizon)
    records = []

    run_scenario("cold_cache", engine, jobs, records)

    # Keep every other request cached to create a deterministic partial-hit case.
    for i, (request, _, _) in enumerate(jobs):
        if i % 2:
            store.remove(request)
    run_scenario("partial_cache", engine, jobs, records)
    run_scenario("warm_cache", engine, jobs, records)

    out = pd.DataFrame(records)
    output_path = ROOT / "results" / "incremental_cache_benchmark.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
