from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from energy_forecasting.base_forecaster import SeasonalBlendForecaster
from energy_forecasting.chronos_adapter import Chronos2Forecaster
from energy_forecasting.data import make_synthetic_market
from energy_forecasting.experiment import run_experiment

TASKS = {
    "day_ahead": "price_day_ahead",
    "real_time": "price_real_time",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.json")
    p.add_argument("--base-model", choices=["seasonal", "chronos2"], default="seasonal")
    p.add_argument("--task", choices=["day_ahead", "real_time", "both"], default="both")
    a = p.parse_args()

    cfg = json.loads((ROOT / a.config).read_text())
    frame = make_synthetic_market(cfg["n_days"], cfg["seed"])
    selected = list(TASKS) if a.task == "both" else [a.task]

    all_metrics, all_eff, all_preds = [], [], []
    for task in selected:
        forecaster = SeasonalBlendForecaster() if a.base_model == "seasonal" else Chronos2Forecaster()
        metrics, eff, preds = run_experiment(
            frame,
            forecaster=forecaster,
            train_days=cfg["train_days"],
            test_days=cfg["test_days"],
            step_days=cfg["step_days"],
            max_windows=cfg["max_windows"],
            cache_dir=ROOT / cfg["cache_dir"],
            results_dir=ROOT / cfg["results_dir"],
            random_state=cfg["seed"],
            target_col=TASKS[task],
            market_task=task,
        )
        all_metrics.append(metrics); all_eff.append(eff); all_preds.append(preds)

    results_dir = ROOT / cfg["results_dir"]
    metrics = pd.concat(all_metrics, ignore_index=True)
    efficiency = pd.concat(all_eff, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    efficiency.to_csv(results_dir / "efficiency.csv", index=False)
    preds.to_csv(results_dir / "predictions.csv", index=False)

    print(metrics[["task", "window", "test_start", "base_MAE", "corrected_MAE", "MAE_improvement_pct"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nAverage MAE improvement by task:")
    print(metrics.groupby("task")["MAE_improvement_pct"].mean().to_string(float_format=lambda x: f"{x:.2f}%"))


if __name__ == "__main__":
    main()
