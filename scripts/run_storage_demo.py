from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from energy_forecasting.storage import BatterySpec, optimize_storage_dispatch, realized_revenue


def main():
    pred_path = ROOT / "results" / "predictions.csv"
    if not pred_path.exists():
        raise SystemExit("Run `python scripts/run_demo.py --task both` first.")
    pred = pd.read_csv(pred_path, parse_dates=["timestamp"])
    # Storage settles in the synthetic real-time market.
    pred = pred[pred["task"] == "real_time"].copy()
    if pred.empty:
        raise SystemExit("No real_time predictions found. Run the forecasting demo with --task both or --task real_time.")

    spec = BatterySpec()
    rows = []
    pred["date"] = pred["timestamp"].dt.date
    for date, g in pred.groupby("date", sort=True):
        if len(g) != 24:
            continue
        true_price = g["y_true"].to_numpy()
        for source in ["base_forecast", "corrected_forecast", "oracle"]:
            signal = true_price if source == "oracle" else g[source].to_numpy()
            dispatch = optimize_storage_dispatch(signal, spec)
            rows.append({
                "date": date,
                "forecast_source": source,
                "realized_revenue": realized_revenue(dispatch, true_price),
                "forecast_objective_revenue": dispatch["forecast_objective_revenue"],
                "total_charge_mwh": float(dispatch["charge_mw"].sum()),
                "total_discharge_mwh": float(dispatch["discharge_mw"].sum()),
            })

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "storage_dispatch.csv", index=False)
    summary = out.groupby("forecast_source")["realized_revenue"].agg(["mean", "sum", "std"])
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
