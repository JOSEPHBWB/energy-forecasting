from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = ROOT / "results"
f = d / "figures"
f.mkdir(parents=True, exist_ok=True)

m = pd.read_csv(d / "metrics.csv")
for task, g in m.groupby("task"):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(g["window"], g["base_MAE"], marker="o", label="Base forecast")
    ax.plot(g["window"], g["corrected_MAE"], marker="o", label="Base + LightGBM correction")
    ax.set_xlabel("Monthly rolling test window")
    ax.set_ylabel("MAE")
    ax.set_title(f"{task.replace('_', ' ').title()} rolling-window forecasting error")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    out = f / f"rolling_mae_{task}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("created", out)

p = pd.read_csv(d / "predictions.csv", parse_dates=["timestamp"])
for task, g in p.groupby("task"):
    g = g.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    chg = np.where(np.diff(g["regime"].to_numpy()) != 0)[0]
    c = int(chg[0] + 1) if len(chg) else len(g) // 2
    v = g.iloc[max(0, c - 72):min(len(g), c + 120)].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(v["timestamp"], v["y_true"], label="Observed")
    ax.plot(v["timestamp"], v["base_forecast"], label="Base forecast")
    ax.plot(v["timestamp"], v["corrected_forecast"], label="Base + LightGBM correction")
    lc = np.where(np.diff(v["regime"].to_numpy()) != 0)[0]
    if len(lc):
        ax.axvline(v["timestamp"].iloc[int(lc[0] + 1)], linestyle="--", linewidth=1.2, label="Regime change")
    ax.set_xlabel("Time")
    ax.set_ylabel("Synthetic electricity price")
    ax.set_title(f"{task.replace('_', ' ').title()} forecast behavior around regime change")
    ax.legend()
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = f / f"regime_change_{task}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("created", out)

# Optional downstream storage figure if the storage experiment has been run.
storage_path = d / "storage_dispatch.csv"
if storage_path.exists():
    s = pd.read_csv(storage_path, parse_dates=["date"])
    daily = s.pivot(index="date", columns="forecast_source", values="realized_revenue").sort_index()
    cumulative = daily.cumsum()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col in cumulative.columns:
        ax.plot(cumulative.index, cumulative[col], label=col.replace("_", " ").title())
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative realized revenue")
    ax.set_title("Synthetic battery dispatch value under alternative price signals")
    ax.legend()
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = f / "storage_cumulative_revenue.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("created", out)
