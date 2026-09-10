from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def relative_improvement(baseline: float, candidate: float) -> float:
    return float("nan") if baseline == 0 else 100.0 * (baseline - candidate) / baseline
