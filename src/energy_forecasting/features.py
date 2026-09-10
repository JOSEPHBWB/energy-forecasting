from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "base_forecast",
    "load",
    "temperature",
    "fuel_cost",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def build_feature_table(frame: pd.DataFrame, base_forecast: np.ndarray) -> pd.DataFrame:
    if len(frame) != len(base_forecast):
        raise ValueError("frame and base_forecast must have equal length")

    ts = pd.to_datetime(frame["timestamp"])
    hour = ts.dt.hour.to_numpy(dtype=float)
    dow = ts.dt.dayofweek.to_numpy(dtype=float)

    return pd.DataFrame(
        {
            "base_forecast": np.asarray(base_forecast, dtype=float),
            "load": frame["load"].to_numpy(dtype=float),
            "temperature": frame["temperature"].to_numpy(dtype=float),
            "fuel_cost": frame["fuel_cost"].to_numpy(dtype=float),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        },
        index=frame.index,
    )[FEATURE_COLUMNS]
