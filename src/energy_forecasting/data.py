from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_market(n_days: int = 570, seed: int = 7) -> pd.DataFrame:
    """Generate hourly synthetic day-ahead and real-time electricity prices.

    The construction is intentionally simple but includes daily/weekly seasonality,
    weather/load/fuel covariates, scarcity spikes, and a later structural regime shift.
    Real-time prices are more volatile and contain additional imbalance shocks.
    """
    rng = np.random.default_rng(seed)
    n = int(n_days) * 24
    time = pd.date_range("2024-01-01", periods=n, freq="h")

    hour = time.hour.to_numpy()
    day = np.arange(n) / 24.0
    dow = time.dayofweek.to_numpy()

    temperature = (
        16
        + 10 * np.sin(2 * np.pi * day / 365 - 0.8)
        + 3 * np.sin(2 * np.pi * day / 7)
        + rng.normal(0, 1.8, n)
    )

    morning = np.exp(-0.5 * ((hour - 9) / 2.8) ** 2)
    evening = np.exp(-0.5 * ((hour - 19) / 3.0) ** 2)
    weekday = (dow < 5).astype(float)

    load = (
        72
        + 15 * morning
        + 23 * evening
        + 7 * weekday
        + 0.35 * np.maximum(temperature - 24, 0) ** 1.4
        + rng.normal(0, 3.0, n)
    )

    fuel = (
        28
        + 0.012 * np.arange(n)
        + 2.5 * np.sin(2 * np.pi * day / 45)
        + rng.normal(0, 0.7, n)
    )

    daily_price = 6 * np.sin(2 * np.pi * (hour - 7) / 24)
    weekly_price = 3 * np.cos(2 * np.pi * day / 7)

    regime = (np.arange(n) >= int(0.70 * n)).astype(float)
    load_beta = 0.55 + 0.20 * regime
    fuel_beta = 0.75 - 0.25 * regime

    scarcity_prob = 0.004 + 0.012 * (load > np.quantile(load, 0.90))
    scarcity = rng.random(n) < scarcity_prob
    da_spikes = scarcity * rng.gamma(shape=2.0, scale=24.0, size=n)

    price_day_ahead = (
        5
        + daily_price
        + weekly_price
        + load_beta * load
        + fuel_beta * fuel
        + 0.18 * np.maximum(25 - temperature, 0)
        + da_spikes
        + rng.normal(0, 3.5, n)
    )

    imbalance = rng.normal(0, 5.0, n)
    rt_scarcity = (rng.random(n) < (scarcity_prob * 1.35)) * rng.gamma(
        shape=2.1, scale=31.0, size=n
    )
    price_real_time = (
        price_day_ahead
        + 0.45 * imbalance
        + 0.12 * (load - pd.Series(load).rolling(24, min_periods=1).mean().to_numpy())
        + rt_scarcity
        + rng.normal(0, 3.2, n)
    )

    return pd.DataFrame(
        {
            "timestamp": time,
            "price_day_ahead": price_day_ahead.astype(float),
            "price_real_time": price_real_time.astype(float),
            # Backward-compatible default target.
            "price": price_day_ahead.astype(float),
            "load": load.astype(float),
            "temperature": temperature.astype(float),
            "fuel_cost": fuel.astype(float),
            "regime": regime.astype(int),
        }
    )
