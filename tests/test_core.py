from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_forecasting.correction import CorrectionRegressor
from energy_forecasting.data import make_synthetic_market
from energy_forecasting.metrics import regression_metrics, relative_improvement
from energy_forecasting.storage import BatterySpec, optimize_storage_dispatch, realized_revenue
from energy_forecasting.windows import rolling_windows


def test_synthetic_market_shape():
    df = make_synthetic_market(n_days=10, seed=1)
    assert len(df) == 240
    assert {"timestamp", "price_day_ahead", "price_real_time", "load", "temperature", "fuel_cost", "regime"} <= set(df.columns)


def test_rolling_windows_are_chronological():
    windows = list(rolling_windows(570 * 24, 180 * 24, 30 * 24, 30 * 24, 12))
    assert len(windows) == 12
    for w in windows:
        assert w.train_end == w.test_start


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0])
    m = regression_metrics(y, pred)
    assert abs(m["MAE"] - 1 / 3) < 1e-9
    assert relative_improvement(2.0, 1.0) == 50.0


def test_lightgbm_correction_backend():
    X = np.arange(80, dtype=float).reshape(40, 2)
    base = np.linspace(10, 20, 40)
    y = base + 0.1 * X[:, 0]
    model = CorrectionRegressor(random_state=1).fit(X, y, base)
    pred = model.predict(X, base)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_storage_milp_feasible_and_terminal_soc():
    prices = np.array([10.0] * 12 + [100.0] * 12)
    spec = BatterySpec(capacity_mwh=4.0, max_power_mw=1.0, initial_soc_mwh=2.0, terminal_soc_mwh=2.0)
    d = optimize_storage_dispatch(prices, spec)
    assert np.all(d["soc_mwh"] >= -1e-7)
    assert np.all(d["soc_mwh"] <= spec.capacity_mwh + 1e-7)
    assert abs(d["soc_mwh"][-1] - spec.terminal_soc_mwh) < 1e-6
    assert np.all(d["charge_mw"] * d["discharge_mw"] < 1e-7)
    assert realized_revenue(d, prices) > 0

from energy_forecasting.base_forecaster import SeasonalBlendForecaster
from energy_forecasting.forecast_engine import ForecastEngine, ForecastRequest, PredictionStore


def _forecast_job(frame, model, origin, horizon=24):
    import pandas as pd
    history = frame.iloc[:origin].copy()
    history["price"] = history["price_real_time"]
    future = frame.iloc[origin:origin + horizon].copy()
    future["price"] = future["price_real_time"]
    req = ForecastRequest.from_frames(
        series_id="test_rt", history=history, future=future, model_name=model.name
    )
    return req, history, future


def test_request_cache_supports_partial_hits(tmp_path):
    frame = make_synthetic_market(n_days=40, seed=3)
    model = SeasonalBlendForecaster()
    store = PredictionStore(tmp_path / "forecast_cache")
    engine = ForecastEngine(model, store)
    jobs = [_forecast_job(frame, model, 30 * 24 + i * 24) for i in range(4)]

    cold = engine.run(jobs)
    assert cold.cache_hits == 0
    assert cold.cache_misses == 4

    store.remove(jobs[1][0])
    store.remove(jobs[3][0])
    partial = engine.run(jobs)
    assert partial.cache_hits == 2
    assert partial.cache_misses == 2
    assert partial.hit_rate == 0.5

    warm = engine.run(jobs)
    assert warm.cache_hits == 4
    assert warm.cache_misses == 0
    for a, b in zip(cold.predictions, warm.predictions):
        assert np.allclose(a, b)


def test_request_key_changes_when_context_changes():
    frame = make_synthetic_market(n_days=40, seed=4)
    model = SeasonalBlendForecaster()
    a = _forecast_job(frame, model, 30 * 24)[0]
    changed = frame.copy()
    changed.loc[10, "price_real_time"] += 1.0
    b = _forecast_job(changed, model, 30 * 24)[0]
    assert a.cache_key() != b.cache_key()
