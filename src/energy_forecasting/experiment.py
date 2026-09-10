from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .cache import ForecastCache
from .correction import CorrectionRegressor
from .features import build_feature_table
from .metrics import regression_metrics, relative_improvement
from .profiler import profile_stage
from .windows import rolling_windows


def _historical_base_features(frame, forecaster, min_history=168):
    if hasattr(forecaster, "one_step_historical"):
        return forecaster.one_step_historical(frame, min_history=min_history)
    out = np.full(len(frame), np.nan, dtype=float)
    for i in range(min_history, len(frame)):
        out[i] = forecaster.predict(frame.iloc[:i], frame.iloc[i:i + 1])[0]
    return out


def _target_frame(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = frame.copy()
    if target_col not in out.columns:
        raise KeyError(f"target column {target_col!r} not found")
    out["price"] = out[target_col].astype(float)
    return out


def run_experiment(
    frame,
    *,
    forecaster,
    train_days,
    test_days,
    step_days,
    max_windows,
    cache_dir,
    results_dir,
    random_state=7,
    target_col="price_day_ahead",
    market_task="day_ahead",
):
    frame = _target_frame(frame, target_col)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache = ForecastCache(cache_dir)
    metrics_rows, efficiency_rows, prediction_rows = [], [], []

    for w in rolling_windows(len(frame), train_days * 24, test_days * 24, step_days * 24, max_windows):
        train = frame.iloc[w.train_start:w.train_end].copy()
        test = frame.iloc[w.test_start:w.test_end].copy()
        tr_key = {
            "kind": "historical_base", "task": market_task, "target": target_col,
            "window": w.window_id, "train_end": w.train_end, "model": forecaster.name,
        }
        with profile_stage(f"{market_task}:window_{w.window_id}:train_base", efficiency_rows):
            tr_base = cache.load(tr_key)
            tr_hit = tr_base is not None
            if tr_base is None:
                tr_base = _historical_base_features(train, forecaster)
                cache.save(tr_key, tr_base)

        te_key = {
            "kind": "test_rollout", "task": market_task, "target": target_col,
            "window": w.window_id, "train_end": w.train_end,
            "test_end": w.test_end, "model": forecaster.name,
        }
        with profile_stage(f"{market_task}:window_{w.window_id}:test_base", efficiency_rows):
            te_base = cache.load(te_key)
            te_hit = te_base is not None
            if te_base is None:
                te_base = forecaster.predict(train, test)
                cache.save(te_key, te_base)

        valid = np.isfinite(tr_base)
        with profile_stage(f"{market_task}:window_{w.window_id}:features", efficiency_rows):
            Xtr = build_feature_table(train.loc[valid], tr_base[valid])
            Xte = build_feature_table(test, te_base)
            ytr = train.loc[valid, "price"].to_numpy(dtype=float)
            yte = test["price"].to_numpy(dtype=float)

        model = CorrectionRegressor(random_state=random_state)
        with profile_stage(f"{market_task}:window_{w.window_id}:fit", efficiency_rows):
            model.fit(Xtr.to_numpy(), ytr, tr_base[valid])
        with profile_stage(f"{market_task}:window_{w.window_id}:predict", efficiency_rows):
            corr = model.predict(Xte.to_numpy(), te_base)

        bm = regression_metrics(yte, te_base)
        cm = regression_metrics(yte, corr)
        metrics_rows.append({
            "task": market_task,
            "window": w.window_id + 1,
            "test_start": test["timestamp"].iloc[0],
            "test_end": test["timestamp"].iloc[-1],
            "base_model": forecaster.name,
            "base_MAE": bm["MAE"],
            "corrected_MAE": cm["MAE"],
            "MAE_improvement_pct": relative_improvement(bm["MAE"], cm["MAE"]),
            "base_RMSE": bm["RMSE"],
            "corrected_RMSE": cm["RMSE"],
            "train_base_cache_hit": tr_hit,
            "test_base_cache_hit": te_hit,
        })
        prediction_rows.append(pd.DataFrame({
            "task": market_task,
            "window": w.window_id + 1,
            "base_model": forecaster.name,
            "timestamp": test["timestamp"].to_numpy(),
            "regime": test["regime"].to_numpy(),
            "y_true": yte,
            "base_forecast": te_base,
            "corrected_forecast": corr,
        }))

    metrics = pd.DataFrame(metrics_rows)
    efficiency = pd.DataFrame(efficiency_rows)
    preds = pd.concat(prediction_rows, ignore_index=True)
    return metrics, efficiency, preds
