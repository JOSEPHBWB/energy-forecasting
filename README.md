# Electricity Price Forecasting + Storage Dispatch

This repository is an independent public implementation of forecasting and computational-efficiency ideas I worked on during my research. The full experiment is designed to run on a normal laptop, with Chronos-2 available as an optional backend.

The project has four parts:

1. synthetic day-ahead (DA) and real-time (RT) electricity prices;
2. a base time-series forecast followed by a LightGBM residual correction;
3. 12 chronological monthly evaluation windows;
4. a small battery-dispatch MILP that uses the RT forecasts.

The repository also includes a request-level forecast cache, incremental recomputation, optional batched Chronos-2 inference, and CUDA-aware runtime/memory measurements. These components are implemented independently for this public project.

## Forecast correction

The correction model does not replace the base forecast. It learns the remaining error:

```text
residual = observed price - base forecast
corrected forecast = base forecast + LightGBM residual prediction
```

The default base model is deliberately simple (daily/weekly seasonal lags) so the repository runs without a GPU. There is also an optional Chronos-2 adapter in `chronos_adapter.py` for machines with the required model dependencies.

## DA and RT experiment

The synthetic generator creates separate DA and RT series. RT has extra imbalance noise and scarcity shocks, so I evaluate the two targets separately.

The default setup starts with 180 days of training data and then evaluates 12 consecutive 30-day windows. Training history expands after each month. There is no random train/test split.

Run it with:

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py --base-model seasonal --task both
```

Outputs are saved under `results/`:

```text
metrics.csv
predictions.csv
efficiency.csv
```

To try the optional Chronos-2 base model:

```bash
python -m pip install -r requirements-chronos.txt
python scripts/run_demo.py --base-model chronos2 --task both
```

I have not rerun the original lab GPU experiments in this public repository. The profiler records CPU memory and wall time everywhere, and records CUDA peak memory when CUDA is available.

## Incremental inference and cache benchmark

Repeated foundation-model inference can dominate rolling evaluation. The public implementation separates inference from cache policy: each forecast request is keyed by its series, forecast origin, horizon, model, and a fingerprint of the historical context. A batch is resolved into cache hits and misses; only unresolved requests are sent to the model, then the new predictions are merged back into the store.

This makes partial reuse measurable rather than treating an entire experiment window as one all-or-nothing cache entry. The benchmark constructs three deterministic scenarios:

- **cold cache:** every request requires inference;
- **partial cache:** half of the requests are reused and only the missing half are recomputed;
- **warm cache:** every request is reused and the model is not called.

Run:

```bash
python benchmarks/benchmark_incremental_cache.py
```

The output includes cache-hit rate, number of model calls, wall time, process RSS, and CUDA peak allocated/reserved memory when CUDA is available. CUDA timing is synchronized around measured stages so asynchronous execution is included in wall time. The optional Chronos-2 adapter exposes batched prediction for compatible requests.

The older whole-window cache benchmark remains available for comparison:

```bash
python benchmarks/benchmark_cache.py --base-model seasonal --task day_ahead
```

## Storage experiment

The storage example uses the synthetic RT forecasts to schedule a battery. I formulate a 24-hour MILP with:

- state-of-charge dynamics;
- charge/discharge efficiencies;
- power and energy limits;
- a terminal SOC constraint;
- a binary mode variable to prevent charging and discharging at the same time.

Run forecasting first, then:

```bash
python scripts/run_storage_demo.py
```

The script compares dispatch based on the base forecast, corrected forecast, and an oracle using realized prices. Decisions are evaluated using the realized synthetic RT price.

## Files I would look at first

```text
src/energy_forecasting/data.py          synthetic DA/RT data
src/energy_forecasting/experiment.py    rolling forecasting experiment
src/energy_forecasting/correction.py    LightGBM residual model
src/energy_forecasting/storage.py       battery MILP
src/energy_forecasting/forecast_engine.py request cache / incremental inference
src/energy_forecasting/runtime_monitor.py CUDA-safe benchmark measurements
src/energy_forecasting/profiler.py      experiment-stage profiling
scripts/run_demo.py                     main forecasting script
scripts/run_storage_demo.py             storage experiment
```

`notes/experiment_notes.md` explains the main modeling choices and limitations.

## Important limitation

This repository is an independent synthetic implementation. It does not contain FutureBoosting source code, private lab code, private electricity-market data, or the original Shandong / Guangdong / Hainan / NSW experiments. Numbers produced here are synthetic-demo results and should not be interpreted as results from those markets. Original research benchmarks are not presented as results reproduced by this repository.
