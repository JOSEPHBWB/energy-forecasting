# Electricity Price Forecasting + Storage Dispatch

This is a small public version of the type of forecasting experiments I worked on as an RA. The original research code and market data are not included here. I rewrote the main ideas with synthetic data so the full experiment can be run on a normal laptop.

The project has four parts:

1. synthetic day-ahead (DA) and real-time (RT) electricity prices;
2. a base time-series forecast followed by a LightGBM residual correction;
3. 12 chronological monthly evaluation windows;
4. a small battery-dispatch MILP that uses the RT forecasts.

I also kept the runtime/cache profiling because computational cost was one of the things I looked at in the research project.

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

## Cache/runtime check

Base forecasts are cached because repeated foundation-model inference can dominate runtime. The cache benchmark compares the first run with a second run that reuses the saved forecasts:

```bash
python benchmarks/benchmark_cache.py --base-model seasonal --task day_ahead
```

For CUDA models, timing is synchronized before/after a measured stage so asynchronous GPU execution is included in wall time.

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
src/energy_forecasting/profiler.py      runtime / memory measurements
scripts/run_demo.py                     main forecasting script
scripts/run_storage_demo.py             storage experiment
```

`notes/experiment_notes.md` explains the main modeling choices and limitations.

## Important limitation

This repository is an independent synthetic example. It does not contain private lab code, private electricity-market data, or the original Shandong / Guangdong / Hainan / NSW experiments. Numbers produced here are synthetic-demo results and should not be interpreted as results from those markets.
