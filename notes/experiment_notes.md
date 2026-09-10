# Experiment notes

These are the main choices behind the public version of the project.

## Why use a simple seasonal base model by default?

I wanted the public example to run without a GPU. Electricity prices have strong hourly and weekly structure, so lag-24 and lag-168 give a reasonable sanity-check baseline. The Chronos adapter is optional rather than required.

## Why predict residuals with LightGBM?

The base forecast already contains temporal information. Instead of asking LightGBM to relearn the whole price series, I train it on `y - base_forecast`. The features include the base prediction and synthetic load/weather/fuel variables. This makes it easy to check whether the second model adds information beyond the base forecast.

## Why rolling windows instead of a random split?

A random split would let future observations appear in the training set. For forecasting I keep chronological order. The first window has 180 days of history; after each 30-day test month, the available training history expands.

## DA versus RT

I keep two targets because real-time prices should be harder and noisier than day-ahead prices. In the synthetic generator I add imbalance noise and additional scarcity shocks to RT. This is only a controlled toy market, not a calibrated model of a particular ISO.

## Why cache forecasts?

When the base model is expensive, rerunning the same historical forecast while changing the correction model wastes time. The cache is keyed by model, target, task, and window boundaries. The cache benchmark is mainly there to make this cost visible.

## Storage experiment

The storage experiment is deliberately separate from forecasting. For each day, the optimizer receives a 24-hour price signal and chooses charge/discharge decisions. I then evaluate those fixed decisions against the realized RT price. The oracle is useful as an upper reference but is not a feasible forecasting strategy.

## What this repo does not reproduce

- real Shandong, Guangdong, Hainan, or NSW market data;
- cross-market transfer experiments;
- the original lab's multi-GPU benchmark numbers;
- a dedicated extreme-price classifier;
- dynamic-programming storage scheduling.

Those pieces are intentionally not claimed by this public synthetic implementation.
