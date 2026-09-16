from __future__ import annotations

import numpy as np
import pandas as pd


class Chronos2Forecaster:
    """Optional Chronos-family backend used by the public experiment."""

    name = "chronos2"

    def __init__(self, model_id="amazon/chronos-2", device_map="auto", quantile=0.5):
        self.model_id = model_id
        self.device_map = device_map
        self.quantile = float(quantile)
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-chronos.txt before using --base-model chronos2"
            ) from exc
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id, device_map=self.device_map
        )

    def _reduce_forecast(self, forecast, batch_size: int) -> list[np.ndarray]:
        if hasattr(forecast, "detach"):
            forecast = forecast.detach().cpu().numpy()
        arr = np.asarray(forecast)
        if arr.ndim == 3:
            return [np.quantile(arr[i], self.quantile, axis=0).astype(float) for i in range(batch_size)]
        if arr.ndim == 2 and batch_size == 1:
            return [arr[0].astype(float)]
        if arr.ndim == 1 and batch_size == 1:
            return [arr.astype(float)]
        raise RuntimeError(f"Unexpected Chronos output shape: {arr.shape}")

    def predict_batch(self, histories: list[pd.DataFrame], futures: list[pd.DataFrame]) -> list[np.ndarray]:
        """Run compatible forecast requests in one model call.

        Requests in one batch must share a horizon. Context lengths may differ.
        """
        if len(histories) != len(futures):
            raise ValueError("histories and futures must have equal length")
        if not histories:
            return []
        horizons = {len(f) for f in futures}
        if len(horizons) != 1:
            raise ValueError("all requests in a Chronos batch must share a horizon")

        self._load()
        contexts = [h["price"].to_numpy(dtype=np.float32) for h in histories]
        forecast = self._pipeline.predict(
            context=contexts, prediction_length=int(next(iter(horizons)))
        )
        return self._reduce_forecast(forecast, len(histories))

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        return self.predict_batch([history], [future])[0]

    def one_step_historical(self, frame: pd.DataFrame, min_history=168, stride=24) -> np.ndarray:
        out = np.full(len(frame), np.nan, dtype=float)
        start = min_history
        while start < len(frame):
            end = min(start + stride, len(frame))
            context = frame.iloc[:start].copy()
            future = frame.iloc[start:end].copy()
            out[start:end] = self.predict(context, future)[: end - start]
            start = end
        return out
