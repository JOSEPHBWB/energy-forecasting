from __future__ import annotations
import numpy as np
import pandas as pd

class Chronos2Forecaster:
    """Optional Chronos-family adapter. All model-specific code lives here."""
    name = "chronos2"
    def __init__(self, model_id="amazon/chronos-2", device_map="auto", quantile=0.5):
        self.model_id=model_id; self.device_map=device_map; self.quantile=float(quantile); self._pipeline=None

    def _load(self):
        if self._pipeline is not None: return
        try:
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-chronos.txt before using --base-model chronos2") from exc
        self._pipeline=ChronosPipeline.from_pretrained(self.model_id, device_map=self.device_map)

    def _predict_array(self, context: np.ndarray, horizon: int) -> np.ndarray:
        self._load()
        forecast=self._pipeline.predict(context=[context.astype(np.float32)], prediction_length=int(horizon))
        if hasattr(forecast, "detach"): forecast=forecast.detach().cpu().numpy()
        forecast=np.asarray(forecast)
        if forecast.ndim==3: return np.quantile(forecast[0], self.quantile, axis=0).astype(float)
        if forecast.ndim==2: return forecast[0].astype(float)
        if forecast.ndim==1: return forecast.astype(float)
        raise RuntimeError(f"Unexpected Chronos output shape: {forecast.shape}")

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        return self._predict_array(history["price"].to_numpy(dtype=float), len(future))

    def one_step_historical(self, frame: pd.DataFrame, min_history=168, stride=24) -> np.ndarray:
        out=np.full(len(frame), np.nan, dtype=float)
        start=min_history
        while start < len(frame):
            end=min(start+stride, len(frame))
            context=frame.iloc[:start]["price"].to_numpy(dtype=float)
            out[start:end]=self._predict_array(context, end-start)[:end-start]
            start=end
        return out
