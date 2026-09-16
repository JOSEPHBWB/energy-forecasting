from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastRequest:
    """One forecast job identified by its information set and horizon."""

    series_id: str
    origin: str
    horizon: int
    model_name: str
    context_fingerprint: str

    @classmethod
    def from_frames(cls, *, series_id: str, history: pd.DataFrame,
                    future: pd.DataFrame, model_name: str) -> "ForecastRequest":
        values = history["price"].to_numpy(dtype=np.float64)
        digest = hashlib.sha256(values.tobytes()).hexdigest()[:20]
        if "timestamp" in history.columns and len(history):
            origin = str(history["timestamp"].iloc[-1])
        else:
            origin = str(len(history))
        return cls(series_id, origin, len(future), model_name, digest)

    def cache_key(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]


class PredictionStore:
    """Small disk-backed store for independent forecast requests."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, request: ForecastRequest) -> Path:
        return self.directory / f"{request.cache_key()}.npy"

    def get(self, request: ForecastRequest) -> np.ndarray | None:
        path = self._path(request)
        return np.load(path) if path.exists() else None

    def put(self, request: ForecastRequest, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != (request.horizon,):
            raise ValueError(f"forecast has shape {values.shape}; expected {(request.horizon,)}")
        np.save(self._path(request), values)

    def remove(self, request: ForecastRequest) -> None:
        self._path(request).unlink(missing_ok=True)


@dataclass
class ForecastBatchResult:
    predictions: list[np.ndarray]
    cache_hits: int
    cache_misses: int
    model_calls: int

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0


class ForecastEngine:
    """Resolve forecast jobs from cache and compute only unresolved requests.

    The engine owns cache policy; the forecasting model only owns inference.
    This separation makes cache behavior testable without a GPU or Chronos.
    """

    def __init__(self, forecaster, store: PredictionStore):
        self.forecaster = forecaster
        self.store = store

    def run(self, jobs: Iterable[tuple[ForecastRequest, pd.DataFrame, pd.DataFrame]]) -> ForecastBatchResult:
        jobs = list(jobs)
        output: list[np.ndarray | None] = [None] * len(jobs)
        missing_indices: list[int] = []

        for i, (request, _history, _future) in enumerate(jobs):
            cached = self.store.get(request)
            if cached is None:
                missing_indices.append(i)
            else:
                output[i] = cached

        model_calls = 0
        if missing_indices:
            missing_jobs = [jobs[i] for i in missing_indices]
            histories = [job[1] for job in missing_jobs]
            futures = [job[2] for job in missing_jobs]

            if hasattr(self.forecaster, "predict_batch"):
                fresh = self.forecaster.predict_batch(histories, futures)
                model_calls = 1
            else:
                fresh = [self.forecaster.predict(h, f) for h, f in zip(histories, futures)]
                model_calls = len(missing_jobs)

            if len(fresh) != len(missing_jobs):
                raise RuntimeError("forecaster returned the wrong number of forecasts")

            for idx, values in zip(missing_indices, fresh):
                request = jobs[idx][0]
                values = np.asarray(values, dtype=float)
                self.store.put(request, values)
                output[idx] = values

        return ForecastBatchResult(
            predictions=[np.asarray(x, dtype=float) for x in output],
            cache_hits=len(jobs) - len(missing_indices),
            cache_misses=len(missing_indices),
            model_calls=model_calls,
        )
