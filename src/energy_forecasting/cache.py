from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class ForecastCache:
    """Simple file cache for base-model predictions."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def path_for(self, payload: dict) -> Path:
        return self.directory / f"{self._key(payload)}.npy"

    def load(self, payload: dict) -> np.ndarray | None:
        path = self.path_for(payload)
        return np.load(path) if path.exists() else None

    def save(self, payload: dict, values: np.ndarray) -> Path:
        path = self.path_for(payload)
        np.save(path, np.asarray(values, dtype=float))
        return path
