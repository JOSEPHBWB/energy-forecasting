from __future__ import annotations
from typing import Protocol
import numpy as np
import pandas as pd

class BaseForecaster(Protocol):
    name: str
    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray: ...

class SeasonalBlendForecaster:
    name = "seasonal_blend"
    def __init__(self, daily_weight=0.50, weekly_weight=0.35):
        if daily_weight < 0 or weekly_weight < 0 or daily_weight + weekly_weight > 1:
            raise ValueError("invalid weights")
        self.daily_weight=float(daily_weight); self.weekly_weight=float(weekly_weight)
        self.mean_weight=1.0-self.daily_weight-self.weekly_weight

    def one_step_historical(self, frame: pd.DataFrame, min_history: int = 168) -> np.ndarray:
        price=frame["price"].to_numpy(dtype=float)
        out=np.full(len(price), np.nan, dtype=float)
        prefix=np.concatenate([[0.0], np.cumsum(price)])
        for i in range(min_history, len(price)):
            lag24=price[i-24]; lag168=price[i-168]
            recent=(prefix[i]-prefix[i-24])/24.0
            out[i]=self.daily_weight*lag24+self.weekly_weight*lag168+self.mean_weight*recent
        return out

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        prices=history["price"].to_numpy(dtype=float).tolist(); preds=[]
        for _ in range(len(future)):
            arr=np.asarray(prices,dtype=float)
            lag24=arr[-24] if len(arr)>=24 else arr[-1]
            lag168=arr[-168] if len(arr)>=168 else lag24
            recent=float(np.mean(arr[-24:]))
            pred=self.daily_weight*lag24+self.weekly_weight*lag168+self.mean_weight*recent
            preds.append(float(pred)); prices.append(float(pred))
        return np.asarray(preds,dtype=float)
