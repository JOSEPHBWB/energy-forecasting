import numpy as np
from lightgbm import LGBMRegressor


class CorrectionRegressor:
    """LightGBM model trained on the error of the base forecast."""

    def __init__(self, random_state=7):
        self.model = LGBMRegressor(
            objective="regression_l2",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_lambda=0.2,
            random_state=random_state,
            verbosity=-1,
        )

    def fit(self, X, y, base_forecast):
        y = np.asarray(y, dtype=float)
        base_forecast = np.asarray(base_forecast, dtype=float)
        residual = y - base_forecast
        self.model.fit(X, residual)
        return self

    def predict(self, X, base_forecast):
        base_forecast = np.asarray(base_forecast, dtype=float)
        residual_prediction = self.model.predict(X)
        return base_forecast + residual_prediction
