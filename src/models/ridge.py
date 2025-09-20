from __future__ import annotations
import numpy as np
import pandas as pd

EXCLUDE = {"entry_price", "exit_price", "premium"}

class RidgeModel:
    """Minimal ridge on past features → premium; returns a 24-length hourly vector."""

    def __init__(self, alpha: float = 1.0, standardize: bool = True):
        self.alpha = float(alpha)
        self.standardize = bool(standardize)
        self.cols: list[str] | None = None
        self.mu: np.ndarray | None = None
        self.sig: np.ndarray | None = None
        self.w: np.ndarray | None = None 

    def _select_cols(self, df: pd.DataFrame) -> list[str]:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return [c for c in nums if c not in EXCLUDE]

    def fit(self, df_past: pd.DataFrame) -> "RidgeModel":
        if df_past.empty:
            self.cols, self.mu, self.sig, self.w = [], None, None, None
            return self

        # columns and X/y
        cols = self._select_cols(df_past)
        X = df_past[cols].astype("float64")
        y = df_past["premium"].astype("float64").to_numpy()

        allnan = X.columns[X.isna().all()].tolist()
        if allnan:
            cols = [c for c in cols if c not in allnan]
            X = X[cols]
        X = X.fillna(0.0)

        # standardization
        if self.standardize and len(cols) > 0:
            mu = X.mean(axis=0).to_numpy(dtype="float64")
            sig = X.std(axis=0, ddof=0).to_numpy(dtype="float64")
            sig[sig == 0.0] = 1.0
            Xn = (X.to_numpy(dtype="float64") - mu) / sig
            self.mu, self.sig = mu, sig
        else:
            Xn = X.to_numpy(dtype="float64")
            self.mu = self.sig = None

        # bias column
        Xb = np.c_[np.ones((Xn.shape[0], 1), dtype="float64"), Xn]

        # Tikhonov augmentation for ridge
        p = Xb.shape[1]
        R = np.eye(p, dtype="float64")
        R[0, 0] = 0.0
        Xa = np.vstack([Xb, np.sqrt(self.alpha) * R])
        ya = np.concatenate([y, np.zeros(p, dtype="float64")])

        w, *_ = np.linalg.lstsq(Xa, ya, rcond=None)

        self.cols = cols
        self.w = w  # (p,)
        return self

    def predict(self, df_today: pd.DataFrame) -> np.ndarray:
        vec = np.zeros(24, dtype="float64")
        if self.w is None or self.cols is None or df_today.empty:
            return vec

        Xt = df_today[self.cols].astype("float64").fillna(0.0).to_numpy()
        if self.standardize and (self.mu is not None) and (self.sig is not None):
            Xt = (Xt - self.mu) / self.sig

        Xb = np.c_[np.ones((Xt.shape[0], 1), dtype="float64"), Xt]
        yhat = Xb @ self.w  # (#rows_today,)

        hours = df_today.index.get_level_values("hour").astype(int).to_numpy()
        for h, pred in zip(hours, yhat):
            if 0 <= h < 24:
                vec[h] = float(pred)
        return vec
