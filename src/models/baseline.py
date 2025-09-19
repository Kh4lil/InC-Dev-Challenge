from __future__ import annotations
import numpy as np
import pandas as pd
from pandas import IndexSlice as S

class YesterdayPremiumModel:
    """
    Predict today's expected premium per hour as yesterday's realized premium
    for the same hour, where available; otherwise 0.
    """

    def __init__(self):
        self._yesterday: pd.Series | None = None
        self._last_day: pd.Timestamp | None = None

    def fit(self, df_past: pd.DataFrame) -> "YesterdayPremiumModel":
        # df_past indexed by (entry_time, hour) and includes 'premium'
        days = df_past.index.get_level_values("entry_time").unique().sort_values()
        if len(days) >= 1:
            self._last_day = days[-1]
            self._yesterday = df_past.loc[S[self._last_day, :], "premium"].astype(float)
        else:
            self._yesterday = None
            self._last_day = None
        return self

    def predict(self, df_today: pd.DataFrame) -> np.ndarray:
        if self._yesterday is None:
            return np.zeros(24)
        vec = np.zeros(24)
        for h in range(24):
            if h in self._yesterday.index:
                vec[h] = float(self._yesterday.at[h])
            else:
                vec[h] = 0.0
        return vec
