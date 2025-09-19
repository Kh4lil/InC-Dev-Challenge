from __future__ import annotations
import pandas as pd
from pathlib import Path

REQUIRED = {
    "entry_time", "exit_time", "entry_price", "exit_price"
}

def _to_utc(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC")

class CSVDayaheadDataSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, *, end_now: pd.Timestamp | None = None) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["entry_time", "exit_time"]).copy()
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["entry_time"] = _to_utc(df["entry_time"])
        df["exit_time"]  = _to_utc(df["exit_time"])

        df["hour"] = df["exit_time"].dt.hour
        df["premium"] = df["exit_price"] - df["entry_price"]
        df = df.set_index(["entry_time", "hour"]).sort_index()
        if end_now is not None:
            end_now = pd.Timestamp(end_now)
            end_now = end_now.tz_convert("UTC") if end_now.tzinfo else end_now.tz_localize("UTC")
            df = df.loc[pd.IndexSlice[: end_now, :], :]

        return df
