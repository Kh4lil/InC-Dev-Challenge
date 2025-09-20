from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, Union

REQUIRED = {
    "entry_time", "exit_time", "entry_price", "exit_price"
}

# only features KNOWN AT ENTRY are allowed to flow downstream.
ALLOWED_FEATURE_PREFIXES: tuple[str, ...] = (
    "wind_forecast_",
    "solar_forecast_",
    "load_forecast_",
)

def _to_utc(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC")

class CSVDayaheadDataSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, end_now: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        df = pd.read_csv(self.path)

        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Parse timestamps
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=False, errors="raise")
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=False, errors="raise")

        df["entry_time"] = _to_utc(df["entry_time"])
        df["exit_time"] = _to_utc(df["exit_time"])

        # Compute delivery hour and realized premium
        df["hour"] = df["exit_time"].dt.hour.astype(int)
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
        df["premium"] = df["exit_price"] - df["entry_price"]

        # Known-at-entry feature whitelist
        keep_cols: list[str] = ["entry_time", "exit_time", "entry_price", "exit_price", "hour", "premium"]
        for c in df.columns:
            if any(c.startswith(pfx) for pfx in ALLOWED_FEATURE_PREFIXES):
                keep_cols.append(c)
        keep_cols = list(dict.fromkeys(keep_cols))
        df = df[keep_cols].copy()

        # MultiIndex by (entry_time, hour), sorted
        df = df.set_index(["entry_time", "hour"]).sort_index()

        # Filter by end_now
        if end_now is not None:
            end_now = pd.Timestamp(end_now)
            end_now = end_now.tz_convert("UTC") if end_now.tzinfo else end_now.tz_localize("UTC")
            df = df[df.index.get_level_values("entry_time") <= end_now]

        return df
