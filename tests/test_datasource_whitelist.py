from __future__ import annotations
import pandas as pd
import numpy as np

from src.data.datasource import CSVDayaheadDataSource


def _rows_for_day(entry_day_utc: pd.Timestamp, tz: str = "Europe/Berlin"):
    """
    Build a tiny 3-hour synthetic DAH slice for a single entry day.
    entry_time = 11:50 local. We care about:
      - (entry_time, hour) MultiIndex
      - UTC normalization of entry/exit
      - known-at-entry feature whitelisting
    """
    rows = []
    entry_local = entry_day_utc.tz_convert(tz)
    entry_local = entry_local.normalize() + pd.Timedelta(hours=11, minutes=50)
    for h in [0, 1, 2]:
        rows.append(
            {
                "entry_time": entry_local,  # loader will normalize to UTC
                "exit_time": (entry_local + pd.Timedelta(days=1)).replace(hour=h),
                "entry_price": 50.0 + h,
                "exit_price": 55.0 + h,
                # Allowed, known-at-entry feature prefixes:
                "wind_forecast_provider1": 10.0 + h,
                "solar_forecast_provider2": 20.0 + h,
                "load_forecast_provider3": 30.0 + h,
                # Disallowed (should be dropped by whitelist):
                "actual_load": 999.0,              # leakage
                "post_delivery_feature": 888.0,     # leak
            }
        )
    return rows


def test_datasource_whitelist_and_end_now(tmp_path):
    # CSV
    d0 = pd.Timestamp("2059-01-05 00:00:00", tz="UTC")
    d1 = pd.Timestamp("2059-01-06 00:00:00", tz="UTC")

    df = pd.DataFrame(_rows_for_day(d0) + _rows_for_day(d1))
    csv_path = tmp_path / "tiny_dah.csv"
    df.to_csv(csv_path, index=False)

    # Use a cutoff after the day’s entry cut so day0 is included
    end_now_include_day0 = d0 + pd.Timedelta(hours=12)

    # load with end_now set after the day's entry cut => expect only day0 rows
    ds = CSVDayaheadDataSource(str(csv_path))
    out = ds.load(end_now=end_now_include_day0)

    # MultiIndex 
    assert isinstance(out.index, pd.MultiIndex)
    assert out.index.names == ["entry_time", "hour"]

    etimes = out.index.get_level_values("entry_time")
    assert etimes.tz is not None and str(etimes.tz) in ("UTC", "utc")
    assert (etimes <= end_now_include_day0).all()

    hours = sorted(out.index.get_level_values("hour").unique().tolist())
    assert len(hours) == 3
    assert all(isinstance(h, (int, np.integer)) and 0 <= h <= 23 for h in hours)

    cols = set(out.columns)

    assert {"exit_time", "entry_price", "exit_price", "premium"}.issubset(cols)

    assert "wind_forecast_provider1" in cols
    assert "solar_forecast_provider2" in cols
    assert "load_forecast_provider3" in cols

    assert "actual_load" not in cols
    assert "post_delivery_feature" not in cols

    # Premium = exit - entry
    prem = out["premium"].to_numpy()
    exp = out["exit_price"].to_numpy() - out["entry_price"].to_numpy()
    np.testing.assert_allclose(prem, exp, rtol=0, atol=1e-12)
