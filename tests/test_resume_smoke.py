from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import yaml

from src.engine.backtest_engine import BacktestEngine
from src.utils.config import Config


def _tiny_csv(path: Path, days: int = 3):
    rows = []
    for d in pd.date_range("2059-01-01", periods=days, tz="Europe/Berlin"):
        entry = pd.Timestamp(d.date()) + pd.Timedelta(hours=11, minutes=50)
        for h in range(24):
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": (entry + pd.Timedelta(days=1))
                    .tz_localize(None)
                    .tz_localize("Europe/Berlin")
                    .replace(hour=h),
                    "entry_price": 50.0,
                    "exit_price": 55.0,
                    "wind_forecast_provider1": h,
                    "solar_forecast_provider2": 2 * h,
                    "load_forecast_provider3": 3 * h,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_engine_resume_writes_empty_second_run(tmp_path: Path):
    data_csv = tmp_path / "tiny.csv"
    _tiny_csv(data_csv, days=5)

    base_cfg = {
        "run": {"run_id": "resume_test", "artifacts_dir": str(tmp_path)},
        "data": {"path": str(data_csv)},
        "engine": {
            "start": "2059-01-01",
            "end": "2059-01-05",
            "end_now": "2059-01-05 11:50:00+01:00",
            "resume": True,
        },
        "risk": {"per_hour_bounds": [-10, 10], "daily_abs_cap": 60},
        "costs": {"fee_per_mwh": 0.0, "bps_per_trade": 0.0},
        "components": {
            "datasource": "src.data.datasource:CSVDayaheadDataSource",
            "model": "src.models.baseline:YesterdayPremiumModel",
            "policy": "src.strategy.baseline:SignPolicy",
            "risk": "src.risk.manager:SimpleRiskManager",
            "broker": "src.exec.broker:SimpleBroker",
            "reporter": "src.metrics.reporter:Reporter",
        },
    }

    # First run processes all days
    cfg1 = Config(base_cfg)
    BacktestEngine(cfg1).run()

    # Second run with same end_now should skip all days metrics days=0
    cfg2 = Config(base_cfg)
    BacktestEngine(cfg2).run()

    out = cfg2.artifacts_dir
    m = json.loads((out / "metrics.json").read_text())
    assert m["days"] == 0
    assert (out / "cumulative_pnl.png").exists()
    assert (out / "summary.md").exists()
    assert (out / "summary.html").exists()
