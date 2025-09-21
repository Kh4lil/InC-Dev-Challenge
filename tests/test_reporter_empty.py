from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from src.metrics.reporter import Reporter


def test_reporter_handles_empty_series(tmp_path: Path):
    rep = Reporter(artifacts_dir=tmp_path)
    empty = pd.Series(dtype="float64")  # no days

    # Also i pass a split_date to make sure IS/OOS code path tolerates empty
    rep.save(empty, config={"report": {"split_date": "2060-01-01"}})

    # Artifacts exist
    metrics_path = tmp_path / "metrics.json"
    png_path = tmp_path / "cumulative_pnl.png"
    md_path = tmp_path / "summary.md"
    html_path = tmp_path / "summary.html"

    for p in (metrics_path, png_path, md_path, html_path):
        assert p.exists(), f"Missing artifact: {p}"

    # Metrics should be all zeros for empty run
    m = json.loads(metrics_path.read_text())
    assert m == {
        "days": 0,
        "total_pnl": 0.0,
        "mean_daily": 0.0,
        "stdev_daily": 0.0,
        "sharpe": 0.0,
        "min_drawdown": 0.0,
    }

    # Summary.md should include the friendly resume message
    md_text = md_path.read_text()
    assert "No data to plot" in (tmp_path / "summary.html").read_text() or "no new days were processed" in md_text
