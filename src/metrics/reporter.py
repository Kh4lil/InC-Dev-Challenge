from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Reporter:
    def __init__(self, artifacts_dir: Path):
        self.artifacts = artifacts_dir

    def save(self, daily_pnl: pd.Series) -> None:
        # Metrics
        cum = daily_pnl.cumsum()
        metrics = {
            "days": int(len(daily_pnl)),
            "total_pnl": float(daily_pnl.sum()),
            "mean_daily": float(daily_pnl.mean()),
            "stdev_daily": float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0,
            "sharpe": float(np.sqrt(252) * (daily_pnl.mean() / (daily_pnl.std(ddof=1) + 1e-12))) if len(daily_pnl) > 1 else 0.0,
            "min_drawdown": float((cum - cum.cummax()).min() if len(cum) else 0.0),
        }
        (self.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # Plot
        plt.figure()
        cum.plot()
        plt.title("Cumulative P&L")
        plt.xlabel("Date")
        plt.ylabel("P&L")
        plt.tight_layout()
        out = self.artifacts / "cumulative_pnl.png"
        plt.savefig(out)
        plt.close()
