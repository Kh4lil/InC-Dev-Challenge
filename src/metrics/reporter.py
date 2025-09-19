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

    def save(self, daily_pnl: pd.Series, config: dict | None = None) -> None:
        """
        Writes:
          - metrics.json   (numeric run metrics)
          - cumulative_pnl.png (plot of cumulative P&L)
          - summary.md     (concise markdown summary)
          - summary.html   (simple HTML summary with inline CSS)
          - config.snapshot.yaml (if config provided and pyyaml available)
        """
        # Metrics JSON
        cum = daily_pnl.cumsum()
        days = int(len(daily_pnl))
        total_pnl = float(daily_pnl.sum())
        mean_daily = float(daily_pnl.mean()) if days > 0 else 0.0
        stdev_daily = float(daily_pnl.std(ddof=1)) if days > 1 else 0.0
        sharpe = float(np.sqrt(252) * (mean_daily / (stdev_daily + 1e-12))) if days > 1 else 0.0
        min_dd = float((cum - cum.cummax()).min() if days > 0 else 0.0)

        metrics = {
            "days": days,
            "total_pnl": total_pnl,
            "mean_daily": mean_daily,
            "stdev_daily": stdev_daily,
            "sharpe": sharpe,
            "min_drawdown": min_dd,
        }
        (self.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # Cumulative P&L plot
        plt.figure()
        cum.plot()
        plt.title("Cumulative P&L")
        plt.xlabel("Date")
        plt.ylabel("P&L")
        plt.tight_layout()
        plt.savefig(self.artifacts / "cumulative_pnl.png")
        plt.close()

        # Markdown summary
        md = [
            "# Backtest Summary",
            "",
            f"- Days: {metrics['days']}",
            f"- Total P&L: {metrics['total_pnl']:.2f}",
            f"- Mean Daily: {metrics['mean_daily']:.6f}",
            f"- Stdev Daily: {metrics['stdev_daily']:.6f}",
            f"- Sharpe (≈252d): {metrics['sharpe']:.2f}",
            f"- Min Drawdown: {metrics['min_drawdown']:.2f}",
            "",
            "Notes:",
            "- Chronological walk-forward (fit uses strictly prior days; results are OOS).",
            "- Per-hour bounds and optional daily cap enforced via RiskManager.",
            "- Broker applies round-trip fees/slippage at entry and exit.",
            "",
            "Artifacts:",
            "- `metrics.json`",
            "- `cumulative_pnl.png`",
            "- `summary.html`",
        ]
        (self.artifacts / "summary.md").write_text("\n".join(md))

        # HTML summary
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Backtest Summary</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
 h1 {{ margin: 0 0 12px; }}
 .grid {{ display: grid; grid-template-columns: 200px 1fr; gap: 8px 16px; max-width: 540px; }}
 .label {{ color: #555; }}
 img {{ max-width: 960px; width: 100%; height: auto; margin-top: 16px; border: 1px solid #eee; }}
</style>
</head>
<body>
<h1>Backtest Summary</h1>
<div class="grid">
  <div class="label">Days</div><div>{metrics['days']}</div>
  <div class="label">Total P&L</div><div>{metrics['total_pnl']:.2f}</div>
  <div class="label">Mean Daily</div><div>{metrics['mean_daily']:.6f}</div>
  <div class="label">Stdev Daily</div><div>{metrics['stdev_daily']:.6f}</div>
  <div class="label">Sharpe (≈252d)</div><div>{metrics['sharpe']:.2f}</div>
  <div class="label">Min Drawdown</div><div>{metrics['min_drawdown']:.2f}</div>
</div>

<h2>Notes</h2>
<ul>
  <li>Chronological walk-forward (fit uses strictly prior days; results are OOS).</li>
  <li>Per-hour bounds and optional daily cap enforced via RiskManager.</li>
  <li>Broker applies round-trip fees/slippage at entry and exit.</li>
</ul>

<h2>Plot</h2>
<img src="cumulative_pnl.png" alt="Cumulative P&L"/>
</body>
</html>"""
        (self.artifacts / "summary.html").write_text(html)

        # config snapshot
        if config is not None:
            try:
                import yaml
                (self.artifacts / "config.snapshot.yaml").write_text(
                    yaml.safe_dump(config, sort_keys=False)
                )
            except Exception:
                pass
