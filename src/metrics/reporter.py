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
        Writes per run:
          - metrics.json
          - cumulative_pnl.png
          - summary.md
          - summary.html
          - config.snapshot.yaml (if config provided and PyYAML available)

        includes an IS vs OOS metric section computed by that split.
        """
        # metrics
        days = int(len(daily_pnl))
        cum = daily_pnl.cumsum() if days > 0 else pd.Series(dtype="float64")
        total_pnl = float(daily_pnl.sum()) if days > 0 else 0.0
        mean_daily = float(daily_pnl.mean()) if days > 0 else 0.0
        stdev_daily = float(daily_pnl.std(ddof=1)) if days > 1 else 0.0
        sharpe = float(np.sqrt(252) * (mean_daily / (stdev_daily + 1e-12))) if days > 1 else 0.0
        min_dd = float((cum - cum.cummax()).min()) if days > 0 else 0.0

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
        if days > 0:
            cum.plot()
            plt.title("Cumulative P&L")
            plt.xlabel("Date")
            plt.ylabel("P&L")
        else:
            plt.title("Cumulative P&L")
            plt.axis("off")
            plt.text(
                0.5, 0.5,
                "No data to plot\n(resume caught up to end_now)",
                ha="center", va="center", fontsize=12
            )
        plt.tight_layout()
        plt.savefig(self.artifacts / "cumulative_pnl.png")
        plt.close()

        # IS vs OOS split metrics
        split_md_lines: list[str] = []
        split_html_rows: list[str] = []
        split_cfg = (config or {}).get("report", {}) if config else {}
        split_date = split_cfg.get("split_date")
        if split_date and days > 0:
            try:
                split_ts = pd.Timestamp(split_date)
                if hasattr(daily_pnl.index, "tz") and daily_pnl.index.tz is not None:
                    split_ts = split_ts.tz_convert("UTC") if split_ts.tzinfo else split_ts.tz_localize("UTC")
            except Exception:
                split_ts = None

            if split_ts is not None:
                is_mask = daily_pnl.index < split_ts
                oos_mask = daily_pnl.index >= split_ts

                def slice_metrics(s: pd.Series) -> dict:
                    n = int(len(s))
                    if n == 0:
                        return {"days": 0, "total": 0.0, "mean": 0.0, "stdev": 0.0, "sharpe": 0.0, "min_dd": 0.0}
                    cum_s = s.cumsum()
                    total = float(s.sum())
                    mean_ = float(s.mean())
                    std_ = float(s.std(ddof=1)) if n > 1 else 0.0
                    shrp = float(np.sqrt(252) * (mean_ / (std_ + 1e-12))) if n > 1 else 0.0
                    dd = float((cum_s - cum_s.cummax()).min())
                    return {"days": n, "total": total, "mean": mean_, "stdev": std_, "sharpe": shrp, "min_dd": dd}

                is_metrics = slice_metrics(daily_pnl[is_mask])
                oos_metrics = slice_metrics(daily_pnl[oos_mask])

                # Markdown section
                split_md_lines.extend([
                    "",
                    "## In-Sample vs Out-of-Sample",
                    "",
                    f"Split date: `{split_ts}`",
                    "",
                    "| Slice | Days | Total P&L | Mean | Stdev | Sharpe | Min DD |",
                    "|------:|-----:|----------:|-----:|------:|-------:|-------:|",
                    f"| IS (< split) | {is_metrics['days']} | {is_metrics['total']:.2f} | {is_metrics['mean']:.4f} | {is_metrics['stdev']:.4f} | {is_metrics['sharpe']:.2f} | {is_metrics['min_dd']:.2f} |",
                    f"| OOS (≥ split) | {oos_metrics['days']} | {oos_metrics['total']:.2f} | {oos_metrics['mean']:.4f} | {oos_metrics['stdev']:.4f} | {oos_metrics['sharpe']:.2f} | {oos_metrics['min_dd']:.2f} |",
                ])

                # HTML grid rows
                def html_row(label: str, m: dict) -> str:
                    return (
                        f"<tr><th>{label}</th>"
                        f"<td>{m['days']}</td><td>{m['total']:.2f}</td>"
                        f"<td>{m['mean']:.6f}</td><td>{m['stdev']:.6f}</td>"
                        f"<td>{m['sharpe']:.2f}</td><td>{m['min_dd']:.2f}</td></tr>"
                    )
                split_html_rows.append(html_row("IS (&lt; split)", is_metrics))
                split_html_rows.append(html_row("OOS (&ge; split)", oos_metrics))

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
        if days == 0:
            md.append("\n_Resume caught up to `end_now`; no new days were processed in this run._\n")
        if split_md_lines:
            md.extend(split_md_lines)
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
 table {{ border-collapse: collapse; margin-top: 16px; }}
 th, td {{ border: 1px solid #eee; padding: 6px 10px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
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
"""
        if days == 0:
            html += "\n<p><em>Resume caught up to <code>end_now</code>; no new days were processed in this run.</em></p>\n"

        if split_html_rows:
            html += """
<h2>In-Sample vs Out-of-Sample</h2>
<table>
  <thead>
    <tr><th>Slice</th><th>Days</th><th>Total P&L</th><th>Mean</th><th>Stdev</th><th>Sharpe</th><th>Min DD</th></tr>
  </thead>
  <tbody>
"""
            html += "\n".join(split_html_rows)
            html += """
  </tbody>
</table>
"""

        html += """
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
                # snapshot is optional; ignore if PyYAML unavailable
                pass
