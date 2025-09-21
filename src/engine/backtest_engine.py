from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd

from src.utils.config import Config, build_component
from src.utils.logging import setup_logging
from src.utils.calendar import entry_days
from src.engine.portfolio import PortfolioEngine
from src.utils.checkpoint import Checkpoint

log = logging.getLogger(__name__)


@dataclass
class BacktestEngine:
    cfg: Config

    def run(self) -> None:
        artifacts = self.cfg.artifacts_dir
        setup_logging(artifacts)
        log.info("Artifacts dir: %s", artifacts)

        # Build components
        data_cfg = self.cfg.raw["data"]
        comp = self.cfg.raw["components"]
        ds = build_component(comp["datasource"], path=data_cfg["path"])  # type: ignore[arg-type]
        risk = build_component(comp["risk"], **self.cfg.raw.get("risk", {}))  # type: ignore[arg-type]
        broker = build_component(comp["broker"], **self.cfg.raw.get("costs", {}))  # type: ignore[arg-type]
        reporter = build_component(comp["reporter"], artifacts_dir=artifacts)  # type: ignore[call-arg]

        # portfolio or single-strategy?
        portfolio_cfg = self.cfg.raw.get("portfolio")
        use_portfolio = bool(portfolio_cfg and isinstance(portfolio_cfg.get("strategies"), list))

        # Single-strategy components
        model = policy = None
        if not use_portfolio:
            model = build_component(comp["model"])   # type: ignore[call-arg]
            policy = build_component(comp["policy"]) # type: ignore[call-arg]
        else:
            portfolio = PortfolioEngine(self.cfg.raw)

        # Load data up to end_now
        engine_cfg = self.cfg.raw.get("engine", {})
        end_now = None
        if engine_cfg.get("end_now"):
            end_now = pd.Timestamp(engine_cfg["end_now"])
            end_now = end_now.tz_convert("UTC") if end_now.tzinfo else end_now.tz_localize("UTC")
        full = ds.load(end_now=end_now)

        if engine_cfg.get("start"):
            start = pd.Timestamp(engine_cfg["start"])
            start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        else:
            start = full.index.get_level_values(0).min()

        if engine_cfg.get("end"):
            end = pd.Timestamp(engine_cfg["end"])
            end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
        else:
            end = full.index.get_level_values(0).max()

        # resume-from-checkpoint
        resume_enabled = bool(engine_cfg.get("resume"))
        ckpt = Checkpoint(path=artifacts / "checkpoint.json")
        last_done = ckpt.load_last_day() if resume_enabled else None
        if last_done is not None:
            log.info("Resume enabled. Last completed day in checkpoint: %s", last_done)

        days = [d for d in entry_days(full) if (d >= start and d <= end)]
        log.info("Running backtest from %s to %s (%d days)", start.date(), end.date(), len(days))

        daily_pnl: dict[pd.Timestamp, float] = {}
        for i, day in enumerate(days):
            # Skip days already processed if resuming
            if last_done is not None and day <= last_done:
                log.debug("Skipping day %s (already completed in checkpoint).", day)
                continue

            past = full.loc[pd.IndexSlice[: day - pd.Timedelta("1ns"), :], :]
            today = full.loc[pd.IndexSlice[[day], :], :]

            log.debug("Day %s: fit on %d rows, predict on %d", day, len(past), len(today))

            hours_today = today.index.get_level_values("hour").astype(int).to_numpy()

            # Policy -> targets (single or portfolio)
            lo, hi = tuple(self.cfg.raw.get("risk", {}).get("per_hour_bounds", [-10, 10]))

            if use_portfolio:
                # Fit all strategies on prior data combine targets for today
                portfolio.fit_all(past)
                targets = portfolio.predict_and_combine(today, (lo, hi)).astype(float)
            else:
                # Single-strat path
                assert model is not None and policy is not None
                model.fit(past)
                preds = model.predict(today).astype(float)

                if len(preds) != len(hours_today):
                    preds = preds[hours_today]

                targets = policy.target_positions(preds, (lo, hi)).astype(float)

            if len(targets) != len(hours_today):
                targets = targets[hours_today]

            targets = risk.enforce(targets)

            res = broker.settle(today, targets)
            daily_pnl[day] = res["daily_pnl"]

            # update checkpoint after successful day
            if resume_enabled:
                ckpt.save_last_day(day)

        pnl_series = pd.Series(daily_pnl).sort_index()
        # pass config so reporter can snapshot and write a summary
        reporter.save(pnl_series, config=self.cfg.raw)
        log.info("Done. Metrics and plot saved to %s", artifacts)
