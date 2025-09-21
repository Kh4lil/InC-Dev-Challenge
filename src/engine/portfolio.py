from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any

from src.utils.config import build_component


class PortfolioEngine:
    """
    Run multiple model+policy pairs in parallel and combine their target positions
    before passing through RiskManager and Broker.

    Config format (YAML):
      portfolio:
        strategies:
          - name: "baseline"
            model: "src.models.baseline:YesterdayPremiumModel"
            policy: "src.strategy.baseline:SignPolicy"
            weight: 0.5
          - name: "ridge"
            model: "src.models.ridge:RidgeModel"
            policy: "src.strategy.linear_scaler:LinearScalerPolicy"
            weight: 0.5
    """

    def __init__(self, cfg: dict[str, Any]):
        self.strategies_cfg = cfg["portfolio"]["strategies"]
        self.strategies: list[dict[str, Any]] = []

        # models+policies
        for strat in self.strategies_cfg:
            model = build_component(strat["model"])
            policy = build_component(strat["policy"], **strat.get("policy_args", {}))
            self.strategies.append(
                {
                    "name": strat["name"],
                    "model": model,
                    "policy": policy,
                    "weight": float(strat.get("weight", 1.0)),
                }
            )

    def fit_all(self, df_past: pd.DataFrame) -> None:
        """Fit all models on strictly prior data."""
        for s in self.strategies:
            s["model"].fit(df_past)

    def predict_and_combine(self, df_today: pd.DataFrame, bounds: tuple[int, int]) -> np.ndarray:
        """
        Run each model+policy, weight their target positions, and combine.
        Returns vector of 24 target positions
        """
        agg = np.zeros(24, dtype=float)
        for s in self.strategies:
            preds = s["model"].predict(df_today)
            pos = s["policy"].target_positions(preds, bounds).astype(float)
            agg += s["weight"] * pos
        return agg
