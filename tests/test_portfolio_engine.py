from __future__ import annotations
import numpy as np
import pandas as pd

from src.engine.portfolio import PortfolioEngine


# This is a test for dummy components used via fake build_component

class _DummyModel:
    """Returns a constant 24-vector so we can verify weighting."""
    def __init__(self, value: float = 1.0, **_):
        self.value = float(value)
        self.fit_called = False

    def fit(self, df_past: pd.DataFrame):
        self.fit_called = True
        return self

    def predict(self, df_today: pd.DataFrame) -> np.ndarray:
        # Always 24 entries; engine would subset on DST in real flow
        return np.full(24, self.value, dtype=float)


class _DummyPolicy:
    """Maps preds -> integer targets within bounds by scaling to the hi bound."""
    def __init__(self, scale: float = 1.0, **_):
        self.scale = float(scale)

    def target_positions(self, preds: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
        lo, hi = bounds
        const = int(np.clip(round(self.scale), lo, hi))
        return np.full(24, const, dtype=int)


def test_portfolio_combines_weighted_targets(monkeypatch):
    # values don't matter for dummy classes
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2059-01-05 11:50:00+00:00")], list(range(24))],
        names=["entry_time", "hour"],
    )
    today = pd.DataFrame({"entry_price": 50.0, "exit_price": 55.0}, index=idx)
    past = today.iloc[:0].copy()  # empty

    def _fake_build_component(target: str, **kwargs):
        if target.endswith(":_DummyModel"):
            return _DummyModel(**kwargs)
        if target.endswith(":_DummyPolicy"):
            return _DummyPolicy(**kwargs)
        raise AssertionError(f"Unexpected target: {target}")

    monkeypatch.setattr("src.engine.portfolio.build_component", _fake_build_component, raising=True)

    cfg = {
        "portfolio": {
            "strategies": [
                {
                    "name": "s1",
                    "model": "any.module.path:_DummyModel",
                    "policy": "any.module.path:_DummyPolicy",
                    "weight": 0.25,
                    "policy_args": {"scale": 6.0},  # targets=6 bounded by hi
                },
                {
                    "name": "s2",
                    "model": "any.other.path:_DummyModel",
                    "policy": "any.other.path:_DummyPolicy",
                    "weight": 0.75,
                    "policy_args": {"scale": 4.0},  # targets=4
                },
            ]
        }
    }

    pe = PortfolioEngine(cfg)
    pe.fit_all(past)

    # bounds (lo, hi) = (-10, 10) targets are 6 and 4
    combined = pe.predict_and_combine(today, bounds=(-10, 10))
    assert combined.shape == (24,)
    # Expected: 0.25*6 + 0.75*4 = 1.5 + 3 = 4.5 per hour
    assert np.allclose(combined, np.full(24, 4.5, dtype=float))
