from __future__ import annotations
import numpy as np
import pandas as pd

from src.exec.broker import SimpleBroker


def test_pnl_no_costs_linearity():
    broker = SimpleBroker(fee_per_mwh=0.0, bps_per_trade=0.0)

    entry = np.array([50.0, 50.0, 50.0, 50.0], dtype=float)
    exit_ = np.array([55.0, 49.0, 60.0, 40.0], dtype=float)
    df = pd.DataFrame({"entry_price": entry, "exit_price": exit_})

    q1 = np.array([1,  0, -2,  3], dtype=float)
    q2 = 2 * q1

    r1 = broker.settle(df, q1)
    r2 = broker.settle(df, q2)

    # hour definition: q * (exit - entry) - costs
    # double quantity doubles P&L exactly
    assert np.isclose(r2["daily_pnl"], 2.0 * r1["daily_pnl"], rtol=0, atol=1e-10)

    # Hourly vector must sum to daily
    assert np.isclose(r1["daily_pnl"], np.sum(r1["hourly_pnl"]), rtol=0, atol=1e-10)
    assert np.isclose(r2["daily_pnl"], np.sum(r2["hourly_pnl"]), rtol=0, atol=1e-10)


def test_round_trip_costs_applied():
    broker = SimpleBroker(fee_per_mwh=0.1, bps_per_trade=10)

    entry = np.array([100.0] * 4, dtype=float)
    exit_  = np.array([100.0] * 4, dtype=float)  # zero premium
    q = np.array([10, 10, 10, 10], dtype=float)

    df = pd.DataFrame({"entry_price": entry, "exit_price": exit_})
    res = broker.settle(df, q)

    per_hour_cost = 2 * (np.abs(q) * 0.1 + np.abs(q) * entry * 0.001)
    expected = -float(np.sum(per_hour_cost))
    assert np.isclose(res["daily_pnl"], expected, rtol=0, atol=1e-10)
