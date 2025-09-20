from __future__ import annotations
import numpy as np
import pytest

from src.risk.manager import SimpleRiskManager


def test_per_hour_clip_logs_and_bounds(caplog):
    # bounds [-10, 10], no daily cap so we isolate per-hour clipping
    risk = SimpleRiskManager(per_hour_bounds=(-10, 10), daily_abs_cap=None)

    # Some targets exceed bounds; some inside
    targets = np.array([ -25, -11, -10, -3, 0, 2, 9, 10, 11, 25 ], dtype=float)

    caplog.clear()
    with caplog.at_level("INFO"):
        out = risk.enforce(targets)

    # All outputs must be within [-10, 10] and integer
    assert out.dtype.kind in ("i", "u")
    assert out.min() >= -10 and out.max() <= 10

    # check
    text = " ".join(r.message for r in caplog.records)
    assert "risk.per_hour_clip" in text

    # Original inside-bounds indices: [2,3,4,5,6,7]
    assert out[3] == -3 and out[4] == 0 and out[5] == 2 and out[6] == 9 and out[7] == 10
    # clipped
    assert out[0] == -10 and out[1] == -10 and out[8] == 10 and out[9] == 10


def test_daily_cap_clip_logs_and_cap(caplog):
    risk = SimpleRiskManager(per_hour_bounds=(-100, 100), daily_abs_cap=50)

    # L1 sum is 1+2+...+10 = 55 > 50 -> must be scaled to <= 50
    targets = np.arange(1, 11, dtype=float)

    caplog.clear()
    with caplog.at_level("INFO"):
        out = risk.enforce(targets)

    # within bounds
    assert out.dtype.kind in ("i", "u")
    assert out.min() >= -100 and out.max() <= 100

    # L1 exposure is capped
    assert int(np.sum(np.abs(out))) <= 50

    # Log
    text = " ".join(r.message for r in caplog.records)
    assert "risk.daily_cap_clip" in text


def test_idempotence():
    risk = SimpleRiskManager(per_hour_bounds=(-10, 10), daily_abs_cap=60)
    rng = np.random.default_rng(7)
    targets = rng.integers(-25, 25, size=24).astype(float)

    once = risk.enforce(targets)
    twice = risk.enforce(once.copy())

    assert np.array_equal(once, twice)
