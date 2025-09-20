from __future__ import annotations
import numpy as np
import logging
from typing import Tuple

log = logging.getLogger(__name__)

class SimpleRiskManager:
    def __init__(self, per_hour_bounds: Tuple[int, int] = (-10, 10), daily_abs_cap: int | None = 120):
        self.bounds = per_hour_bounds
        self.daily_abs_cap = daily_abs_cap

    def enforce(self, targets: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds
        t = targets.astype(int, copy=True)

        # Per-hour clip
        before = t.copy()
        t = np.clip(t, lo, hi)
        clipped_idx = np.where(before != t)[0]
        if clipped_idx.size:
            log.info(
                "risk.per_hour_clip",
                extra={
                    "hours": clipped_idx.tolist(),
                    "before": before[clipped_idx].tolist(),
                    "after": t[clipped_idx].tolist(),
                    "bounds": [int(lo), int(hi)],
                },
            )

        # Daily absolute exposure cap (L1)
        if self.daily_abs_cap is not None:
            abs_sum = int(np.sum(np.abs(t)))
            cap = int(self.daily_abs_cap)
            if abs_sum > cap and abs_sum > 0:
                scale = cap / abs_sum
                scaled = np.rint(t.astype(float) * scale).astype(int)

                # If rounding exceeds cap, trim the largest magnitudes toward zero
                over = int(np.sum(np.abs(scaled))) - cap
                if over > 0:
                    order = np.argsort(-np.abs(scaled))[:over]
                    for idx in order:
                        if scaled[idx] > 0:
                            scaled[idx] -= 1
                        elif scaled[idx] < 0:
                            scaled[idx] += 1

                if not np.array_equal(scaled, t):
                    log.info(
                        "risk.daily_cap_clip",
                        extra={
                            "pre_abs_sum": abs_sum,
                            "cap": cap,
                            "scale": float(scale),
                            "post_abs_sum": int(np.sum(np.abs(scaled))),
                        },
                    )
                t = scaled

        return t
