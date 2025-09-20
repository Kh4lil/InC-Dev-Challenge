from __future__ import annotations
import numpy as np
from typing import Tuple

class LinearScalerPolicy:
    """
    Map predicted hourly premiums -> target positions with:
      - scale:    linear scaling of signal magnitude
      - threshold:don't trade if |pred| < threshold (ignore noise)
      - MAD norm: robust normalization so the scale is stable across days
      - bounds:   final hard clip into [lo, hi] (engine/risk will still enforce)

    Returns an int vector (24,) of target positions within [lo, hi].
    """

    def __init__(self, scale: float = 1.0, threshold: float = 0.0, mad_eps: float = 1e-9):
        self.scale = float(scale)
        self.threshold = float(threshold)
        self.mad_eps = float(mad_eps)

    def target_positions(self, preds: np.ndarray, bounds: Tuple[int, int]) -> np.ndarray:
        lo, hi = bounds
        x = preds.astype(float)

        # Median Absolute Deviation to stabilize scaling
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + self.mad_eps

        # Scale predictions into position units
        pos = self.scale * (x / mad)

        # Ignore micro-signals and soft-clip to bounds
        pos[np.abs(x) < self.threshold] = 0.0


        pos = np.clip(pos, lo, hi)

        return pos.astype(int)
