from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import pandas as pd


@dataclass
class Checkpoint:
    """
    daily checkpoint to support resume:
      - stores the last processed entry_time
      - lives under artifacts/<run_id>/checkpoint.json
    """
    path: Path

    def load_last_day(self) -> Optional[pd.Timestamp]:
        """
        Returns the last processed entry_time,
        or None if no checkpoint exists or file is malformed.
        """
        try:
            if not self.path.exists():
                return None
            data = json.loads(self.path.read_text())
            v = data.get("last_entry_time")
            if not v:
                return None
            ts = pd.Timestamp(v)
            # Normalize to UTC
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts
        except Exception:
            return None

    def save_last_day(self, entry_time: pd.Timestamp) -> None:
        """
        last processed entry_time
        """
        # Ensure tz-aware UTC
        ts = entry_time
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        payload = {"last_entry_time": ts.isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))
