from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.utils.checkpoint import Checkpoint


def test_checkpoint_roundtrip_utc(tmp_path: Path):
    ckpt_path = tmp_path / "checkpoint.json"
    ckpt = Checkpoint(path=ckpt_path)

    # Save a timestam
    naive = pd.Timestamp("2060-05-31 10:50:00")
    ckpt.save_last_day(naive)

    assert ckpt_path.exists()
    raw = ckpt_path.read_text()
    assert "Z" in raw or "+00:00" in raw

    # Load
    loaded = ckpt.load_last_day()
    assert loaded is not None
    assert loaded.tz is not None
    assert str(loaded.tz) in ("UTC", "utc")
    assert loaded == pd.Timestamp("2060-05-31 10:50:00", tz="UTC")


def test_checkpoint_handles_missing_or_malformed(tmp_path: Path):
    # Missing file -> returns None
    ckpt = Checkpoint(path=tmp_path / "missing.json")
    assert ckpt.load_last_day() is None

    # Malformed content -> returns None
    bad = tmp_path / "bad.json"
    bad.write_text("{not: json")
    ckpt_bad = Checkpoint(path=bad)
    assert ckpt_bad.load_last_day() is None
