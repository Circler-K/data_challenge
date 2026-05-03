"""TDMS I/O for KSPHM-KIMM 2026 vibration data.

Each file holds 1 minute of 4-channel vibration at 25.6 kHz, stored under
the group ``Vibration`` with channels ``CH1..CH4``.

Standard usage (per challenge guide):

    from nptdms import TdmsFile
    import pandas as pd

    def load_tdms_file(file_path):
        tdms_file = TdmsFile.read(file_path)
        df = tdms_file.as_dataframe()
        return df

This module wraps that pattern and adds a small helper for code paths that
need a NumPy ``(4, N)`` matrix layout (e.g. signal-processing pipelines).

Install nptdms with either:
    conda install conda-forge::nptdms
    pip install nptdms
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile

FS = 25_600
DURATION_SEC = 60
SAMPLES_PER_CH = FS * DURATION_SEC  # 1_536_000
N_CHANNELS = 4
CHANNEL_NAMES = ("CH1", "CH2", "CH3", "CH4")
GROUP_NAME = "Vibration"


def load_tdms_file(file_path: str | Path) -> pd.DataFrame:
    """Load a TDMS file as a pandas DataFrame (one column per channel)."""
    tdms_file = TdmsFile.read(str(file_path))
    df = tdms_file.as_dataframe()
    return df


def tdms_to_array(df: pd.DataFrame,
                  channels: tuple[str, ...] = CHANNEL_NAMES) -> np.ndarray:
    """Convert the DataFrame returned by ``load_tdms_file`` to a ``(C, N)``
    float32 NumPy array in the order given by ``channels``.

    nptdms names columns like ``/'Vibration'/'CH1'`` — we match by suffix so
    minor format variations are tolerated.
    """
    cols = []
    for ch in channels:
        match = [c for c in df.columns if c.endswith(f"'{ch}'") or c.endswith(f"/{ch}")]
        if not match:
            raise KeyError(f"channel {ch!r} not found in {list(df.columns)}")
        cols.append(match[0])
    arr = df[cols].to_numpy(dtype=np.float32, copy=False).T
    return arr


def time_axis(n_samples: int = SAMPLES_PER_CH, fs: int = FS) -> np.ndarray:
    """Time axis in seconds for a single channel of one file."""
    return np.arange(n_samples, dtype=np.float64) / fs


def file_start_seconds(file_idx: int, period_min: int = 10) -> int:
    """Start time of a vibration file relative to the test start.

    File ``i`` (1-based numbering on disk) begins ``(i-1) * 10`` minutes after
    the test start: 10-minute period with 1-minute capture each.
    """
    return (file_idx - 1) * period_min * 60


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else (
        "c:/Users/User/WorkSpace/data_challenge/Train/Train1_Vibration/000001.tdms"
    )
    df = load_tdms_file(test_path)
    print(f"DataFrame shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    arr = tdms_to_array(df)
    print(f"matrix shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"per-channel RMS: {np.sqrt((arr**2).mean(axis=1))}")
