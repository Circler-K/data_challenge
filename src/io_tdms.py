"""TDMS I/O for KSPHM-KIMM 2026 vibration data.

Each file holds 1 minute of 4-channel vibration at 25.6 kHz, stored as
sequential float32 blocks per channel under the group ``Vibration``.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from nptdms import TdmsFile

FS = 25_600
DURATION_SEC = 60
SAMPLES_PER_CH = FS * DURATION_SEC  # 1_536_000
N_CHANNELS = 4
CHANNEL_NAMES = ("CH1", "CH2", "CH3", "CH4")
GROUP_NAME = "Vibration"


def read_tdms(path: str | Path) -> np.ndarray:
    """Load a TDMS file as ``(4, 1_536_000)`` float32 array via nptdms."""
    with TdmsFile.open(str(path)) as tf:
        group = tf[GROUP_NAME]
        out = np.empty((N_CHANNELS, SAMPLES_PER_CH), dtype=np.float32)
        for i, name in enumerate(CHANNEL_NAMES):
            data = group[name][:]
            out[i, : len(data)] = data
    return out


def read_tdms_raw(path: str | Path) -> np.ndarray:
    """Fast binary reader bypassing nptdms.

    Assumes the layout confirmed via the EDA: ToC=0x0E (no interleaving),
    metadata block size = 225 bytes, and channels concatenated as
    ``[CH1 | CH2 | CH3 | CH4]`` little-endian float32. Used to cross-check
    nptdms output and to speed up bulk feature extraction if needed.
    """
    with open(path, "rb") as fp:
        header = fp.read(28)
        tag, _toc, _ver, _next_off, data_off = struct.unpack("<4sIIQQ", header)
        if tag != b"TDSm":
            raise ValueError(f"Not a TDMS file: {path}")
        fp.read(data_off)
        raw = fp.read(SAMPLES_PER_CH * N_CHANNELS * 4)
    return np.frombuffer(raw, dtype="<f4").reshape(N_CHANNELS, SAMPLES_PER_CH)


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
    a = read_tdms(test_path)
    b = read_tdms_raw(test_path)
    rms_a = np.sqrt((a ** 2).mean(axis=1))
    rms_b = np.sqrt((b ** 2).mean(axis=1))
    print(f"shape: {a.shape}, dtype: {a.dtype}")
    print(f"nptdms RMS:  {rms_a}")
    print(f"raw    RMS:  {rms_b}")
    print(f"max abs diff: {np.max(np.abs(a - b)):.3e}")
