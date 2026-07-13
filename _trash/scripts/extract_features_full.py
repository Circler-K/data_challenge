"""Comprehensive feature extraction for all 4 trains × all TDMS files.

Output: outputs/features_full/train{k}.parquet, one row per file.
Per channel features (×4 channels):
  Time-domain: rms, peak, p2p, mean_abs, std, kurt, skew, crest, impulse,
               shape, margin, energy
  Frequency-domain band RMS (8 bands): 0-500, 500-1500, 1500-2500,
               2500-3000, 3000-5000, 5000-7500, 7500-10000, 10000-12800 Hz
  Spectral statistics: centroid, spread, skewness, kurtosis, entropy, rolloff
  Envelope (Hilbert on bandpass via kurtogram BP):
               env_rms, env_kurt, env_peak,
               env_BPFI_1x/2x/3x, env_BPFO_1x/2x/3x, env_BSF_1x/2x/3x
Operation features (one row): rpm_mean/std, torque_mean/min/std, tcf/tcr_mean/max
"""
from __future__ import annotations
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.io_tdms import load_tdms_file, tdms_to_array, FS
from src.operation import load_operation, list_vibration_files, align_to_vibration

OUT_DIR = ROOT / "outputs" / "features_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS_HZ = [
    (0, 500),
    (500, 1500),
    (1500, 2500),
    (2500, 3000),
    (3000, 5000),
    (5000, 7500),
    (7500, 10000),
    (10000, 12800),
]

# Bearing characteristic frequencies at 1000 RPM (matches features_utils)
BPFx_AT_1000 = {"BPFI": 140.0, "BPFO": 93.0, "BSF": 78.0}
HARMONICS = (1, 2, 3)
ENV_HALF_PCT = 0.05  # ±5% around fc

# Selected bandpass per (train, channel) — load from selected_bands.csv
BAND_CSV = ROOT / "outputs" / "features_utils" / "selected_bands.csv"


def load_selected_bands() -> dict:
    """Returns {(train, ch_name): (lo, hi)}."""
    df = pd.read_csv(BAND_CSV)
    out = {}
    for _, r in df.iterrows():
        out[(r["train"], r["channel"])] = (float(r["lo"]), float(r["hi"]))
    return out


def time_features(x: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    p2p = float(np.max(x) - np.min(x))
    mean_abs = float(np.mean(np.abs(x)))
    std = float(np.std(x))
    kurt = float(sp_kurt(x, fisher=False))
    skew = float(sp_skew(x))
    eps = 1e-12
    crest = peak / (rms + eps)
    impulse = peak / (mean_abs + eps)
    shape = rms / (mean_abs + eps)
    sqrt_mean = float(np.mean(np.sqrt(np.abs(x))))
    margin = peak / (sqrt_mean ** 2 + eps)
    energy = float(np.sum(x * x))
    return dict(rms=rms, peak=peak, p2p=p2p, mean_abs=mean_abs, std=std,
                kurt=kurt, skew=skew, crest=crest, impulse=impulse,
                shape=shape, margin=margin, energy=energy)


def freq_features(x: np.ndarray, fs: int = FS) -> dict:
    N = len(x)
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / N
    f = np.fft.rfftfreq(N, 1.0 / fs)
    out: dict = {}

    for lo, hi in BANDS_HZ:
        m = (f >= lo) & (f < hi)
        out[f"band_{lo}_{hi}_rms"] = float(np.sqrt(psd[m].sum())) if m.any() else 0.0

    psd_norm = psd / (psd.sum() + 1e-30)
    centroid = float(np.sum(f * psd_norm))
    spread = float(np.sqrt(np.sum((f - centroid) ** 2 * psd_norm)))
    if spread > 0:
        spec_skew = float(np.sum(((f - centroid) / spread) ** 3 * psd_norm))
        spec_kurt = float(np.sum(((f - centroid) / spread) ** 4 * psd_norm))
    else:
        spec_skew = 0.0
        spec_kurt = 0.0
    spec_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-30)))

    # 95% rolloff frequency
    cum = np.cumsum(psd_norm)
    idx = int(np.searchsorted(cum, 0.95))
    rolloff = float(f[min(idx, len(f) - 1)])

    out["spec_centroid"] = centroid
    out["spec_spread"] = spread
    out["spec_skew"] = spec_skew
    out["spec_kurt"] = spec_kurt
    out["spec_entropy"] = spec_entropy
    out["spec_rolloff_95"] = rolloff
    return out


def envelope_features(x: np.ndarray, fs: int, lo: float, hi: float,
                     rpm: float) -> dict:
    """Hilbert envelope of bandpassed x. Compute env stats + BPFx amplitudes."""
    nyq = fs / 2.0
    lo = max(50.0, min(lo, nyq - 200.0))
    hi = max(lo + 100.0, min(hi, nyq - 50.0))
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    x_bp = sosfiltfilt(sos, x)
    env = np.abs(hilbert(x_bp))
    env = env - env.mean()  # remove DC for spectral analysis

    out = dict(
        env_rms=float(np.sqrt(np.mean(env * env))),
        env_kurt=float(sp_kurt(env, fisher=False)),
        env_peak=float(np.max(np.abs(env))),
    )

    # Envelope spectrum at BPFx harmonics
    N = len(env)
    F = np.fft.rfft(env)
    freq_env = np.fft.rfftfreq(N, 1.0 / fs)
    amp = np.abs(F) / (N / 2)

    rpm_safe = max(float(rpm), 100.0)
    scale = rpm_safe / 1000.0
    df = freq_env[1] - freq_env[0]
    half_bins = max(2, int(round(2.0 / df)))  # ±2 Hz tolerance
    for name, f0 in BPFx_AT_1000.items():
        for h in HARMONICS:
            fc = f0 * h * scale
            i_c = int(round(fc / df))
            i_lo = max(0, i_c - half_bins)
            i_hi = min(len(amp), i_c + half_bins + 1)
            out[f"env_{name}_{h}x"] = float(amp[i_lo:i_hi].max()) if i_hi > i_lo else 0.0
    return out


def per_file_features(sig4: np.ndarray, rpm: float, train_id: int,
                      bands: dict) -> dict:
    """Compute all features for a single TDMS file (4 channels)."""
    out: dict = {}
    for i in range(4):
        ch = f"CH{i+1}"
        x = sig4[i].astype(np.float64)
        for k, v in time_features(x).items():
            out[f"{ch}_{k}"] = v
        for k, v in freq_features(x, FS).items():
            out[f"{ch}_{k}"] = v
        lo, hi = bands.get((f"Train{train_id}", ch), (1000.0, 10000.0))
        for k, v in envelope_features(x, FS, lo, hi, rpm).items():
            out[f"{ch}_{k}"] = v
    return out


def process_train(train_id: int) -> pd.DataFrame:
    bands = load_selected_bands()
    files = list_vibration_files(train_id)
    op = load_operation(train_id)
    op_aligned = align_to_vibration(op, n_files=len(files), period_min=10)

    rows = []
    n = len(files)
    print(f"[Train{train_id}] {n} files")
    t0 = time.time()
    for i, fp in enumerate(files, start=1):
        sig4 = tdms_to_array(load_tdms_file(str(fp)))
        rpm = float(op_aligned.iloc[i - 1].get("rpm_mean", 1000.0))
        feats = per_file_features(sig4, rpm, train_id, bands)
        feats["train_id"] = train_id
        feats["file_idx"] = i
        feats["file_name"] = fp.name
        feats["t_start_sec"] = (i - 1) * 600
        # operation
        for col in ["rpm_mean", "rpm_std", "torque_mean", "torque_min",
                    "torque_std", "tcf_mean", "tcr_mean", "tcf_max", "tcr_max"]:
            if col in op_aligned.columns:
                feats[col] = float(op_aligned.iloc[i - 1][col])
        rows.append(feats)
        if i % 20 == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n - i) / rate if rate > 0 else 0.0
            print(f"  [{i:>3}/{n}] {elapsed:5.1f}s elapsed,"
                  f" {rate:4.1f} files/s, ETA {eta:5.1f}s")

    df = pd.DataFrame(rows)
    # Reorder: id columns first
    id_cols = ["train_id", "file_idx", "file_name", "t_start_sec"]
    other = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + other]
    return df


def main():
    for train_id in [1, 2, 3, 4]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = process_train(train_id)
        out = OUT_DIR / f"train{train_id}.parquet"
        df.to_parquet(out, index=False)
        print(f"  -> wrote {out} ({df.shape[0]} rows × {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
