"""Validate a DENOISING frontend: extract OT features from wavelet-denoised
signals for the 4 training bearings, so we can measure whether HI monotonicity
improves vs the non-denoised features. Original TDMS is never modified; output
goes to a separate folder. Denoised features are kept ONLY if they measurably
improve HI quality (objective guardrail against destroying fault signal).

Denoiser: wavelet soft-threshold (db4, universal threshold) — the canonical
"remove noise while preserving impulsive transients" method for vibration. It
attenuates broadband noise but keeps the bearing-fault impulse content.

Run:  python scripts/extract_denoised.py          # extract train (denoised)
      python scripts/extract_denoised.py --test    # also extract 6 validation
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from nptdms import TdmsFile

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
INHWAN = Path("c:/Users/User/WorkSpace/INHWAN")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(INHWAN))

import rpm_estimator as R                              # noqa: E402
from ot_rpm_impact import order_track, ot_features, CHS  # noqa: E402
from src.operation import list_vibration_files          # noqa: E402
from predict_test_wiener import bearing_files, BEARINGS  # noqa: E402

OUT = ROOT / "outputs" / "ot_features" / "est_denoised"
TEST_OUT = OUT / "test"
FILE_PERIOD = 600


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Soft-threshold wavelet denoise. Preserves impulsive (fault) transients,
    attenuates broadband noise. Threshold from the finest-detail noise estimate."""
    x = np.asarray(x, dtype=float)
    coeff = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745          # robust noise std
    thr = sigma * np.sqrt(2.0 * np.log(max(len(x), 2)))    # universal threshold
    coeff[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeff[1:]]
    rec = pywt.waverec(coeff, wavelet)
    return rec[: len(x)]


def extract(files, denoise: bool) -> pd.DataFrame:
    ch0, sig_cache = [], []
    for f in files:
        chs = [c[:] for c in TdmsFile.read(str(f)).groups()[0].channels()][:4]
        sig_cache.append(chs)
        ch0.append(chs[0])
    est_rpm = R.refine_stepwise(R.estimate_rpm_series(ch0))
    rows = []
    for k, chs in enumerate(sig_cache):
        row = {"File_Index": k + 1}
        for ch in CHS:
            if ch < len(chs):
                sig = wavelet_denoise(chs[ch]) if denoise else chs[ch]
                row.update(ot_features(order_track(sig, float(est_rpm[k])), ch))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["t_sec"] = (df["File_Index"] - 1) * FILE_PERIOD
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for tr in (1, 2, 3, 4):
        t0 = time.time()
        df = extract(list_vibration_files(tr), denoise=True)
        df.to_csv(OUT / f"Train{tr}.csv", index=False)
        print(f"Train{tr} denoised ({len(df)} files, {time.time()-t0:.0f}s)")
    if args.test:
        TEST_OUT.mkdir(parents=True, exist_ok=True)
        for name in BEARINGS:
            t0 = time.time()
            df = extract(bearing_files(name), denoise=True)
            df.to_csv(TEST_OUT / f"{name}.csv", index=False)
            print(f"{name} denoised ({len(df)} files, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
