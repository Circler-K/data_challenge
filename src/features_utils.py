"""Feature extraction reusing utils.feature() and fast_kurtogram.

Pipeline per file:
  - For each of 4 vibration channels:
      * utils.feature(x, fs, [low, high], envelope_band_matrix)
        which returns 4 raw time + 1 raw band-RMS + 4 filtered time + N envelope-band RMS
      * The bandpass [low, high] for envelope filtering is determined ONCE per
        (Train, channel) pair by running fast_kurtogram on a late-life file.
        This bypasses the cost of running kurtogram on every file.
      * Envelope band-energy ranges = BPFI/BPFO/BSF at 1x/2x/3x harmonics, each
        ±10%, scaled to the file's mean RPM.

Output: outputs/features_utils/train{1,2,3,4}.parquet, one row per TDMS file.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(ROOT / "utils"))

import utils as ku_utils  # utils.py
from kurtogram import fast_kurtogram

from src.io_tdms import (
    FS,
    SAMPLES_PER_CH,
    CHANNEL_NAMES,
    load_tdms_file,
    tdms_to_array,
    file_start_seconds,
)
from src.operation import (
    DATA_ROOT,
    align_to_vibration,
    list_vibration_files,
    load_operation,
)

# Bearing 30306 fault frequencies given at 1000 RPM (per challenge spec).
# FTF (cage) included for reference even though we don't probe it in env bands.
BPFx_AT_1000 = dict(BPFI=140.0, BPFO=93.0, BSF=78.0, FTF=6.7)
ENV_TARGETS = ("BPFI", "BPFO", "BSF")
HARMONICS = (1, 2, 3)
BAND_HALF_PCT = 0.10  # ±10% around each fault line / harmonic
KURT_NLEVEL = 6
KURT_DOWNSAMPLE = 512_000  # use first 512k samples for kurtogram (0.9s vs 2.7s for full)
KURT_FALLBACK_BAND = (1000.0, 10000.0)  # if kurtogram returns degenerate band

OUT_DIR = ROOT / "outputs" / "features_utils"


def select_envelope_band(x: np.ndarray, fs: int = FS,
                         nlevel: int = KURT_NLEVEL,
                         n_use: int = KURT_DOWNSAMPLE) -> tuple[float, float, dict]:
    """Run fast_kurtogram on a representative signal to get bandpass [lo, hi].

    Returns (lo_hz, hi_hz, info_dict). Falls back to KURT_FALLBACK_BAND when
    the chosen band is too narrow or out of range.
    """
    sig = np.asarray(x[: min(len(x), n_use)], dtype=np.float64).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Kwav, Lvl, freq_w, c, kmax, bw, lvl = fast_kurtogram(sig, fs, nlevel=nlevel)

    # fc is the column of the kurtogram with the highest kurtosis at the
    # best-level row. freq_w gives the column-to-Hz mapping directly.
    max_level_index = np.argmax(Kwav[np.arange(Kwav.shape[0]),
                                     np.argmax(Kwav, axis=1)])
    J = int(np.argmax(Kwav[max_level_index, :]))
    fc = float(freq_w[J])
    lo = fc - bw / 2.0
    hi = fc + bw / 2.0

    info = dict(kmax=float(kmax), bw=float(bw), fc=float(fc),
                level=float(lvl), lo=float(lo), hi=float(hi))

    # Sanity rules:
    #  - level==0 means the kurtogram picked "no decomposition" (entire band) —
    #    that often happens when one huge transient dominates; not useful for
    #    envelope-band selection. Use the safe default.
    #  - very narrow bands (<150 Hz) are dominated by leakage; widen.
    #  - lo must stay above ~300 Hz to escape the fundamental shaft tones.
    if lvl < 1.0 or (hi - lo) < 150 or lo < 300 or hi > fs / 2 - 100:
        info["fallback"] = True
        return KURT_FALLBACK_BAND[0], KURT_FALLBACK_BAND[1], info
    info["fallback"] = False
    return float(lo), float(hi), info


def envelope_bands_at_rpm(rpm: float) -> np.ndarray:
    """Build the (N, 2) matrix of [lo, hi] frequency ranges for utils.feature.

    BPFI / BPFO / BSF at 1x, 2x, 3x harmonics, each ±BAND_HALF_PCT of center.
    """
    rpm_safe = max(float(rpm), 100.0)
    scale = rpm_safe / 1000.0
    bands = []
    for name in ENV_TARGETS:
        f0 = BPFx_AT_1000[name] * scale
        for h in HARMONICS:
            fc = f0 * h
            bands.append([fc * (1 - BAND_HALF_PCT), fc * (1 + BAND_HALF_PCT)])
    return np.matrix(bands)


def _band_names() -> list[str]:
    """Names for the envelope band features in the same order as
    envelope_bands_at_rpm()."""
    out = []
    for name in ENV_TARGETS:
        for h in HARMONICS:
            out.append(f"{name}_{h}x")
    return out


def channel_features(x: np.ndarray, rpm: float,
                     band_lo: float, band_hi: float) -> dict[str, float]:
    """Run utils.feature() on one channel with the given filter band.

    Returns a flat {feature_name: value} dict. Raw output of utils.feature() is:
        [RMS, Skew, Kurt, CF, Band1,  RMS_filter, Skew_filter, Kurt_filter, CF_filter,
         Band1_env, Band2_env, ..., BandN_env]
    We rename the band features to match the bearing harmonics naming.
    """
    band_filter = [band_lo, band_hi]
    env_bands = envelope_bands_at_rpm(rpm)

    feat_vec, feat_names = ku_utils.feature(
        x.astype(np.float64).copy(), FS, band_filter, env_bands,
    )
    # Rename "Band1" (raw band RMS) and "BandK_env" (envelope band RMS)
    band_names = _band_names()
    out: dict[str, float] = {}
    j = 0  # index into band_names for envelope features
    for name, val in zip(feat_names, feat_vec):
        if name == "Band1":
            out["band_filter_rms"] = float(val)
        elif name.endswith("_env"):
            out[f"env_{band_names[j]}"] = float(val)
            j += 1
        else:
            out[name.lower()] = float(val)
    return out


def per_train_bands(train_id: int, root: Path = DATA_ROOT,
                    use_idx: int | None = None) -> dict[str, tuple[float, float, dict]]:
    """For each channel, run fast_kurtogram on a late-life file to find the
    envelope filter band. Returns {channel_name: (lo, hi, info)}.
    """
    files = list_vibration_files(train_id, root=root)
    if use_idx is None:
        # Pick a file ~95% through life (avoid the very last in case it post-failure noise dominates)
        use_idx = max(0, int(len(files) * 0.92) - 1)
    arr = tdms_to_array(load_tdms_file(files[use_idx]))
    bands: dict[str, tuple[float, float, dict]] = {}
    for i, ch in enumerate(CHANNEL_NAMES):
        lo, hi, info = select_envelope_band(arr[i])
        info["src_file"] = files[use_idx].name
        info["src_idx"] = use_idx
        bands[ch] = (lo, hi, info)
    return bands


def build_train_features(train_id: int, root: Path = DATA_ROOT,
                         verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    files = list_vibration_files(train_id, root=root)
    op = load_operation(train_id, root=root)
    op_agg = align_to_vibration(op, len(files))

    bands = per_train_bands(train_id, root=root)
    if verbose:
        for ch, (lo, hi, info) in bands.items():
            tag = " (fallback)" if info.get("fallback") else ""
            print(f"  Train{train_id} {ch} band [{lo:7.0f}, {hi:7.0f}] Hz "
                  f"kurt={info['kmax']:.2f} level={info['level']:.2f}{tag}")

    eol_sec = file_start_seconds(len(files)) + 60
    rows = []
    t0 = time.time()
    for i, path in enumerate(files, start=1):
        arr = tdms_to_array(load_tdms_file(path))
        rpm = op_agg.loc[op_agg.file_idx == i, "rpm_mean"].iloc[0]
        row: dict = dict(
            train_id=train_id,
            file_idx=i,
            file_name=path.name,
            t_start_sec=file_start_seconds(i),
            time_to_eol_sec=eol_sec - file_start_seconds(i),
            life_frac=file_start_seconds(i) / max(eol_sec, 1),
        )
        for j, ch in enumerate(CHANNEL_NAMES):
            lo, hi, _ = bands[ch]
            feats = channel_features(arr[j], rpm, lo, hi)
            for k, v in feats.items():
                row[f"{ch}_{k}"] = v
        rows.append(row)
        if verbose and (i % 20 == 0 or i == len(files)):
            print(f"  Train{train_id} {i}/{len(files)}  ({time.time()-t0:.1f}s)")

    df_feat = pd.DataFrame(rows)
    df = df_feat.merge(op_agg, on=["file_idx", "t_start_sec"], how="left")
    # info dict already carries lo/hi from select_envelope_band
    bands_summary = {ch: dict(info) for ch, (_, _, info) in bands.items()}
    return df, bands_summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_bands = {}
    for tr in [1, 2, 3, 4]:
        print(f"=== Train{tr} ===")
        df, bands = build_train_features(tr)
        out_path = OUT_DIR / f"train{tr}.parquet"
        df.to_parquet(out_path, index=False)
        all_bands[f"Train{tr}"] = bands
        print(f"  saved {out_path}  shape={df.shape}")
    # Save band selections for reproducibility
    band_rows = []
    for tr_name, bands in all_bands.items():
        for ch, info in bands.items():
            band_rows.append(dict(train=tr_name, channel=ch, **info))
    pd.DataFrame(band_rows).to_csv(OUT_DIR / "selected_bands.csv", index=False)
    print(f"  saved {OUT_DIR / 'selected_bands.csv'}")


if __name__ == "__main__":
    main()
