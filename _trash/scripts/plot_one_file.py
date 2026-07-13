"""Quick-look visualization + RMS for a single TDMS file.

Usage:
    python scripts/plot_one_file.py                            # default sample
    python scripts/plot_one_file.py Train/Train1_Vibration/000095.tdms
    python scripts/plot_one_file.py <path> --out outputs/quicklook
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_tdms import (
    FS,
    CHANNEL_NAMES,
    load_tdms_file,
    tdms_to_array,
    time_axis,
)


def channel_rms(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(arr.astype(np.float64) ** 2, axis=1))


def fft_mag(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    X = np.fft.rfft(x.astype(np.float64))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = (2.0 / n) * np.abs(X)
    mag[0] /= 2.0
    return f, mag


def plot_waveform(arr: np.ndarray, t: np.ndarray, rms: np.ndarray,
                  title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, arr[i], lw=0.3, color="C0")
        ax.set_ylabel(f"{CHANNEL_NAMES[i]}")
        ax.grid(True, alpha=0.3)
        ax.text(0.99, 0.92, f"RMS = {rms[i]:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_spectrum(arr: np.ndarray, fs: int, title: str, out_path: Path,
                  fmax: float = 6000.0) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for i, ax in enumerate(axes):
        f, mag = fft_mag(arr[i], fs)
        sel = f <= fmax
        ax.semilogy(f[sel], mag[sel], lw=0.6, color="C3")
        ax.set_ylabel(f"{CHANNEL_NAMES[i]}")
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle(title + f" — FFT (0–{int(fmax)} Hz)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?",
                    default="Train/Train1_Vibration/000095.tdms",
                    help="TDMS path (relative to data_challenge/ or absolute)")
    ap.add_argument("--out", default="outputs/quicklook",
                    help="Output directory (relative to data_challenge/)")
    ap.add_argument("--fmax", type=float, default=6000.0,
                    help="Max frequency on the spectrum plot [Hz]")
    args = ap.parse_args()

    in_path = Path(args.file)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_tdms_file(in_path)
    arr = tdms_to_array(df)            # shape (4, N)
    t = time_axis(arr.shape[1], FS)
    rms = channel_rms(arr)

    print(f"file       : {in_path}")
    print(f"shape      : {arr.shape}  fs={FS} Hz  duration={arr.shape[1]/FS:.3f} s")
    print("per-channel RMS:")
    for ch, r in zip(CHANNEL_NAMES, rms):
        print(f"  {ch}: {r:.6f}")

    stem = in_path.parent.name + "_" + in_path.stem    # e.g. Train1_Vibration_000095
    title = f"{in_path.parent.name} / {in_path.name}"
    plot_waveform(arr, t, rms, title, out_dir / f"{stem}_waveform.png")
    plot_spectrum(arr, FS, title, out_dir / f"{stem}_spectrum.png", fmax=args.fmax)
    print(f"saved      : {out_dir / (stem + '_waveform.png')}")
    print(f"saved      : {out_dir / (stem + '_spectrum.png')}")


if __name__ == "__main__":
    main()
