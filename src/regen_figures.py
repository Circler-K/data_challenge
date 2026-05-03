"""Regenerate EDA figures 03 and 04 to show ALL 4 channels per Train.

Previously these showed only the failure channel per Train; the user asked
to plot every channel.

Layout (both figures): rows = CH1..CH4, cols = Train1..Train4.

  - 03_waveform_fft_compare.png : 8 rows x 4 cols
        rows 0-3 = early vs late waveform per channel (1 second)
        rows 4-7 = early vs late |FFT| per channel (5-5000 Hz, log-y)
  - 04_envelope_spectra.png : 4 rows x 4 cols
        rows = CH1..CH4 envelope spectrum (1-500 Hz, log-y)
        with BPFI/BPFO/BSF marker lines scaled to that file's RPM
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(ROOT / "utils"))

from src.io_tdms import FS, CHANNEL_NAMES, load_tdms_file, tdms_to_array
from src.operation import list_vibration_files, load_operation, align_to_vibration
from src.features_utils import BPFx_AT_1000

FIG_DIR = ROOT / "outputs" / "figures"
TRAIN_IDS = (1, 2, 3, 4)
FAILURE_SIDE = {1: "Front", 2: "Rear", 3: "Front", 4: "Rear"}


def get_endpoints(tr: int):
    """Return (early_arr, late_arr, rpm_late) for one Train."""
    files = list_vibration_files(tr)
    op = load_operation(tr)
    agg = align_to_vibration(op, len(files))
    early = tdms_to_array(load_tdms_file(files[0]))
    late  = tdms_to_array(load_tdms_file(files[-1]))
    rpm_late = float(agg["rpm_mean"].iloc[-1])
    return early, late, rpm_late


def fig03_waveform_fft(endpoints):
    """8 rows (4 wave + 4 FFT) x 4 cols (Trains)."""
    fig, axes = plt.subplots(8, 4, figsize=(22, 24))
    n_show = FS  # 1 second
    t = np.arange(n_show) / FS

    for j, tr in enumerate(TRAIN_IDS):
        early, late, _ = endpoints[tr]
        for i, ch in enumerate(CHANNEL_NAMES):
            # Top half: waveforms
            ax = axes[i, j]
            ax.plot(t, early[i, :n_show], color="tab:blue",
                    lw=0.6, alpha=0.9, label="early")
            ax.plot(t, late[i, :n_show], color="tab:red",
                    lw=0.6, alpha=0.7, label="late")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"Train{tr}  ({FAILURE_SIDE[tr]} bearing failure)",
                             fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{ch}\namplitude", fontsize=10)
            if i == 3:
                ax.set_xlabel("time [s]")
            if i == 0 and j == 3:
                ax.legend(loc="upper right", fontsize=8)

            # Bottom half: |FFT| 5-5000 Hz, log-y
            ax = axes[4 + i, j]
            for sig, lbl, col in [(early[i], "early", "tab:blue"),
                                  (late[i],  "late",  "tab:red")]:
                spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
                freqs = np.fft.rfftfreq(len(sig), d=1.0 / FS)
                m = (freqs >= 5) & (freqs <= 5000)
                ax.semilogy(freqs[m], spec[m] + 1e-3,
                            color=col, lw=0.5, alpha=0.85, label=lbl)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"{ch}\n|FFT|", fontsize=10)
            if i == 3:
                ax.set_xlabel("frequency [Hz]")
            if i == 0 and j == 3:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Early vs late: waveform (top 4 rows) + |FFT| (bottom 4 rows) — all 4 channels",
                 fontsize=14, y=0.998)
    fig.tight_layout()
    out = FIG_DIR / "03_waveform_fft_compare.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def fig04_envelope(endpoints):
    """4 rows (channels) x 4 cols (Trains) — Hilbert envelope spectra."""
    sos = butter(4, [1000.0, 10000.0], btype="band", fs=FS, output="sos")
    fig, axes = plt.subplots(4, 4, figsize=(22, 14))

    for j, tr in enumerate(TRAIN_IDS):
        early, late, rpm_late = endpoints[tr]
        scale = max(rpm_late, 100.0) / 1000.0
        for i, ch in enumerate(CHANNEL_NAMES):
            ax = axes[i, j]
            for sig, lbl, col in [(early[i], "early", "tab:blue"),
                                  (late[i],  "late",  "tab:red")]:
                env = np.abs(hilbert(sosfiltfilt(sos, sig)))
                env = env - env.mean()
                spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
                freqs = np.fft.rfftfreq(len(env), d=1.0 / FS)
                m = (freqs >= 1) & (freqs <= 500)
                ax.plot(freqs[m], spec[m], color=col, lw=0.6, alpha=0.85, label=lbl)
            for name, color in [("BPFI", "green"), ("BPFO", "magenta"), ("BSF", "gray")]:
                ax.axvline(BPFx_AT_1000[name] * scale,
                           color=color, ls="--", lw=0.7, alpha=0.7, label=name)
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"Train{tr}  ({FAILURE_SIDE[tr]}, rpm~{rpm_late:.0f})",
                             fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{ch}\n|env spec|", fontsize=10)
            if i == 3:
                ax.set_xlabel("envelope frequency [Hz]")
            if i == 0 and j == 3:
                ax.legend(loc="upper right", fontsize=7, ncol=2)

    fig.suptitle("Hilbert envelope spectra (early vs late) at fault frequencies — all 4 channels",
                 fontsize=14, y=0.998)
    fig.tight_layout()
    out = FIG_DIR / "04_envelope_spectra.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    print("Loading endpoint TDMS files for all 4 Trains ...")
    endpoints = {tr: get_endpoints(tr) for tr in TRAIN_IDS}
    print("Generating fig 03 (waveform + FFT) ...")
    fig03_waveform_fft(endpoints)
    print("Generating fig 04 (envelope spectra) ...")
    fig04_envelope(endpoints)
    print("done.")


if __name__ == "__main__":
    main()
