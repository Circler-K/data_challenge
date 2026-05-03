"""Regenerate EDA figures with larger per-Train layouts.

Per user feedback ("사진이 너무 작아"): instead of one giant grid jamming all
channels and Trains together, produce:

  outputs/figures/03_Train{1..4}.png    (4 files)
      Per-Train detail. 4 rows (CH1..CH4) x 3 cols (waveform 1s, |FFT|,
      Hilbert envelope). Each panel large enough to read.

  outputs/figures/04_failure_summary.png  (1 file)
      Side-by-side comparison across 4 Trains using ONLY the failure
      channel of each Train (top Spearman ρ vs life_frac):
          Train1 = CH2   (Front Axial, BPFI)
          Train2 = CH3   (Rear Vertical, complex fault)
          Train3 = CH1   (Front Vertical, BPFO + impact)
          Train4 = CH4   (Rear Axial, BPFI from pre-existing defect)

Run:  python -m src.regen_figures
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
# Top Spearman ρ vs life_frac (from features_utils analysis)
FAILURE_CH = {1: "CH2", 2: "CH3", 3: "CH1", 4: "CH4"}
CH_INDEX = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}


def get_endpoints(tr: int):
    """Return (early_arr, late_arr, rpm_late) for one Train."""
    files = list_vibration_files(tr)
    op = load_operation(tr)
    agg = align_to_vibration(op, len(files))
    early = tdms_to_array(load_tdms_file(files[0]))
    late = tdms_to_array(load_tdms_file(files[-1]))
    rpm_late = float(agg["rpm_mean"].iloc[-1])
    return early, late, rpm_late


def _envelope_spec(sig: np.ndarray, sos):
    env = np.abs(hilbert(sosfiltfilt(sos, sig)))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / FS)
    return freqs, spec


def fig03_per_train(endpoints):
    """4 figures, one per Train. 4 rows (channels) x 3 cols (wave/FFT/env)."""
    sos = butter(4, [1000.0, 10000.0], btype="band", fs=FS, output="sos")
    n_show = FS  # 1 second
    t = np.arange(n_show) / FS

    for tr in TRAIN_IDS:
        early, late, rpm_late = endpoints[tr]
        scale = max(rpm_late, 100.0) / 1000.0

        fig, axes = plt.subplots(4, 3, figsize=(18, 14))
        for i, ch in enumerate(CHANNEL_NAMES):
            # Col 0: waveform (1 s)
            ax = axes[i, 0]
            ax.plot(t, early[i, :n_show], color="tab:blue",
                    lw=0.55, alpha=0.9, label="early")
            ax.plot(t, late[i, :n_show],  color="tab:red",
                    lw=0.55, alpha=0.7, label="late")
            ax.set_ylabel(f"{ch}\namplitude")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title("Waveform (first 1 s)", fontsize=12)
                ax.legend(fontsize=9)
            if i == 3:
                ax.set_xlabel("time [s]")

            # Col 1: |FFT| 5-5000 Hz, log-y
            ax = axes[i, 1]
            for sig, lbl, col in [(early[i], "early", "tab:blue"),
                                  (late[i],  "late",  "tab:red")]:
                spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
                freqs = np.fft.rfftfreq(len(sig), d=1.0 / FS)
                m = (freqs >= 5) & (freqs <= 5000)
                ax.semilogy(freqs[m], spec[m] + 1e-3,
                            color=col, lw=0.45, alpha=0.85, label=lbl)
            ax.set_ylabel("|FFT|")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title("FFT 5-5000 Hz (log)", fontsize=12)
            if i == 3:
                ax.set_xlabel("frequency [Hz]")

            # Col 2: envelope spectrum 1-500 Hz, log-y
            ax = axes[i, 2]
            for sig, lbl, col in [(early[i], "early", "tab:blue"),
                                  (late[i],  "late",  "tab:red")]:
                freqs, spec = _envelope_spec(sig, sos)
                m = (freqs >= 1) & (freqs <= 500)
                ax.plot(freqs[m], spec[m], color=col, lw=0.55, alpha=0.85, label=lbl)
            for name, color in [("BPFI", "green"), ("BPFO", "magenta"), ("BSF", "gray")]:
                ax.axvline(BPFx_AT_1000[name] * scale, color=color,
                           ls="--", lw=0.7, alpha=0.7, label=name)
            ax.set_yscale("log")
            ax.set_ylabel("|env spec|")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"Hilbert envelope spectrum  (BPFx scaled to rpm~{rpm_late:.0f})",
                             fontsize=12)
                ax.legend(fontsize=8, ncol=2)
            if i == 3:
                ax.set_xlabel("envelope freq [Hz]")

        # Highlight the failure channel row in the title
        fail_ch = FAILURE_CH[tr]
        fig.suptitle(
            f"Train{tr} — {FAILURE_SIDE[tr]} bearing failure  "
            f"(strongest signal: {fail_ch})  ·  early(blue) vs late(red)",
            fontsize=15, y=1.0,
        )
        fig.tight_layout()
        out = FIG_DIR / f"03_Train{tr}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def fig04_failure_summary(endpoints):
    """1 row x 4 cols: failure-channel envelope spectrum per Train."""
    sos = butter(4, [1000.0, 10000.0], btype="band", fs=FS, output="sos")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=False)

    for j, tr in enumerate(TRAIN_IDS):
        early, late, rpm_late = endpoints[tr]
        ch = FAILURE_CH[tr]
        idx = CH_INDEX[ch]
        scale = max(rpm_late, 100.0) / 1000.0
        ax = axes[j]
        for sig, lbl, col in [(early[idx], "early", "tab:blue"),
                              (late[idx],  "late",  "tab:red")]:
            freqs, spec = _envelope_spec(sig, sos)
            m = (freqs >= 1) & (freqs <= 500)
            ax.plot(freqs[m], spec[m], color=col, lw=0.7, alpha=0.85, label=lbl)
        for name, color in [("BPFI", "green"), ("BPFO", "magenta"), ("BSF", "gray")]:
            ax.axvline(BPFx_AT_1000[name] * scale, color=color,
                       ls="--", lw=0.7, alpha=0.7, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("envelope freq [Hz]")
        ax.set_title(f"Train{tr}  /  {ch}  ({FAILURE_SIDE[tr]}, rpm~{rpm_late:.0f})",
                     fontsize=12)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("|env spec|")
        ax.legend(fontsize=8, ncol=2, loc="upper right")

    fig.suptitle(
        "Failure-channel envelope spectra — 4 Trains side by side  "
        "(early=blue, late=red; dashed lines = BPFI/BPFO/BSF at running rpm)",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    out = FIG_DIR / "04_failure_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    print("Loading endpoint TDMS files for all 4 Trains ...")
    endpoints = {tr: get_endpoints(tr) for tr in TRAIN_IDS}
    print("Generating per-Train detail (03_TrainN.png) ...")
    fig03_per_train(endpoints)
    print("Generating failure-channel summary (04_failure_summary.png) ...")
    fig04_failure_summary(endpoints)
    print("done.")


if __name__ == "__main__":
    main()
