"""Regenerate EDA figures with clearer per-Train layouts.

Per user feedback: early/late + BPFI/BPFO/BSF on a single envelope panel
overlapped too much. Each plot is now split so signals don't compete:

  outputs/figures/03_Train{1..4}.png    (4 files, one per Train)
      4 rows (CH1..CH4) x 4 cols
        col 0: waveform 1 s            — early(blue) + late(red) overlay
        col 1: |FFT| 5-5000 Hz log-y   — early + late overlay
        col 2: envelope spec EARLY     — blue + BPFI/BPFO/BSF shaded bands
        col 3: envelope spec LATE      — red  + BPFI/BPFO/BSF shaded bands

  outputs/figures/04_failure_summary.png  (1 file)
      2 rows (early on top, late on bottom) x 4 cols (Trains).
      Failure channel only:
          Train1=CH2, Train2=CH3, Train3=CH1, Train4=CH4
      BPFx as shaded bands with text labels.

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
# Top Spearman ρ vs life_frac (from feature analysis)
FAILURE_CH = {1: "CH2", 2: "CH3", 3: "CH1", 4: "CH4"}
CH_INDEX = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}

# Color scheme for BPFx markers (also used as legend labels)
BPFX_COLORS = {"BPFI": "tab:green", "BPFO": "tab:purple", "BSF": "tab:gray"}


def get_endpoints(tr: int):
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


def _draw_bpfx_bands(ax, scale: float, *,
                    half_pct: float = 0.04,
                    label_top: bool = False) -> None:
    """Draw BPFI/BPFO/BSF as light vertical bands instead of single lines.

    half_pct: half-width of the band as a fraction of the center frequency.
    """
    for name in ("BPFI", "BPFO", "BSF"):
        fc = BPFx_AT_1000[name] * scale
        ax.axvspan(fc * (1 - half_pct), fc * (1 + half_pct),
                   color=BPFX_COLORS[name], alpha=0.18, lw=0)
        if label_top:
            # Add a small text label at the top of the band
            ymax = ax.get_ylim()[1]
            ax.text(fc, ymax, name, color=BPFX_COLORS[name],
                    fontsize=8, ha="center", va="bottom")


def _bpfx_legend_handles():
    """Reusable legend handles for BPFx bands (Patch artists)."""
    from matplotlib.patches import Patch
    return [Patch(facecolor=BPFX_COLORS[n], alpha=0.4, label=n)
            for n in ("BPFI", "BPFO", "BSF")]


def fig03_per_train(endpoints):
    """4 figures, one per Train. 4 rows (channels) x 4 cols.

    cols: waveform | FFT | envelope-early | envelope-late
    """
    sos = butter(4, [1000.0, 10000.0], btype="band", fs=FS, output="sos")
    n_show = FS  # 1 s
    t = np.arange(n_show) / FS

    for tr in TRAIN_IDS:
        early, late, rpm_late = endpoints[tr]
        scale = max(rpm_late, 100.0) / 1000.0

        fig, axes = plt.subplots(4, 4, figsize=(22, 14))

        for i, ch in enumerate(CHANNEL_NAMES):
            # ------------- Col 0: waveform 1 s -------------
            ax = axes[i, 0]
            ax.plot(t, early[i, :n_show], color="tab:blue",
                    lw=0.55, alpha=0.85, label="early")
            ax.plot(t, late[i, :n_show],  color="tab:red",
                    lw=0.55, alpha=0.7,  label="late")
            ax.set_ylabel(f"{ch}\namplitude")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title("(a) Waveform — first 1 s", fontsize=12)
                ax.legend(fontsize=9, loc="upper right")
            if i == 3:
                ax.set_xlabel("time [s]")

            # ------------- Col 1: |FFT| 5-5000 Hz -------------
            ax = axes[i, 1]
            for sig, lbl, color in [(early[i], "early", "tab:blue"),
                                    (late[i],  "late",  "tab:red")]:
                spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
                freqs = np.fft.rfftfreq(len(sig), d=1.0 / FS)
                m = (freqs >= 5) & (freqs <= 5000)
                ax.semilogy(freqs[m], spec[m] + 1e-3,
                            color=color, lw=0.45, alpha=0.85, label=lbl)
            ax.set_ylabel("|FFT|")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title("(b) FFT 5-5000 Hz (log)", fontsize=12)
            if i == 3:
                ax.set_xlabel("frequency [Hz]")

            # ------------- Col 2: envelope spectrum (early) -------------
            ax = axes[i, 2]
            freqs, spec_e = _envelope_spec(early[i], sos)
            m = (freqs >= 1) & (freqs <= 500)
            ax.plot(freqs[m], spec_e[m], color="tab:blue", lw=0.6, alpha=0.95)
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.set_ylabel("|env spec|")
            _draw_bpfx_bands(ax, scale)
            if i == 0:
                ax.set_title(f"(c) Envelope — EARLY  (rpm~{rpm_late:.0f})",
                             fontsize=12)
                ax.legend(handles=_bpfx_legend_handles(),
                          fontsize=8, loc="upper right", ncol=3)
            if i == 3:
                ax.set_xlabel("envelope freq [Hz]")

            # ------------- Col 3: envelope spectrum (late) -------------
            ax = axes[i, 3]
            freqs, spec_l = _envelope_spec(late[i], sos)
            ax.plot(freqs[m], spec_l[m], color="tab:red", lw=0.6, alpha=0.95)
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.set_ylabel("|env spec|")
            _draw_bpfx_bands(ax, scale)
            if i == 0:
                ax.set_title(f"(d) Envelope — LATE  (rpm~{rpm_late:.0f})",
                             fontsize=12)
            if i == 3:
                ax.set_xlabel("envelope freq [Hz]")

            # Make c and d share y-axis range so the increase is obvious
            yc = axes[i, 2].get_ylim()
            yd = axes[i, 3].get_ylim()
            ymin = min(yc[0], yd[0])
            ymax = max(yc[1], yd[1])
            axes[i, 2].set_ylim(ymin, ymax)
            axes[i, 3].set_ylim(ymin, ymax)

        fig.suptitle(
            f"Train{tr} — {FAILURE_SIDE[tr]} bearing failure  ·  "
            f"early(blue) vs late(red)  ·  shaded bands = BPFI/BPFO/BSF at running rpm",
            fontsize=15, y=1.0,
        )
        fig.tight_layout()
        out = FIG_DIR / f"03_Train{tr}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def fig04_failure_summary(endpoints):
    """2 rows (early/late) x 4 cols (Trains) — failure-channel envelope only."""
    sos = butter(4, [1000.0, 10000.0], btype="band", fs=FS, output="sos")
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))

    # Need shared y-limits per Train so the early-vs-late jump is visible
    train_ylim: dict = {}

    # First pass: compute spectra and find y-range per Train
    cache: dict = {}
    for tr in TRAIN_IDS:
        early, late, rpm_late = endpoints[tr]
        ch = FAILURE_CH[tr]
        idx = CH_INDEX[ch]
        scale = max(rpm_late, 100.0) / 1000.0

        f_e, s_e = _envelope_spec(early[idx], sos)
        f_l, s_l = _envelope_spec(late[idx], sos)
        m = (f_e >= 1) & (f_e <= 500)

        cache[tr] = (f_e[m], s_e[m], s_l[m], scale, ch, rpm_late)
        # Symmetric log range: pad above max and below 1e-2 of max
        s_max = max(s_e[m].max(), s_l[m].max())
        s_min = max(min(s_e[m].min(), s_l[m].min()), s_max * 1e-4)
        train_ylim[tr] = (s_min * 0.5, s_max * 2.0)

    # Second pass: actual plotting
    for j, tr in enumerate(TRAIN_IDS):
        f, s_e, s_l, scale, ch, rpm_late = cache[tr]
        ymin, ymax = train_ylim[tr]

        # Top: early
        ax = axes[0, j]
        ax.plot(f, s_e, color="tab:blue", lw=0.7, alpha=0.95, label="early")
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        _draw_bpfx_bands(ax, scale)
        ax.set_title(f"Train{tr}  /  {ch}  (early — first file)\n"
                     f"{FAILURE_SIDE[tr]} bearing, rpm~{rpm_late:.0f}",
                     fontsize=11)
        if j == 0:
            ax.set_ylabel("|env spec| (early)")
            ax.legend(handles=_bpfx_legend_handles() + [
                plt.Line2D([0], [0], color="tab:blue", lw=1.2, label="early")],
                fontsize=8, loc="upper right", ncol=2)

        # Bottom: late
        ax = axes[1, j]
        ax.plot(f, s_l, color="tab:red", lw=0.7, alpha=0.95, label="late")
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        _draw_bpfx_bands(ax, scale)
        ax.set_xlabel("envelope freq [Hz]")
        ax.set_title(f"Train{tr}  /  {ch}  (late — last file, EOL)",
                     fontsize=11)
        if j == 0:
            ax.set_ylabel("|env spec| (late)")
            ax.legend(handles=_bpfx_legend_handles() + [
                plt.Line2D([0], [0], color="tab:red", lw=1.2, label="late")],
                fontsize=8, loc="upper right", ncol=2)

    fig.suptitle(
        "Failure-channel envelope spectra — top = EARLY (first file), "
        "bottom = LATE (last file). Y-axis shared per Train. "
        "Shaded bands = BPFI/BPFO/BSF at running rpm.",
        fontsize=13, y=1.02,
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
