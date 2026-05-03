"""Export TDMS vibration files to viewable CSV + PNG.

Each TDMS file (1 minute @ 25.6 kHz, 4 channels) is downsampled with
``scipy.signal.decimate`` (anti-aliased) to 800 Hz, then saved as
  - outputs/csv/Train{i}/000XXX.csv  (Excel-friendly: 48,000 rows x 4 cols)
  - outputs/plots/Train{i}/000XXX.png (4-channel time-series)

Operation CSVs are already CSV; we only generate a per-Train overview plot
(RPM / Torque / Front+Rear temperature).

Run:
    python -m src.export_view              # convert all 4 trains
    python -m src.export_view 1            # one train only
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_tdms import (
    FS, CHANNEL_NAMES, load_tdms_file, tdms_to_array, file_start_seconds,
)
from src.operation import list_vibration_files, load_operation

DECIM = 32                       # 25_600 / 32 = 800 Hz
FS_DECIM = FS // DECIM           # 800
CSV_DIR = ROOT / "outputs" / "csv"
PLOT_DIR = ROOT / "outputs" / "plots"


def export_one(path: Path, csv_path: Path, plot_path: Path) -> None:
    """Convert one TDMS file to a decimated CSV and a 4-channel PNG."""
    arr = tdms_to_array(load_tdms_file(path))  # (4, 1_536_000)
    # Anti-aliased decimation per channel
    decim = np.stack([decimate(arr[i], DECIM, ftype="iir", zero_phase=True)
                      for i in range(arr.shape[0])])  # (4, 48_000)
    n = decim.shape[1]
    t = np.arange(n) / FS_DECIM

    df = pd.DataFrame(decim.T.astype(np.float32), columns=list(CHANNEL_NAMES))
    df.insert(0, "time_sec", t.astype(np.float32))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.5g")

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for i, ch in enumerate(CHANNEL_NAMES):
        axes[i].plot(t, decim[i], lw=0.4, color=f"C{i}")
        axes[i].set_ylabel(ch)
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{path.parent.name} / {path.name}  (decimated to {FS_DECIM} Hz)",
                 fontsize=11)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def export_train(train_id: int, verbose: bool = True) -> None:
    files = list_vibration_files(train_id)
    csv_dir = CSV_DIR / f"Train{train_id}_Vibration"
    plot_dir = PLOT_DIR / f"Train{train_id}_Vibration"
    t0 = time.time()
    for i, path in enumerate(files, start=1):
        stem = path.stem  # e.g. "000001"
        export_one(path, csv_dir / f"{stem}.csv", plot_dir / f"{stem}.png")
        if verbose and (i % 20 == 0 or i == len(files)):
            print(f"  Train{train_id} {i}/{len(files)}  ({time.time()-t0:.1f}s)")


def plot_operation(train_id: int) -> None:
    """Save a single overview PNG of the Train{i}_Operation.csv."""
    op = load_operation(train_id)
    t_h = op["Time[sec]"].to_numpy() / 3600.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t_h, op["Motor speed[rpm]"], color="tab:blue", lw=0.6)
    axes[0].set_ylabel("RPM"); axes[0].grid(alpha=0.3)
    axes[1].plot(t_h, op["Torque[Nm]"], color="tab:purple", lw=0.5)
    axes[1].axhline(-20, color="red", ls="--", lw=0.7, label="-20 Nm stop")
    axes[1].set_ylabel("Torque [Nm]"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    axes[2].plot(t_h, op["TC SP Front"], color="tab:orange", lw=0.6, label="Front")
    axes[2].plot(t_h, op["TC SP Rear"],  color="tab:green",  lw=0.6, label="Rear")
    axes[2].axhline(200, color="red", ls="--", lw=0.7, label="200°C stop")
    axes[2].set_ylabel("Temp [°C]"); axes[2].set_xlabel("Time [hours]")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
    fig.suptitle(f"Train{train_id} Operation overview")
    fig.tight_layout()
    out = PLOT_DIR / f"Train{train_id}_Operation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    targets = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    for tr in targets:
        print(f"=== Train{tr} ===")
        plot_operation(tr)
        export_train(tr)
    print("done.")


if __name__ == "__main__":
    main()
