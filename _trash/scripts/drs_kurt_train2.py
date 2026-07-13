"""DRS (Antoni 2004 multi-tap Wiener) → kurtogram on Train2.

Pipeline:
  1. Load Train2 file at chosen idx.
  2. For each channel, fit a multi-tap delayed Wiener filter (p taps at delays
     Δ ... Δ+p-1) by least-squares. The filter models the periodic
     (gear/shaft) component; subtracting it leaves random + impulsive content
     (bearing faults). See src/drs.py.
  3. Run fast_kurtogram on the original and on the DRS residual.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
from io_tdms import load_tdms_file, tdms_to_array  # type: ignore
from kurtogram import fast_kurtogram  # type: ignore
from drs import drs  # type: ignore

FS = 25600
NLEVEL = 6
P = 200
DELAY = 100
N_TRAIN = 80_000

VIB_DIR = ROOT / "Train" / "Train2_Vibration"
# idx 103 = 92% of life (post-impact, used in selected_bands.csv)
# idx 70  = 62% of life (pre-impact, healthy-ish)
FILE_IDXS = [70, 103]


def kurt_band(x: np.ndarray, fs: int = FS, nlevel: int = NLEVEL,
              n_use: int = 512_000) -> dict:
    sig = np.asarray(x[: min(len(x), n_use)], dtype=np.float64).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Kwav, Lvl, freq_w, c, kmax, bw, lvl = fast_kurtogram(sig, fs, nlevel=nlevel)
    max_level_index = int(np.argmax(Kwav[np.arange(Kwav.shape[0]),
                                         np.argmax(Kwav, axis=1)]))
    J = int(np.argmax(Kwav[max_level_index, :]))
    fc = float(freq_w[J])
    return dict(kmax=float(kmax), bw=float(bw), fc=fc, level=float(lvl),
                lo=fc - bw / 2.0, hi=fc + bw / 2.0)


def analyze(path: Path) -> list[dict]:
    df = load_tdms_file(str(path))
    sig4 = tdms_to_array(df)
    rows = []
    for i in range(4):
        x = sig4[i].astype(np.float64)
        rms_o = float(np.sqrt(np.mean(x * x)))
        kurt_o = float(kurtosis(x, fisher=False))
        t0 = time.time()
        r, _, _ = drs(x, fs=FS, delay=DELAY, p=P, n_train=N_TRAIN)
        dt_drs = time.time() - t0
        rms_r = float(np.sqrt(np.mean(r * r)))
        kurt_r = float(kurtosis(r, fisher=False))
        keep = (rms_r / rms_o) ** 2 if rms_o > 0 else 0.0
        info_o = kurt_band(x)
        info_r = kurt_band(r)
        rows.append(dict(ch=f"CH{i+1}", rms_o=rms_o, rms_r=rms_r, keep=keep,
                         kurt_o=kurt_o, kurt_r=kurt_r,
                         info_o=info_o, info_r=info_r, dt=dt_drs))
    return rows


def print_table(idx: int, fname: str, rows: list[dict]) -> None:
    print(f"\n=== file idx {idx} ({fname}) ===")
    print(f"{'ch':<5}{'rms_o':>9}{'rms_r':>9}{'keep%':>8}"
          f"{'kurt_o':>10}{'kurt_r':>10}{'drs_s':>7}")
    for r in rows:
        print(f"{r['ch']:<5}{r['rms_o']:>9.3f}{r['rms_r']:>9.3f}"
              f"{r['keep']*100:>7.1f}%{r['kurt_o']:>10.2f}"
              f"{r['kurt_r']:>10.2f}{r['dt']:>7.2f}")
    print(f"\nkurtogram (orig -> DRS residual):")
    print(f"{'ch':<5}{'kmax_o':>9}{'lvl_o':>6}{'lo_o':>8}{'hi_o':>8}"
          f"  | {'kmax_r':>9}{'lvl_r':>6}{'lo_r':>8}{'hi_r':>8}")
    for r in rows:
        a, b = r['info_o'], r['info_r']
        print(f"{r['ch']:<5}{a['kmax']:>9.2f}{a['level']:>6.2f}"
              f"{a['lo']:>8.0f}{a['hi']:>8.0f}"
              f"  | {b['kmax']:>9.2f}{b['level']:>6.2f}"
              f"{b['lo']:>8.0f}{b['hi']:>8.0f}")


def main() -> None:
    files = sorted(VIB_DIR.glob("*.tdms"))
    print(f"AR(p={P}, delay={DELAY}, n_train={N_TRAIN}) on Train2")
    for idx in FILE_IDXS:
        f = files[idx]
        rows = analyze(f)
        print_table(idx, f.name, rows)


if __name__ == "__main__":
    main()
