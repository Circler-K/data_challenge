"""Verify whether Train2 CH3/CH4 kurtogram outcome is genuine.

The selected_bands.csv shows kmax=1448 (CH3) and 2458 (CH4) at level 0
for Train2 file 000104.tdms (idx 103). That triggered the [1,10] kHz fallback.
This script checks WHY by printing raw stats of the signal so we can decide
whether the kurtogram itself misbehaved or the input has a pathological spike.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_tdms import load_tdms_file, tdms_to_array  # type: ignore

FS = 25600
F = ROOT / "Train" / "Train2_Vibration" / "000104.tdms"
df = load_tdms_file(str(F))
sig4 = tdms_to_array(df)  # (4, 1_536_000)

print(f"file: {F.name}, shape: {sig4.shape}, dtype: {sig4.dtype}")
print()
print(f"{'ch':<4}{'rms':>10}{'peak':>10}{'kurt(60s)':>12}{'argmax_t[s]':>14}"
      f"{'kurt_excl_max±0.05s':>22}")
for i in range(4):
    x = sig4[i].astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    k = float(kurtosis(x, fisher=False))  # Pearson, not excess
    j = int(np.argmax(np.abs(x)))
    t = j / FS
    # mask ±0.05s around the global peak and recompute kurtosis
    half = int(0.05 * FS)
    mask = np.ones_like(x, dtype=bool)
    mask[max(0, j - half): min(len(x), j + half)] = False
    k_excl = float(kurtosis(x[mask], fisher=False))
    print(f"CH{i+1:<3}{rms:>10.4f}{peak:>10.4f}{k:>12.2f}{t:>14.4f}"
          f"{k_excl:>22.2f}")
