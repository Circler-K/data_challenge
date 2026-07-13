"""Sanity test: DRS on synthetic (sinusoid + white noise + impulses)."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from drs import drs  # type: ignore

FS = 25600
N = 1_536_000
t = np.arange(N) / FS
rng = np.random.default_rng(0)

# Deterministic: 200 Hz tone (gear-like), amplitude 1.0
det = 1.0 * np.sin(2 * np.pi * 200 * t)
# Random: white noise, std 0.3
noise = 0.3 * rng.standard_normal(N)
# Impulses: 5 Hz Dirac train, amplitude 5
impulses = np.zeros(N)
impulse_times = np.arange(0, N, FS // 5)
impulses[impulse_times] = 5.0

x = det + noise + impulses

print(f"Synthetic signal: {N/FS:.1f}s @ {FS} Hz")
print(f"  RMS det:  {np.sqrt(np.mean(det**2)):.3f}")
print(f"  RMS rand: {np.sqrt(np.mean(noise**2)):.3f}")
print(f"  RMS imp:  {np.sqrt(np.mean(impulses**2)):.3f}")
print(f"  RMS x:    {np.sqrt(np.mean(x**2)):.3f}")

from drs import drs as drs_fn, drs_kernel_response  # type: ignore
r, d, w = drs_fn(x, fs=FS, delay=100, p=200)

print(f"\nAfter DRS (multi-tap Wiener):")
print(f"  RMS d (det estimate): {np.sqrt(np.mean(d**2)):.3f}  (truth: {np.sqrt(np.mean(det**2)):.3f})")
print(f"  RMS r (residual):     {np.sqrt(np.mean(r**2)):.3f}  (truth noise+imp: {np.sqrt(np.mean((noise+impulses)**2)):.3f})")
print(f"  Energy d/x:           {(np.sum(d**2)/np.sum(x**2))*100:.1f}%")

# Filter response
f, mag = drs_kernel_response(w, delay=100, n_fft=8192, fs=FS)
i_tone = int(np.argmin(np.abs(f - 200)))
i_far = int(np.argmin(np.abs(f - 7000)))
print(f"\nWiener kernel |H(f)|:")
print(f"  |H(200 Hz)| = {mag[i_tone]:.3f}  (expect ~1)")
print(f"  |H(7000 Hz)| = {mag[i_far]:.4f}  (expect ~0)")

# Recovery of impulses: kurtosis of residual should be high
from scipy.stats import kurtosis
print(f"\n  kurtosis(x):      {kurtosis(x, fisher=False):.2f}")
print(f"  kurtosis(r):      {kurtosis(r, fisher=False):.2f}")
print(f"  kurtosis(noise+imp): {kurtosis(noise+impulses, fisher=False):.2f}")
