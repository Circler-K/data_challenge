"""Discrete Random Separation (DRS) for vibration signals.

Reference: Antoni, J. & Randall, R. B. (2004),
  "Unsupervised noise cancellation for vibration signals: Part II — A novel
   frequency-domain algorithm", Mechanical Systems and Signal Processing.

Why we use a multi-tap Wiener filter
------------------------------------
A naive single-delay cross-spectrum filter

    H(f) = S_{x,x_Δ}(f) / S_{x_Δ,x_Δ}(f)

reduces to a pure phase rotation exp(-j 2π f Δ / fs) for any wide-sense
stationary process and therefore does NOT separate deterministic from random
content (verified on synthetic data; see scripts/drs_sanity.py).

The converged solution of Antoni 2004 Part II's block algorithm is the
multi-tap Wiener predictor of x(n) from {x(n-Δ-k)}_{k=0..p-1}, identical to
time-domain SANC. We compute it directly:

    1. Solve, in least-squares sense, the AR(p) predictor with decorrelation
       delay Δ:  x(n) ≈ Σ_{k=0}^{p-1} w[k] · x(n-Δ-k).
       The optimal w produces zero output for any signal component that is
       uncorrelated with x(n-Δ-...) — i.e. random / impulsive / aperiodic
       transients.
    2. Build FIR kernel h of length Δ+p with h[Δ+k]=w[k], else 0.
    3. Apply h via FFT-based fast convolution (this is the "frequency-domain"
       step in Antoni's terminology — the FILTER is applied via FFT).
    4. residual = x − d, where d = h * x is the deterministic estimate.

Δ is chosen larger than the autocorrelation length of the random part
(typically 50–200 samples for bearing vibration at 25.6 kHz).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _fit_wiener(x: np.ndarray, delay: int, p: int,
                n_train: int, seed: int) -> np.ndarray:
    """Fit multi-tap delayed Wiener predictor by random-row LSQ."""
    N = len(x)
    M = N - delay - p + 1
    if M <= p + 10:
        raise ValueError(f"signal too short for delay={delay}, p={p}: N={N}")
    n_train = int(min(n_train, M))
    rng = np.random.default_rng(seed)
    rows = rng.choice(M, n_train, replace=False)
    A = np.empty((n_train, p), dtype=np.float64)
    for k in range(p):
        A[:, k] = x[rows + p - 1 - k]
    y = x[rows + p - 1 + delay]
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def drs(x: np.ndarray, fs: int = 25600, delay: int = 100, p: int = 200,
        n_train: int = 80_000, seed: int = 0
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete Random Separation via multi-tap delayed Wiener filter.

    Parameters
    ----------
    x : 1-D array
        Input signal.
    fs : int
        Sampling rate (only stored for reference; not used in computation).
    delay : int
        Decorrelation delay Δ in samples.
    p : int
        Number of taps in the AR predictor.
    n_train : int
        Number of LSQ training rows. ≥ p · 100 for stable fit.
    seed : int
        Random seed for the training subset.

    Returns
    -------
    residual : np.ndarray (length N)
        Random + impulsive component (bearing fault signature lives here).
    d : np.ndarray (length N)
        Estimated deterministic component (gear/shaft).
    w : np.ndarray (length p)
        Wiener filter coefficients.
    """
    x = np.asarray(x, dtype=np.float64)
    w = _fit_wiener(x, delay=delay, p=p, n_train=n_train, seed=seed)

    # Build FIR kernel: h[delay + k] = w[k]
    h = np.zeros(delay + p, dtype=np.float64)
    h[delay : delay + p] = w

    # FFT-based fast convolution; trim to original length, matching lfilter
    # convention (causal: h applied to past samples of x).
    d_full = fftconvolve(x, h, mode="full")
    d = d_full[: len(x)]
    residual = x - d
    return residual, d, w


def drs_kernel_response(w: np.ndarray, delay: int, n_fft: int = 8192,
                        fs: int = 25600) -> tuple[np.ndarray, np.ndarray]:
    """Frequency response of the deterministic-part predictor h.

    Returns (freqs_Hz, |H(f)|) for inspection. |H| close to 1 at deterministic
    frequencies, close to 0 elsewhere.
    """
    h = np.zeros(delay + len(w), dtype=np.float64)
    h[delay : delay + len(w)] = w
    H = np.fft.rfft(h, n=n_fft)
    f = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return f, np.abs(H)
