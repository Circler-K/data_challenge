"""Comprehensive denoising benchmark for the KSPHM-KIMM 2026 bearing-RUL task.

Run EVERY practical denoiser through the SAME pipeline (per-channel denoise ->
order tracking -> OT features -> M/T/P-weighted HI) and rank by HI quality
(monotonicity / trendability) on the 4 training bearings. Denoising is adopted
ONLY if it measurably beats the no-denoise baseline. Original TDMS untouched;
features cached per denoiser so the run is resumable.

Denoiser families:
  none            - baseline (no denoising)
  wav_univ        - wavelet soft-threshold, db4, universal threshold (the current one)
  wav_half        - same but 0.5x threshold (gentler; tests if universal over-smooths)
  wav_bayes       - wavelet BayesShrink (per-subband adaptive threshold)
  wav_sym8        - wavelet universal, sym8 (different mother wavelet)
  sk_bandpass     - spectral-kurtosis / kurtogram band-pass (keep most impulsive band)
  cepstral        - cepstral / spectral pre-whitening (DRS family: remove discrete)
  vmd             - VMD reconstruction (HEAVY: ~144s/channel -> probe only)
  emd             - EMD partial reconstruction, drop finest IMF (HEAVY -> probe only)

Run (fast family, all bearings):
    python scripts/denoise_compare.py --denoisers none,wav_univ,wav_half,wav_bayes,wav_sym8,sk_bandpass,cepstral
Probe heavy methods (1 bearing, every Nth file):
    python scripts/denoise_compare.py --denoisers vmd,emd --bearings 2 --stride 6 --tag probe
Compare only (after extraction):
    python scripts/denoise_compare.py --compare-only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from nptdms import TdmsFile
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis as _kurt

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
INHWAN = Path("c:/Users/User/WorkSpace/INHWAN")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(INHWAN))

import rpm_estimator as R                                # noqa: E402
from ot_rpm_impact import order_track, ot_features, CHS, FS  # noqa: E402
from src.operation import list_vibration_files            # noqa: E402

OUT = ROOT / "outputs" / "ot_features" / "denoise"
FILE_PERIOD = 600
HEALTHY_FRAC = 0.15


# ----------------------------- denoisers -----------------------------
def _wavelet(x, wavelet="db4", level=4, scale=1.0):
    x = np.asarray(x, float)
    coeff = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    thr = scale * sigma * np.sqrt(2.0 * np.log(max(len(x), 2)))
    coeff[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeff[1:]]
    return pywt.waverec(coeff, wavelet)[: len(x)]


def _wavelet_bayes(x, wavelet="db4", level=4):
    """BayesShrink: per-subband soft threshold thr = sigma_n^2 / sigma_signal."""
    x = np.asarray(x, float)
    coeff = pywt.wavedec(x, wavelet, level=level)
    sigma_n = np.median(np.abs(coeff[-1])) / 0.6745
    var_n = sigma_n ** 2
    new = [coeff[0]]
    for c in coeff[1:]:
        var_y = np.mean(c ** 2)
        sigma_x = np.sqrt(max(var_y - var_n, 1e-12))
        thr = var_n / sigma_x
        new.append(pywt.threshold(c, thr, mode="soft"))
    return pywt.waverec(new, wavelet)[: len(x)]


def _sk_bandpass(x, fs=FS, nbands=8):
    """Kurtogram-lite: split band into nbands oct(ish) windows, pick the one whose
    envelope is most impulsive (max kurtosis), return that band-passed signal."""
    x = np.asarray(x, float)
    nyq = fs / 2.0
    edges = np.linspace(500.0, nyq * 0.98, nbands + 1)  # skip near-DC shaft orders
    best_k, best = -np.inf, x
    for lo, hi in zip(edges[:-1], edges[1:]):
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        y = sosfiltfilt(sos, x)
        env = np.abs(hilbert(y))
        k = _kurt(env, fisher=True, bias=False)
        if k > best_k:
            best_k, best = k, y
    return best


def _cepstral_prewhiten(x):
    """Spectral pre-whitening: flatten magnitude spectrum, keep phase. Removes
    dominant discrete (shaft/gear) components, emphasises broadband fault content."""
    x = np.asarray(x, float)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    Xw = X / (mag + 1e-12 * mag.max())
    y = np.fft.irfft(Xw, n=len(x))
    return y * np.std(x) / (np.std(y) + 1e-12)


def _vmd_denoise(x, K=5, alpha=2000):
    from vmdpy import VMD
    x = np.asarray(x, float)
    n = len(x)
    xp = x[: n - (n % 2)]  # vmdpy wants even length
    u, _, _ = VMD(xp, alpha, 0.0, K, 0, 1, 1e-7)
    # keep impulsive modes (envelope kurtosis > 0), drop low-info noise modes
    keep = [m for m in u if _kurt(np.abs(hilbert(m)), fisher=True, bias=False) > 0]
    rec = np.sum(keep, axis=0) if keep else np.sum(u, axis=0)
    out = np.zeros(n)
    out[: len(rec)] = rec
    return out


def _emd_denoise(x, max_imf=6):
    from PyEMD import EMD
    x = np.asarray(x, float)
    imfs = EMD(max_imf=max_imf).emd(x)
    if len(imfs) <= 1:
        return x
    return np.sum(imfs[1:], axis=0)  # drop finest (noise) IMF


DENOISERS = {
    "none": lambda x: np.asarray(x, float),
    "wav_univ": lambda x: _wavelet(x, "db4", 4, 1.0),
    "wav_half": lambda x: _wavelet(x, "db4", 4, 0.5),
    "wav_bayes": lambda x: _wavelet_bayes(x, "db4", 4),
    "wav_sym8": lambda x: _wavelet(x, "sym8", 4, 1.0),
    "sk_bandpass": _sk_bandpass,
    "cepstral": _cepstral_prewhiten,
    "vmd": _vmd_denoise,
    "emd": _emd_denoise,
}


# ----------------------------- extraction -----------------------------
def extract_bearing(tr, denoiser_names, stride=1):
    files = list_vibration_files(tr)[::stride]
    sig_cache, ch0 = [], []
    for f in files:
        chs = [c[:] for c in TdmsFile.read(str(f)).groups()[0].channels()][:4]
        sig_cache.append(chs)
        ch0.append(chs[0])
    est_rpm = R.refine_stepwise(R.estimate_rpm_series(ch0))
    results = {}
    for name in denoiser_names:
        fn = DENOISERS[name]
        rows = []
        for k, chs in enumerate(sig_cache):
            row = {"File_Index": (k * stride) + 1}
            for ch in CHS:
                if ch < len(chs):
                    sig = fn(chs[ch])
                    row.update(ot_features(order_track(sig, float(est_rpm[k])), ch))
            rows.append(row)
        df = pd.DataFrame(rows)
        df["t_sec"] = (df["File_Index"] - 1) * FILE_PERIOD
        results[name] = df
    return results


# ----------------------------- HI quality -----------------------------
def _mono(x):
    d = np.diff(x)
    return abs((np.sum(d > 0) - np.sum(d < 0)) / max(len(d), 1))


def _trend(x):
    t = np.arange(len(x))
    return abs(np.corrcoef(x, t)[0, 1]) if np.std(x) > 0 else 0.0


def _mtp_weights(data):
    cols = [c for c in next(iter(data.values())).columns if c.startswith("Ch")]
    w = {}
    for c in cols:
        mons, trs, eol, rng = [], [], [], []
        for df in data.values():
            x = pd.Series(df[c].to_numpy(float)).rolling(5, min_periods=1).median().to_numpy()
            mons.append(_mono(x)); trs.append(_trend(x)); eol.append(x[-1]); rng.append(np.ptp(x) + 1e-9)
        prog = float(np.exp(-np.std(eol) / (np.mean(rng) + 1e-9)))
        w[c] = np.mean(mons) + np.mean(trs) + prog
    return w


def _build_hi(data, sel, wts):
    P = np.vstack([data[t][sel].to_numpy(float)[:max(3, int(len(data[t]) * HEALTHY_FRAC))] for t in data])
    mu, sd = P.mean(0), P.std(0) + 1e-9
    his = {}
    for tr, df in data.items():
        z = (df[sel].to_numpy(float) - mu) / sd
        h = (z * wts).sum(1)
        his[tr] = pd.Series(h).rolling(5, min_periods=1).median().cummax().to_numpy()
    return his


def hi_quality(data):
    w = _mtp_weights(data)
    sel = sorted(w, key=lambda c: -w[c])[:10]
    wts = np.array([w[c] for c in sel]); wts = wts / wts.sum()
    his = _build_hi(data, sel, wts)
    return np.mean([_mono(his[t]) for t in data]), np.mean([_trend(his[t]) for t in data])


# ----------------------------- driver -----------------------------
def csv_path(tag, name, tr):
    d = OUT / tag / name
    return d, d / f"Train{tr}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoisers", default="none,wav_univ,wav_half,wav_bayes,wav_sym8,sk_bandpass,cepstral")
    ap.add_argument("--bearings", default="1,2,3,4")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    names = args.denoisers.split(",")
    bearings = [int(b) for b in args.bearings.split(",")]

    if not args.compare_only:
        for tr in bearings:
            todo = [n for n in names if not csv_path(args.tag, n, tr)[1].exists()]
            if not todo:
                print(f"Train{tr}: all cached, skip"); continue
            t0 = time.time()
            res = extract_bearing(tr, todo, stride=args.stride)
            for n, df in res.items():
                d, p = csv_path(args.tag, n, tr)
                d.mkdir(parents=True, exist_ok=True)
                df.to_csv(p, index=False)
            print(f"Train{tr}: {todo} done ({len(res[todo[0]])} files, {time.time()-t0:.0f}s)", flush=True)

    # compare
    print(f"\n=== Denoising HI-quality ranking (tag={args.tag}, bearings={bearings}) ===")
    print(f"{'denoiser':>12} {'monotonic':>10} {'trend':>8} {'sum':>8}   {'verdict'}")
    base = None
    rows = []
    for n in names:
        data = {}
        for tr in bearings:
            p = csv_path(args.tag, n, tr)[1]
            if p.exists():
                data[tr] = pd.read_csv(p).sort_values("File_Index").reset_index(drop=True)
        if len(data) < len(bearings):
            print(f"{n:>12}  (incomplete: have {sorted(data)})"); continue
        m, t = hi_quality(data)
        rows.append((n, m, t))
        if n == "none":
            base = (m, t)
    for n, m, t in sorted(rows, key=lambda r: -(r[1] + r[2])):
        if base and n != "none":
            dm, dt = m - base[0], t - base[1]
            v = "KEEP" if (dm > 0.005 or dt > 0.01) and dm > -0.005 and dt > -0.01 else "reject"
            tail = f"   (mono {dm:+.3f}, trend {dt:+.3f}) {v}"
        else:
            tail = "   <- baseline" if n == "none" else ""
        print(f"{n:>12} {m:>10.3f} {t:>8.3f} {m+t:>8.3f}{tail}")


if __name__ == "__main__":
    main()
