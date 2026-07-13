"""Extract within-minute (sub-file) dynamics features per file. The key one found:
sub-file PEAK CREST FACTOR (crest_max) transfers better across bearings (EOL CV 0.20
vs RMS 0.41) and separates healthy vs EOL (sep 3.21) — captures impulsiveness, which
is the physical bearing-fault signature, more transferably than energy (RMS).

Each 1-min file (1.5M samples) is split into n_sub windows; per window we compute
RMS / kurtosis / crest, then per-file we summarise (max over the 4 channels):
  crest_max    : peak crest over sub-windows  (main indicator)
  nonstat      : std/mean of sub-window RMS    (non-stationarity)
  kurt_mean    : mean sub-window kurtosis       (impulsiveness)

Run:  python scripts/subfile_extract.py --bearing 1          # train bearing
      python scripts/subfile_extract.py --test Test1          # validation bearing
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.stats import kurtosis

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
INHWAN = Path("c:/Users/User/WorkSpace/INHWAN")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(INHWAN))
from src.operation import list_vibration_files  # noqa: E402
from predict_test_wiener import bearing_files     # noqa: E402

OUT = ROOT / "outputs" / "ot_features" / "subfile"
N_SUB = 20


def subfeats(sig):
    sig = np.asarray(sig, float); L = len(sig) // N_SUB
    rms, ku, cr = [], [], []
    for i in range(N_SUB):
        w = sig[i * L:(i + 1) * L]; r = np.sqrt(np.mean(w ** 2)) + 1e-12
        rms.append(r); ku.append(kurtosis(w, fisher=True, bias=False)); cr.append(np.max(np.abs(w)) / r)
    rms, ku, cr = np.array(rms), np.array(ku), np.array(cr)
    return {"nonstat": rms.std() / rms.mean(), "kurt_mean": ku.mean(), "crest_max": cr.max()}


def file_feat(path):
    chs = [c[:] for c in TdmsFile.read(str(path)).groups()[0].channels()][:4]
    fs = [subfeats(c) for c in chs]
    return {k: max(f[k] for f in fs) for k in fs[0]}


def extract(files):
    rows = []
    for k, f in enumerate(files):
        row = {"File_Index": k + 1}
        row.update(file_feat(f))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bearing", type=int)
    ap.add_argument("--test", type=str)
    args = ap.parse_args()
    t0 = time.time()
    if args.test:
        (OUT / "test").mkdir(parents=True, exist_ok=True)
        df = extract(bearing_files(args.test))
        df.to_csv(OUT / "test" / f"{args.test}.csv", index=False)
        print(f"{args.test}: {len(df)} files, {time.time()-t0:.0f}s", flush=True)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        df = extract(list_vibration_files(args.bearing))
        df.to_csv(OUT / f"Train{args.bearing}.csv", index=False)
        print(f"Train{args.bearing}: {len(df)} files, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
