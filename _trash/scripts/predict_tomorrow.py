"""TOMORROW (6/4) CANDIDATE: Spectral-Entropy degradation-fraction × LOO-optimal
conservative factor 0.75. Best test-faithful validation of any model built:
cut-40/50/60 LOO = 0.711 (others 0.4-0.6); Spectral_Entropy transfers best
(EOL CV 0.02). The 0.75 shrink (LOO-optimal, clear peak) confirms the
"numbers were too big" intuition and aligns with the asymmetric metric (over ÷30).

Writes a PREVIEW file (does NOT overwrite today's RMS submission). Tomorrow, after
seeing today's RMS real score, decide RMS-view (short Test5/6) vs this (even moderate)
and copy the chosen vector into 아이사_validation.xlsx.

Run:  python scripts/predict_tomorrow.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
EST = ROOT / "outputs/ot_features/est"; ESTT = ROOT / "outputs/ot_features/test"
OUT = ROOT / "outputs" / "아이사_validation_spectral_x075.xlsx"
TESTS = [f"Test{i}" for i in range(1, 7)]
CAP, FLOOR, FACTOR = 53153.0, 3600.0, 0.75


def spec(which, ident):
    d = EST if which == "train" else ESTT
    fn = f"Train{ident}.csv" if which == "train" else f"{ident}.csv"
    df = pd.read_csv(d / fn).sort_values("File_Index")
    cols = [c for c in df.columns if c.endswith("Spectral_Entropy")]
    v = df[cols].to_numpy(float).max(1)
    return pd.Series(v).rolling(5, min_periods=1).median().to_numpy()


def main():
    tr = {t: spec("train", t) for t in (1, 2, 3, 4)}
    heal = float(np.median([np.median(tr[t][:max(3, int(len(tr[t]) * 0.15))]) for t in tr]))
    eol = float(np.median([tr[t][-1] for t in tr]))
    rows = []
    for i, n in enumerate(TESTS):
        v = float(spec("test", n)[-1])
        lf = min(max((v - heal) / (eol - heal), 0.0), 1.0)
        rul = (CAP - lf * (CAP - FLOOR)) * FACTOR
        rows.append({"File": f"Validation{i+1}", "RUL_Score": int(round(max(600, rul)))})
    df = pd.DataFrame(rows)
    print(f"S_healthy={heal:.2f} S_EOL={eol:.2f} factor={FACTOR}")
    print(df.to_string(index=False))
    df.to_excel(OUT, index=False)
    print(f"\nPREVIEW written: {OUT}  (today's submission 아이사_validation.xlsx = RMS, untouched)")


if __name__ == "__main__":
    main()
