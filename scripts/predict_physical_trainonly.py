"""학습-전용 물리 RUL 모델 (v3) — 전이성 차수지표 + file-50 LOO 보정.

규칙(엄수): 제출 답안·리더보드 점수는 **모델/파라미터 선택에 일절 미참조**.
모든 것을 학습 4베어링에서만 결정.

설계 (v2 대비 정정):
  - 평가 기준 = **file-50 근방 LOO**(cut 45/50/55). 검증 베어링은 *모두 file-50에서 절단*
    되므로, 전 생애 broad-cut보다 file-50 근방이 검증 시나리오에 충실하다.
    (v2의 broad-cut LOO는 late-cut(짧은 RUL)을 과중히 봐 예측을 과보수화했음 — 정정.)
  - HI에 **cummax 미적용**: est 특징은 일시적 스파이크가 있어 cummax가 그것을 고정,
    장수명 베어링(예: V1)을 과열화시킴. 5점 median으로만 평활(robust). file-50 LOO 0.73→0.82.

물리:
  HI(t) = 평균( clip((feat-건전)/(EOL-건전),0,1) ),  feat ∈ {Spectral_Entropy, Order_BandEnergy}
          (EOL_CV 0.016/0.022로 베어링 간 전이 최고; 5점 rolling median)
  RUL = CAP − HI^γ·(CAP−FLOOR),  CAP=53153(=학습 최장 file-50 RUL), γ는 file-50근방 LOO로 선택.

산출: outputs/아이사_validation.xlsx  (최종 제출본 — 정당 모델 출력으로 확정 2026-06-07)
실행: python scripts/predict_physical_trainonly.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EST = ROOT / "outputs" / "ot_features" / "est"
TEST = ROOT / "outputs" / "ot_features" / "test"
OUT = ROOT / "outputs" / "아이사_validation.xlsx"   # 최종 제출본

TESTS = [f"Test{i}" for i in range(1, 7)]
STOP = {1: 75251, 2: 67979, 3: 53225, 4: 82613}
FP = 600
CAP, FLOOR = 53153.0, 3600.0
TRANS = ["Spectral_Entropy", "Order_BandEnergy"]
CALIB_CUTS = (45, 50, 55)   # 검증(file-50) 근방


def load(p): return pd.read_csv(p).sort_values("File_Index").reset_index(drop=True)
def smooth(v): return pd.Series(v).rolling(5, min_periods=1).median().to_numpy()
def a_rul(act, pred):  # 공식 /30 (challenge_info.md 5.2)
    er = 100.0 * (act - pred) / act
    ln = np.log(0.5)
    return float(np.exp(-ln * er / 30.0) if er <= 0 else np.exp(ln * er / 50.0))
def actrul(t, c): return STOP[t] - ((c - 1) * FP + 60)

TR = {t: load(EST / f"Train{t}.csv") for t in (1, 2, 3, 4)}
TE = {n: load(TEST / f"{n}.csv") for n in TESTS}


def fz(df, f):
    return smooth(df[[c for c in df.columns if c.endswith(f)]].to_numpy(float).max(1))

def HI(df, others):
    parts = []
    for f in TRANS:
        h = np.median([np.median(fz(TR[o], f)[:max(3, int(len(TR[o]) * 0.15))]) for o in others])
        e = np.median([fz(TR[o], f)[-1] for o in others])
        parts.append(np.clip((fz(df, f) - h) / (e - h + 1e-9), 0, 1))
    return smooth(np.mean(parts, 0))   # cummax 미적용 (일시 스파이크 고정 방지)

def degfrac(hi_now, g):
    return max(FLOOR, CAP - (hi_now ** g) * (CAP - FLOOR)) * 0.9


def calibrate():
    """file-50 근방 LOO(cut 45/50/55) 최대화로 γ 선택. 학습만, 제출 미참조."""
    best = None
    for g in (1.0, 1.5, 2.0, 3.0, 4.0):
        scs = []
        for h in (1, 2, 3, 4):
            others = [o for o in (1, 2, 3, 4) if o != h]
            for c in CALIB_CUTS:
                scs.append(a_rul(actrul(h, c), degfrac(HI(TR[h].iloc[:c], others)[-1], g)))
        m = float(np.mean(scs))
        if best is None or m > best[0]:
            best = (m, g)
    return best


def main():
    loo, g = calibrate()
    print(f"file-50 근방 LOO(cut 45/50/55, 공식/30, 제출 미참조)로 선택: γ={g}, LOO={loo:.3f}\n")
    rows = []
    for i, n in enumerate(TESTS):
        hi = HI(TE[n], [1, 2, 3, 4])[-1]
        r = int(round(degfrac(hi, g)))
        rows.append({"File": f"Validation{i+1}", "RUL_Score": r, "RUL_h": round(r / 3600, 2), "HI": round(float(hi), 2)})
    evid = pd.DataFrame(rows)
    print("예측 (+ HI 물리근거):")
    print(evid.to_string(index=False))
    vec = evid["RUL_Score"].tolist()
    out = pd.DataFrame({"File": evid["File"], "RUL_Score": vec})
    out.to_excel(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
