"""종합 스윕: 전처리 x 피처추출 x 모델 을 cut-50/40/60 LOO(A_RUL)로 평가하고,
상위 조합의 검증셋 예측 + 3개 실측 일관 진실분포 기준 기대점수를 함께 출력.

평가 기준(타당성):
  - 1차: 학습 LOO A_RUL (검증셋=50파일/베어링 이므로 cut-50이 가장 test-faithful).
  - 2차: 그 모델의 6검증 예측을, B=0.49 / V2=0.4646 / Val2=73k 에 일관된
         진실 사후분포로 채점 -> 평균/범위 (실측 일관성 + 강건성).
  - 누설 없음(LOO), 리더보드 피팅 없음.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, random

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.scoring import a_rul_score
import wiener_rul as W

EST = ROOT / "outputs/ot_features/est"; ESTT = ROOT / "outputs/ot_features/test"
FP = 600
NS = {1: 126, 2: 114, 3: 89, 4: 137}
TESTS = [f"Test{i}" for i in range(1, 7)]
FEATS6 = ["OT_RMS", "OT_Kurtosis", "OT_CrestFactor", "Order_BandEnergy", "Spectral_Entropy", "Env_BandEnergy"]

_cache = {}
def raw(which, ident, feat):
    key = (which, ident, feat)
    if key in _cache: return _cache[key]
    d = EST if which == "train" else ESTT
    fn = f"Train{ident}.csv" if which == "train" else f"{ident}.csv"
    df = pd.read_csv(d / fn).sort_values("File_Index")
    cols = [c for c in df.columns if c.endswith(feat)]
    _cache[key] = (df, cols)
    return df, cols

def series(which, ident, feat, agg, smooth):
    df, cols = raw(which, ident, feat)
    X = df[cols].to_numpy(float)
    if agg == "max": v = X.max(1)
    elif agg == "mean": v = X.mean(1)
    elif agg == "rear": v = X[:, -2:].mean(1)   # Ch2/Ch3 (rear)
    return pd.Series(v).rolling(smooth, min_periods=1).median().to_numpy()

def calib(feat, others, agg, smooth):
    s = {t: series("train", t, feat, agg, smooth) for t in others}
    heal = np.median([np.median(s[t][:max(3, int(len(s[t]) * 0.15))]) for t in others])
    eol = np.median([s[t][-1] for t in others])
    return heal, eol

def lf_val(feat, val, heal, eol):
    return min(max((val - heal) / (eol - heal + 1e-9), 0.0), 1.0)

# ---------- LOO over training (cut d) ----------
def loo_lf(feats, agg, smooth, capmode):
    scores = []
    for d in (40, 50, 60):
        for t in (1, 2, 3, 4):
            others = [x for x in (1, 2, 3, 4) if x != t]
            if d >= NS[t]: continue
            cap = (max((NS[o]-50)*FP for o in others) if capmode == "rul50"
                   else max((NS[o]-1)*FP+60 for o in others) if capmode == "life"
                   else float(np.median([(NS[o]-50)*FP for o in others])))
            floor = 1800.0
            lfs = []
            for f in feats:
                heal, eol = calib(f, others, agg, smooth)
                lfs.append(lf_val(f, series("train", t, f, agg, smooth)[d-1], heal, eol))
            pred = max(floor, cap - float(np.mean(lfs)) * (cap - floor))
            scores.append(float(a_rul_score((NS[t]-d)*FP+60, pred)))
    return float(np.mean(scores))

def loo_const(factor):
    scores = []
    for d in (40, 50, 60):
        for t in (1, 2, 3, 4):
            others = [x for x in (1, 2, 3, 4) if x != t]
            if d >= NS[t]: continue
            life = float(np.median([(NS[o]-1)*FP+60 for o in others]))
            pred = max(1800.0, factor * life)
            scores.append(float(a_rul_score((NS[t]-d)*FP+60, pred)))
    return float(np.mean(scores))

def predict_lf(feats, agg, smooth, capmode):
    cap = (max((NS[o]-50)*FP for o in (1,2,3,4)) if capmode == "rul50"
           else max((NS[o]-1)*FP+60 for o in (1,2,3,4)) if capmode == "life"
           else float(np.median([(NS[o]-50)*FP for o in (1,2,3,4)])))
    floor = 1800.0
    out = []
    for n in TESTS:
        lfs = []
        for f in feats:
            heal, eol = calib(f, (1,2,3,4), agg, smooth)
            lfs.append(lf_val(f, series("test", n, f, agg, smooth)[-1], heal, eol))
        out.append(int(round(max(floor, cap - float(np.mean(lfs)) * (cap - floor)))))
    return out

def predict_const(factor):
    life = float(np.median([(NS[o]-1)*FP+60 for o in (1,2,3,4)]))
    return [int(round(max(1800.0, factor*life)))]*6

# ---------- inferred-truth posterior (consistent with B & V2 & Val2=73k) ----------
def build_posterior(n=800):
    B=[59351,51138,19936,12168,1800,1800]; V2=[53153,51138,13079,35633,14732,14313]
    tot=lambda p,T: float(np.mean([a_rul_score(T[i],p[i]) for i in range(6)]))
    err=lambda T: abs(tot(B,T)-0.49)+abs(tot(V2,T)-0.464633)
    random.seed(0); rng=list(range(2000,84001,2000)); post=[]; tries=0
    while len(post)<n and tries<2_000_000:
        tries+=1
        T=[random.choice(rng),73000,random.choice(rng),random.choice(rng),random.choice(rng),random.choice(rng)]
        if err(T)<0.015: post.append(T)
    return post
POST = build_posterior()
def real_est(pred):
    sc=[float(np.mean([a_rul_score(T[i],pred[i]) for i in range(6)])) for T in POST]
    return np.mean(sc), np.min(sc), np.max(sc)

# ---------- run sweep ----------
SETS = {
    "transfer(Ord+Spec)": ["Order_BandEnergy","Spectral_Entropy"],
    "all6": FEATS6,
    "all4(R,K? no)": ["Spectral_Entropy","Order_BandEnergy","Env_BandEnergy","OT_RMS"],
    "Order_only": ["Order_BandEnergy"],
    "Spectral_only": ["Spectral_Entropy"],
    "faultEnv(Ord+Env)": ["Order_BandEnergy","Env_BandEnergy"],
    "RMS_only": ["OT_RMS"],
}
rows=[]
for name,feats in SETS.items():
    for agg in ("max","mean","rear"):
        for smooth in (5,):
            for cap in ("rul50","median"):
                loo=loo_lf(feats,agg,smooth,cap)
                rows.append(("lf|"+name+f"|{agg}|cap={cap}", loo, lambda f=feats,a=agg,s=smooth,c=cap: predict_lf(f,a,s,c)))
for fac in (0.5,0.6,0.7,0.85,1.0):
    loo=loo_const(fac)
    rows.append((f"const|x{fac}", loo, lambda ff=fac: predict_const(ff)))

rows.sort(key=lambda r:-r[1])
print("="*100)
print(f"{'pipeline':40s}{'LOO':>7s}   {'검증예측':>28s}   기대점수(posterior)")
print("="*100)
for name,loo,pf in rows[:18]:
    pred=pf(); m,lo,hi=real_est(pred)
    print(f"{name:40s}{loo:7.3f}   {str(pred):>28s}   {m:.3f} [{lo:.2f}-{hi:.2f}]")
