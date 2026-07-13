"""모델 설정 대량 스윕 -> 5개 실측 점수를 가장 잘 재현하는(진실 근접) 모델 탐색.
재현편차 = sum|모델출력을정답으로채점한값 - 실제점수| (5앵커). 작을수록 진실근접+고득점.
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("c:/Users/User/WorkSpace/data_challenge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from src.scoring import a_rul_score
import wiener_rul as W, predict_robust as R
FP=600; NS={1:126,2:114,3:89,4:137}; TESTS=[f"Test{i}" for i in range(1,7)]; EST=W.load("est")
ALL=["OT_RMS","OT_Kurtosis","OT_CrestFactor","Order_BandEnergy","Spectral_Entropy","Env_BandEnergy"]
ANCH=[([59351,51138,19936,12168,1800,1800],0.48972),([53153,51138,13079,35633,14732,14313],0.46463),
      ([53153,15118,13079,35633,14732,14313],0.41),([59351,72200,34515,26437,12636,6615],0.424539),
      ([48187,71382,35989,19158,44242,46357],0.54923)]
def dev(T):
    d=0
    for S,r in ANCH: d+=abs(float(np.mean([a_rul_score(T[i],S[i]) for i in range(6)]))-r)
    return d
_t={}
def tdf(n):
    if n not in _t:_t[n]=pd.read_csv(R.ESTT/f"{n}.csv").sort_values("File_Index")
    return _t[n]
def cl(df,f): return [c for c in df.columns if any(c.endswith(x) for x in f)]
def lf1(n,feat,agg):
    def red(X):return X.max(1) if agg=="max" else (X.mean(1) if agg=="mean" else X[:,-2:].mean(1))
    val=pd.Series(red(tdf(n)[cl(tdf(n),[feat])].to_numpy(float))).rolling(5,1).median().to_numpy()[-1]
    s={t:pd.Series(red(EST[t][cl(EST[t],[feat])].to_numpy(float))).rolling(5,1).median().to_numpy() for t in (1,2,3,4)}
    heal=np.median([np.median(s[t][:max(3,int(len(s[t])*0.15))]) for t in (1,2,3,4)]);eol=np.median([s[t][-1] for t in (1,2,3,4)])
    return min(max((val-heal)/(eol-heal+1e-9),0),1)
SUB={"transfer":["Order_BandEnergy","Spectral_Entropy"],"order":["Order_BandEnergy"],"spectral":["Spectral_Entropy"],
 "all6":ALL,"energy":["OT_RMS","Order_BandEnergy","Env_BandEnergy"],"no_order":["OT_RMS","OT_Kurtosis","OT_CrestFactor","Spectral_Entropy","Env_BandEnergy"],
 "ord_spec_env":["Order_BandEnergy","Spectral_Entropy","Env_BandEnergy"],"rms":["OT_RMS"],"kurt":["OT_Kurtosis"]}
res=[]
# lf 모델
for sk,f in SUB.items():
  for agg in ("max","mean"):
    lfm={n:[lf1(n,x,agg) for x in f] for n in TESTS}
    for combine in ("mean","max"):
      base={n:(max(lfm[n]) if combine=="max" else np.mean(lfm[n])) for n in TESTS}
      for cap in (53153.,71640.,81648.,45000.):
        for g in (0.5,1.0,2.0):
          T=[int(round(max(1800.,cap-(base[n]**g)*(cap-1800.)))) for n in TESTS]
          res.append((dev(T),f"lf|{sk}|{agg}|{combine}|cap{int(cap)}|g{g}",T))
res.sort()
print("=== 5앵커 재현편차 작은 모델 상위 15 (작을수록 진실근접) ===")
print(f'{"config":40s}{"편차":>6s}  벡터(h)')
for d,n,T in res[:15]:
    print(f'{n:40s}{d:6.3f}  {[round(x/3600,1) for x in T]}')
print()
print(f'(참고) candB 편차={dev([48187,71382,35989,19158,44242,46357]):.3f}, max-전이성+Val2 편차={dev([53153,72200,27416,37484,53153,47469]):.3f}')
