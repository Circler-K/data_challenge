"""자동 탐색: 다양한 모델/전처리/매핑 -> 각 출력을 정답가정 -> 5앵커 재현편차 평가.
편차 작은 후보(=진실근접 + 모델근거) 순으로 보고. 보정은 안 함(순수 모델 출력).
"""
from __future__ import annotations
import sys
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
def dev(T): return sum(abs(float(np.mean([a_rul_score(T[i],S[i]) for i in range(6)]))-r) for S,r in ANCH)
_t={}
def tdf(n):
    if n not in _t:_t[n]=pd.read_csv(R.ESTT/f"{n}.csv").sort_values("File_Index")
    return _t[n]
def cl(df,f): return [c for c in df.columns if any(c.endswith(x) for x in f)]
_lf={}
def lf1(n,feat,agg):
    k=(n,feat,agg)
    if k in _lf: return _lf[k]
    def red(X):return X.max(1) if agg=="max" else (X.mean(1) if agg=="mean" else X[:,-2:].mean(1))
    val=pd.Series(red(tdf(n)[cl(tdf(n),[feat])].to_numpy(float))).rolling(5,1).median().to_numpy()[-1]
    s={t:pd.Series(red(EST[t][cl(EST[t],[feat])].to_numpy(float))).rolling(5,1).median().to_numpy() for t in (1,2,3,4)}
    h=np.median([np.median(s[t][:max(3,int(len(s[t])*0.15))]) for t in (1,2,3,4)]);e=np.median([s[t][-1] for t in (1,2,3,4)])
    _lf[k]=min(max((val-h)/(e-h+1e-9),0),1); return _lf[k]
FEATS6=["OT_RMS","OT_Kurtosis","OT_CrestFactor","Order_BandEnergy","Spectral_Entropy","Env_BandEnergy"]
import itertools
SUBSETS=[]
for r in (1,2,3):
    for c in itertools.combinations(FEATS6,r): SUBSETS.append(list(c))
SUBSETS.append(FEATS6)
res=[]
for f in SUBSETS:
  for agg in ("max","mean"):
    lfm={n:[lf1(n,x,agg) for x in f] for n in TESTS}
    for comb in ("mean","max"):
      base={n:(max(lfm[n]) if comb=="max" else float(np.mean(lfm[n]))) for n in TESTS}
      for cap in (53153.,65000.,73000.,81648.):
        for g in (0.5,1.0,2.0,3.0):
          T=[int(round(max(1800.,cap-(base[n]**g)*(cap-1800.)))) for n in TESTS]
          res.append((dev(T),f"lf|{'+'.join(x[:3] for x in f)}|{agg}|{comb}|cap{int(cap)}|g{g}",T))
res.sort()
print(f"총 {len(res)}개 모델설정 평가. 5앵커 재현편차 작은 순:")
print(f'{"config":48s}{"편차":>6s}  벡터(h)')
seen=set()
shown=0
for d,n,T in res:
    key=tuple(round(x/3600) for x in T)
    if key in seen: continue
    seen.add(key); print(f'{n:48s}{d:6.3f}  {[round(x/3600,1) for x in T]}'); shown+=1
    if shown>=18: break
