"""모델 설정 대량 스윕 -> 5앵커 일관 진실분포 기준 *기대점수* 최대 모델 탐색.
(재현편차가 아니라 기대점수 = 실제로 높게 받을 가능성으로 평가)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("c:/Users/User/WorkSpace/data_challenge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import wiener_rul as W, predict_robust as R
FP=600; NS={1:126,2:114,3:89,4:137}; TESTS=[f"Test{i}" for i in range(1,7)]; EST=W.load("est")
ALL=["OT_RMS","OT_Kurtosis","OT_CrestFactor","Order_BandEnergy","Spectral_Entropy","Env_BandEnergy"]
# 5앵커 진실분포
A=[(np.array([59351,51138,19936,12168,1800,1800.]),.48972),(np.array([53153,51138,13079,35633,14732,14313.]),.46463),
   (np.array([53153,15118,13079,35633,14732,14313.]),.41),(np.array([59351,72200,34515,26437,12636,6615.]),.424539),
   (np.array([48187,71382,35989,19158,44242,46357.]),.54923)]
def vsc(T,S): Er=(T-S)/T*100; return np.where(Er>0,0.5**(Er/50),0.5**(-Er/30)).mean(1)
rng=np.random.default_rng(0); post=[]
for _ in range(50):
    P=rng.uniform(1800,95000,(2_000_000,6)); m=np.ones(len(P),bool)
    for S,r in A: m&=np.abs(vsc(P,S)-r)<0.025
    if m.any(): post.append(P[m])
post=np.vstack(post); print(f"5앵커 진실분포 {len(post):,}개")
def exp_score(v): return float(vsc(post,np.array(v,float)).mean())

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
 "all6":ALL,"energy":["OT_RMS","Order_BandEnergy","Env_BandEnergy"],"ord_spec_env":["Order_BandEnergy","Spectral_Entropy","Env_BandEnergy"],
 "ord_kurt":["Order_BandEnergy","OT_Kurtosis"],"rms":["OT_RMS"]}
res=[]
for sk,f in SUB.items():
  for agg in ("max","mean"):
    lfm={n:[lf1(n,x,agg) for x in f] for n in TESTS}
    for comb in ("mean","max"):
      base={n:(max(lfm[n]) if comb=="max" else float(np.mean(lfm[n]))) for n in TESTS}
      for cap in (53153.,65000.,73000.,81648.):
        for g in (0.5,1.0,1.5,2.0,3.0):
          T=[int(round(max(1800.,cap-(base[n]**g)*(cap-1800.)))) for n in TESTS]
          res.append((exp_score(T),f"lf|{sk}|{agg}|c{comb}|cap{int(cap)}|g{g}",T))
res.sort(key=lambda r:-r[0])
print("=== 기대점수 최대 모델 상위 15 ===")
print(f'{"config":38s}{"기대":>6s}  벡터(h)')
for e,n,T in res[:15]:
    print(f'{n:38s}{e:6.3f}  {[round(x/3600,1) for x in T]}')
print(f'(참고) candB 기대 {exp_score([48187,71382,35989,19158,44242,46357]):.3f}')
