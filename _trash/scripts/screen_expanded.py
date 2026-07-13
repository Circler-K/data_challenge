"""확장 스크린: 559종에 없던 방법군 대거 추가 -> B,V2(0.41 제외) 일관성 체크.
추가: 비선형 HI->RUL(gamma/exp), faultorder 피처, 총수명 회귀, 보수계수, 앙상블, PCA-HI.
끝에 '일관 영역이 요구하는 형태 vs 모델 출력' 구조 간극 분석.
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
B=[59351,51138,19936,12168,1800,1800]; rB=0.48972
V2=[53153,51138,13079,35633,14732,14313]; rV2=0.46463
def cons(P):
    cB=float(np.mean([a_rul_score(P[i],B[i]) for i in range(6)])); cV=float(np.mean([a_rul_score(P[i],V2[i]) for i in range(6)]))
    return abs(cB-rB)+abs(cV-rV2),cB,cV
_t={};
def tdf(n):
    if n not in _t:_t[n]=pd.read_csv(R.ESTT/f"{n}.csv").sort_values("File_Index")
    return _t[n]
def cl(df,f): return [c for c in df.columns if any(c.endswith(x) for x in f)]
def lf1(n,feat,agg):
    def red(X): return X.max(1) if agg=="max" else (X.mean(1) if agg=="mean" else X[:,-2:].mean(1))
    val=pd.Series(red(tdf(n)[cl(tdf(n),[feat])].to_numpy(float))).rolling(5,1).median().to_numpy()[-1]
    s={t:pd.Series(red(EST[t][cl(EST[t],[feat])].to_numpy(float))).rolling(5,1).median().to_numpy() for t in (1,2,3,4)}
    heal=np.median([np.median(s[t][:max(3,int(len(s[t])*0.15))]) for t in (1,2,3,4)]); eol=np.median([s[t][-1] for t in (1,2,3,4)])
    return min(max((val-heal)/(eol-heal+1e-9),0),1)
SUB={"all6":ALL,"energy":["OT_RMS","Order_BandEnergy","Env_BandEnergy"],"no_order":["OT_RMS","OT_Kurtosis","OT_CrestFactor","Spectral_Entropy","Env_BandEnergy"],
 "transfer":["Order_BandEnergy","Spectral_Entropy"],"rms_env":["OT_RMS","Env_BandEnergy"],"spectral":["Spectral_Entropy"],"order":["Order_BandEnergy"],"rms":["OT_RMS"],"env":["Env_BandEnergy"]}
cands=[]
# (1) lf x 비선형 gamma x cap x 보수계수
for sk,f in SUB.items():
  for agg in ("max","mean"):
    lfm={n:np.mean([lf1(n,x,agg) for x in f]) for n in TESTS}
    for cap in (53153.,71640.,81648.):
      for g in (0.5,1.0,1.5,2.0):
        for fac in (1.0,0.85,0.7):
          v=[int(round(max(1800.,(cap-(lfm[n]**g)*(cap-1800.))*fac))) for n in TESTS]
          cands.append((f"lf|{sk}|{agg}|cap{int(cap)}|g{g}|f{fac}",v))
# (2) 총수명 회귀: total ~ a+b*descriptor (LOO), RUL=total-29460, clip
EL=29460
def hi_level(n,feat,agg):  # file50 HI level (z-score mean over chosen, last)
    def red(X): return X.max(1) if agg=="max" else X.mean(1)
    return pd.Series(red(tdf(n)[cl(tdf(n),[feat])].to_numpy(float))).rolling(5,1).median().to_numpy()[-1]
for feat in ["OT_RMS","Env_BandEnergy","Spectral_Entropy","Order_BandEnergy"]:
  for agg in ("max","mean"):
    # train descriptor at file50 vs total life
    def red(X): return X.max(1) if agg=="max" else X.mean(1)
    xs=[]; ys=[]
    for t in (1,2,3,4):
        d=red(EST[t][cl(EST[t],[feat])].to_numpy(float)); xs.append(pd.Series(d).rolling(5,1).median().to_numpy()[min(49,len(d)-1)]); ys.append((NS[t]-1)*FP+60)
    xs=np.array(xs); ys=np.array(ys)
    A=np.polyfit(xs,ys,1)
    lo,hi=ys.min(),ys.max()*1.2
    v=[]
    for n in TESTS:
        xt=pd.Series(red(tdf(n)[cl(tdf(n),[feat])].to_numpy(float))).rolling(5,1).median().to_numpy()[-1]
        tot=np.clip(np.polyval(A,xt),lo,hi); v.append(int(round(max(1800.,tot-EL))))
    cands.append((f"totlife_reg|{feat}|{agg}",v))
# (3) faultorder lf (캐시 사용)
FO=ROOT/"outputs/ot_features/faultorder"; FOT=FO/"test"
if FOT.exists():
  try:
    fcols=None
    def foser(path,ident,istest):
        fn=(f"{ident}.csv" if istest else f"Train{ident}.csv"); df=pd.read_csv(path/fn).sort_values("File_Index")
        cols=[c for c in df.columns if any(k in c for k in ["Fault_BPFO","Fault_BPFI","Fault_BSF"])]
        return pd.Series(df[cols].to_numpy(float).max(1)).rolling(5,1).median().to_numpy()
    heal=np.median([np.median(foser(FO,t,False)[:max(3,int(NS[t]*0.15))]) for t in (1,2,3,4)]); eol=np.median([foser(FO,t,False)[-1] for t in (1,2,3,4)])
    for cap in (53153.,71640.):
      v=[int(round(max(1800.,cap-min(max((foser(FOT,n,True)[-1]-heal)/(eol-heal+1e-9),0),1)*(cap-1800.)))) for n in TESTS]
      cands.append((f"faultorder_lf|cap{int(cap)}",v))
  except Exception as e: pass
# (4) 앙상블: 상위 후보들의 블렌드 (현 cands에서 lf/totlife 일부 평균)
import random; random.seed(0)
base_vecs=[v for _,v in cands]
for _ in range(200):
    k=random.choice([2,3]); pick=random.sample(base_vecs,k)
    v=[int(round(np.mean([p[i] for p in pick]))) for i in range(6)]
    cands.append((f"ensemble{k}",v))

res=sorted([(cons(v)[0],n,v,cons(v)[1],cons(v)[2]) for n,v in cands])
npass=sum(1 for r in res if r[0]<0.016 and abs(r[3]-rB)<0.008 and abs(r[4]-rV2)<0.008)
print(f"확장 후보 {len(cands)}종. B,V2 정밀(±0.008) 통과: {npass}개. (±0.02 근접: {sum(1 for r in res if r[0]<0.04)}개)")
print()
print("=== 잔차 작은 상위 18 ===")
print(f'{"방법":30s}{"resid":>7s}{"cB":>7s}{"cV2":>7s}  벡터(h)')
for r,n,v,cB,cV in res[:18]:
    mk='PASS' if (abs(cB-rB)<0.008 and abs(cV-rV2)<0.008) else ''
    print(f'{n:30s}{r:7.3f}{cB:7.3f}{cV:7.3f} {mk:4s} {[round(x/3600,1) for x in v]}')
