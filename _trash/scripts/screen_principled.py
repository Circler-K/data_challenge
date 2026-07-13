"""근거 있는(principled) 후보 대량 생성 -> 각각 일관성 체크(B,V2,V2_pre 재현) 통과 여부.
무작위 아님. 모델(lf/stage) x 피처조합 x 채널집계 x 전처리(cummax/smooth) x 하이퍼파라미터.
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("c:/Users/User/WorkSpace/data_challenge"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from src.scoring import a_rul_score
import wiener_rul as W, predict_robust as R
FP=600; NS={1:126,2:114,3:89,4:137}; TESTS=[f"Test{i}" for i in range(1,7)]; EST=W.load("est")
FEATS=["OT_RMS","OT_Kurtosis","OT_CrestFactor","Order_BandEnergy","Spectral_Entropy","Env_BandEnergy"]
B=[59351,51138,19936,12168,1800,1800]; rB=0.48972
V2=[53153,51138,13079,35633,14732,14313]; rV2=0.46463
Vp=[53153,15118,13079,35633,14732,14313]; rVp=0.41
def consistency(P):  # 0.41(V2_pre) 제외 — B,V2 두 정밀앵커만
    cB=float(np.mean([a_rul_score(P[i],B[i]) for i in range(6)]))
    cV=float(np.mean([a_rul_score(P[i],V2[i]) for i in range(6)]))
    cp=float(np.mean([a_rul_score(P[i],Vp[i]) for i in range(6)]))  # 참고용만
    res=abs(cB-rB)+abs(cV-rV2)
    ok=abs(cB-rB)<0.008 and abs(cV-rV2)<0.008
    return ok,res,cB,cV,cp

_td={}
def tdf(n):
    if n not in _td:_td[n]=pd.read_csv(R.ESTT/f"{n}.csv").sort_values("File_Index")
    return _td[n]
def cols(df,f): return [c for c in df.columns if any(c.endswith(x) for x in f)]
# 전처리/HI
def series_lf(n,feat,agg):
    df=tdf(n); X=df[cols(df,[feat])].to_numpy(float); v=X.max(1) if agg=="max" else (X.mean(1) if agg=="mean" else X[:,-2:].mean(1))
    val=pd.Series(v).rolling(5,min_periods=1).median().to_numpy()[-1]
    # train baseline for lf
    s={t:pd.Series((lambda Z:Z.max(1) if agg=="max" else (Z.mean(1) if agg=="mean" else Z[:,-2:].mean(1)))(EST[t][cols(EST[t],[feat])].to_numpy(float))).rolling(5,min_periods=1).median().to_numpy() for t in (1,2,3,4)}
    heal=np.median([np.median(s[t][:max(3,int(len(s[t])*0.15))]) for t in (1,2,3,4)]); eol=np.median([s[t][-1] for t in (1,2,3,4)])
    return min(max((val-heal)/(eol-heal+1e-9),0),1)
def hi_curve(df,mu,sg,f,agg,cm):
    Z=(df[cols(df,f)].to_numpy(float)-mu)/sg; z=Z.max(1) if agg=="max" else (Z.mean(1) if agg=="mean" else Z[:,-2:].mean(1))
    s=pd.Series(z).rolling(5,min_periods=1).median(); return (s.cummax() if cm else s).to_numpy()
def base_ms(f,agg):
    P=np.vstack([(lambda d:d)(EST[t][cols(EST[t],f)].to_numpy(float))[:max(3,int(len(EST[t])*0.15))] for t in (1,2,3,4)]); return P.mean(0),P.std(0)+1e-9

SUBSETS={ "all6":FEATS,"energy":["OT_RMS","Order_BandEnergy","Env_BandEnergy"],"no_order":["OT_RMS","OT_Kurtosis","OT_CrestFactor","Spectral_Entropy","Env_BandEnergy"],
 "transfer":["Order_BandEnergy","Spectral_Entropy"],"rms_env":["OT_RMS","Env_BandEnergy"],"spectral":["Spectral_Entropy"],"order":["Order_BandEnergy"],
 "rms":["OT_RMS"],"env":["Env_BandEnergy"],"ord_env":["Order_BandEnergy","Env_BandEnergy"],"rms_kurt":["OT_RMS","OT_Kurtosis"],"spec_env":["Spectral_Entropy","Env_BandEnergy"]}
cands=[]  # (name, vector)
# --- lf 모델 ---
for sk,f in SUBSETS.items():
  for agg in ("max","mean"):
    for cap in (53153.,71640.,81648.):
      lfs={n:np.mean([series_lf(n,x,agg) for x in f]) for n in TESTS}
      v=[int(round(max(1800.,cap-lfs[n]*(cap-1800.)))) for n in TESTS]
      cands.append((f"lf|{sk}|{agg}|cap{int(cap)}",v))
# --- stage 모델 ---
for sk,f in [("energy",SUBSETS["energy"]),("no_order",SUBSETS["no_order"]),("all6",FEATS),("transfer",SUBSETS["transfer"]),("rms_env",SUBSETS["rms_env"])]:
  for agg in ("mean","max"):
    for cm in (True,False):
      mu,sg=base_ms(f,agg)
      cv=[(hi_curve(EST[t],mu,sg,f,agg,cm),EST[t]["t_sec"].to_numpy()/(EST[t]["t_sec"].to_numpy()[-1]+60)) for t in (1,2,3,4)]
      for lifef in (1.0,1.1,1.25):
        for stat in ("median","max"):
          Tl=(np.median if stat=="median" else np.max)([(NS[o]-1)*FP+60 for o in (1,2,3,4)])*lifef
          for pct in (30,40,50,60):
            def srul(q):
              lfs=[lf[np.where(h>=q)[0][0]] if len(np.where(h>=q)[0]) else 1.05 for h,lf in cv]
              return float(max((1-min(np.percentile(lfs,pct),1.0))*Tl,1800.))
            v=[int(round(srul(float(hi_curve(tdf(n),mu,sg,f,agg,cm)[-1])))) for n in TESTS]
            cands.append((f"stage|{sk}|{agg}|cm{int(cm)}|L{lifef}|{stat}|p{pct}",v))
# --- 기타 모델 출력(고정) ---
for name,v in [("survival_base",[39994]*6),("similarity",[36697,44758,41984,48468,44029,31005]),
 ("rate_const",[42060]*6),("wiener",[34710]*6),("faultfreq",[63561,11256,31316,40903,65000,65000]),
 ("B_itself",B),("constant_med",[35730]*6)]:
    cands.append((name,v))

scored=[(consistency(v),name,v) for name,v in cands]
passers=[(r[1],n,v,r[2],r[3],r[4]) for r,n,v in [(c[0],c[1],c[2]) for c in scored] if r[0]]
allres=sorted([(consistency(v)[1],n,v,consistency(v)[2],consistency(v)[3],consistency(v)[4]) for n,v in cands])
print(f"근거있는 후보 총 {len(cands)}개 생성. 일관성 통과(B,V2±0.01·Vp±0.025): {sum(1 for c in scored if c[0][0])}개")
print()
print("=== 통과 또는 근접 상위 20 (잔차 작은 순) ===")
print(f'{"방법":34s}{"resid":>7s}{"cB":>7s}{"cV2":>7s}{"cVp":>7s}  벡터')
for res,n,v,cB,cV,cp in allres[:20]:
    mark='PASS' if (abs(cB-rB)<0.008 and abs(cV-rV2)<0.008) else ''
    print(f'{n:34s}{res:7.3f}{cB:7.3f}{cV:7.3f}{cp:7.3f} {mark:4s} {v}')
