"""스테이지/궤적 모델 전수 스윕: HI피처 x 채널집계 x cummax x 기준수명 x lifefrac퍼센타일 x CAP
각 설정의 (1) 6검증 예측, (2) 타깃과의 거리, (3) cut40/50/60 LOO A_RUL, (4) Val2 예측 출력.
목표: 타깃 [59351,72119,19936,12168,1800,1800] 형태를 근거있게 내는 설정 탐색.
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
from src.scoring import a_rul_score
import wiener_rul as W

FP=600; NS={1:126,2:114,3:89,4:137}
TESTS=[f"Test{i}" for i in range(1,7)]
TARGET=[59351,72119,19936,12168,1800,1800]
EST=W.load("est")
import predict_robust as R
_td={}
def test_df(n):
    if n not in _td: _td[n]=pd.read_csv(R.ESTT/f"{n}.csv").sort_values("File_Index")
    return _td[n]

FEATSETS={
 "all6":["OT_RMS","OT_Kurtosis","OT_CrestFactor","Order_BandEnergy","Spectral_Entropy","Env_BandEnergy"],
 "no_order":["OT_RMS","OT_Kurtosis","OT_CrestFactor","Spectral_Entropy","Env_BandEnergy"],
 "energy":["OT_RMS","Order_BandEnergy","Env_BandEnergy"],
 "rms_env_kurt":["OT_RMS","Env_BandEnergy","OT_Kurtosis"],
 "transfer":["Order_BandEnergy","Spectral_Entropy"],
}
def cols_of(df,feats): return [c for c in df.columns if any(c.endswith(f) for f in feats)]
def hi_curve(df,mu,sigma,feats,agg,cummax):
    X=df[cols_of(df,feats)].to_numpy(float)
    Z=(X-mu)/sigma
    if agg=="max": z=Z.max(1)
    elif agg=="mean": z=Z.mean(1)
    elif agg=="rear": z=Z[:,[c for c in range(Z.shape[1])]][:, -max(1,Z.shape[1]//2):].mean(1)
    s=pd.Series(z).rolling(5,min_periods=1).median()
    return (s.cummax() if cummax else s).to_numpy()
def baseline(train,feats,agg):
    P=np.vstack([df[cols_of(df,feats)].to_numpy(float)[:max(3,int(len(df)*0.15))] for df in train.values()])
    return P.mean(0),P.std(0)+1e-9
def stage_rul(q,curves,Tlife,pct,floor,cap):
    lfs=[]
    for h,lf in curves:
        idx=np.where(h>=q)[0]; lfs.append(lf[idx[0]] if len(idx) else 1.05)
    rul=(1-min(np.percentile(lfs,pct),1.0))*Tlife
    return float(min(cap,max(rul,floor)))

def lifestat(train,stat):
    L=[ (NS[t]-1)*FP+60 for t in train ] if isinstance(train,dict) else [ (NS[t]-1)*FP+60 for t in train]
    return float(np.median(L)) if stat=="median" else float(np.max(L)) if stat=="max" else float(np.percentile(L,75))

def run(feats,agg,cummax,stat,pct,cap,floor=1800.0):
    # predict test
    mu,sg=baseline(EST,feats,agg)
    curves=[(hi_curve(df,mu,sg,feats,agg,cummax), df["t_sec"].to_numpy()/(df["t_sec"].to_numpy()[-1]+60)) for df in EST.values()]
    Tlife=lifestat(EST,stat)
    pred=[int(round(stage_rul(float(hi_curve(test_df(n),mu,sg,feats,agg,cummax)[-1]),curves,Tlife,pct,floor,cap))) for n in TESTS]
    # LOO
    sc=[]
    for d in (40,50,60):
        for t in (1,2,3,4):
            others=[x for x in (1,2,3,4) if x!=t]
            if d>=NS[t]: continue
            tr={o:EST[o] for o in others}
            mu2,sg2=baseline(tr,feats,agg)
            cv=[(hi_curve(tr[o],mu2,sg2,feats,agg,cummax), tr[o]["t_sec"].to_numpy()/(tr[o]["t_sec"].to_numpy()[-1]+60)) for o in others]
            Tl=float(np.median([(NS[o]-1)*FP+60 for o in others]) if stat=="median" else np.max([(NS[o]-1)*FP+60 for o in others]) if stat=="max" else np.percentile([(NS[o]-1)*FP+60 for o in others],75))
            hh=hi_curve(EST[t],mu2,sg2,feats,agg,cummax)[d-1]
            p=stage_rul(float(hh),cv,Tl,pct,floor,cap)
            sc.append(float(a_rul_score((NS[t]-d)*FP+60,p)))
    loo=float(np.mean(sc))
    dist=float(np.mean([abs(pred[i]-TARGET[i])/3600 for i in range(6)]))  # mean abs err in hours
    return pred,loo,dist

rows=[]
caps=[71640.0, 81648.0, 53153.0]
for feats in FEATSETS:
    for agg in ("max","mean"):
        for cummax in (True,False):
            for stat in ("median","max"):
                for pct in (40,50,60):
                    for cap in caps:
                        try:
                            pred,loo,dist=run(FEATSETS[feats],agg,cummax,stat,pct,cap)
                        except Exception as e:
                            continue
                        rows.append((f"{feats}|{agg}|cmax={int(cummax)}|life={stat}|pct{pct}|cap={int(cap)}",pred,loo,dist))
rows.sort(key=lambda r:r[3])  # closest to target first
print("TARGET:",TARGET)
print("="*120)
print(f"{'config':52s}{'타깃거리(h)':>10s}{'LOO':>7s}   예측  [Val2]")
print("="*120)
for name,pred,loo,dist in rows[:20]:
    print(f"{name:52s}{dist:10.2f}{loo:7.3f}   {pred}")
print()
print(">>> LOO 상위(타당성) 중 타깃근접:")
for name,pred,loo,dist in sorted(rows,key=lambda r:-r[2])[:8]:
    print(f"  {name:52s} LOO={loo:.3f} dist={dist:.2f}h  {pred}")
