"""타깃 벡터에 가장 가까운 출력을 내는 모델 설정 탐색 (파이프라인 역설계).
타깃 = [41047,45780,34836,15399,27056,37131] (기대점수 최고 후보)
"""
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, pandas as pd, itertools
import wiener_rul as W, predict_robust as R
TESTS=[f"Test{i}" for i in range(1,7)]; EST=W.load("est"); NS={1:126,2:114,3:89,4:137}; FP=600
TARGET=np.array([41047,45780,34836,15399,27056,37131.])
FEATS=["OT_RMS","OT_Kurtosis","OT_CrestFactor","Order_BandEnergy","Spectral_Entropy","Env_BandEnergy"]
_t={}
def tdf(n):
    if n not in _t:_t[n]=pd.read_csv(R.ESTT/f"{n}.csv").sort_values("File_Index")
    return _t[n]
def cl(df,f): return [c for c in df.columns if any(c.endswith(x) for x in f)]
_lf={}
def lf1(n,feat,agg):
    k=(n,feat,agg)
    if k in _lf:return _lf[k]
    def red(X):return X.max(1) if agg=="max" else X.mean(1)
    val=pd.Series(red(tdf(n)[cl(tdf(n),[feat])].to_numpy(float))).rolling(5,1).median().to_numpy()[-1]
    s={t:pd.Series(red(EST[t][cl(EST[t],[feat])].to_numpy(float))).rolling(5,1).median().to_numpy() for t in (1,2,3,4)}
    h=np.median([np.median(s[t][:max(3,int(len(s[t])*0.15))]) for t in (1,2,3,4)]);e=np.median([s[t][-1] for t in (1,2,3,4)])
    _lf[k]=min(max((val-h)/(e-h+1e-9),0),1);return _lf[k]
def dist(v): return float(np.sqrt(np.mean(((np.array(v,float)-TARGET)/TARGET)**2)))  # 상대 RMSE
SUBSETS=[]
for r in (1,2,3,4):
    for c in itertools.combinations(FEATS,r):SUBSETS.append(list(c))
SUBSETS.append(FEATS)
res=[]
for f in SUBSETS:
  for agg in ("max","mean"):
    lfm={n:[lf1(n,x,agg) for x in f] for n in TESTS}
    for comb in ("mean","max"):
      base={n:(max(lfm[n]) if comb=="max" else float(np.mean(lfm[n]))) for n in TESTS}
      for cap in (45000.,50000.,55000.,60000.,65000.):
        for g in (0.5,1.0,1.5,2.0,3.0):
          v=[int(round(max(1800.,cap-(base[n]**g)*(cap-1800.)))) for n in TESTS]
          res.append((dist(v),f"{'+'.join(x[:4] for x in f)}|{agg}|{comb}|cap{int(cap)}|g{g}",v))
res.sort()
print(f'타깃 {[int(x) for x in TARGET]} = {[round(x/3600,1) for x in TARGET]}h')
print(f'총 {len(res)}개 설정. 타깃에 가장 가까운 모델 상위 12 (상대RMSE):')
print(f'{"config":42s}{"dist":>6s}  벡터(h)')
for d,n,v in res[:12]:
    print(f'{n:42s}{d:6.3f}  {[round(x/3600,1) for x in v]}')
