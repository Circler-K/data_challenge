"""기대점수 추정기를 LOO로 보정: 여러 prior를 시험 -> 이전 5점수를 가장 잘 예측하는 prior 선택.
그 prior로 EV-최적 벡터 재계산.
"""
import sys; sys.path.insert(0,'.')
import numpy as np
ALL=[('B',np.array([59351,51138,19936,12168,1800,1800.]),.48972),
     ('V2',np.array([53153,51138,13079,35633,14732,14313.]),.46463),
     ('V2_pre',np.array([53153,15118,13079,35633,14732,14313.]),.41),
     ('candA',np.array([59351,72200,34515,26437,12636,6615.]),.424539),
     ('candB',np.array([48187,71382,35989,19158,44242,46357.]),.54923)]
def vsc(T,S):Er=(T-S)/T*100;return np.where(Er>0,0.5**(Er/50),0.5**(-Er/30)).mean(1)

def sample(prior,n):
    rng=np.random.default_rng(0)
    if prior[0]=='uni':
        return rng.uniform(prior[1],prior[2],(n,6))
    if prior[0]=='loguni':
        return np.exp(rng.uniform(np.log(prior[1]),np.log(prior[2]),(n,6)))
    if prior[0]=='tri':  # 삼각: 최빈 prior[3]
        return rng.triangular(prior[1],prior[3],prior[2],(n,6))

def post_from(anchors,prior,tol,target=6000):
    parts=[];tot=0;tries=0
    while tot<target and tries<30:
        tries+=1;P=sample(prior,2_000_000);m=np.ones(len(P),bool)
        for S,r in anchors:m&=np.abs(vsc(P,S)-r)<tol
        if m.any():parts.append(P[m]);tot+=int(m.sum())
    return np.vstack(parts) if parts else None

PRIORS=[('uni',1800,95000),('uni',1800,73000),('uni',1800,55000),('uni',5000,45000),
        ('loguni',1800,95000),('tri',1800,95000,15000),('tri',1800,60000,12000)]
print('=== prior별 LOO 오차 (작을수록 이전점수 잘 예측) ===')
best=None
for prior in PRIORS:
    errs=[]
    for i in range(5):
        others=[(S,r) for j,(_,S,r) in enumerate(ALL) if j!=i]
        post=post_from(others,prior,0.03,4000)
        if post is None or len(post)<200: errs=[9]; break
        pred=vsc(post,ALL[i][1]).mean(); errs.append(abs(pred-ALL[i][2]))
    mae=float(np.mean(errs))
    print(f'  {str(prior):28s} LOO평균오차 {mae:.4f}')
    if best is None or mae<best[0]: best=(mae,prior)
print(f'\\n최적 prior: {best[1]} (오차 {best[0]:.4f})')
# 최적 prior로 EV-최적 재계산 (5앵커 전체)
post=post_from([(S,r) for _,S,r in ALL],best[1],0.03,10000)
def g1(t,p):Er=(t-p)/t*100;return np.where(Er>0,0.5**(Er/50),0.5**(-Er/30))
opt=[]
for i in range(6):
    cand=np.linspace(1800,90000,1000);opt.append(int(round(cand[np.argmax([g1(post[:,i],c).mean() for c in cand])])))
print(f'보정된 EV-최적: {opt} = {[round(x/3600,1) for x in opt]}h')
print(f'  예상점수(보정 prior 기준) {vsc(post,np.array(opt,float)).mean():.3f}  (집합 {len(post)})')
