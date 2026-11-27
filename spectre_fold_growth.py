#!/usr/bin/env python3
"""
Two further attempts at a closed Spectre substrate with many pair qubits (open problem 4 of quantgravbits.tex; Stage 9e).

1. Closing rate of random disc patches against their size, and guided growth: start from closable 10-tile patches, add
   one tile at a time, keep the patches that still glue shut (beam of 40, ranked by the number of pairs).
2. Doubles: two copies of one patch glued along their boundaries (a pillow). With boundary angles a_i this needs a
   shift s with a_i + a_(s-i) <= 360 for every i.
Counted, random samples. Needs brentharts/ballgen. Usage: python3 spectre_fold_growth.py
"""
import os
import sys,random,collections,json,time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.argv=['x']
import spectre_fold_substrate as SS
from spectre.patch import SpectrePatch
from spectre.search import PatchPool
patch=SpectrePatch(levels=3); pool=PatchPool(patch); rng=random.Random(3)
t0=time.time()
# 1 closing rate against N
rate={}
for N in (8,10,12,14,16,18,20,24):
    c=0; n=400
    for _ in range(n):
        if SS.fold(pool,pool.random_subset(N,rng,rng.choice([0.3,0.7,0.9]))): c+=1
    rate[N]=c/n; print('N',N,'closing rate',c/n,round(time.time()-t0),flush=True)
# 2 growth: from closable seeds add one tile at a time, keep closable patches (beam of 40, ranked by pairs)
seeds=[]
while len(seeds)<30:
    f=SS.fold(pool,pool.random_subset(10,rng,0.7))
    if f: seeds.append(f)
beam=seeds; hist=[]
for step in range(30):
    nxt={}
    for f in beam:
        T=set(f['tiles'])
        fr=sorted(set().union(*[pool.nbrs[t] for t in T])-T)
        rng.shuffle(fr)
        for u in fr[:25]:
            f2=SS.fold(pool,sorted(T|{u}))
            if f2: nxt[f2['tiles']]=f2
    if not nxt: print('growth stops at',len(next(iter(beam))['tiles']) if beam else None); break
    beam=sorted(nxt.values(),key=lambda f:-len(f['pairs']))[:40]
    N=len(beam[0]['tiles']); hist.append((N,len(nxt),max(len(f['pairs']) for f in beam),sorted(collections.Counter(len(f['deficits']) for f in beam).items())))
    print('grown to N=%d: %d closable patches found, max pairs %d, cone counts %s  (%ds)'%(*hist[-1],time.time()-t0),flush=True)
json.dump(dict(closing_rate=rate,growth=hist),open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectre_fold_growth.json'), 'w'),indent=1)

# ---- doubles
def shifts(A):
    m=len(A); A=np.array(A); out=[]
    for s in range(m):
        B=A[(s-np.arange(m))%m]
        if (A+B<=360).all(): out.append(s)
    return out
import numpy as np; rng=random.Random(5); t0=time.time(); rows=[]
for N in (5,10,20,40,80,150):
    ok=0;n=0;ns=[]
    for _ in range(300):
        g=pool.graph(pool.random_subset(N,rng,rng.choice([0.3,0.7,0.9])))
        if g is None: continue
        n+=1
        A=[int(a) for a in g.loops[0][1]]
        S=shifts(A)
        if S: ok+=1; ns.append(len(S))
    rows.append((N,n,ok)); print('N',N,'disc patches',n,'that close as a double',ok,'shifts',collections.Counter(ns).most_common(3),round(time.time()-t0),flush=True)
