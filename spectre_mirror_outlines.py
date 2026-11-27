#!/usr/bin/env python3
"""
Does reflection act on the Spectre's states? (open problem 2 of quantgravbits.tex, second half; ROADMAP Stage 9d)

On the hat sphere the kite grid is mirror symmetric, so a reflection maps tilings to tilings and one can ask for a
mirror-odd function of the bits (chi4). For the Spectre the first question is whether the mirror image of a patch can be
tiled by Spectres of the same hand at all. Test: every connected cluster of 1 to 5 tiles around 40 central tiles of the
level-3 patch; its outline as a cyclic word of angles; the reversed word is the mirror image; both are given to the exact
combinatorial tiler of spectre_fold_flips.py (one hand of tile only). The tiler misses tilings in which a tile splits the
region, so clusters whose own tiling it does not find are set aside, and "mirror not tiled" is a search result, not a proof.
Needs brentharts/ballgen. Usage: python3 spectre_mirror_outlines.py
"""
import sys,collections,random,json,numpy as np
sys.argv=['x']
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectre_fold_flips as SF
from spectre.patch import SpectrePatch
from spectre.search import PatchPool
from spectre.fold import PatchGraph
patch=SpectrePatch(levels=3); pool=PatchPool(patch); g0=PatchGraph(patch)
tau=[int(x) for x in g0.angle[0]]
S0=SF.Surface(g0)
cen=np.array([patch.straight_polygon(t).mean(0) for t in range(g0.n_tiles)])
seeds=[int(t) for t in np.argsort(np.linalg.norm(cen-cen.mean(0),axis=1))[:40]]
def canon(w):
    n=len(w); return min(tuple(w[k:]+w[:k]) for k in range(n))
res=collections.Counter(); ex=[]
seenw=set()
for C in S0.clusters(5,seeds):
    w=S0.word(C)
    if w is None: continue
    word=list(w[0]); k=len(C)
    cw=canon(word)
    if (k,cw) in seenw: continue
    seenw.add((k,cw))
    n0,_=SF.count_tilings(tuple(word),k,tau)
    if n0==0: res[(k,'original not found (search limit)')]+=1; continue
    mw=word[::-1]
    sym = canon(mw)==cw
    n1,_=SF.count_tilings(tuple(mw),k,tau)
    res[(k,'outline mirror-symmetric' if sym else 'asymmetric','mirror tiled' if n1 else 'mirror not tiled')]+=1
    if n1 and not sym and len(ex)<5: ex.append((k,word))
for key,v in sorted(res.items(),key=str): print(key,v)
print('examples',ex[:2])
json.dump({str(k):v for k,v in res.items()},open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectre_mirror_outlines.json'), 'w'),indent=1)
