#!/usr/bin/env python3
"""
Where an active strand stops, and what the hubs are.  (ROADMAP Stage 10b.)

The network of active ribbons is a forest of finite trees (Stage 9w), and a flip ribbon is not finite. So an active
strand must stop somewhere and the ribbon must go on in another form. This script looks at every place where a flipped
active strand stops inside a generic-phase patch (not at the edge of the disc):

  where     at a root or an end of the network (no further link exists in the tiling), or in the middle of a chain
            (the next Mystic exists but is not flipped by this step);
  what next the distance, counted in flipped tiles, from the stopping Mystic to the nearest flipped stack
            (a silent strand).

It also reports, for the hub of each patch (the Mystic removed by the most step directions), its supertile label and
its role in the network.

Usage: python3 ribbon_joins.py dump.json [...] [--tests]      (dumps from generic_phase.py DUMP=...)
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stack_trace as ST  # noqa: E402
import tile_stacks as TS  # noqa: E402
import ribbon_trees as RT  # noqa: E402

MARGIN, DEPTH = 25.0, 8


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    tab, hubs, events = Counter(), [], 0
    for f in files:
        dump = json.load(open(f))[0]
        d = ST.prepare(f)
        P, L, lab, par, nb = d["P"], d["L"], d["lab"], d["par"], d["nb"]
        sp, u = defaultdict(list), 0
        for a, pl in enumerate(dump["tiles"]):
            for Q in pl:
                for _ in range(1 if len(Q) == 14 else 2):
                    sp[a].append(u); u += 1
        G = [i for i in range(len(P)) if lab[i] == "Gamma1"]
        s_, p_ = RT.links(P[G], L[G])
        succ = {G[a]: {k: G[b] for k, b in v.items()} for a, v in s_.items()}
        pred = {G[b]: G[a] for b, a in p_.items()}
        count = np.zeros(len(P), int)
        for st in dump["steps"]:
            R = {v for a in st["removed"] for v in sp[a]}
            for v in R:
                count[v] += 1
            stacked = {t for s, _ in TS.stacks(P, L, sorted(R)) for t in s}
            act = {u for u in R if lab[u] == "Gamma1" and u not in stacked
                   and ((u in pred and pred[u] in R) or any(v in R for v in succ.get(u, {}).values()))}
            for u in act:
                if np.linalg.norm(P[u]) > d["R"] - MARGIN:
                    continue
                has_p = u in pred and pred[u] in act
                has_s = any(v in act for v in succ.get(u, {}).values())
                if has_p and has_s:
                    continue
                if not has_p:
                    where = "start, at a root of the network" if u not in pred else "start, in mid-chain"
                else:
                    where = "stop, at an end of the network" if u not in succ else "stop, in mid-chain"
                seen, fr, found = {u}, [u], None
                for dep in range(DEPTH):
                    nx = []
                    for x in fr:
                        for y in nb.get(x, []):
                            if y in R and y not in seen:
                                seen.add(y); nx.append(y)
                                if y in stacked and found is None:
                                    found = dep + 1
                    fr = nx
                tab[(where, "stack within 3" if found and found <= 3 else "stack within 4 to 8" if found else "no stack within 8")] += 1
                events += 1
        h = int(np.argmax(np.where(np.array([x == "Gamma1" for x in lab]), count, -1)))
        role = "isolated" if h not in pred and h not in succ else "root" if h not in pred else "end" if h not in succ else ("fork" if len(succ[h]) == 2 else "middle")
        hubs.append(dict(file=os.path.basename(f), removed_by=int(count[h]), of=len(dump["steps"]), supertile=par[h], network_role=role,
                         flipped_neighbours={k: v for k, v in Counter(lab[y] for y in nb.get(h, []) if count[y] > 0).items()}))
    out = dict(events=events, table={" | ".join(k): v for k, v in sorted(tab.items())},
               followed_by_stack=sum(v for k, v in tab.items() if k[1] != "no stack within 8"), hubs=hubs)
    for k, v in out["table"].items():
        print("%3d  %s" % (v, k))
    print("active strands that stop inside the patch: %d; followed by a silent stack within 8 flipped tiles: %d" % (events, out["followed_by_stack"]))
    for h in hubs:
        print("hub of %s: removed by %d of %d; supertile %s; role in the active network: %s; flipped neighbours %s" % (
            h["file"], h["removed_by"], h["of"], h["supertile"], h["network_role"], h["flipped_neighbours"]))
    json.dump(out, open(os.path.join(HERE, "ribbon_joins.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        assert events > 0 and out["followed_by_stack"] == events
        print("ribbon_joins: tests passed (every active strand that stops inside the patch is followed by a silent stack)")


if __name__ == "__main__":
    main()
