#!/usr/bin/env python3
"""
Outlines around stacks, and the arrow of a ribbon.  (ROADMAP Stage 9s.)

Port of `trace_tiles` / `trace_inner_edges` of spectre/spectre_tiles_blender.py (B. Hartshorn) to numpy:

  environment   a stack (tile_stacks.py) with the tiles that touch it. Two stacks are in the same class when the
                surrounding tiles have the same poses in the stack's frame.
  smoothing     the tiles are welded into one mesh; three times, every vertex moves to the mean of its neighbours
                (Blender's SMOOTH modifier with factor 1, 3 iterations). Reported per class: area and outline of the
                stack after smoothing, the outer outline, and the shift of the stack's centroid (an arrow).
  stack vector  a stack runs from a tile to its translate by w. w is fixed in the tile's frame, so it is a directed
                vector in the plane: every stack has an arrow before any smoothing.

Ribbons (dumps from generic_phase.py DUMP=...): the stacks of a ribbon are ordered along it; reported are the turn of w
from one stack to the next, and whether the next stack lies ahead of or behind w (the same for all stacks = the ribbon
has an arrow).

Usage: python3 stack_trace.py dump.json [...] [--figure] [--tests]
"""
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tile_stacks as TS  # noqa: E402
REF, W = TS.REF, TS.W


def prepare(path):
    """Poses, centroids, neighbours and ribbon clusters of a dump (cached beside it; the first call takes minutes)."""
    cache = path + ".prep.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import ribbon_hierarchy as RH
    import hole_uniqueness as HU
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    dump = json.load(open(path))[0]
    S = HU.ccw(REF * np.array([-1.0, 1.0]))
    tiles, owner = [], []
    for a, pl in enumerate(dump["tiles"]):
        for Q in RH.split(pl, S):
            tiles.append(RH.pose_of(Q)); owner.append(a)
    P = np.array([T[:, 2] for T in tiles]); L = np.array([T[:, :2] for T in tiles])
    polys = [Polygon(REF @ T[:, :2].T + T[:, 2]) for T in tiles]
    cent = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    pr = STRtree(polys).query(polys, predicate="dwithin", distance=1e-6)
    nb = defaultdict(list)
    for a, b in zip(*pr):
        if a != b:
            nb[int(a)].append(int(b))
    sp = defaultdict(list)
    for u, a in enumerate(owner):
        sp[a].append(u)
    ribs, seen = [], set()
    for st in dump["steps"]:
        rem = sorted({u for a in st["removed"] for u in sp[a]}); rs = set(rem); comp = {}
        for u in rem:
            if u in comp:
                continue
            stack = [u]; comp[u] = u
            while stack:
                x = stack.pop()
                for y in nb[x]:
                    if y in rs and y not in comp:
                        comp[y] = u; stack.append(y)
        groups = defaultdict(list)
        for u, c in comp.items():
            groups[c].append(u)
        for g in groups.values():
            g = sorted(g)
            if len(g) >= 20 and (len(g), tuple(g[:8])) not in seen:
                seen.add((len(g), tuple(g[:8]))); ribs.append(g)
    out = dict(P=P, L=L, cent=cent, nb=dict(nb), ribs=ribs, R=dump["R"])
    pickle.dump(out, open(cache, "wb"))
    return out


def smooth_measure(Ps, Ls, n_centre, iters=3):
    """Weld the tiles (the first n_centre are the stack), smooth, and measure."""
    vid, V, faces = {}, [], []
    for P, L in zip(Ps, Ls):
        f = []
        for q in REF @ L.T + P:
            k = (round(q[0], 3), round(q[1], 3))
            if k not in vid:
                vid[k] = len(V); V.append(q)
            f.append(vid[k])
        faces.append(f)
    V = np.array(V)
    eface = defaultdict(list)
    for fi, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            eface[(min(a, b), max(a, b))].append(fi)
    nbr = defaultdict(set)
    for a, b in eface:
        nbr[a].add(b); nbr[b].add(a)
    X = V.copy()
    for _ in range(iters):
        X = np.array([X[list(nbr[i])].mean(0) for i in range(len(X))])

    def stats(f, Y):
        Q = Y[f]; x, y = Q[:, 0], Q[:, 1]
        cr = x * np.roll(y, -1) - np.roll(x, -1) * y
        A = cr.sum() / 2
        return abs(A), np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * A)

    def centre(Y):
        st = [stats(faces[i], Y) for i in range(n_centre)]
        A = sum(a for a, _ in st)
        return A, sum(a * c for a, c in st) / A

    A0, c0 = centre(V); A1, c1 = centre(X)
    inner = [e for e, fs in eface.items() if sum(f < n_centre for f in fs) == 1]
    outer = [e for e, fs in eface.items() if len(fs) == 1]
    ln = lambda es, Y: float(sum(np.linalg.norm(Y[a] - Y[b]) for a, b in es))
    return dict(area_before=A0, area_after=A1, shift=c1 - c0, outline_before=ln(inner, V), outline_after=ln(inner, X),
                outer_before=ln(outer, V), outer_after=ln(outer, X))


def env_key(d, s):
    P, L, nb = d["P"], d["L"], d["nb"]
    cor = sorted({v for u in s for v in nb.get(u, [])} - set(s))
    Lt = L[s[0]].T
    key = (len(s), tuple(sorted(tuple(np.round(Lt @ (P[v] - P[s[0]]), 2)) + tuple(np.round((Lt @ L[v]).ravel(), 2)) for v in cor)))
    return key, cor


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    data = [prepare(f) for f in files]
    # ---- environments of all stacks ----
    classes = defaultdict(list)
    for di, d in enumerate(data):
        for s, axis in TS.stacks(d["P"], d["L"]):
            if np.linalg.norm(d["P"][s[0]]) > d["R"] - 20:
                continue
            key, cor = env_key(d, s)
            classes[key].append((di, s, cor))
    order = sorted(classes, key=lambda k: -len(classes[k]))
    cid = {k: i for i, k in enumerate(order)}
    env = []
    for k in order:
        di, s, cor = classes[k][0]; d = data[di]
        idx = list(s) + cor
        m = smooth_measure(d["P"][idx], d["L"][idx], len(s))
        sh = d["L"][s[0]].T @ m["shift"]
        rot = Counter(int(round(math.degrees(math.atan2(*(d["L"][s[0]].T @ d["L"][v])[::-1, 0])))) % 360 for v in cor)
        env.append(dict(cls=cid[k], stack_tiles=len(s), surrounding_tiles=len(cor), occurrences=len(classes[k]), area_after=m["area_after"],
                        outline_after=m["outline_after"], outer_before=m["outer_before"], outer_after=m["outer_after"],
                        shift=[float(sh[0]), float(sh[1])], shift_angle_from_w=float((math.degrees(math.atan2(sh[1], sh[0]) - math.atan2(W[1], W[0])) + 180) % 360 - 180),
                        relative_rotations={str(a): c for a, c in sorted(rot.items())}))
        e = env[-1]
        print("class %d: %d tiles, %2d around, %4d times | after smoothing: area %.4f outline %.4f outer %.4f (before %.0f) | shift %.4f at %+.0f deg from w" % (
            e["cls"], e["stack_tiles"], e["surrounding_tiles"], e["occurrences"], e["area_after"], e["outline_after"], e["outer_after"], e["outer_before"],
            math.hypot(*e["shift"]), e["shift_angle_from_w"]))
    # ---- ribbons ----
    ribs, turns_all, turns_simple, keepfig = [], Counter(), Counter(), None
    for di, d in enumerate(data):
        P, L, nb, cent = d["P"], d["L"], d["nb"], d["cent"]
        for g in d["ribs"]:
            gs = set(g)
            st = TS.stacks(P, L, g)
            if len(st) < 5:
                ribs.append(dict(file=os.path.basename(files[di]), tiles=len(g), stacks=len(st))); continue
            c = cent[g]; X = c - c.mean(0)
            w_, v_ = np.linalg.eigh(X.T @ X)
            start = g[int(np.argmin(X @ v_[:, 1]))]
            dist = {start: 0}; q = [start]
            while q:
                x = q.pop(0)
                for y in nb.get(x, []):
                    if y in gs and y not in dist:
                        dist[y] = dist[x] + 1; q.append(y)
            seq = sorted((dist[s[0]], k) for k, (s, _) in enumerate(st) if s[0] in dist)
            ws, cs, cl = {}, {}, Counter()
            for _, k in seq:
                s = st[k][0]
                ww = L[s[0]] @ W
                ws[k] = round(math.degrees(math.atan2(ww[1], ww[0])) % 360); cs[k] = cent[s].mean(0)
                kk, _ = env_key(d, s)
                cl[cid.get(kk, -1)] += 1
            turns, ahead = Counter(), []
            for (d1, k1), (d2, k2) in zip(seq, seq[1:]):
                if d2 - d1 > 8:
                    continue                                   # a jump to another branch of the cluster
                t = (ws[k2] - ws[k1] + 180) % 360 - 180
                turns[int(t)] += 1
                a1, a2 = math.radians(ws[k1]), math.radians(ws[k2])
                ahead.append(float((cs[k2] - cs[k1]) @ np.array([math.cos(a1) + math.cos(a2), math.sin(a1) + math.sin(a2)])))
            ahead = np.array(ahead)
            cons = float(max((ahead > 0).mean(), (ahead < 0).mean())) if len(ahead) else None
            turns_all.update(turns)
            mw = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in ws.values()]).mean(0)
            ribs.append(dict(file=os.path.basename(files[di]), tiles=len(g), stacks=len(st), stack_share=sum(len(s) for s, _ in st) / len(g),
                             consistency=cons, steps=len(ahead), turns={str(k): v for k, v in sorted(turns.items())},
                             w_directions={str(k): v for k, v in sorted(Counter(ws.values()).items())},
                             mean_w_angle=float(math.degrees(math.atan2(mw[1], mw[0])) % 360), mean_w_length=float(np.linalg.norm(mw)),
                             ribbon_angle=float(math.degrees(math.atan2(v_[1, 1], v_[0, 1])) % 180), classes={str(k): v for k, v in sorted(cl.items())}))
            r = ribs[-1]
            print("ribbon %3d tiles, %2d stacks (share %.2f): same sense in %.2f of %d steps; turns %s; w directions %s; mean w at %.0f (length %.2f), ribbon axis %.0f" % (
                r["tiles"], r["stacks"], r["stack_share"], r["consistency"] or 0, r["steps"], r["turns"], r["w_directions"], r["mean_w_angle"], r["mean_w_length"], r["ribbon_angle"]))
            if r["consistency"] and r["consistency"] >= 0.95 and r["stack_share"] > 0.6:
                turns_simple.update(turns)
                if keepfig is None or len(g) > keepfig[0]:
                    keepfig = (len(g), d, g, st)
    pure = [r for r in ribs if r.get("stack_share", 0) > 0.45 and r.get("consistency") is not None]
    dev = []
    for r in pure:
        x = abs(r["mean_w_angle"] % 180 - r["ribbon_angle"]); dev.append(min(x, 180 - x))
    summ = dict(stack_classes=len(order), ribbons=len(ribs), with_stacks=sum(r["stacks"] >= 5 for r in ribs), clean=len(pure),
                clean_consistency=[r["consistency"] for r in pure], turns_simple={str(k): v for k, v in sorted(turns_simple.items())}, turns_all={str(k): v for k, v in sorted(turns_all.items())},
                clean_axis_minus_mean_w=dev)
    print(summ)
    json.dump(dict(environments=env, ribbons=ribs, summary=summ), open(os.path.join(HERE, "stack_trace.json"), "w"), indent=1)
    if "--figure" in sys.argv and keepfig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titlelocation": "left", "savefig.dpi": 300})
        _, d, g, st = keepfig
        P, L, cent = d["P"], d["L"], d["cent"]
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1.9, 1]))
        ax = axs[0]
        c = cent[g].mean(0); X = cent[g] - c
        w_, v_ = np.linalg.eigh(X.T @ X); axv = v_[:, 1]; R = np.array([axv, [-axv[1], axv[0]]])
        tr = lambda Q: (Q - c) @ R.T
        gs = set(g)
        hy = np.abs(X @ R[1]).max() + 5
        near = [u for u in range(len(P)) if abs((cent[u] - c) @ R[1]) < hy and abs((cent[u] - c) @ R[0]) < np.abs(X @ axv).max() + 4]
        ax.add_collection(PolyCollection([tr(REF @ L[u].T + P[u]) for u in near], facecolors=["#bdbdbd" if u in gs else "#f2f2f2" for u in near], edgecolors="white", linewidths=0.2))
        cols = {45: "#1f77b4", 105: "#d95f02", 165: "#1b9e77", 225: "#1f77b4", 285: "#d95f02", 345: "#1b9e77"}
        for s, _ in st:
            ww = L[s[0]] @ W; a = round(math.degrees(math.atan2(ww[1], ww[0])) % 360)
            p0 = tr(cent[s[0]][None])[0]; p1 = tr(cent[s[-1]][None])[0]
            ax.annotate("", xy=p1 + 0.5 * (p1 - p0), xytext=p0 - 0.5 * (p1 - p0), arrowprops=dict(arrowstyle="-|>", color=cols.get(a, "k"), lw=1.3, mutation_scale=9))
        ax.set_xlim(-np.abs(X @ axv).max() - 3, np.abs(X @ axv).max() + 3); ax.set_ylim(-hy, hy); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title("a  One ribbon (grey) and the vector of each of its stacks")
        ax = axs[1]
        ks = [-120, -60, 0, 60, 120, 180]
        ax.bar(range(len(ks)), [turns_simple.get(k, 0) + (turns_simple.get(-180, 0) if k == 180 else 0) for k in ks], color="#4d4d4d", width=0.7)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(["%d°" % k for k in ks])
        ax.set_xlabel("turn of the stack vector to the next stack"); ax.set_ylabel("steps (simple ribbons)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("b  Steps turn by 60°")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "stack_trace.png")); plt.close(fig)
    if "--tests" in sys.argv:
        assert summ["stack_classes"] == 9
        print("stack_trace: tests passed (nine stack environments)")


if __name__ == "__main__":
    main()
