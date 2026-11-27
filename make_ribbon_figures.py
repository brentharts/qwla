#!/usr/bin/env python3
"""Figures for the ribbon results (generic_phase*_tile.json, *_rib.json). Run from the repo folder:
SPECTRE_DIR=.. WADA_DIR=../wada-coudene python3 make_ribbon_figures.py
Writes figures/ribbon_*.png."""
import glob as _glob, os, pathlib, json, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from matplotlib.collections import PatchCollection

_plugins = os.environ.get("CS_PLUGINS") or (os.environ["CS_WORKDIR"].rpartition("/claude-science-work/")[0] if os.environ.get("CS_WORKDIR") else os.environ.get("HOME", "")) + "/.claude/plugins/synced/*"
try:
    _helper_files = [f for g in ("", "~g*") for f in _glob.glob(_plugins + "/claude-science" + g + "/skills/sci-core-figure-style/kernel.py") if os.path.isfile(f) and os.access(f, os.R_OK) and (_plugins == os.environ.get("CS_PLUGINS") or os.name == "posix" and all(not s.st_mode & (0o22 if s.st_gid else 2) and s.st_uid in (0, os.getuid()) for s in map(os.stat, [f, *list(pathlib.Path(f).parents)[:8]])))]
    exec(pathlib.Path(max(_helper_files, key=os.path.getctime) if len({f.rpartition("/claude-science")[0] for f in _helper_files}) == 1 else "/helpers-not-found").read_text(encoding="utf-8"))
except OSError:
    print("figure helpers not loaded")
    apply_figure_style = lambda **k: None
    set_frame = panel_letter = goodness_arrow = lambda *a, **k: None
    panel_crops = lambda *a, **k: {}
    render_check = lambda *a, **k: dict.fromkeys(("overlaps", "outside", "cut", "small", "on_data"), ())

HERE = os.path.dirname(os.path.abspath(__file__))
WADA = os.environ.get("WADA_DIR", os.path.join(HERE, "..", "wada-coudene"))
sys.path.insert(0, WADA)
from lakes_of_wada import spectre_window as SWR

PAIR, NOPAIR, GREY = "#E69F00", "#0072B2", "#999999"
SILENT_C = "#56B4E9"
apply_figure_style()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)


def load(delta, tag="_tile", R=60):
    d = json.load(open(os.path.join(HERE, "generic_phase_d%s_R%d%s.json" % (delta, R, tag))))
    u = {}
    for x in d["rows"]:
        if x.get("pv") and x["pv"]["len_in"] > 10:
            u.setdefault((delta, x["phase"], x["sig"]), x)
    return list(u.values())


def nev(x):
    return len(x["pv"]["dest"]) + len(x["pv"]["crea"])


RIB = load("0.03") + load("0.06")
SIL = [x for x in RIB if nev(x) == 0]
ACT = [x for x in RIB if nev(x) > 0]


# ---------------------------------------------------------------- figure 1: two ribbons in the tiling
def straighten(coords, mean, ax):
    p = np.array(coords) - np.array(mean)
    a = np.array(ax)
    return np.c_[p @ a, p @ np.array([-a[1], a[0]])]


def draw_ribbon(axs, x, title):
    g = x["pv"]["geo"]
    m, a = g["mean"], g["axis"]
    for coords, (npair, nl) in zip(g["band"], g["band_np"]):
        axs.add_patch(MPoly(straighten(coords, m, a), closed=True, fc="#FBE8C2" if npair else "#F0F0F0", ec="white", lw=0.3))
    for coords, (npair, nl) in zip(g["rem"], g["rem_np"]):
        axs.add_patch(MPoly(straighten(coords, m, a), closed=True, fc=PAIR if npair else NOPAIR, ec="white", lw=0.4))
    ev = np.array(x["pv"]["dest_xy"]) if x["pv"]["dest_xy"] else np.zeros((0, 2))
    if len(ev):
        q = straighten(ev, m, a)
        axs.plot(q[:, 0], q[:, 1], "k.", ms=2.5, zorder=5)
    axs.set_aspect("equal")
    axs.set_xlim(-56, 56); axs.set_ylim(-17, 17)
    axs.set_yticks([])
    set_frame(axs, "open")
    axs.spines["left"].set_visible(False)
    axs.set_title(title, loc="left")


def pick(L, nrem):
    c = [x for x in L if x["nrem"] == nrem and x["pv"]["geo"]]
    return c[0]


sil = pick(SIL, 60) if any(x["nrem"] == 60 and x["pv"]["geo"] for x in SIL) else [x for x in SIL if x["pv"]["geo"]][0]
act = pick(ACT, 85) if any(x["nrem"] == 85 and x["pv"]["geo"] for x in ACT) else [x for x in ACT if x["pv"]["geo"]][0]
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
draw_ribbon(axs[0], sil, "A silent ribbon: no tile of it sits at a pair site (%d tiles)" % sil["nrem"])
draw_ribbon(axs[1], act, "An active ribbon: orange tiles are pair sites, dots mark pairs destroyed (%d tiles)" % act["nrem"])
axs[1].set_xlabel("distance along the ribbon (tile edge lengths)")
panel_letter(axs[0], "a", dx=-0.04); panel_letter(axs[1], "b", dx=-0.04)
from matplotlib.patches import Patch
axs[1].legend(handles=[Patch(fc=PAIR, label="removed tile with a pair-site leaf"), Patch(fc=NOPAIR, label="removed tile without"),
                       Patch(fc="#FBE8C2", label="neighbour with pair site"), Patch(fc="#F0F0F0", label="neighbour without")],
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4, handlelength=1.2, columnspacing=1.2)
fig.savefig(os.path.join(HERE, "figures", "ribbon_1_silent_vs_active.png"))
f1 = fig
print("fig1 done", sil["nrem"], act["nrem"])
res1 = render_check(fig, final_width_in=7.2, min_pt=7); print({k: v for k, v in res1.items() if v})
plt.close(fig)

# ---------------------------------------------------------------- figure 2: window position decides the pair site
O = np.array([(t[2], t[3], t[8] > 0, nev(x) == 0) for x in RIB for t in x["pv"]["trem"] if t[1] == "O"], float)
pts = SWR.points("O", 0, 6)[0]
rng = np.random.default_rng(0)
cloud = pts[rng.choice(len(pts), 60000, replace=False)]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1.0, 0.85], wspace=0.62))
ax = axs[0]
ax.plot(cloud[:, 0], cloud[:, 1], ".", ms=0.4, color="#DDDDDD", rasterized=True)
j = rng.permutation(len(O))
for sel, c, lab in ((O[j, 2] == 0, NOPAIR, "no pair site"), (O[j, 2] == 1, PAIR, "pair site")):
    q = O[j][sel]
    ax.plot(q[::2, 0], q[::2, 1], ".", ms=1.6, color=c, label=lab, alpha=0.8, rasterized=True)
ax.set_aspect("equal"); ax.set_xlabel("window coordinate $z_x$"); ax.set_ylabel("window coordinate $z_y$")
ax.legend(loc="upper right", markerscale=4, handletextpad=0.2)
ax.set_title("Pair sites: low window edge", loc="left")
ax = axs[1]
bins = np.linspace(-1.5, 2.8, 23)
ax.hist(O[O[:, 2] == 1, 1], bins=bins, color=PAIR, alpha=0.9, label="pair site")
ax.hist(O[O[:, 2] == 0, 1], bins=bins, color=NOPAIR, alpha=0.7, label="no pair site")
ax.set_xlabel("window coordinate $z_y$"); ax.set_ylabel("removed tiles of the O shape")
ax.axvline(0.0, color="k", lw=0.6, ls=":")
ax.set_xlim(-1.5, 2.8); ax.set_ylim(0, 420)
ax.legend(loc="upper left", ncol=1)
ax.set_title("A cut at $z_y$ = 0: 91%", loc="left")
ax = axs[2]
fr = lambda x: np.mean([t[3] < 0.0 for t in x["pv"]["trem"] if t[1] == "O"])
vs = [[fr(x) for x in SIL], [fr(x) for x in ACT]]
for k, (v, c) in enumerate(zip(vs, (SILENT_C, PAIR))):
    ax.plot(k + (np.random.default_rng(k).random(len(v)) - 0.5) * 0.25, v, "o", ms=3, color=c, alpha=0.8)
    ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, "k-", lw=1.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["silent\n(n=%d)" % len(SIL), "active\n(n=%d)" % len(ACT)])
ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.85)
ax.set_ylabel("share of O tiles with $z_y$ < 0")
ax.set_title("Silent ribbons stay high", loc="left")
for a, l in zip(axs, "abc"):
    panel_letter(a, l, dx=-0.2 if l != "a" else -0.16)
fig.savefig(os.path.join(HERE, "figures", "ribbon_2_window_position.png"))
res2 = render_check(fig, final_width_in=7.2, min_pt=7); print("fig2", {k: v for k, v in res2.items() if v})
plt.close(fig)

# ---------------------------------------------------------------- figure 3: ribbon statistics
def ribbons(f, R):
    d = json.load(open(os.path.join(HERE, f)))
    u = {}
    for x in d["rows"]:
        if x.get("n", 0) >= 20 and x.get("ends") is not None:
            u.setdefault((x["phase"], x["sig"]), x)
    return list(u.values())


r60 = ribbons("generic_phase_d0.03_R60_rib.json", 60)
r100 = ribbons("generic_phase_d0.03_R100_rib.json", 100)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.6))
ax = axs[0]
for L, c, m, lab in ((r60, "#0072B2", "o", "region radius 60"), (r100, "#D55E00", "s", "region radius 100")):
    ax.plot([x["diam"] for x in L], [x["nrem"] for x in L], m, ms=3.5, color=c, label=lab, alpha=0.8)
xx = np.array([0, 260]); ax.plot(xx, 0.95 * xx, "k:", lw=0.8)
ax.set_xlabel("ribbon extent"); ax.set_ylabel("tiles in the ribbon")
ax.set_ylim(-20, 460); ax.legend(loc="upper left"); ax.set_title("Longer region, longer ribbon", loc="left")
ax = axs[1]
e60 = [np.hypot(*e["pos"]) for x in r60 for e in x["ends"]]
e100 = [np.hypot(*e["pos"]) for x in r100 for e in x["ends"]]
ax.hist(e60, bins=np.arange(60, 130, 3), color="#0072B2", alpha=0.85, label="radius 60")
ax.hist(e100, bins=np.arange(60, 130, 3), color="#D55E00", alpha=0.85, label="radius 100")
for r in (76, 116):
    ax.axvline(r, color="k", ls=":", lw=0.8)
ax.set_xlabel("end distance from origin"); ax.set_ylabel("ends"); ax.set_ylim(0, 22); ax.legend(loc="upper left")
ax.set_title("Ends follow the edge", loc="left")
ax = axs[2]
gaps = []
for x in ACT:
    t = np.sort([a for a in x["pv"]["dest"] + x["pv"]["crea"] if a is not None])
    if len(t) > 5:
        g = np.split(t, np.where(np.diff(t) > 1.0)[0] + 1); c = np.array([a.mean() for a in g]); gaps += list(np.diff(c))
ax.hist(gaps, bins=np.arange(0, 12, 0.5), color="#009E73", alpha=0.9)
ax.axvline(np.mean(gaps), color="k", ls="--", lw=0.8)
ax.set_xlabel("gap between pair sites"); ax.set_ylabel("gaps")
ax.set_title("Two spacings, no period", loc="left")
for a, l in zip(axs, "abc"):
    panel_letter(a, l, dx=-0.2)
fig.savefig(os.path.join(HERE, "figures", "ribbon_3_statistics.png"))
res3 = render_check(fig, final_width_in=7.2, min_pt=7); print("fig3", {k: v for k, v in res3.items() if v}, "n gaps", len(gaps), "mean %.2f" % np.mean(gaps), "ends", len(e60), len(e100))
plt.close(fig)
