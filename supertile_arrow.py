#!/usr/bin/env python3
"""
Does the lopsided shape of a Spectre supertile give the tiling an arrow, and does it fix the direction of a ribbon?
(ROADMAP Stage 9p.)

Observation (B. Hartshorn): the substitution patch has more tiles on one side, at every iteration.

 1  The arrow of one supertile: centroid of its tiles minus the centre of its quad (the four key points used by the
    substitution), in units of lambda^n; long axis, aspect ratio and skewness along the long axis.          (Exact numbers)
 2  The arrows in a tiling: orientations of the level-k supertiles inside a level-6 patch. If they are spread evenly
    over the allowed orientations the arrows cancel and the tiling has no arrow of its own.                  (Counted)
 3  Ribbons: do the physical directions of the phason ribbons (ribbon_directions.json) cluster modulo 30 or 60
    degrees, as they would if a ribbon ran at a fixed angle to the supertile arrows?        (Numeric, only 8 ribbons)

Usage: python3 supertile_arrow.py [--tests]       (about a minute)
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recognisability_radius as RR  # noqa: E402
SP, LAM = RR.SP, RR.LAM


def one_supertile(n, label="Delta"):
    tiles = SP.buildSpectreTiles(n, 1.0, 1.0)
    tc = np.array(SP.get_spectre_points(1.0, 1.0)).mean(0)
    root = tiles[label]
    C = []
    root.forEachTile(lambda T, l: C.append(T[:, :2] @ tc + T[:, 2]))
    C = np.array(C); cen = C.mean(0)
    a = cen - np.array(root.quad).mean(0)
    X = C - cen
    w, v = np.linalg.eigh(X.T @ X / len(X))
    ax = v[:, 1]
    return dict(level=n, label=label, tiles=len(C), arrow_length=float(np.linalg.norm(a) / LAM ** n), arrow_angle=float(math.degrees(math.atan2(a[1], a[0]))),
                long_axis_angle=float(math.degrees(math.atan2(ax[1], ax[0])) % 180), aspect=float(math.sqrt(w[1] / w[0])),
                skewness=float(((X @ ax) ** 3).mean() / w[1] ** 1.5 * np.sign(ax @ a)))


def main():
    out = dict(supertile=[one_supertile(n, lab) for n in range(1, 7) for lab in ("Delta", "Gamma")])
    for r in out["supertile"]:
        print("level %d %-5s %6d tiles: arrow %.3f lambda^n at %6.1f deg; long axis %5.1f deg, aspect %.2f, skewness %+.3f" % (
            r["level"], r["label"], r["tiles"], r["arrow_length"], r["arrow_angle"], r["long_axis_angle"], r["aspect"], r["skewness"]))
    levels, leaves = RR.build(6)
    out["orientations"] = []
    for k in (1, 2, 3, 4):
        ori = np.array([round(math.degrees(math.atan2(T[1, 0], T[0, 0]))) % 360 for T, _, _ in levels[k]])
        cnt = np.bincount(ori // 30, minlength=12)
        R = abs(np.exp(1j * np.radians(ori)).mean())
        out["orientations"].append(dict(level=k, units=len(ori), counts_per_30_degrees=cnt.tolist(), resultant=float(R), noise=float(0.886 / math.sqrt(len(ori)))))
        print("level %d supertiles in a level-6 patch: %5d; per 30 degrees %s; resultant %.4f (random: about %.4f)" % (k, len(ori), cnt.tolist(), R, 0.886 / math.sqrt(len(ori))))
    d = json.load(open(os.path.join(HERE, "ribbon_directions.json")))
    ang = np.array([r["physical_angle"] for r in d["ribbons"] if r["physical_angle"] is not None])
    out["ribbons"] = dict(n=len(ang), angles=ang.tolist(), resultant_mod_30=float(abs(np.exp(1j * np.radians(12 * ang)).mean())),
                          resultant_mod_60=float(abs(np.exp(1j * np.radians(6 * ang)).mean())), noise=float(0.886 / math.sqrt(len(ang))))
    print("ribbons: %d angles; resultant modulo 30 degrees %.3f, modulo 60 degrees %.3f (random: about %.2f)" % (len(ang), out["ribbons"]["resultant_mod_30"], out["ribbons"]["resultant_mod_60"], out["ribbons"]["noise"]))
    json.dump(out, open(os.path.join(HERE, "supertile_arrow.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        D = [r for r in out["supertile"] if r["label"] == "Delta"]
        assert abs(D[-1]["arrow_length"] - D[-3]["arrow_length"]) < 1e-3 and abs(D[-1]["aspect"] - D[-3]["aspect"]) < 1e-2      # converged, same parity
        assert all(sum(o["counts_per_30_degrees"][1::2]) == 0 for o in out["orientations"])                                    # six orientations only
        print("supertile_arrow: tests passed (arrow and aspect converged; supertiles occur in six orientations)")


if __name__ == "__main__":
    main()
