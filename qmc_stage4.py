#!/usr/bin/env python3
"""
Stage 4 with QMC: runs `qmc_pairs.run_qmc_pt` on the flow Hamiltonian of the exact patches at 82 and
804 pairs and measures how the Z correlations follow the geometry. Entanglement itself is not measured
(see `qmc_pairs`); the geometric quantities are the ones that need only <Z_i Z_j> and <X_i>.

    python3 qmc_stage4.py run  GEN N FLOW BETA SEED SWEEPS [THERM]     # writes qmc_results/*.npz
    python3 qmc_stage4.py analyse FILE [FILE ...]                      # chain comparison + geometry
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import pair_hamiltonian as ph
import qmc_pairs as q
from entanglement_geometry import build_cluster, delaunay_links, partial_spearman

OUT = "qmc_results"


def _name(gen, n, flow, beta, seed):
    return os.path.join(OUT, f"g{gen}_n{n}_f{flow:g}_b{beta:g}_s{seed}.npz")  # (replica settings are in meta)


def run(gen, n, flow, beta, seed, sweeps, therm=None, replicas=8, bmin=0.1):
    os.makedirs(OUT, exist_ok=True)
    fp = ph.flow_parameters(flow)
    K = ph.flow_couplings(gen, n, flow)
    t = time.time()
    r = q.run_qmc_pt(K, fp.J, beta, 0.1, sweeps=sweeps, therm=therm if therm is not None else max(500, sweeps // 5),
                     seed=seed, measure_every=5, n_replicas=replicas, beta_min_factor=bmin)
    np.savez_compressed(_name(gen, n, flow, beta, seed), ZZ=r.ZZ, X=r.X, E=r.energy_series, J=fp.J, K=K,
                        meta=np.array([gen, n, flow, beta, seed, sweeps, r.swap_rate, time.time() - t]))
    print(f"gen {gen} n {n} flow {flow} beta {beta} seed {seed}: E/n = {r.energy / n:.5f} +- {r.energy_err / n:.5f}, "
          f"swap {r.swap_rate:.2f}, {time.time() - t:.0f}s")


def analyse(files):
    runs = [np.load(f) for f in files]
    for f, d in zip(files, runs):
        gen, n, flow, beta, seed, sweeps, swap, secs = d["meta"]
        E = d["E"]
        h = len(E) // 2
        print(f"{os.path.basename(f)}: E/n {E.mean() / n:.5f} (halves {E[:h].mean() / n:.5f} {E[h:].mean() / n:.5f}), "
              f"<X> {d['X'].mean():.4f}, mean|C_ij| nn {np.abs(d['ZZ'][np.triu_indices(int(n), 1)]).mean():.4f}")
    if len(runs) >= 2:
        iu = np.triu_indices(runs[0]["ZZ"].shape[0], 1)
        a, b = runs[0]["ZZ"][iu], runs[1]["ZZ"][iu]
        print(f"chain-to-chain: rms dZZ {np.sqrt(np.mean((a - b) ** 2)):.4f}, max {np.abs(a - b).max():.4f}, "
              f"corr {np.corrcoef(a, b)[0, 1]:.4f}; dX max {np.abs(runs[0]['X'] - runs[1]['X']).max():.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        g, n, fl, be, se, sw = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        run(g, n, fl, be, se, sw, int(sys.argv[8]) if len(sys.argv) > 8 else None, int(sys.argv[9]) if len(sys.argv) > 9 else 8, float(sys.argv[10]) if len(sys.argv) > 10 else 0.1)
    else:
        analyse(sys.argv[2:])
