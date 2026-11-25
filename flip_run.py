#!/usr/bin/env python3
"""
Stage 3, measurements: the flip rules on the pair events of `flip_pairs.py`.

For each flip with a change of the pair set the state before is the ground state of H(pi0) on the pairs of the
region (couplings from the centroids at b(tau), J and kappa of the Spectre-to-Hat flow at tau). Created pairs
start in |+>. Everything below is computed on the register of the present pairs by pure-state formulas, which
`check_against_dense` compares with the density-matrix rules of `flip_rule.py` on the small events.

  dE         E_after - E0: the energy of the old state under H(pi1), minus its energy under H(pi0)
  excess     E_after - E1: the same above the new ground energy (>= 0)
  S_loss     entanglement entropy, in bits, between the persisting pairs and the destroyed ones in the old state;
             the information the reset and trace rules discard and the dormant rule keeps
  F_ret      overlap of the state after (flip, reversed flip) with the old one: 1 for dormant, ||M^T M u||^2 for reset
  F_new      overlap of the persisting-pair marginal with the new ground state (the same for the three rules)

Null: the same numbers of destroyed and created pairs, but the destroyed pairs drawn at random from the pairs
of the old region.
Usage: python3 flip_run.py [flip_pairs json] [tau]
"""
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from flip_rule import pair_W, evaluate_flip, ptrace
from pair_hamiltonian import flow_parameters, build_hamiltonian

HERE = os.path.dirname(os.path.abspath(__file__))


def ground(W, J, kappa):
    n = W.shape[0]
    if n == 0:
        return 0.0, np.ones(1)
    if n <= 10:
        H = build_hamiltonian(W, J, kappa).toarray()
        w, v = np.linalg.eigh(H)
        return float(w[0]), v[:, 0]
    H = build_hamiltonian(W, J, kappa)
    w, v = spla.eigsh(H, k=2, which="SA")
    i = int(np.argmin(w))
    return float(w[i]), v[:, i]


def expect_X(psi, n, p):
    idx = np.arange(2 ** n)
    return float(psi @ psi[idx ^ (1 << (n - 1 - p))])


def expect_ZZ(psi, n, p, q):
    idx = np.arange(2 ** n)
    s = (1 - 2 * ((idx >> (n - 1 - p)) & 1)) * (1 - 2 * ((idx >> (n - 1 - q)) & 1))
    return float((psi * psi) @ s)


def ids(ps):
    return [tuple(map(tuple, p["id"])) for p in ps]


def event_slots(rec, key):
    """Slots = persisting + destroyed + created, with the centroids at the chosen b."""
    p0 = {tuple(map(tuple, p["id"])): p for p in rec["pairs0"]}
    p1 = {tuple(map(tuple, p["id"])): p for p in rec["pairs1"]}
    persist = sorted(set(p0) & set(p1))
    destroyed = sorted(set(p0) - set(p1))
    created = sorted(set(p1) - set(p0))
    return p0, p1, persist, destroyed, created


def measure(rec, J, kappa, key="ctau", rng=None, null=False):
    p0, p1, persist, destroyed, created = event_slots(rec, key)
    reg0 = persist + destroyed                      # register of pi0 (slot order: persisting first)
    reg1 = persist + created
    cents0 = [tuple(map(np.array, p0[i][key])) for i in reg0]
    cents1 = [tuple(map(np.array, p1[i][key])) for i in reg1]
    if null:
        # random choice of the destroyed pairs among those of the old region (same number); created count kept
        order = rng.permutation(len(reg0))
        d = len(destroyed)
        keep = [reg0[i] for i in order[d:]]
        drop = [reg0[i] for i in order[:d]]
        reg0 = keep + drop
        persist = keep
        cents0 = [tuple(map(np.array, p0[i][key])) for i in reg0]
        cents1 = cents0[:len(persist)] + [tuple(map(np.array, p1[i][key])) for i in created]
        reg1 = persist + created
    n0, n1 = len(reg0), len(reg1)
    nP, nD, nC = len(persist), len(destroyed), len(created)
    W0 = pair_W(cents0)
    E0, psi0 = ground(W0, J, kappa)
    # terms of the old Hamiltonian that touch a destroyed pair, in the old state
    touch = 0.0
    for d in range(nP, n0):
        touch += -J * expect_X(psi0, n0, d)
    for p in range(n0):
        for q in range(p + 1, n0):
            if q >= nP:                                 # q is destroyed (p may be anything)
                touch += kappa * W0[p, q] * expect_ZZ(psi0, n0, p, q)
    E_after = E0 - touch - J * nC                       # created in |+>: <X> = 1, <Z> = 0, so only -J each
    W1 = pair_W(cents1)
    E1, psi1 = ground(W1, J, kappa)
    # persisting marginal
    M = psi0.reshape(2 ** nP, 2 ** nD) if nD else psi0.reshape(-1, 1)
    sv = np.linalg.svd(M, compute_uv=False)
    pr = sv ** 2
    pr = pr[pr > 1e-14]
    S_loss = float(-(pr * np.log2(pr)).sum())
    # F_new: <psi1| rho_P (x) |+><+|_C |psi1>
    if nC:
        T = psi1.reshape(2 ** nP, 2 ** nC)
        u = np.ones(2 ** nC) / math.sqrt(2 ** nC)
        phi = T @ u
    else:
        phi = psi1
    F_new = float(np.linalg.norm(M.T @ phi) ** 2)
    # F_ret for reset: after reset the state is rho_P (x) |+><+|_D; overlap with psi0
    if nD:
        u = np.ones(2 ** nD) / math.sqrt(2 ** nD)
        F_ret = float(np.linalg.norm(M.T @ (M @ u)) ** 2)
    else:
        F_ret = 1.0
    return dict(nP=nP, nD=nD, nC=nC, E0=E0, E1=E1, E_after=E_after, dE=E_after - E0, excess=E_after - E1,
                S_loss=S_loss, F_new=F_new, F_ret_reset=F_ret, F_ret_dormant=1.0)


def check_against_dense(rec, J, kappa, key="ctau"):
    """The pure-state formulas of `measure` against `flip_rule.evaluate_flip` (density matrices) on one event."""
    p0, p1, persist, destroyed, created = event_slots(rec, key)
    slots = persist + destroyed + created
    n = len(slots)
    cents = [tuple(map(np.array, (p0.get(i) or p1[i])[key])) for i in slots]
    W = pair_W(cents)
    pres0 = [True] * (len(persist) + len(destroyed)) + [False] * len(created)
    pres1 = [True] * len(persist) + [False] * len(destroyed) + [True] * len(created)
    res = evaluate_flip(W, pres0, pres1, J, kappa)
    m = measure(rec, J, kappa, key)
    return abs(res["reset_E"] - m["E_after"]) + abs(res["dormant_E"] - m["E_after"]) + abs(res["reset_return_F"] - m["F_ret_reset"]), res, m


# ---- checks -----------------------------------------------------------------------------

def _synthetic_event(nP, nD, nC, seed):
    rng = np.random.default_rng(seed)
    def pair(i, x0):
        c = rng.normal(size=2) * 3 + x0
        return dict(id=[[i, 0], [i, 1]], c1=[c.tolist(), (c + [1.3, 0.4]).tolist()], ctau=[c.tolist(), (c + [1.3, 0.4]).tolist()])
    keep = [pair(i, 0.0) for i in range(nP)]
    dead = [pair(100 + i, 0.0) for i in range(nD)]
    new = [pair(200 + i, 0.0) for i in range(nC)]
    return dict(pairs0=keep + dead, pairs1=keep + new)


def test_pure_state_formulas_match_the_dense_rules():
    for nP, nD, nC, seed in ((3, 1, 1, 1), (4, 2, 1, 2), (3, 1, 2, 3), (4, 0, 2, 4), (4, 2, 0, 5)):
        rec = _synthetic_event(nP, nD, nC, seed)
        dev, res, m = check_against_dense(rec, 0.3, 2.0)
        assert dev < 1e-9, (nP, nD, nC, dev)
        assert m["excess"] > -1e-9


def test_created_only_and_destroyed_only_limits():
    rec = _synthetic_event(4, 0, 2, 7)
    m = measure(rec, 0.3, 2.0)
    assert m["S_loss"] < 1e-9 and abs(m["F_ret_reset"] - 1.0) < 1e-12 and abs(m["dE"] + 2 * 0.3) < 1e-9


def test_energy_of_the_new_state_is_never_below_the_new_ground_energy():
    for seed in range(6):
        rec = _synthetic_event(4, 1 + seed % 2, 1 + seed % 3, seed)
        m = measure(rec, 0.4, 1.5)
        assert m["excess"] >= -1e-9


def run_tests(verbose=True):
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        if verbose:
            print("ok", t.__name__)
    return len(tests)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else os.path.join(HERE, "flip_pairs_c0_d0.03.json")
    tau = float(args[1]) if len(args) > 1 else 0.05
    fp = flow_parameters(tau)
    J, kappa = fp.J, fp.kappa
    d = json.load(open(path))
    rng = np.random.default_rng(11)
    rows = []
    for rec in d["records"]:
        if rec["label"] != "changed":
            continue
        p0, p1, persist, destroyed, created = event_slots(rec, "ctau")
        if not destroyed and not created:
            continue
        if len(persist) + max(len(destroyed), len(created)) > 13:
            continue
        m = measure(rec, J, kappa)
        m.update(sample=rec["sample"], tau=tau)
        nm = measure(rec, J, kappa, rng=rng, null=True)
        for k in ("dE", "excess", "S_loss", "F_new", "F_ret_reset"):
            m["null_" + k] = nm[k]
        rows.append(m)
    import csv
    out = os.path.join(HERE, "flip_rules_c0_tau%g.csv" % tau)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("tau", tau, "J %.4f kappa %.4f" % (J, kappa), "events", len(rows), "->", out)


if __name__ == "__main__":
    if "--tests-only" in sys.argv:
        print(run_tests(), "checks passed")
    else:
        main()
