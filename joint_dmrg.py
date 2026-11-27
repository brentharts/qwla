#!/usr/bin/env python3
"""
Ground state of the joint geometry-matter Hamiltonian of the Hat Trick sphere at kappa != 0, by DMRG
(open problem 3 of quantgravbits.tex; ROADMAP Stage 9e).

    H = -Gamma sum_p X_p + g sum_p (1 - sigma_p) - J sum_e n_e tau^x_e - kappa sum_{e<e'} n_e n_e' W_ee' tau^z_e tau^z_e'

12 geometry qubits (one per cone point, sigma_p in {0,1}) and 30 slots (one per icosahedron edge e = pq, present iff
n_e = (1 - sigma_p)(1 - sigma_q) = 1); 2^42 states. W is the dipole matrix of the 30 pairs (hat_sphere_couplings.json,
exported from spectre-monotile-py/research/toymodel_v12). The earlier results were two variational bounds
(quantgravalt.tex, Result "sphere-kappa"). Here the 42 sites are put on a line (each slot right after the later of its
two cone points) and the ground state is found by two-site DMRG (TeNPy) at bond dimensions chi = 32, 64, 128, ...; the
energy is variational at every chi, and the change between the last two chi is the error estimate.

Checks: at kappa = 0 the matter is the product state |+...+> and the energy must equal the exact ground energy of the
12-qubit model  -Gamma sum X + g sum n - J sum_<pq> n_p n_q  (4096 states); at kappa = 0.3 the energy must lie below
both earlier bounds.

Needs physics-tenpy. Usage: python3 joint_dmrg.py [--tests] [--chi 32,64,128]
"""
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "hat_sphere_couplings.json")))
EDGES = [tuple(e) for e in C["edges"]]
W = np.array(C["W"])
NP_, NE = 12, 30
OLD_BOUNDS = {(0.0, 0.2): (-32.12, -32.7677), (0.0, 1.0): (-34.14, -34.4655), (0.0, 3.0): (-48.95, -47.9022),
              (2.5, 0.2): (None, -2.8227), (2.5, 1.0): (None, -6.0522), (2.5, 3.0): (None, -28.3152)}   # (CI(2) joint, dressed CI(4))


def exact_geometry(Gamma, g, J):
    """Ground energy of the 12-qubit model that the joint model reduces to at kappa = 0."""
    s = np.arange(4096)
    n = 1 - ((s[:, None] >> np.arange(12)[None, :]) & 1)
    d = g * n.sum(1) - J * sum(n[:, i] * n[:, j] for i, j in EDGES)
    H = np.diag(d.astype(float))
    for p in range(12):
        H[s, s ^ (1 << p)] -= Gamma
    return float(np.linalg.eigvalsh(H)[0])


def site_order():
    """Cone points in breadth-first order from point 0; slot e right after the later of its two cone points.
    With ORDER=geofirst all 12 cone points come first, then the 30 slots in the same order, so that the bond after
    site 12 is the cut between geometry and matter."""
    adj = {i: [] for i in range(NP_)}
    for i, j in EDGES:
        adj[i].append(j)
        adj[j].append(i)
    order, seen, queue = [], {0}, [0]
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    rank = {p: k for k, p in enumerate(order)}
    if os.environ.get("ORDER") == "geofirst":
        slots = [("e", e) for p in order for e, (i, j) in enumerate(EDGES) if max(rank[i], rank[j]) == rank[p]]
        return [("p", p) for p in order] + slots
    sites = []
    for p in order:
        sites.append(("p", p))
        for e, (i, j) in enumerate(EDGES):
            if max(rank[i], rank[j]) == rank[p]:
                sites.append(("e", e))
    return sites


def build(Gamma, g, J, kappa):
    from tenpy.models.lattice import Chain
    from tenpy.models.model import MPOModel
    from tenpy.networks import mpo
    from tenpy.networks.site import SpinHalfSite
    from tenpy.networks.terms import TermList
    sites = site_order()
    pos = {s: k for k, s in enumerate(sites)}
    L = len(sites)
    site = SpinHalfSite(conserve=None)
    site.add_op("P0", np.diag([1.0, 0.0]))            # 'up' is sigma = 0 (the pair-bearing state); for a slot 'Sigmax' = tau^x
    terms, strengths = [], []

    def add(ops, c):
        if c != 0:
            terms.append(sorted(ops, key=lambda t: t[1]))
            strengths.append(float(c))
    for p in range(NP_):
        add([("Sigmax", pos[("p", p)])], -Gamma)
        add([("P0", pos[("p", p)])], g)
    for e, (i, j) in enumerate(EDGES):
        add([("P0", pos[("p", i)]), ("P0", pos[("p", j)]), ("Sigmax", pos[("e", e)])], -J)
    for a, b in itertools.combinations(range(NE), 2):
        if W[a, b] == 0 or kappa == 0:
            continue
        pts = sorted(set(EDGES[a]) | set(EDGES[b]))
        add([("P0", pos[("p", q)]) for q in pts] + [("Sigmaz", pos[("e", a)]), ("Sigmaz", pos[("e", b)])], -kappa * W[a, b])
    lat = Chain(L, site, bc="open", bc_MPS="finite")
    tl = TermList(terms, strengths)
    ot, ct = tl.to_OnsiteTerms_CouplingTerms([site] * L)
    H = mpo.MPOGraph.from_terms((ot, ct), [site] * L, bc="finite").build_MPO()
    return MPOModel(lat, H), sites, pos


def observables(psi, sites, pos):
    P0 = psi.expectation_value("P0")
    geo = [pos[("p", p)] for p in range(NP_)]
    n0 = float(sum(P0[k] for k in geo))
    pairs = 0.0
    for i, j in EDGES:
        pairs += float(psi.expectation_value_term([("P0", pos[("p", i)]), ("P0", pos[("p", j)])]))
    tx = psi.expectation_value("Sigmax")
    slots = [pos[("e", e)] for e in range(NE)]
    # entanglement entropy between geometry and matter is not a cut of the chain; report the largest bond entropy instead
    S = psi.entanglement_entropy() / np.log(2)
    p_all0 = float(psi.expectation_value_term([("P0", k) for k in sorted(geo)]))
    cut = None
    if sorted(geo) == list(range(NP_)):                     # geometry first: bond 11 separates geometry from matter
        cut = float(S[NP_ - 1])
    return dict(mean_rotors_in_0=n0, mean_pairs=pairs, P_all0=p_all0, mean_tau_x_of_slots=float(np.mean([tx[k] for k in slots])),
                max_bond_entropy_bits=float(S.max()), S_geometry_matter_bits=cut)


def run(Gamma, g, J=1.0, kappa=0.3, chis=(32, 64, 128), verbose=True):
    from tenpy.algorithms import dmrg
    from tenpy.networks.mps import MPS
    t0 = time.time()
    model, sites, pos = build(Gamma, g, J, kappa)
    L = len(sites)
    # start: geometry in 'up' (sigma = 0) rotated toward +x by the first sweeps, slots in |+> (the kappa = 0 ground state of the matter)
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    up = np.array([1.0, 0.0])
    init = [up * 0.95 + plus * 0.3 if s[0] == "p" else plus for s in sites]
    init = [v / np.linalg.norm(v) for v in init]
    psi = MPS.from_product_state(model.lat.mps_sites(), init, bc="finite")
    out = []
    for chi in chis:
        eng = dmrg.TwoSiteDMRGEngine(psi, model, {"mixer": True, "mixer_params": {"amplitude": 1e-4, "decay": 2.0, "disable_after": 6},
                                                  "trunc_params": {"chi_max": int(chi), "svd_min": 1e-10}, "max_sweeps": 14, "min_sweeps": 4,
                                                  "max_E_err": 1e-8, "combine": True, "max_trunc_err": None})
        E, psi = eng.run()
        row = dict(chi=int(chi), E=float(E), chi_used=int(max(psi.chi)), seconds=round(time.time() - t0, 1))
        out.append(row)
        if verbose:
            print("  Gamma=%g g=%g kappa=%g chi=%d: E = %.6f (bond %d, %.0f s)" % (Gamma, g, kappa, chi, E, max(psi.chi), time.time() - t0), flush=True)
    res = dict(Gamma=Gamma, g=g, J=J, kappa=kappa, mpo_bond=int(max(model.H_MPO.chi)), sweeps=out, E=out[-1]["E"],
               E_change_last_two_chi=(out[-1]["E"] - out[-2]["E"]) if len(out) > 1 else None)
    res.update(observables(psi, sites, pos))
    return res


def run_tests():
    for Gamma, g in ((1.0, 0.0), (0.2, 2.5)):
        r = run(Gamma, g, kappa=0.0, chis=(16, 32), verbose=False)
        ex = exact_geometry(Gamma, g, 1.0)
        assert abs(r["E"] - ex) < 1e-6, (r["E"], ex)
        print("kappa = 0, Gamma = %g, g = %g: DMRG %.8f, exact 12-qubit model %.8f" % (Gamma, g, r["E"], ex))
    print("tests passed")


if __name__ == "__main__":
    if "--tests" in sys.argv:
        run_tests()
        sys.exit(0)
    chis = (32, 64, 128)
    if "--chi" in sys.argv:
        chis = tuple(int(x) for x in sys.argv[sys.argv.index("--chi") + 1].split(","))
    results = []
    for g, Gamma in ((0.0, 0.2), (0.0, 1.0), (0.0, 3.0), (2.5, 1.0)):
        r = run(Gamma, g, chis=chis)
        r["earlier_bounds_CI2_dressed"] = OLD_BOUNDS.get((g, Gamma))
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "sweeps"}), flush=True)
        json.dump(results, open(os.path.join(HERE, "joint_dmrg%s.json" % ("_geofirst" if os.environ.get("ORDER") == "geofirst" else "")), "w"), indent=1)
    print("-> joint_dmrg.json")
