"""
Figures for Stages 2 and 4 of the roadmap. Writes PNGs to figures/.

    python3 make_figures_entanglement.py             # both (about 1 minute)
    python3 make_figures_entanglement.py tau         # Stage 2: the tau scan
    python3 make_figures_entanglement.py geometry    # Stage 4: mutual information against distance

Every number drawn comes from `pair_hamiltonian.py` and `entanglement_geometry.py`.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import entanglement_geometry as eg
import pair_hamiltonian as ph

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_tau_scan():
    """Stage 2: concurrence, CHSH and the cat-doublet splitting against tau = J / (kappa w)."""
    taus = np.geomspace(0.1, 3.0, 22)
    sizes = (8, 10, 12, 14)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for n in sizes:
        W = ph.central_couplings(3, n)
        w = ph.coupling_scale(W)
        rows = ph.tau_scan(W, taus)
        ax[0].plot(taus, [r["max_concurrence"] for r in rows], "o-", ms=3, label=f"{n} pairs")
        ax[1].plot(taus, [r["chsh_best"] for r in rows], "o-", ms=3, label=f"{n} pairs")
        ax[2].semilogy(taus, [max(r["gap"] / (r["tau"] * w), 1e-16) for r in rows], "o-", ms=3, label=f"{n} pairs")
    ax[0].set_ylabel("largest pairwise concurrence")
    ax[1].axhline(2.0, color="k", lw=0.8, ls="--")
    ax[1].set_ylabel("best CHSH value (above 2 violates)")
    ax[2].set_ylabel("cat-doublet splitting / J")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel(r"$\tau = J / (\kappa w)$")
        a.legend(fontsize=8)
    fig.suptitle("Stage 2: ground state of the pair Hamiltonian, central pairs of generation 3 (exact diagonalisation)")
    _save(fig, "fig_stage2_tau_scan.png")


def fig_geometry():
    """Stage 4: mutual information against distance, with the shuffled-position control."""
    n = 16
    patch, W, mids, D = eg.build_cluster(4, n)
    iu = np.triu_indices(n, 1)
    r = np.linalg.norm(mids[:, None] - mids[None], axis=-1)[iu]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    rs = np.linalg.norm(mids[perm][:, None] - mids[perm][None], axis=-1)[iu]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for col, tau in enumerate((1.0, 3.0)):
        psi = eg.ground_state_at(W, tau)
        I = eg.mutual_information(psi, n)
        nul = eg.geometry_vs_null(I, mids, W)
        a = ax[col]
        a.loglog(r, np.clip(I[iu], 1e-6, None), "o", ms=4, label="real positions")
        a.loglog(rs, np.clip(I[iu], 1e-6, None), "x", ms=4, alpha=0.5, label="shuffled positions")
        a.set_xlabel("distance r between pair midpoints")
        a.set_ylabel("mutual information I(p:q) (bits)")
        a.set_title(rf"$\tau$ = {tau:g}:  $\rho$ = {nul['rho_I_r']:+.2f} (null {nul['null_rho_mean']:+.2f} $\pm$ {nul['null_rho_sd']:.2f}), "
                    rf"partial given |W| = {nul['partial_rho_I_r_given_W']:+.2f}", fontsize=9)
        a.legend(fontsize=8)
    links = eg.delaunay_links(mids)
    psi = eg.ground_state_at(W, 1.0)
    regs = eg.random_regions(n, range(2, n // 2 + 1), 250)
    S = np.array([ph.entanglement_entropy(psi, n, list(g)) for g in regs])
    B = np.array([eg.boundary_size(g, links) for g in regs])
    V = np.array([len(g) for g in regs])
    sc = ax[2].scatter(B, S, c=V, cmap="viridis", s=14)
    ax[2].set_xlabel("boundary B(R): Delaunay links cut")
    ax[2].set_ylabel("entanglement entropy S(R) (bits)")
    ax[2].set_title(r"random regions at $\tau$ = 1, colour = volume |R|", fontsize=9)
    fig.colorbar(sc, ax=ax[2], label="|R|")
    fig.suptitle("Stage 4: 16 central pairs of generation 4 (exact ground state)")
    _save(fig, "fig_stage4_geometry.png")


FIGS = {"tau": fig_tau_scan, "geometry": fig_geometry}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for name in names:
        FIGS[name]()
