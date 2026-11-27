# Connections with "An algebraic quantization of gravity from the null cone"

This note explores how the paper of V. Chandrasekaran (arXiv:2611.34924, hep-th, November 2026) might connect to the Spectre and Hat Trick programme of this repository. The paper has strong mathematics and, by its own account, few explicit physical pictures of the states it proves exist. This repository has an explicit, finite, computable model of a closed surface with punctures. The question is whether the second can supply physical content to the first. Some candidate bridges below survive a computation; some do not; most remain hypotheses.

Labels as in the roadmap: **Exact**, **Counted**, **Known** (standard result used, not derived here), **Definition**, **Hypothesis**, **Not done**. Numbers computed here come from `null_cone_checks.py` (`python3 null_cone_checks.py --tests`, about 10 seconds).

## Reading status (read this first)

I did not read the paper end to end. I read it through a summarising tool, in three pieces that covered the abstract and introduction, Sections 2 to 6 and Appendices A to E in outline, and the visible text was truncated in places. I have not checked any derivation. Everything below attributed to the paper is the author's claim as summarised to me, with a section number where I have one; treat the section numbers as pointers to check, not as quotations. The paper's own statements of limitation that I saw are listed in section 2.

## 1. What the paper does

- **Setup.** A background-independent quantisation of vacuum general relativity on a causal diamond: first-order gravity, constraints solved on two intersecting null cones, giving a reduced phase space at finite G_N. Poisson brackets become commutators; the observables form a C*-algebra of relative S-matrices.
- **Corner and clock.** The corner S where the cones meet carries a boost charge K(y) = rho_S(y)^2 / (8 pi G_N) with total K = A/(8 pi G_N), conjugate to a relative boost. The "gravitational clock" u_* is the affine parameter at which incoming null generators reach the corner, and its reading bounds the area of S above (A <= 4 pi l(D)^2/(1 - epsilon)^2 on the diamond).
- **Hilbert space.** Gaussian states of canonical pairs give a representation in which the algebra generates a type I factor; amplitudes compose by sums over intermediate states that converge absolutely at finite G_N.
- **Finite dimension.** A diamond of bounded clock reading has a finite-dimensional Hilbert space H_D, hence a discrete area spectrum A_1, ..., A_m with m <= d_D = dim H_D (Section 5.1 and 5.2). The argument shows existence and finiteness; it does not compute d_D.
- **Entropy.** The thermal density on H_D gives S = <A>/(4 G_N hbar) + log Z_{D,1}, and for marginally trapped surfaces on Schwarzschild a logarithmic term with coefficient q/(2(1-q)) in log(<A>/4 G_N hbar), with q = 257/302, which the author says matches Sen's one-loop result (106/45 after subtracting 1/2) (Section 5.3).
- **Topology.** For two boundaries the connected-topology component is a null state, so K_Sigma = H_Sigma (x) H_Sigma^* and each side has entropy log d_A (Section 5.4).
- **Semiclassical limit.** One-loop semiclassical Einstein equations for coherent states (Section 6).

## 2. Where the paper leaves physical content open (its own statements)

As summarised to me, the author states or implies these gaps:

1. The area eigenvalues and their spacing are not given; the only explicit scale is the dimensionless operator s = A/(4 G_N hbar) and its bound s <= pi l(D)^2 / (G_N hbar (1-epsilon)^2).
2. d_D is not computed, and no explicit basis of H_D is exhibited.
3. The finite-dimension argument is an existence argument; the convergence step behind it is called strong evidence, not proof, at finite G_N and hbar, and some steps are conditional on dim H_D = infinity being excluded.
4. The logarithmic coefficient uses a spherical Schwarzschild background and a marginally trapped surface (which may lie inside the horizon); it is a one-loop value, not an all-orders statement.
5. Backreaction is only to one loop and first order in hbar.
6. Whether different choices of the pure state give unitarily equivalent representations is left open.
7. The connected-topology component is null; whether it has physical meaning beyond that is left open. (The paper does not use the phrase ER = EPR in that section, as far as I saw.)

The first two are what an explicit model could address.

## 3. Candidate connections

### A. A concrete finite corner space: the punctured sphere (Exact counts; the identification is a Hypothesis)

This repository's 12 cone points are the 12 face centres of the Hat Trick ball, and are the vertices of a regular icosahedron, equivalently the 12 face normals of a regular dodecahedron (`HOPF_SITES.md`). Reading each as a spin-1/2 puncture, the gauge-invariant (Gauss-constraint) space of the corner is the 132 singlets (Stage 9b), a Temperley-Lieb module whose level-k quotient has dimension 1, 32, 89, 122, 131, 132 for k = 1 to 6 (Stage 9c, **Exact**). This is the standard way finite-dimensional corner spaces arise in spin-network models (**Known**).

- **What matches.** The paper's H_D is finite-dimensional and decomposes by area, H_D = direct sum over j of H_{D,A_j}. A puncture model has the same structure: fix the spins at the punctures and the area is fixed, and the singlets of those spins form an area sector. For 12 spin-1/2 punctures the sector has dimension 132 (the Catalan number C_6). With 12 spin-1 punctures it would have a different dimension and a different area; the sum over spin assignments is the analogue of the direct sum over A_j.
- **What is a hypothesis.** That the corner of a causal diamond of the paper is described by such a punctured sphere. The paper does not mention punctures, spins or intertwiners, and its Hilbert space is built from Gaussian states of graviton modes; no map between the two pictures is known to me.
- **A possible role for the level k.** The paper bounds the area by the clock reading (an infrared cutoff), and in 3D gravity with a cosmological constant a level k truncation plays the same role for the puncture space (**Known** in outline). If the same were true here, k would be tied to l(D)^2 / (G_N hbar). The data of the Hat Trick ball cannot test this: its space saturates at k = 6. **Not done** and a pure guess.

### B. Logarithmic terms: two different sources (Exact for the counting; the comparison is a Hypothesis)

The singlet space of N spin-1/2 punctures has dimension equal to the Catalan number C_{N/2}, so (**Exact**)

    ln dim = N ln 2 - (3/2) ln N + (3/2) ln 2 - (1/2) ln pi + o(1).

`null_cone_checks.py` checks the constant: the quantity ln dim - N ln 2 + (3/2) ln N is 0.292 at N = 12, 0.412 at N = 40, 0.440 at N = 80, approaching 0.467. The logarithmic term -(3/2) ln N is the familiar one from counting gauge-invariant states, and N is proportional to the area if each puncture carries a fixed area. The paper's logarithmic term (coefficient 106/45 in log s after the subtraction of 1/2, as summarised) comes from integrating over the clock bound and a one-loop graviton computation, not from a gauge constraint. They are different mechanisms with different coefficients. Whether the Gauss-constraint term should be added to the paper's, replaced by it, or is already inside it is not something I can say; the paper's counting does not use punctures. **Hypothesis, not done.**

### C. Entanglement of the corner state follows the boundary of the region (Exact numerics; interpretation open)

For the exact Heisenberg ground state of the 12 punctures (energy -6.1879, the unique singlet of `ring_bits.py`), I computed the entanglement entropy of every region R of punctures (2047 bipartitions up to complement). The entropy depends mainly on the number of punctures in R and on the number of icosahedron edges cut:

- a fit S = 0.077 (edges cut) + 0.234 (punctures in R) + 0.284 has R^2 = 0.963, against R^2 = 0.812 for the number of punctures alone;
- at a fixed region size the correlation of S with the number of edges cut is 0.92 to 0.98 (sizes 2 to 6);
- the largest entropy of a half (6 punctures) is 3.19 nats, below ln 132 = 4.88 and below 6 ln 2 = 4.16.

So the corner state has an area-law-like structure in the puncture graph: more links cut, more entropy, on top of a term proportional to the size of the region. The paper's entropy has the same shape, an area term plus a logarithm (S = <A>/4G + log Z). The comparison is only of form: here "area" of a region is the number of punctures or cut links, with no G_N, and the fit constants carry no physical normalisation. **Not done:** repeating this at level k or with other spins, and asking whether the coefficient per cut edge is fixed by symmetry.

### C2. The same checks at spin 1, and sectors of mixed spin (Exact numerics)

`null_cone_spin1.py` (about 6 minutes) repeats section 3A and 3C with spin-1 punctures.

- **Area sectors (Counted).** Fixing which punctures carry spin 1 and which carry spin 1/2 fixes the area sector, as in the paper's decomposition of H_D by area. The singlet space of one such assignment, by number a of spin-1 punctures out of 12, has dimension 132, 207, 352, 628, 1158, 2188, 4213 for a = 0, 2, 4, ..., 12 (odd a gives 0, since the number of half-integer spins must be even). Counting every labelled assignment, the sectors hold 132, 13662, 174240, 580272, 573210, 144408, 4213 states (a = 0, 2, 4, 6, 8, 10, 12), so the space of mixed assignments is far larger than either pure case. The all-spin-1 value 4213 is the Riordan number; the all-spin-1/2 value is the Catalan number 132.
- **Ground state (Exact).** The Heisenberg ground state of 12 spin-1 punctures on the 30 icosahedron edges, restricted to total spin 0, has energy -18.561 and S_tot^2 = 0 to rounding error.
- **Entropy fit (Exact).** The same fit as for spin 1/2, S = a (edges cut) + b (punctures in R) + c over all 2047 regions, gives 0.092 per cut edge, 0.352 per puncture, constant 0.828, R^2 = 0.942 against 0.831 for size alone. At spin 1/2 the numbers were 0.077, 0.234, 0.284, R^2 = 0.963 against 0.812. The largest half entropy is 4.64 nats, below ln 4213 = 8.35 and below 6 ln 3 = 6.59.

So the pattern holds at the second spin: entropy tracks the boundary of the region and the coefficient per cut edge changes only modestly (0.077 to 0.092). This is two data points, so I cannot say whether the per-edge coefficient is fixed by symmetry or just varies slowly. The per-puncture coefficient grows more (0.23 to 0.35), as expected if it counts internal dimension. Whether the per-edge term is the analogue of the paper's area term is still the Hypothesis of 3C.

### C3. Level 2: the entropy is close to a pure boundary law (Exact)

`null_cone_level2.py` (about 8 seconds) does the same count for the level-2 model of Stage 9c, the free Majorana model with one Majorana per cone point and signed adjacency a, a^2 = -5.

- **Correlations stop at nearest neighbours (Exact).** Because a^2 = -5, the matrix a/sqrt5 is a complex structure, and the ground state is the Gaussian state with covariance <i gamma_p gamma_q> = a_pq / sqrt5. The two-point function is 1/sqrt5 on the 30 edges and exactly zero between cone points that are not neighbours.
- **Entropy of a region (Exact).** From the covariance restricted to the region, over all 2047 regions: S = 0.068 (edges cut) + 0.028 (punctures in R) - 0.012 with R^2 = 0.995, against R^2 = 0.495 for the number of punctures alone. The cut alone, S = 0.075 (edges cut), already has R^2 = 0.982. At a fixed region size the correlation of S with the number of edges cut is 0.996 to 1.000. One cone point has S = (1/2) ln 2; two that are not neighbours have ln 2 exactly, two neighbours 0.590. The largest entropy of a half is 1.52 nats, below ln 32 = 3.47.
- **A convention.** A region with an odd number of Majoranas has no Hilbert space of its own (its algebra has a centre). I used the usual Gaussian formula, which gives such a region (1/2) ln 2 for the unpaired mode, the logarithm of the quantum dimension sqrt2 of one Ising anyon. Restricted to the 1023 regions of even size the fit is 0.068 per cut edge, 0.025 per puncture, R^2 = 0.995, so the conclusion does not rest on the convention.

Set beside 3C and 3C2, the three ground states give 0.068 (level 2), 0.077 (spin 1/2, level infinity) and 0.092 (spin 1) per cut edge, and 0.028, 0.234 and 0.352 per puncture. So the per-edge coefficient is not fixed by symmetry: it varies by about a third over the three cases. What changes most is the volume term, which nearly vanishes at level 2, where the state is a free state with nearest-neighbour correlations only. Of the three, the level-2 state is the one whose entropy has the form of an area law. Levels 3 and up are in 3C4.

### C4. Levels 3 and up: the volume term grows smoothly with the level (Exact numerics on 75 regions)

`anyon_entropy.py` (about 100 seconds) computes the entropy of a region at any level. Above level 2 the state space is not a tensor product, so the entropy is the anyonic one: the operators of a disc A form a Temperley-Lieb algebra, the space splits by the total charge c of A, and S(A) = sum_c p_c [ -ln p_c + S(rho_c) + ln d_c ] with d_c the quantum dimension (**Known**, the entropy defined with the quantum trace; I used the formula from memory and did not re-read a source, and it is tested at both ends below).

- **Regions.** A set of punctures needs a disc, and discs differ by braids. The unambiguous ones are neighbourhoods of arcs of consecutive punctures along the Hamiltonian cycle that orders them. Two cycles give 75 distinct bipartitions (57 arise from both, and agree to 1e-10). Sets that are not such arcs are not treated.
- **Tests (Exact).** At level 2 the result equals the Majorana formula of 3C3 on every arc, including (1/2) ln 2 for an odd arc, which here is not a convention but ln d of the charge-1/2 sector. In the classical limit it equals the ordinary entropy of the spins, computed independently in the 4096-dimensional space. An arc and its complement, computed separately, agree at every level. One puncture has S = ln d_k exactly.
- **Result.** Fit S = a (edges cut) + b (punctures) + c on the 75 arcs:

      level        2      3      4      5      6      8      10     20     50     classical
      per edge     0.069  0.099  0.114  0.122  0.126  0.130  0.131  0.128  0.119  0.113
      per puncture 0.025  0.042  0.057  0.068  0.078  0.092  0.103  0.137  0.183  0.216
      R^2          0.998  0.997  0.997  0.996  0.996  0.995  0.995  0.993  0.992  0.992
      largest half 1.35   1.96   2.29   2.48   2.60   2.74   2.82   2.93   3.01   3.05

  The per-puncture term grows smoothly and monotonically from level 2 to the classical limit, and slowly: it is still 0.18 at level 50 against 0.22. The per-edge term rises to about 0.13 near level 10 and then falls back a little. So the near-area law of level 2 is the end of a smooth family, not a special feature of free fermions, and the volume term is controlled by the level.
- **Caveat.** These coefficients are for arcs only and are not comparable with those of 3C, 3C2 and 3C3, which use all 2047 regions (classical: 0.113 and 0.216 on arcs, 0.077 and 0.234 on all regions). On arcs the number of cut edges and the size are strongly tied, so the split between the two terms is less well determined than the trend with the level.

### C5. An area operator in the centre: it exists, and it does not carry the boundary law (Exact numerics; negative)

The anyonic entropy of 3C4 has the form

    S(A) = <L_A> + S_bulk,     L_A = sum_c ln d_c P_c,     S_bulk = sum_c p_c [ -ln p_c + S(rho_c) ],

where P_c projects on total charge c of the region. L_A lies in the centre of the region's algebra and the space is a direct sum over its eigenvalues. This is the structure of the paper's H_D = direct sum over area sectors with S = <A>/4G + (rest), and of entropy formulas with an area operator in the centre of an algebra (**Known** in outline). So the model has a natural candidate for the area operator of a region: the logarithm of the quantum dimension of its total charge. `anyon_entropy.py` fits the two parts separately on the 75 arcs:

      level                 2      3      4      5      6      10     20     50     classical
      share of <L> in S     0.31   0.39   0.43   0.45   0.47   0.49   0.50   0.50   0.50
      R^2 of <L> fit        0.03   0.02   0.00   0.05   0.13   0.37   0.52   0.58   0.60
      R^2 of S_bulk fit     0.71   0.90   0.93   0.94   0.94   0.95   0.95   0.95   0.96
      S_bulk per cut edge   0.078  0.106  0.112  0.111  0.108  0.097  0.083  0.069  0.061

- <L> is a third to a half of the entropy, but it is not a function of the cut and the size: the linear fit explains almost nothing of it at low level and 60% classically. At level 2 it is exactly (1/2) ln 2 for a region with an odd number of punctures and 0 for an even one, because charges 0 and 1 both have d = 1.
- The dependence on the cut sits in S_bulk, the distribution over charges and the entanglement inside each charge sector.
- So the hoped-for reading "area term = central operator, proportional to the boundary" fails here: the central operator is there, and the boundary law is real (3C4), but they are different parts of the entropy. **Negative.** The total S, not <L>, is what follows the cut.
- Scope: arcs only, one Hamiltonian (nearest-neighbour), ground state only.

### D. The ground state is not a coherent polyhedron (Exact numerics; negative)

The standard classical picture of 12 equal-area faces with normals along the dodecahedron has a quantum counterpart, the coherent intertwiner: a product of spin-1/2 coherent states along the 12 normals, projected to the singlets (**Known**, Livine and Speziale). I computed its fidelity with the exact ground state: |<gs|coherent>|^2 / |P coherent|^2 = 6.5e-5, against 0.0076 for the uniform average over the 132 singlets. It is essentially orthogonal. Over 200 random sets of 12 directions the mean is 0.0074 and the maximum 0.114, and over all 4096 sign patterns of the dodecahedron normals the best is 0.036. So at spin-1/2 the Heisenberg ground state is not close to the regular-dodecahedron geometry, nor to any single coherent state tried. That is expected for spin-1/2, where nothing is semiclassical, and it says that the puncture model reaches the paper's semiclassical regime (Section 6) only for large spins or large level. **Spin 1** (`null_cone_spin1_fidelity.py`, spin-1 coherent states as the symmetric square of the spin-1/2 ones): the fidelity of the dodecahedron state with the spin-1 singlet ground state is 1.1e-4, against a uniform baseline of 1/4213 = 2.4e-4 and a random-direction mean of 4.4e-4 (30 draws, maximum 6.7e-3); the best of 256 sampled sign patterns (one in sixteen of the 4096) is 1.4e-3. So the dodecahedron is again at or below chance and nowhere near the ground state. Doubling the spin has not moved the ground state toward the classical geometry; I expect that this needs spins much larger than 1, or a Hamiltonian that favours it, and I have not tested that. **Not done:** spins above 1, and all 4096 sign patterns at spin 1.

### E. Boosts, area quantisation and compact angles (Hypothesis; I have not checked how the paper treats it)

As summarised, boosts shift the corner area by 8 pi G_N hbar k, so K = A/(8 pi G_N) has integer-spaced spectrum in units of hbar. If so, the boost angle conjugate to K is periodic, as an angle conjugate to an integer-valued charge is. In this repository the analogous statements are: the level k fixes q = e^{i pi/(k+2)}, a root of unity, and complex conjugation of the level-k operators (A -> 1/A) is the time-reversal-odd choice that the tiling does not make (`HAT_TIME.md`). A common structure would be: a discrete charge, a compact conjugate angle, and a complex phase that is odd under time reversal. I have no map between k here and the paper's integer k, and it is possible they are unrelated. **Hypothesis only.**

### F. Light-cone caustics and sphere eversion (Hypothesis, with a precise geometric kernel)

The paper's example of a Schwarzschild photon sphere has circular null geodesics meeting at the antipode and uses a second diamond to continue past the caustic (as summarised). A spherical wavefront that shrinks to a focus and re-expands has passed through the antipodal map with its normal reversed: a point on the sphere goes to its antipode and inside and outside are exchanged. Passing through a point focus gives the wave a phase of pi (the Gouy phase, **Known**).

The `ray` fork's sphere eversion (Bednorz and Bednorz, `docs/EVERSION.md` there) ends at exactly this pair of states: an embedded sphere at t = -1.5 that is the antipodal image of the sphere at t = +1.5, with the parametrisation normal reversed relative to the geometric outside. The difference is the path. A focusing wavefront reaches the end state through a singular point (the whole sphere collapses to a point), so it is not a regular homotopy; an eversion reaches it as a regular homotopy through immersions with double curves, with no point singularity (Smale). So the two share endpoints and differ in route, and the Gouy phase is a property of the singular route. **Hypothesis:** a regular route would carry a different phase, and whether the quantum amplitude knows the difference is open. This is the same question as the conjecture in `ray`'s `QWLA_CONNECTIONS.md`, section 5, in a more physical setting, and like that one it is not supported by an argument.

### G. Closed amplitudes as traces and the tiers (Not done)

The paper's closed amplitudes are traces on H_D (finite because dim H_D is finite), and its gluing law sums over intermediate states along a cut. The nested tiers of Stage 4i give a concrete sequence of regions with finite Hilbert spaces (6, 82 and 804 pair qubits) that glue region to region. A test of the gluing law in that model, a sum over a complete basis on a cut reproducing the amplitude of the bigger region, is the natural finite check; it needs a Hamiltonian that evolves the regions and is not built here.

## 4. What I would do next

In order of cost: (1) regions that are not arcs at levels 3 and up, which need a choice of disc (arcs are done, section 3C4); (2) fidelity of the coherent dodecahedron above spin 1 (3D); (3) a ground state or a thermal state on the mixed-spin sectors of 3C2, as the model of H_D = direct sum of area sectors (3A); (4) read the paper's sections 5.1 to 5.3 in full to see whether the area spectrum can be pinned to a puncture picture at all, since everything in 3A and 3E depends on that.

## 5. What this note does not show

- No part of the paper has been verified here, and I read it second hand.
- No map from the paper's Hilbert space to the puncture model is known; sections 3A, 3B and 3E are the hoped-for connections, not found ones.
- Sections 3C and 3D are computations on this repository's model alone. They say nothing about gravity unless the puncture identification of 3A holds.
- The Hat Trick ball has no spin, no metric signature and no null cone. Its role here is only that its 12 cone points give a small, exactly solvable punctured sphere.

## Reproduce

    python3 null_cone_checks.py --tests        # Gauss-Bonnet sum, Catalan constant, coherent fidelity, entropy table (about 10 s)
    python3 null_cone_spin1_fidelity.py        # spin-1 coherent fidelity (about 15 min)
    python3 null_cone_level2.py --tests        # level-2 Majorana ground state: covariance and entropy fit (about 8 s)
    python3 anyon_entropy.py --tests           # anyonic entropy of arcs at levels 2 to 50 and classical (about 100 s)
    python3 null_cone_spin1.py                 # spin-1 sectors and entropy fit (about 6 min; --tests runs only the counting tests)
