# Horned trees on the Hat Trick ball

The 12 cone points of the Hat Trick ball ([`HAT_TIME.md`](HAT_TIME.md)) are also the places where a horned sphere can hang its children. The `ray` fork ([`brentharts/ray`](https://github.com/brentharts/ray), `fractalkit/hattrick_horned.py`) builds such trees: a copy of the ball's seam network, `shrink` times smaller, is hung on the cone points, and then again on the copy. This note records what was measured there that touches this repository. The details, scripts and figures are in that repository's `docs/HAT_TRICK_HORN_SITES.md`; the numbers below are copied from it.

Labels as in the roadmap: **Exact**, **Counted**, **Known**, **Definition**, **Not done**.

## What the construction is

A seam tree takes the outlines of the ball's 72 panels (60 hats and 12 pentagons) as a network of curves on the unit sphere, and hangs scaled copies on chosen cone points. The copy's own cone point 0 is turned back onto the parent's, so the copy is attached by the point it shares with its parent. The seams are drawn as thin tubes. A copy *clasps* its sibling when some panel loop of one links some panel loop of the other (Gauss linking number, +1 or -1). This is the property of Alexander's horned sphere that the construction is imitating: the two children hang on the two ends of a ring and link each other.

The ring bits are not involved: the trees use the standard tiling's outlines, and a ring flip changes the panels but not the 12 cone points.

## Results (all **Counted**)

**All 12 cone points** (12 children on the root, 11 on every later ball; `shrink` 0.3, tube 0.025): the similarity dimension is log 11 / log(1/0.3) = 1.99, and the box-counting dimension of the seams over three generations is 1.93. 60 pairs of tubes fuse at depth 2.

**Two children per ball.** Every ball has exactly 2 children, so the balls per generation are 1, 2, 4, 8 and the similarity dimension is log 2 / log(1/`shrink`) (1.94 at `shrink` 0.7).

- *Children on an antipodal pair of cone points* (the 6 clasp pairs of [`HOPF_SITES.md`](HOPF_SITES.md)). The two children never link each other: 0 of 72 settings by pentagon loops, 0 of 8 by all 72 panel loops. They are about 2 units apart and much smaller than that, so there is nothing to link; the Hopf clasp of antipodal Bloch points is a statement about fibres in the 3-sphere, not about balls in space. A child does link its parent through the hat loops (10 to 30 linked loop pairs per child in the 8 settings tried), though not through the 12 pentagon loops. Tubes stay clear at depth 2 in 3 of 72 settings.
- *Children on a pair of adjacent cone points* (63 degrees apart). The siblings link from `shrink` about 0.6 up (18 of 32 settings per placement), but with the default tube radius the tubes of every linked pair fuse. With tube radius 0.004 or 0.008, 9 of 144 runs have clear tubes at depth 2 and a positive sibling gap; in all 9 the two children link at every generation down to depth 2 (the root, and both of its children). The best has a sibling gap of +0.010 and a depth-2 clearance of +0.005. Recomputed at a finer step, the Gauss integrals of the linked pairs checked are +1 or -1 to within 0.001. The margins are thin and depth 3 was not tested for clearance.

So a binary horned tree on the Hat Trick ball with clasping siblings exists to depth 2 with thin tubes, when the children hang on adjacent cone points.

## How this relates to the tiers

The nested tiers of Stage 4i (9 in 71 in 559 in 4401 tiles) and a horned tree are both recursive, but differently:

- **Exact (arithmetic).** The tile counts follow `N_k = 8 N_(k-1) - N_(k-2)` with ratio `lambda^2 = 4 + sqrt 15 = 7.873`, and the linear scale factor is `lambda`, so the similarity dimension of the tiers is log(`lambda^2`) / log(`lambda`) = 2: the tiers fill a region of the plane.
- A seam tree attaches its copies *outside* the parent. Its dimension is fractional (1.2, 1.99 or 1.94 in the cases above), and the copies can collide, which is what the fused tubes are.
- **Not done:** a nested variant of the tree (copies inside the parent, as the tiers are), and any way of using the tree's clasps as couplings between tiers.

## What this does not show

- No quantum quantity comes out of these trees. They are geometric statements about curves in space.
- The ring bits do not enter. Whether the number of linked loop pairs between a ball and its child depends on the tiling (a function of the 12 bits) is **Not done**; if it did, it would be a new observable on the 4096 tilings to compare with the vertex count `V = 80 - sum n_i n_j`.
- Alexander's horned sphere has a Cantor set of wild points. These trees are finite-depth stand-ins, and, as noted in Stage 9c, a Cantor set in a surface is tame; nothing here is a wild embedding.
