# Does the geometry of the seams see `chi4`?

[`HAT_TIME.md`](HAT_TIME.md) found that the hand of the hat first appears in the ring bits as the 4-ring term `chi4`, and `chi4_operator.py` turned it into an operator. All of that is about the *combinatorics* of the tilings. This note asks the geometric question: if each of the 4096 tilings is drawn on the sphere, is the mirror asymmetry of the picture related to `chi4`? The answer measured here is no, with the caveat that the measure used is coarse. Script: `hat_seam_chirality.py`.

Labels as in the roadmap: **Exact**, **Counted**, **Known**, **Definition**, **Not done**.

## The measures (Definitions)

For a ring pattern `b`, `N(b)` is the seam network of its tiling: the outline of every hat plus the 12 pentagon loops, as points on the unit sphere (spacing about 0.01 radian). With the improper rotation `s` of `hat_chirality.py` acting on the bits:

- `M(b)`: the mean nearest-point distance, in degrees, from `N(b)` to its own mirror image, minimised over rotations (8 random starts of Nelder-Mead, 1200 sampled points). `M = 0` for a mirror-symmetric network.
- `D(b)`: the same between `N(b)` and the mirror image of `N(s.b)`. `D = 0` would mean the tiling of `s.b` is exactly the mirror image of the tiling of `b`, so `D` is the geometric version of the question in `hat_chirality.py` (does `T(s.b)` equal the mirror image of `T(b)`?). It is even under `b <-> s.b`, as `|chi4|` is. For an achiral pattern `N(s.b)` is a rotation of `N(b)` and `D = M`.

## Results

**Counted** (`python3 hat_seam_chirality.py 300 --tests`, 300 random patterns, about 10 minutes):

- The standard tiling has `M` = 0.99 degrees. This is the "nearly achiral network" figure (0.96 in the `ray` fork's measurement of the same network).
- Random patterns are much further from mirror symmetric: `M` has mean 2.30 degrees and `D` mean 2.34, and one flipped ring already gives about 2.6. The seam network is nearly mirror symmetric only at the symmetric tiling.
- Both measures follow the number of flipped rings (Spearman 0.31 for `M`, 0.28 for `D`, p below 1e-6) and not `chi4`: the Spearman correlation with `|chi4|` is 0.043 for `M` (p = 0.46) and 0.049 for `D` (p = 0.40). A run of the `ray` fork's version of the same test (300 patterns, independent random starts) gave 0.04 (p = 0.52) for `D`; an earlier unsigned test on 120 patterns gave 0.18 (p = 0.047) and 0.17 (p = 0.057) after removing the mean per flip count, which the larger samples do not reproduce.
- Checks that pass in every run: `chi4(s.b) = -chi4(b)`, and `chi4` is non-zero only on chiral patterns (never on the 218 achiral ones in the `ray` run; 0 in this run too).

## What this does and does not show

- The seam network's mirror distance does not track `chi4` on these samples. The hat's hand is not visible to this measure as the order-4 term.
- That does not show that the hand is invisible in the geometry. The mean nearest-point distance is dominated by the large-scale shape of the network (a flipped ring changes the outline of five hats), and `chi4` is a small effect that, by `HAT_TIME.md`, appears only in counts out to third neighbours. A **local** measure is the fair test and is **Not done**: for every pentagon, the multiset of shapes of the hats around it, compared with the mirror multiset, summed with the signs of the helical paths.
- The sample is 300 of 4096 patterns, and the minimisation is a random search, so `M` and `D` carry a spread of a few hundredths of a degree between runs.

## Reproduce

    HEXASPHERE_DIR=/path/to/hexasphere python3 hat_seam_chirality.py 120 --tests    # about 4 minutes
