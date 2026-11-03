# Geodesic distance along APL between glomeruli's microglomeruli

2026-10-23 · Yang asked · hemibrain and male CNS only · script `../scripts/apl_geodesic.py` · this folder = official ROIs;
`../with_outside_ROI_synapses/apl_geodesic/` = the same with synapses outside the official ROIs added (numbers nearly the same)

**Short answer.** Measured along APL's skeleton, APL's synapses on the microglomeruli of two glomeruli are on average
150–230 µm apart, 6–10 times the straight-line distance (about 25 µm). Within one glomerulus they are closer (median 96,
205 and 121 µm in hemibrain R, male CNS L and R), so with any passive decay self-inhibition is stronger than
cross-glomerular inhibition. The pattern across glomerulus pairs is set mostly by how APL's calyx arbor is split into
sub-arbors, and it differs between the three APLs (Spearman 0.08–0.37 over glomerulus pairs).

## How it is computed
- **Skeletons.** Hemibrain v1.2 APL (425790257, 182,514 nodes, 80 mm of cable) and male CNS v1.0 APL_L (10977) and APL_R
  (10540) (about 45,000 nodes and 37 mm each; these are coarser, 0.7 µm median node spacing vs 0.4 µm). Files in
  `../data/skeletons/`. The few unconnected pieces were joined by their shortest bridges (at most 2.7 µm).
- **Microglomerulus.** One derived PN bouton (2 µm cluster of a PN's T-bars in the calyx) plus the KC claws whose main
  bouton it is. Its glomerulus is the bouton's PN's glomerulus (uniglomerular PN types only).
- **APL synapses on a microglomerulus.** APL↔PN synapses on the bouton (partner-side site within 2 µm) and APL↔KC synapses
  on those claws (KC-side site within 2 µm). Hemibrain R has 21,536 of them, male CNS L 14,023, male CNS R 12,973.
  Each is placed at the skeleton node nearest its APL-side site (median 0.4 µm away in hemibrain, 0.8 µm in male CNS).
- **Weighted mean distance between glomeruli A and B.** The mean of the geodesic distance over all pairs of synapses, one
  on A's microglomeruli and one on B's, each pair weighted by the product of the two weights. On a tree this is exact
  as the sum over skeleton edges of edge length × (share of A's weight beyond the edge × share of B's on this side + the
  reverse). Checked against shortest paths (Dijkstra) to 10⁻¹².
  - **All synapses** (top row of `APL_geodesic_heatmaps.png`): every synapse weight 1, both directions together.
  - **Inputs → outputs** (bottom row): pairs of an APL input at A (KC→APL, PN→APL) and an APL output at B (APL→KC,
    APL→PN). The pair weight is then input synapses × output synapses, Yang's product, made directional. The two
    versions are nearly the same; A → B and B → A differ by 6–12 µm (median).
  - The diagonal is the same glomerulus and includes pairs on the same microglomerulus.
- 53 glomeruli have ≥ 20 such synapses in every APL and are shown. The order is average-linkage clustering of the mean
  of the three APLs.

## What it shows
- **APL's calyx arbor comes in sub-arbors** (`APL_geodesic_subarbors.png`). Restricted to the calyx, APL's skeleton falls
  into a few connected pieces that join only through the peduncle or lobes. Two synapses in the same sub-arbor are
  48–69 µm apart on average, in different ones 215–356 µm.
  - Hemibrain R: one sub-arbor holds 81% of the synapses on microglomeruli. Male CNS L: six, the largest 34%. Male CNS R:
    three main ones, the largest 58%.
  - Sub-arbors cover different parts of the calyx but overlap, so even two APL synapses on the same microglomerulus are
    37–51 µm apart along APL (mean over microglomeruli; median 24–28 µm), and some are more than 200 µm apart.
- **Glomerulus pairs.** Median weighted distance 156 µm (hemibrain R), 230 µm (male CNS L), 155 µm (male CNS R). A
  glomerulus's distance to others mostly tracks how much of it sits in the main sub-arbor, which is why the heatmaps
  look like bands rather than blocks.
- **Self vs cross.** The diagonal is shorter than the glomerulus's mean distance to the others in 79%, 100% and 89% of
  glomeruli (median ratio 0.73, 0.90, 0.86).
- **Not conserved.** Over glomerulus pairs, Spearman 0.37 between hemibrain R and male CNS R, 0.31 with male CNS L, and
  0.08 between the two male CNS sides. The straight-line distance predicts the geodesic one only weakly (r 0.23–0.52).
- **Passive spread, illustration only** (`APL_geodesic_passive_spread.png`). If APL's signal fell off as
  exp(−distance / λ) along the skeleton, an input at a glomerulus would reach its own outputs this many times more
  strongly than the average other glomerulus's outputs: λ = 10 µm ×6.2–6.7, 25 µm ×2.2–2.8, 50 µm ×1.5–1.9, 100 µm
  ×1.2–1.4 (median over glomeruli). APL's length constant is not known, and this ignores branch loading and diameter.

## Caveats
- The sub-arbor split depends on the skeleton's topology. A connection missed by the segmentation (or a false merge)
  would change it, so the sub-arbors are worth checking in neuroglancer before relying on them. The male CNS skeletons
  are coarser than hemibrain's.
- Calyx nodes for the sub-arbor labels are nodes within 5 µm of a calyx APL synapse of the outside-ROI table, which covers
  the whole calyx in both versions. The labels only describe the arbor; the distances do not depend on them.
- Synapse counts stand in for synapse strength.

## Files
- `APL_geodesic_glomerulus_pairs.csv`: one row per APL × glomerulus A × glomerulus B. Columns: synapse counts, geodesic
  distance (all synapses; inputs at A → outputs at B; symmetric average), straight-line distance, passive spread for
  λ = 10, 25, 50, 100 µm.
- `APL_geodesic_microglomeruli.csv`: one row per microglomerulus (bouton id) × APL: glomerulus, APL inputs and outputs,
  PN and KC synapses, mean geodesic distance between its own APL synapses.
- `APL_geodesic_glomerulus_by_subarbor.csv`: share of each glomerulus's synapses in each sub-arbor.
- `APL_geodesic_summary.json`, `APL_geodesic_heatmaps.png`, `APL_geodesic_subarbors.png`, `APL_geodesic_checks.png`,
  `APL_geodesic_passive_spread.png`.
