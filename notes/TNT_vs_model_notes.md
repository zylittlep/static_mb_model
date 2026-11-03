# TNT vs TNT3c: does APL's connectivity explain the change?

2026-10-15 (per-KC KC→APL counts restricted to the calyx on 2026-10-21) · official ROIs only · figure `TNT_vs_model.png`
· script `../scripts/tnt_compare.py` · model from `APL_block_model_notes.md`. TNT3c is the control with APL intact
(confirmed by Yang, 2026-10-15), and TNT has APL blocked. Since 2026-10-21 the model uses hemibrain and male CNS only, as
Yang asked: 3 mushroom bodies (hemibrain R, male CNS L and R). The four-connectome version (7 mushroom bodies, with
FlyWire and BANC) is in `archive_2026-10-21_four_connectomes/`.

**Short answer.** Partly, and through APL's feedback loop. The test model has a uniform APL and is pinned to two things
from the data: how many units respond to each odor in TNT, and TNT3c's overall level. It predicts which odor pairs gain
the most similarity: r = 0.46 with ORN inputs and 0.32 with GH146. That is about 30% of the pattern that replicates
across recordings. Adding APL's specific wiring changes the fit little:
- per-KC APL synapse counts in the calyx: r +0.02 with ORN inputs, +0.00 with GH146;
- APL→PN bouton synapses at their real share: r +0.01 with ORN inputs, +0.03 with GH146.

Neither beats the same wiring shuffled (p = 0.07 to 0.36). With all four connectomes both did with GH146 inputs, so that
result does not hold with hemibrain and male CNS alone. The model does not predict which odors lose the most responding
units when APL works, and its similarity change is about half the data's.

## What the data show (panels a–d)
- In TNT, 15.0% of units respond (> 1) to an odor on average. In TNT3c it is 6.8%. Per odor, TNT has ×1.65 (euc) to
  ×2.90 (eb) as many.
- All 120 odor pairs are more similar in TNT. Mean cosine similarity rises from 0.16 to 0.34.
- The change is uneven:
  - 65% of its variation across pairs is per odor. Pairs that include eb, 2h, IaA or pa rise most; pairs with -aPine or
    euc rise least.
    - How (Yang asked, 2026-10-21): each pair's change is fit as ΔS_ab = μ + f_a + f_b by least squares over the 120
      pairs, with the f summing to 0, so μ is the mean change (+0.18). f_a is odor a's per-odor effect, how much more its
      pairs rise than the average pair: eb +0.050, 2h +0.046, IaA +0.042, pa +0.022, euc −0.062, -aPine −0.066. 65% is
      the R² of this fit; its residuals are the pair-specific part. Values in `TNT_vs_model_per_odor.csv`
      (`odor_effect_dS_data`).
  - The rest is pair-specific and about half reliable across recordings (split-half estimate 0.55).
  - Pairs that were already similar in TNT3c rise a little less (r −0.13).
- Counting only responses above 1, as in the model's output, gives the same picture. Similarity of the 0/1 responses
  rises from 0.145 to 0.289, every pair still rises, and the pattern of the change correlates 0.94 with the raw one.

## Uniform APL feedback (panels e–h)
- **Odor pairs.** Data and model changes correlate with r = 0.46 over odor pairs with ORN inputs (odor-label permutation
  p = 0.001). Per mushroom body r is 0.44 (hemibrain R), 0.27 (male CNS L) and 0.53 (male CNS R). With GH146 inputs
  (13 odors) r = 0.32 (p = 0.02). Counting units > 0.5 as responding gives 0.31 and 0.13.
- **Ceiling.** The data's split-half reliability caps any model near r = 0.84. The uniform model therefore captures about
  (0.46 / 0.84)² ≈ 30% of the reproducible pattern.
- **Per-odor part** (panel g): ρ = 0.52 with ORN inputs, 0.27 with GH146. This is the Spearman correlation over odors
  between the data's f_a and the same fit on the model's change, blocked − intact (`odor_effect_dS_model_U`).
  - Right: -aPine changes least in both, euc is low in both, and 2h is in the top three of both.
  - Wrong: IaA (third highest in the data, third lowest in the model), eb (highest in the data, middling in the model)
    and 2-but (highest in the model, middling in the data).
- **Pair-specific part.** Data and model residuals correlate with r = 0.35. The model also has already-similar pairs
  rising less.
- **Size.** The model's change is +0.08, vs +0.18 in the data, or +0.10 to +0.14 when the data are thresholded like the
  model's output (panel l).
- **Suppression per odor** (panel h): ρ = 0.18 with ORN inputs and −0.04 with GH146. The data's own ranking is only
  moderately reliable (split-half estimate 0.48), but the model also misses clear cases. 6al and B-myr are among the
  most suppressed in the data (×2.47 and ×2.46; bootstrap 90% ranges from ×1.96) and among the least in the model.
  2h is the reverse: ×2.07 in the data, the model's second highest.

## APL's specific wiring (panels i–k)
Every version gets the same pinning and calibration. Each null shuffles that version's wiring 200 times per mushroom body
and keeps everything else.
- **Per-KC APL counts (C).** Each KC's APL→KC and KC→APL synapses in the calyx. r = 0.48 with ORN inputs and 0.32 with
  GH146, vs 0.46 and 0.32 for uniform APL. C is above uniform in 68% and 52% of bootstrap draws.
  - Shuffling the counts among KCs of the same class does worse than uniform: mean r 0.40 with ORN, 0.22 with GH146.
  - The real counts beat the shuffles with p = 0.11 (ORN) and p = 0.065 (GH146). With all four connectomes it was
    p = 0.08 and 0.005.
  - Until 2026-10-21 the KC→APL counts covered the whole MB (four connectomes then): r 0.49 and 0.33, p 0.10 and 0.005.
    Yang asked for the calyx only; the earlier outputs are in `archive_2026-10-21_KC_to_APL_whole_MB/`.
  - The real counts give more APL→KC synapses to KCs with more claws: within-class Spearman +0.44 to +0.51 in α′β′ KCs
    and +0.19 to +0.32 in γ KCs (αβ KCs are mixed, −0.04 to +0.26). Shuffling destroys this link.
- **Boutons at their real share (M).** APL→PN synapses carry 10–11% of the inhibition, weighted per glomerulus.
  - r = 0.47 with ORN inputs and 0.34 with GH146. M beats uniform APL in 86% and >99% of bootstrap draws, but by little
    (+0.01 and +0.03).
  - Against glomerulus weights shuffled across glomeruli: p = 0.36 (ORN) and p = 0.21 (GH146). With all four
    connectomes it was p = 0.13 and 0.015.
- **All inhibition on boutons (P).** r = 0.49 and 0.33, the highest with ORN inputs, but no better than shuffled
  weights (p = 0.34 for both).
- **On claws (L)** and **feed-forward PN→APL drive (F0, F)** fit worse than uniform feedback: r = 0.24 for L and
  0.34–0.38 for F0/F with ORN inputs. F is the version where APL activity follows each glomerulus's PN→APL synapses.
- **Suppression per odor.** No version predicts it: ρ ≤ 0.26 with ORN inputs and ≤ 0.23 with GH146. The one
  real-vs-shuffle difference there (F with GH146, p = 0.005) still has only ρ = 0.23.
- **Multiple tests.** Of the 8 real-vs-shuffle tests on odor pairs, one passes at p < 0.05: F with ORN inputs
  (p = 0.025), which still fits worse than uniform feedback. Next is C with GH146 inputs (p = 0.065). With all four
  connectomes, C and M with GH146 inputs passed at p < 0.02.

## What it means
- **Feedback.** Most of the unevenness is what one global feedback inhibitor does to odors of different strength and
  overlap. Odors that drive few KCs, like -aPine and euc, recruit little APL, lose fewer units, and gain the least
  similarity when it is blocked, in both the model and the data.
  - An inhibition of fixed size, the same for every odor, fits worse: r = 0.30 vs 0.46 with ORN inputs. Why feedback
    makes the effect uneven is explained in `APL_block_model_notes.md`.
- **Specific wiring.** APL's KC-specific and glomerulus-specific wiring is reproducible across connectomes (more APL→KC
  synapses on multi-claw KCs, a glomerular pattern at PN boutons). With hemibrain and male CNS it does not fit the data
  better than the same wiring shuffled. With all four connectomes, two features improved the fit slightly with GH146
  inputs. Either way, neither is needed for the main pattern.
- **Not explained:** which odors are suppressed most (eb ×2.9 vs euc ×1.65), and the size of the change.

## Rank order of similarity (Yang's question, 2026-10-16)
Script `../scripts/tnt_rank_order.py`; numbers in `TNT_rank_order.csv`.
- **In principle.** Uniform inhibition keeps the rank order exactly only in special cases. One is inhibition that just
  scales every KC's response to an odor, which leaves cosine similarity unchanged. The other is a single threshold raise
  when all odors drive KCs with the same statistics; similarity then depends only on input correlation. Real odors differ
  in strength and in how their overlap is spread across weakly and strongly driven KCs, so some pairs swap places.
  - **Toy simulations of these cases** (Yang asked, 2026-10-21): `rank_order_toy_simulation.png`, script
    `../scripts/rank_order_toy_sim.py`, numbers in `rank_order_toy_simulation.json` and
    `rank_order_toy_simulation_pairs.csv`. No PN responses or connectome: 1,000,000 model KCs get random inputs for 16
    odors, which overlap through shared random inputs. 20% of KCs respond to an odor with APL blocked and 10% with it
    intact. A swap is two odor pairs whose order flips (7,140 comparisons between the 120 pairs).
    - Scaling (1): every KC's response to odor o is divided by the same number, s_o = 1 + a·A_o. A_o is the odor's APL
      activity: the mean over KCs of the odor's response with APL intact, the same for every KC. It is solved
      self-consistently, because APL activity sets the responses and the responses set APL activity: A_o must equal the
      mean of the responses it produces, A_o = m_o / (1 + a·A_o), where m_o is the odor's mean response with APL
      blocked, so A_o = (√(1 + 4a·m_o) − 1) / (2a). a is one gain for all odors and all KCs, set so that each odor's
      intact / blocked ratio (1/s_o) averages 0.5 over the 16 odors (Yang asked, 2026-10-23). Strong odors drive APL
      more, so s_o runs from 1.4 for the weakest odor to 2.9 for the strongest, and the mean response over all KCs and
      odors falls to 0.44 of its blocked value, a little less than half. No KC stops responding. Cosine ignores the length of each odor's response
      vector, cos(y_a/s_a, y_b/s_b) = cos(y_a, y_b), so every similarity is unchanged (to 10⁻¹⁴) and the order is kept
      exactly.
      - A factor per unit instead (Yang asked, 2026-10-22; numbers under "1 with a factor per KC" in the JSON, not in
        the figure): KC k's responses to every odor are divided by its own π_k. The cosine then weights each KC by
        1/π_k², so what happens depends on how π_k relates to the responses.
        - π_k random, in the same range (1.4 to 2.9): every similarity moves by at most 0.0014 and the order is kept
          (Spearman 1.000; 16 comparisons swap, all between pairs less than 0.002 apart).
        - π_k growing with the number of odors KC k responds to (1 + 0.2 × that number, so 1.2 to 4.2): every pair
          loses similarity (−0.04 to −0.10, mean −0.07). A KC that adds to a pair's overlap responds to at least those
          two odors, so it is divided more than the average responder. The order changes a little: Spearman 0.995,
          2.7% of comparisons swap.
        - A random factor per KC that also grows with the odor's APL activity, 1 + a·u_k·A_o (u_k with mean 1 and CV
          0.5), keeps the order too: Spearman 1.000, every similarity within 0.003.
    - One threshold raise, odors alike (2): at each threshold all pairs sit on one curve of similarity against input
      correlation, so the order is kept. Spearman 1.000; 26 comparisons (0.4%) swap, all between pairs less than 0.003
      apart, which is simulation noise.
    - Odors that differ in strength (3a; each odor's inputs scaled ×0.65 to ×1.5): the same raise leaves the strongest
      odor 67% of its responders and the weakest 22%, so pairs of weak odors lose more similarity (−0.108 vs −0.080 on
      average). Spearman 0.976, 6.1% of comparisons swap; with feedback instead of one raise, 0.956 and 9.0%.
    - Overlap in strongly driven KCs (3b; odors alike in strength, but odors 9–16 share a sparse input that is large in
      a few KCs and zero in the rest): every odor keeps the same share of its own response (37% of its sum of squares),
      so similarity follows the overlap, the sum over KCs of the product of the two responses. A KC driven hard by
      both odors keeps most of its product; one driven hard by only one odor loses it. Odors 10 and 11 keep 40% of
      their overlap and go from 0.345 to 0.372, while odors 1 and 8 keep 28% and drop from 0.417 to 0.317. Spearman
      0.980, 5.6% swap, and 11 pairs become more similar with APL intact.
- **Model.** The swaps are few. Spearman between blocked and intact similarity over the odor pairs is 0.98 (0.97–0.98
  across the 3 mushroom bodies, ORN or GH146 inputs), for uniform feedback and for a fixed threshold raise alike.
- **Data.** Spearman between TNT and TNT3c similarity is 0.87.
  - With half the recordings, TNT vs TNT3c gives 0.83, while two halves of the same condition give 0.90. The
    between-condition value is lower in 97% of splits, about 0.92 after correcting for noise.
  - Counting only responses > 1 (0/1) weakens this: 0.85 vs 0.88, lower in 81% of splits.
  - So TNT reorders odor pairs more than uniform inhibition would.

## Data checks: double counting
Script `../scripts/tnt_data_checks.py`; results in `TNT_data_checks.csv` and `TNT_data_checks.json`.
- **Identical units.** No unit is identical to another unit in its recording, and no unit appears in two recordings.
- **Near-duplicate units.** For each unit, its nearest neighbour in its own recording was compared with its nearest
  neighbours in the other recordings of the same condition, with the same number of candidates. A duplicated cell would
  have an unusually close partner in its own recording.
  - Own-recording partners closer than 99% of other-recording ones: 2.6% of units in TNT, 3.9% in TNT3c.
  - Among responsive units (max > 1), the same check flags 0.5% in TNT and 1.6% in TNT3c.
  - These rates are low and similar between conditions. They cannot rule out duplicates without ROI positions or
    traces. Extra duplicates in TNT3c would, if anything, raise TNT3c similarity and shrink the measured change.
- **Two hemispheres of one fly.** TNT has one fly with both sides recorded (20240814_3, 2 of 8 recordings). TNT3c has
  three (20240710_2, 20240801_3, 20240809_3, 6 of 11 recordings). So treating recordings as independent counts
  some flies twice, unevenly between conditions.
  - In TNT3c, two of the three same-fly pairs have more similar odor-similarity structure (r 0.94, 0.93) than any
    pair of different flies (median 0.85).
  - Weighting each fly once instead barely changes anything. The mean change is 0.182 instead of 0.178, the pair
    pattern correlates 0.99 with the original, and model fits move by at most 0.004 (U 0.460, M 0.468). Per-odor
    suppression ranks correlate 0.97 between the two weightings.
  - The double counting therefore matters only for the error bars, which are slightly too narrow for TNT3c.

## Caveats
- **Recordings as samples.** Each recording is treated as an independent hemisphere, as Yang asked (2026-10-15). The
  bootstrap and split-half tests resample recordings. The label 20240802_5 in both files names two different flies.
- **Connectomes.** Only 3 mushroom bodies from 2 connectomes. Their uniform-model fits range from 0.27 to 0.53, so the
  wiring tests, which average over them, have little power.
- **Model inputs.** The inputs are ORN (or GH146) means from other flies, not these flies' PNs. Imaged units are not
  identified KCs, and one threshold is used for all KCs.
- **Noise.** Data similarity includes measurement noise, which lowers it more when responses are smaller (TNT3c). Some of
  the data's larger change may therefore be noise.
- **Odor count.** 16 odors (13 with GH146) leave little room for per-odor tests.

## Files
All in this folder:
- `TNT_vs_model_fits.csv`: one row per threshold, input and APL version. Columns: r over odor pairs with bootstrap
  range and odor-permutation p, difference from U, suppression ρ, and the shuffle nulls.
- `TNT_vs_model_nulls.csv`: r and ρ for the real wiring (shuffle 0) and for each shuffle.
- `TNT_vs_model_odor_pairs.csv`: data similarity and change, per-odor fit and pair-specific part, and the model change
  per version.
- `TNT_vs_model_per_odor.csv`: fraction responding, suppression with bootstrap range, per-odor effects, and model
  suppression per version.
- `TNT_vs_model_summary.json`.
- `TNT_data_checks.csv`, `TNT_data_checks.json`: the double-counting checks (script `../scripts/tnt_data_checks.py`).
- `TNT_rank_order.csv`: rank-order comparison of similarity, data and model (script `../scripts/tnt_rank_order.py`).
- `rank_order_toy_simulation.png`, `rank_order_toy_simulation.json`, `rank_order_toy_simulation_pairs.csv`: the toy
  simulations of the three rank-order cases (script `../scripts/rank_order_toy_sim.py`).
- `TNT_feedback_vs_fixed.csv`, `TNT_feedback_vs_fixed.json`: feedback vs inhibition of fixed size and other per-odor
  rules, and how the data's per-odor gains relate to odor strength (script `../scripts/tnt_feedback_vs_fixed.py`).
- `archive_2026-10-21_four_connectomes/`: the same outputs and notes with all four connectomes (7 mushroom bodies).
