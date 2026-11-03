# Can APL's connectivity explain the TNT / TNT3c nonuniformity?

2026-10-15 · official ROIs only · figure `APL_block_model.png` · method in `../README.md` ("APL block model") · script
`../scripts/apl_block_model.py` · updated 2026-10-21: similarity is now cosine, as in the TNT comparison (Pearson r
before), and the per-KC KC→APL counts are calyx only. Also since 2026-10-21, hemibrain and male CNS only (3 mushroom
bodies), as Yang asked; the four-connectome version (7 mushroom bodies, with FlyWire and BANC) is in
`archive_2026-10-21_four_connectomes/`.

**Short answer.** In a KC model built on each connectome's PN→KC claws, APL's feedback loop by itself makes the effect of
blocking APL uneven across odors. APL's specific wiring adds little. The one wiring feature with a consistent glomerular
pattern is its synapses on PN boutons, and at their real share that effect is small. When this was written (2026-10-15)
no TNT / TNT3c data were in the project; the comparison is now in `TNT_vs_model_notes.md`.

## Why feedback makes the effect uneven (Yang's question, 2026-10-16)
The TNT / TNT3c data have since arrived; the comparison is in `TNT_vs_model_notes.md`. The numbers below come from the
model pinned to those data (official ROIs, threshold 1). Script `../scripts/tnt_feedback_vs_fixed.py`; tables
`TNT_feedback_vs_fixed.csv` and `TNT_feedback_vs_fixed.json`.
- **Same for every KC, not for every odor.** A uniform APL gives every KC the same inhibition for a given odor. APL is
  driven by the KCs it inhibits, though, so the amount depends on the odor. An odor that drives many KCs recruits a lot of
  APL and is held down hard; one that drives few recruits little. With APL intact this evens out how many KCs the odors
  activate, like a normalization. Blocking APL undoes it, so the strongest odors gain the most.
  - With ORN inputs, blocking multiplies the responding KCs by 2.4–2.5 for 1-6ol, 2h, eb and ep and by 1.7–1.9 for
    -aPine, euc, 6al and B-myr. Over odors, Spearman +0.96 between an odor's fraction responding with APL blocked and
    its gain.
  - The spread of the fraction responding across odors (coefficient of variation) is 0.32 blocked and 0.22 intact.
- **From density to similarity.** Denser codes overlap more even by chance. For two unrelated odors, the cosine
  similarity of their 0/1 responses is about √(p_a·p_b), where p is the fraction responding. An odor that gains many KCs
  therefore becomes more similar to every other odor, and its whole row of the similarity matrix rises.
  - In the data, each pair's change in similarity tracks the change in √(p_a·p_b), computed per recording: r = 0.68 for
    raw responses and 0.79 for 0/1 responses.
  - **Why √(p_a·p_b)** (Yang asked for a proof, 2026-10-21). For 0/1 responses, cosine = n_ab / √(n_a·n_b), where n_a
    and n_b units respond to each odor and n_ab to both. If odor b's responders are placed at random, independently of
    odor a's, the overlap is hypergeometric with mean n_a·n_b / N. The expected cosine is then √(n_a·n_b) / N =
    √(p_a·p_b) exactly, with SD √((1 − p_a)(1 − p_b) / (N − 1)), 0.01–0.02 for N = 2,000.
  - **Simulation** (`chance_overlap_simulation.png`, script `../scripts/chance_overlap_sim.py`, tables
    `chance_overlap_simulation.csv`, `chance_overlap_data_pairs.csv`, `chance_overlap_simulation.json`). Random 0/1 codes
    with 2,000 units match √(p_a·p_b) within 0.001 for p from 0.02 to 0.4. Shuffling the units of each recording makes
    the odors unrelated while keeping p, and puts every data pair on √(p_a·p_b) within 0.003.
  - **A floor when units differ.** The formula assumes every unit is equally likely to respond. With unit k responding
    with chance π_k, the expected cosine is about (p_a·p_b + cov_k(π_a, π_b)) / √(p_a·p_b), which is higher whenever the
    same units respond easily to both odors. In the connectome model, unrelated odors (random glomerular patterns) have a
    cosine of 1.59 × √(p_a·p_b) at p = 0.1 with the ORN-table glomeruli, and 1.18 × √(p_a·p_b) when every glomerulus
    responds; sparser codes show more. With the ORN table, 11% of KCs never respond, because most of their claws come
    from glomeruli without a response. With every glomerulus responding, KCs with fewer effective glomerular inputs
    respond more often (Spearman −0.62). Each KC's response probability over the random odors predicts the overlap
    within 6%.
    - **In numbers.** When two unrelated odors each drive 10% of KCs, the formula gives a cosine of 0.10. In the model
      their KC codes have a cosine of 0.16 with the ORN-table glomeruli (1.59 ×) and 0.12 with every glomerulus
      responding (1.18 ×). Since both codes have the same size, they share 1.59 or 1.18 times as many KCs as they would
      if all KCs were equally excitable.
    - **Sparser codes.** The factor grows as fewer KCs respond, because a sparse code is mostly made of the few KCs that
      are easiest to drive, and two unrelated odors then pick the same ones. With the ORN-table glomeruli it is 1.22,
      1.37, 1.59, 2.00 and 3.18 at p = 0.4, 0.2, 0.1, 0.05 and 0.02; with every glomerulus responding, 1.04, 1.06, 1.18,
      1.49 and 2.48.
  - The real 0/1 similarity is about twice √(p_a·p_b) in both conditions (0.145 vs 0.066 in TNT3c, 0.289 vs 0.144 in
    TNT), and the change per pair has slope 2.0 against the change in √(p_a·p_b). That excess can come from related odors
    or from units that respond to many odors. A shuffle that keeps each unit's number of responses and each odor's p
    reproduces the mean (0.145 vs 0.145 in TNT3c, 0.295 vs 0.289 in TNT), so with 16 odors the data cannot tell the two
    apart.
- **Feedback vs a fixed inhibition.** An inhibition of fixed size, the same for every odor and KC (one threshold raise set
  to TNT3c's mean level), is also uneven, but the other way round. Weak odors lose the largest share of their few
  responders: ms ×3.8, 6al ×3.3 and -aPine ×3.1, but eb, ep and 2-but ×1.7–1.8 (Spearman −0.62). The spread across
  odors widens to 0.49.
  - Its similarity change fits the data worse: r = 0.30 vs 0.46 with ORN inputs and 0.16 vs 0.32 with GH146. Feedback
    is higher in all 3 mushroom bodies and in every bootstrap draw over recordings.
  - Its per-odor part matches the data less well: ρ = 0.20 vs 0.52 with ORN inputs, 0.07 vs 0.27 with GH146.
  - In between, a raise per odor that leaves every odor the same share of its responders gives r = 0.41 and 0.26.
    Setting each odor's raise to match its TNT3c fraction gives r = 0.46 and 0.26, no better than feedback.
- **In the data.** The signature is weaker than in the model.
  - When APL is blocked, the number of responding units grows by about the same factor for strong and weak odors. The
    spread across odors is 0.32 in TNT and 0.31 in TNT3c. With odor strength and gain taken from separate halves of the
    recordings, so that they share no noise, the two are unrelated (mean Spearman +0.06 over 2000 splits).
  - Response size grows more for strong odors: Spearman +0.32 for the mean positive response and +0.37 for the root
    mean square response, positive in 99% of splits.
  - **Definitions.** Gain is per odor: the fraction of units responding (> 1) in TNT divided by that fraction in TNT3c.
    Each fraction is computed per recording and averaged over recordings; with all recordings the gain ranges from ×1.65
    (euc) to ×2.90 (eb). Odor strength is the odor's fraction responding, averaged over the two conditions. In each of
    the 2000 splits, 4 TNT and 5 TNT3c recordings give the strength, and the other 4 TNT and 6 TNT3c recordings give the
    gain; Spearman is over the 16 odors (5–95% of splits: −0.22 to +0.33). The response-size gains use the same ratio
    with the mean positive response or the root mean square response instead of the fraction. The spread is the
    coefficient of variation of the fraction responding across the 16 odors (SD ÷ mean).

## What the model shows
20% of KCs respond with APL blocked, 10% with it intact (10%→5% and 30%→10% give the same picture). Responses: Yang's ORN
table (16 odors) and GH146 means (13 odors). Three mushroom bodies: hemibrain R, male CNS L and R (FlyWire and BANC
dropped on 2026-10-21; their results are in the archive named at the top).

- **Uniform APL** (panels c, d, h, i). Strong odors recruit the most APL, so blocking it roughly doubles the responding
  KCs for pa, 1-6ol, paa, eb and 2h (×2.1–2.4) but adds few for weak odors such as ms and 1-8ol. Odor-pair similarity
  (cosine) rises most for pairs involving the strongest odors (mean +0.076 with ORN, +0.041 with GH146, up to +0.20),
  with the same pattern in all 3 mushroom bodies (median r 0.87 and 0.81).
- **APL synapse counts per KC** (APL→KC and KC→APL synapses in the calyx; panels e, j). Which odors gain is unchanged
  (r ≥ 0.99). The counts shrink the pair changes somewhat, but the same counts shuffled among KCs of the same class do
  much the same (maps r 0.72 with ORN, 0.71 with GH146), and what is left differs between mushroom bodies (median r
  0.34 and 0.20), so the specific KC-by-KC wiring adds nothing consistent.
  - Until 2026-10-21 the KC→APL counts covered the whole MB; Yang asked for the calyx only. With Pearson r, the measure
    used then, and all four connectomes, the switch moved pair similarities by at most 0.014 and these numbers by at most
    0.01. That is expected: APL is one unit here, so which KCs drive it only sets how strongly each odor recruits it.
- **APL on PN boutons** (panels a, f, k; pair by pair in `APL_bouton_pairs.png`). APL→PN synapses per PN→KC synapse vary
  3–4-fold across glomeruli (10th to 90th percentile), in the same order in hemibrain and male CNS (Spearman 0.83). DM2
  gets 1.7–3.6× more than DM3, VM2 and DM5; DM1, VP1m, DL2v, VC3 and DL1 are high, DC4, DA4l and VP1d low. If all of
  APL's inhibition acted this way, blocking APL would bring pa and eb (and pa and ep) closer together and push eb and ep
  away from 1-6ol and 1-8ol, compared with the same inhibition spread evenly over all glomeruli (mean |change| 0.034
  with ORN, 0.039 with GH146). Spreading it evenly acts like a uniform APL on the KCs (within 0.002–0.003 on average).
  - All six pairs move the same way in all 3 mushroom bodies (`APL_bouton_pairs.png`, panels b and e). With GH146
    inputs they are six of the eight largest changes. With ORN inputs pa–eb, pa–ep and eb–1-8ol are among the largest,
    the 1-6ol pairs move less, eb also moves closer to paa and B-myr, and pa to 1-6ol and 1-5ol.
  - As similarities (GH146, mean of 3 MBs): pa and eb have a cosine of 0.77 with APL intact and spread evenly, 0.65 with
    it per glomerulus, and 0.80 with APL blocked. eb and 1-6ol have 0.63 and 0.75, and 0.73 blocked, so with this
    inhibition blocking APL makes them less similar.
  - Why (`APL_bouton_pairs.png`, panels g and h): bouton inhibition spares low-weight glomeruli. eb, ep, 1-6ol and
    1-8ol share theirs (VM2, DM5, VM7d), while much of what pa shares with eb and ep is in DM2, which has a higher
    weight.
- **Boutons at their real share** (panels g, l). APL→PN synapses are 10–11% of APL's calyx output synapses. With that
  share on boutons and the rest uniform on KCs, the same pattern shrinks to a mean |change| of 0.006–0.007, at most
  0.04 (open circles in `APL_bouton_pairs.png`).
- **APL on claws** (panel b). APL→KC synapses on claws, per claw PN synapse, vary only about 2-fold across glomeruli
  (Spearman 0.75 between hemibrain and male CNS). Routing all inhibition through them changes the similarity by
  0.015–0.02 on average.

## How to test it with TNT / TNT3c
- If the per-odor increase in responding KCs (or response size) tracks odor strength, and similarity rises most for pairs
  of strong odors, feedback inhibition alone explains the nonuniformity.
- If specific pairs break that rule, especially pa / eb / ep becoming more similar and eb / ep less similar to 1-6ol and
  1-8ol, that points to glomerulus-specific inhibition, for example at PN boutons.
- Needed: KC responses (or odor-by-odor correlation matrices) for control, TNT and TNT3c on these odors.

## Caveats
- The ORN version uses ORN responses as the PN input; GH146 lacks -aPine, euc and B-myr.
- Claws of glomeruli without a response get no input (about 20% of claw PN synapses with the ORN table).
- One threshold for all KCs. Similarity is the cosine of two odors' KC population responses, and its change is blocked −
  intact. Until 2026-10-21 it was Pearson r; the pair table keeps those values in its r_* columns.
- The coding levels (20% blocked, 10% intact) are assumptions; the TNT data could set them.

Files: `APL_block_model_per_odor.csv` (fraction of KCs responding per odor, mushroom body, APL version and coding level),
`APL_block_model_odor_pairs.csv` (cosine, cos_*, and Pearson r, r_*, per odor pair: KC input, blocked, intact per
version), `APL_block_model_summary.json`, `APL_wiring_by_glomerulus.csv`. Bouton pattern pair by pair:
`APL_bouton_pairs.png`, `APL_bouton_pairs.csv` (the six pairs per mushroom body), `APL_bouton_pairs.json` (script
`../scripts/apl_bouton_pairs_figure.py`).
