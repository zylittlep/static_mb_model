# How the APL block model is built

2026-10-21 · official ROIs only · model `../scripts/apl_block_model.py` · pinned to the TNT / TNT3c data in
`../scripts/tnt_compare.py` · results in `APL_block_model_notes.md` and `TNT_vs_model_notes.md`

A rate model of the calyx: KCs sum their claws' glomerular input, respond above a threshold, and APL inhibits them in
proportion to their summed output. It is built separately for each mushroom body, and "blocked" simply means no APL.

## 1. Wiring: one model per mushroom body
- **Mushroom bodies.** Three: hemibrain R and male CNS L and R, with 1,793, 1,876 and 1,900 KCs. Since 2026-10-21
  only hemibrain and male CNS are used, as Yang asked; the version that also had FlyWire L/R and BANC L/R (7 mushroom
  bodies) is in `archive_2026-10-21_four_connectomes/`. Male CNS has both sides, so its KCs are split by 2-means
  clustering of their claw positions, and each cluster is named after the APL its on-claw synapses go to.
- **Claws.** A claw is a single-linkage cluster (2 µm) of a KC's postsynaptic sites from antennal-lobe PNs in the calyx. It
  belongs to the glomerulus of its main PN, the PN that makes most of its PN synapses, and all its PN synapses count for
  that glomerulus. Only uniglomerular PN types count.
- **Input weights.** W_kg = PN synapses on KC k's claws of glomerulus g ÷ all PN synapses on KC k's claws. Claws whose
  main PN is not a uniglomerular type stay in the denominator but give no input (3–4% of claw PN synapses).

## 2. Odor input
- **Glomerular responses** r_go come from Yang's ORN table (36 glomeruli × 16 odors, fly averages) or the GH146 PN means
  (29 glomeruli × 13 odors; no -aPine, euc or B-myr; paa from the diagnostics panel at −5). Negative responses are set
  to 0.
- **Coverage.** Claws of glomeruli without a response give 0 input. The glomeruli in the ORN table carry 77–78% of claw
  PN synapses, and those in the GH146 table 63–65%.
- **KC drive** x_ko = Σ_g W_kg · r_go. ORN responses stand in for PN responses directly, with no PN-level transformation.

## 3. KCs and APL
- **KC response** (threshold-linear): y_ko = [x_ko − θ − a · u_k · A_o]₊.
- **APL activity** per odor: A_o = mean over KCs of v_k · y_ko. APL is driven by the KCs' output and feeds back on them,
  so A_o and the KC responses are solved together for each odor. The mean output falls as A rises, so there is exactly
  one solution, found by bisection.
- **Blocked (TNT)** is a = 0, so no APL. **Intact** has a > 0. The gain a is one number per mushroom body, the same for
  every odor; the inhibition still differs between odors because A_o does.
- **Uniform APL (U)** has u_k = v_k = 1. Every KC gets the same inhibition for a given odor and drives APL equally, so
  APL acts as a threshold raise that is shared by all KCs but set by each odor's own KC output.

## 4. Versions with APL's connectome wiring
Each version replaces part of the uniform APL with a connectome measurement:
- **C, per-KC counts.** u_k = the KC's APL→KC synapses in the calyx, v_k = its KC→APL synapses in the calyx, each
  divided by its mean (v_k counted the whole MB until 2026-10-21, when Yang asked for the calyx only). Controls: the
  (u, v) pairs shuffled among KCs of the same class (αβ, α′β′, γ, or the few KCs without a subtype; Cs), and class
  means (Cc).
- **P0 / P, on PN boutons.** Inhibition divides glomerulus g's input instead of subtracting: r_go / (1 + b · p_g · A_o),
  with no subtractive term. P0 has p_g = 1. P has p_g = APL→PN synapses per PN→KC synapse of g's PNs.
- **L, on claws.** Same form, with q_g = on-claw APL→KC synapses per PN→KC synapse on g's claws.
- **M0 / M, both at their real share.** U plus P0 or P, with the bouton part at its synapse share f = APL→PN ÷ (APL→PN +
  APL→KC) in the calyx (0.10–0.11 per dataset). The KC gain is (1 − f) · a_U and the bouton gain f · b_P0, and both are
  then scaled together to the calibration target.
- **F0 / F, feed-forward (TNT comparison only).** APL activity comes from the PN input instead of the KCs, so there is
  no feedback: A_o = the odor's mean input over all claws, proportional to the mean KC drive (F0), or that mean with
  each glomerulus weighted by its PN→APL synapses per PN→KC synapse (F). It is subtracted from every KC.
- Glomerulus weights (p, q, PN→APL) are scaled to a claw-input-weighted mean of 1; glomeruli without a value get 1.

## 5. Calibration
Every version is calibrated the same way, so the versions differ only in where the inhibition acts and how it is
weighted.
- **Prediction run** (`APL_block_model_*`, before the TNT data arrived). θ is set so that 20% of KCs respond with APL
  blocked, averaged over odors; each version's APL gain is set so that 10% respond with it intact. 10% → 5% and
  30% → 10% are checks.
- **Pinned to the data** (`TNT_vs_model_*` and the later TNT scripts). θ = 1. Each odor's input is multiplied by its own
  gain so that, with APL blocked, the fraction of KCs responding equals that odor's fraction of units > 1 in TNT. One APL
  gain per version then sets the mean fraction over odors with APL intact to TNT3c's mean (6.8% over the 16 odors). The
  intact fraction of each odor is left free, so it is a prediction. Responding at > 0.5 is a check.

## 6. What is read out
- **Per odor:** the fraction of KCs responding (y > 0), blocked and intact, and their ratio.
- **Per odor pair:** cosine similarity of the two odors' KC population responses, as in Yang's RDMs. It is computed per
  mushroom body and averaged over the 3. The effect of blocking APL is similarity blocked − intact. (The prediction run
  used Pearson r until 2026-10-21; its pair table still has those values in the r_* columns.)
- In the prediction run, odors with < 0.5% of KCs responding intact in any version and mushroom body are left out of the
  pair results.

## What the model leaves out
- One threshold for all KCs, with no differences in excitability between KC classes, and no spikes or time course.
- APL is a single unit with one activity level per odor in every version. P and L weight its output per glomerulus but do
  not make its activity local.
- The inputs are ORN or GH146 averages from other flies, not the recorded flies' PNs. The imaged units are not
  identified KCs.
