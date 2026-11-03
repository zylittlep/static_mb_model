# APL confined to its calyx sub-arbors (version with synapses outside the official ROIs)

Yang asked (2026-10-24): a model in which signals inside APL stay within the calyx sub-arbors of Report 10 (no crossing between them); does it model the reshaping (rank order of odor-pair similarity, inhibition on vs off) better? Hemibrain R, male CNS L and R. Step-by-step report with figures: `reports/12_APL_sub_arbor_compartments.html` / `.pdf`. Figure: `APL_compartment_model.png`.

## Model in short
- Same pinned KC model as Reports 07 and 11: y = [x − 1 − a M y]+, TNT fractions per odor with APL blocked, one APL gain per version for TNT3c's 6.83%.
- Compartments = connected pieces of APL's calyx skeleton (nodes within 5 µm of a calyx APL synapse); every piece is its own compartment.
- Isopotential: M_kj ∝ Σ_c O_ck I_cj (O, I: APL→KC and KC→APL synapses of the KC in compartment c). "÷ length": each term divided by the compartment's cable length. λ inside: Report 11's local kernel with every edge leaving a compartment cut (checked: equals the isopotential formula at λ = ∞ exactly). Bouton route as in Report 11 with the walled signal.
- Null: KCs swap APL synapse sites within class, 100x, for the better isopotential version.

## Answer
**ORN input.** Rank order (data 0.869, ≈ 0.92 noise-corrected): U 0.975, C 0.976, sub-arbors on KCs 0.969–0.979, boutons only in sub-arbors 0.966 (∞) / 0.959 (50 µm), KC + boutons 0.978. Pattern r (ceiling ≈ 0.84): U 0.500, C 0.512, isopotential 0.500, ÷ length 0.558, λ = 10 µm inside 0.567; without walls (Report 11) best 0.596 (λ = 2 µm). Shuffle (compartments inf, per length): r 0.558 vs null 0.490, p = 0.089. Mean change in similarity: data 0.178, U 0.085, sub-arbors 0.056–0.065.

**GH146 input.** Rank order (data 0.873, ≈ 0.92 noise-corrected): U 0.982, C 0.982, sub-arbors on KCs 0.973–0.981, boutons only in sub-arbors 0.951 (∞) / 0.948 (50 µm), KC + boutons 0.979. Pattern r (ceiling ≈ 0.84): U 0.342, C 0.388, isopotential 0.322, ÷ length 0.490, λ = 10 µm inside 0.481; without walls (Report 11) best 0.545 (λ = 2 µm). Shuffle (compartments inf, per length): r 0.490 vs null 0.316, p = 0.01. Mean change in similarity: data 0.201, U 0.057, sub-arbors 0.030–0.036.

In words: confining APL's signal to its calyx sub-arbors does **not** make inhibition reorder odor pairs more; the rank order stays near the uniform model's (about 0.98). Isopotential sub-arbors predict which pairs change no better than one APL. Dividing each sub-arbor's potential by its cable length raises the pattern r above U and C, and the real placement beats within-class shuffles for GH146 but not for ORN. No compartment version beats the local model without walls at short λ, where the walls do not matter.

## All versions
**ORN** | rank (mean of MBs) | pattern r | mean ΔS
---|---|---|---
U | 0.975 | 0.500 | 0.085
C | 0.976 | 0.512 | 0.064
compartments inf | 0.979 | 0.500 | 0.056
compartments inf, per length | 0.969 | 0.558 | 0.065
compartments 200 µm | 0.978 | 0.503 | 0.057
compartments 100 µm | 0.978 | 0.505 | 0.057
compartments 50 µm | 0.976 | 0.511 | 0.058
compartments 25 µm | 0.974 | 0.528 | 0.059
compartments 10 µm | 0.969 | 0.567 | 0.062
boutons, compartments inf | 0.966 | 0.510 | 0.081
KC + boutons, compartments inf | 0.978 | 0.496 | 0.061
boutons, compartments 50 µm | 0.959 | 0.513 | 0.081
KC + boutons, compartments 50 µm | 0.973 | 0.505 | 0.062

**GH146** | rank (mean of MBs) | pattern r | mean ΔS
---|---|---|---
U | 0.982 | 0.342 | 0.057
C | 0.982 | 0.388 | 0.038
compartments inf | 0.981 | 0.322 | 0.036
compartments inf, per length | 0.974 | 0.490 | 0.034
compartments 200 µm | 0.980 | 0.335 | 0.035
compartments 100 µm | 0.980 | 0.348 | 0.034
compartments 50 µm | 0.979 | 0.369 | 0.032
compartments 25 µm | 0.977 | 0.409 | 0.030
compartments 10 µm | 0.973 | 0.481 | 0.031
boutons, compartments inf | 0.951 | 0.264 | 0.062
KC + boutons, compartments inf | 0.979 | 0.314 | 0.039
boutons, compartments 50 µm | 0.948 | 0.304 | 0.057
KC + boutons, compartments 50 µm | 0.977 | 0.362 | 0.035

## Checks
```
{
 "hemibrain R": {
  "cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference": 0.0,
  "r of compartments inf with C": 0.8538,
  "share of M (compartments inf) within KC class": 0.4622,
  "compressed nodes": 16826,
  "compartments holding KC<->APL synapses": 16,
  "KC<->APL synapse nodes outside every piece (share)": 0.0,
  "effective number of compartments (1 / sum of squared APL->KC shares)": 1.84
 },
 "malecns L": {
  "cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference": 0.0,
  "r of compartments inf with C": 0.4446,
  "share of M (compartments inf) within KC class": 0.4876,
  "compressed nodes": 6602,
  "compartments holding KC<->APL synapses": 21,
  "KC<->APL synapse nodes outside every piece (share)": 0.0,
  "effective number of compartments (1 / sum of squared APL->KC shares)": 6.54
 },
 "malecns R": {
  "cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference": 0.0,
  "r of compartments inf with C": 0.6472,
  "share of M (compartments inf) within KC class": 0.4255,
  "compressed nodes": 5773,
  "compartments holding KC<->APL synapses": 18,
  "KC<->APL synapse nodes outside every piece (share)": 0.0,
  "effective number of compartments (1 / sum of squared APL->KC shares)": 3.04
 }
}
```

Files: `APL_compartment_model_summary.json`, `_fits.csv`, `_per_MB_{ORN,GH146}.csv`, `_odor_pairs.csv`, `_nulls.csv`, `_compartments.csv`. Scripts: `scripts/apl_compartment_model.py`, `apl_compartment_model_figure.py`, this file `apl_compartment_model_notes.py`.
