# APL confined to its calyx sub-arbors (official ROIs)

Yang asked (2026-10-24): a model in which signals inside APL stay within the calyx sub-arbors of Report 10 (no crossing between them); does it model the reshaping (rank order of odor-pair similarity, inhibition on vs off) better? Hemibrain R, male CNS L and R. Step-by-step report with figures: `reports/12_APL_sub_arbor_compartments.html` / `.pdf`. Figure: `APL_compartment_model.png`.

## Model in short
- Same pinned KC model as Reports 07 and 11: y = [x − 1 − a M y]+, TNT fractions per odor with APL blocked, one APL gain per version for TNT3c's 6.83%.
- Compartments = connected pieces of APL's calyx skeleton (nodes within 5 µm of a calyx APL synapse); every piece is its own compartment.
- Isopotential: M_kj ∝ Σ_c O_ck I_cj (O, I: APL→KC and KC→APL synapses of the KC in compartment c). "÷ length": each term divided by the compartment's cable length. λ inside: Report 11's local kernel with every edge leaving a compartment cut (checked: equals the isopotential formula at λ = ∞ exactly). Bouton route as in Report 11 with the walled signal.
- Null: KCs swap APL synapse sites within class, 100x, for the better isopotential version.

## Answer
**ORN input.** Rank order (data 0.869, ≈ 0.92 noise-corrected): U 0.977, C 0.984, sub-arbors on KCs 0.979–0.987, boutons only in sub-arbors 0.966 (∞) / 0.961 (50 µm), KC + boutons 0.985. Pattern r (ceiling ≈ 0.84): U 0.455, C 0.475, isopotential 0.469, ÷ length 0.503, λ = 10 µm inside 0.558; without walls (Report 11) best 0.581 (λ = 5 µm). Shuffle (compartments inf, per length): r 0.503 vs null 0.420, p = 0.129. Mean change in similarity: data 0.178, U 0.079, sub-arbors 0.042–0.047.

**GH146 input.** Rank order (data 0.873, ≈ 0.92 noise-corrected): U 0.983, C 0.984, sub-arbors on KCs 0.976–0.983, boutons only in sub-arbors 0.937 (∞) / 0.932 (50 µm), KC + boutons 0.980. Pattern r (ceiling ≈ 0.84): U 0.316, C 0.319, isopotential 0.230, ÷ length 0.399, λ = 10 µm inside 0.418; without walls (Report 11) best 0.528 (λ = 2 µm). Shuffle (compartments inf, per length): r 0.399 vs null 0.233, p = 0.01. Mean change in similarity: data 0.201, U 0.061, sub-arbors 0.029–0.034.

In words: confining APL's signal to its calyx sub-arbors does **not** make inhibition reorder odor pairs more; the rank order stays near the uniform model's (about 0.98). Isopotential sub-arbors predict which pairs change no better than one APL. Dividing each sub-arbor's potential by its cable length raises the pattern r above U and C, and the real placement beats within-class shuffles for GH146 but not for ORN. No compartment version beats the local model without walls at short λ, where the walls do not matter.

## All versions
**ORN** | rank (mean of MBs) | pattern r | mean ΔS
---|---|---|---
U | 0.977 | 0.455 | 0.079
C | 0.984 | 0.475 | 0.049
compartments inf | 0.987 | 0.469 | 0.042
compartments inf, per length | 0.982 | 0.503 | 0.047
compartments 200 µm | 0.986 | 0.471 | 0.042
compartments 100 µm | 0.986 | 0.474 | 0.042
compartments 50 µm | 0.984 | 0.481 | 0.043
compartments 25 µm | 0.982 | 0.504 | 0.044
compartments 10 µm | 0.979 | 0.558 | 0.044
boutons, compartments inf | 0.966 | 0.509 | 0.067
KC + boutons, compartments inf | 0.985 | 0.469 | 0.046
boutons, compartments 50 µm | 0.961 | 0.520 | 0.070
KC + boutons, compartments 50 µm | 0.982 | 0.485 | 0.048

**GH146** | rank (mean of MBs) | pattern r | mean ΔS
---|---|---|---
U | 0.983 | 0.316 | 0.061
C | 0.984 | 0.319 | 0.037
compartments inf | 0.983 | 0.230 | 0.034
compartments inf, per length | 0.981 | 0.399 | 0.032
compartments 200 µm | 0.982 | 0.239 | 0.033
compartments 100 µm | 0.982 | 0.250 | 0.033
compartments 50 µm | 0.982 | 0.274 | 0.032
compartments 25 µm | 0.980 | 0.322 | 0.030
compartments 10 µm | 0.976 | 0.418 | 0.029
boutons, compartments inf | 0.937 | 0.315 | 0.053
KC + boutons, compartments inf | 0.980 | 0.250 | 0.037
boutons, compartments 50 µm | 0.932 | 0.341 | 0.052
KC + boutons, compartments 50 µm | 0.978 | 0.294 | 0.035

## Checks
```
{
 "hemibrain R": {
  "cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference": 0.0,
  "r of compartments inf with C": 0.8548,
  "share of M (compartments inf) within KC class": 0.4626,
  "compressed nodes": 16503,
  "compartments holding KC<->APL synapses": 16,
  "KC<->APL synapse nodes outside every piece (share)": 0.0,
  "effective number of compartments (1 / sum of squared APL->KC shares)": 1.84
 },
 "malecns L": {
  "cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference": 0.0,
  "r of compartments inf with C": 0.6029,
  "share of M (compartments inf) within KC class": 0.5118,
  "compressed nodes": 4534,
  "compartments holding KC<->APL synapses": 15,
  "KC<->APL synapse nodes outside every piece (share)": 0.0,
  "effective number of compartments (1 / sum of squared APL->KC shares)": 5.32
 },
 "malecns R": {
  "cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference": 0.0,
  "r of compartments inf with C": 0.6493,
  "share of M (compartments inf) within KC class": 0.4243,
  "compressed nodes": 5093,
  "compartments holding KC<->APL synapses": 11,
  "KC<->APL synapse nodes outside every piece (share)": 0.0,
  "effective number of compartments (1 / sum of squared APL->KC shares)": 2.75
 }
}
```

Files: `APL_compartment_model_summary.json`, `_fits.csv`, `_per_MB_{ORN,GH146}.csv`, `_odor_pairs.csv`, `_nulls.csv`, `_compartments.csv`. Scripts: `scripts/apl_compartment_model.py`, `apl_compartment_model_figure.py`, this file `apl_compartment_model_notes.py`.
