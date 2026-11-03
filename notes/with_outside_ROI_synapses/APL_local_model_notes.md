# Local, non-spiking APL in the KC model (version with synapses outside the official ROIs)

Yang asked (2026-10-23): does a model in which APL is non-spiking, so its inhibition stays local, and uses its real connectivity, reshape the KC representation (especially the Spearman rank order of odor-pair similarity between inhibition on and off) more like the data than a uniform APL or a simple connectivity APL? Hemibrain R, male CNS L and R. Step-by-step explanation with figures: `reports/11_local_APL_model.html` / `.pdf`. Figure: `APL_local_model.png`.

## Model in short
- Same pinned KC model as `tnt_compare.py`: each odor's input scaled so its fraction of KCs responding with APL blocked equals TNT; one APL gain per version sets the mean intact fraction to TNT3c's (6.83%). y = [x − 1 − a M y]+, solved per odor.
- U: M = 1/N. C: M_kj = u_k v_j / N (calyx APL→KC and KC→APL counts). local λ: M_kj = c Σ exp(−d(s,t)/λ) over APL→KC sites s on KC k and KC→APL sites t of KC j, d = distance along the APL skeleton; λ = ∞ gives C exactly. λ = 200 … 2 µm.
- Boutons: each bouton's output divided by (1 + β Q), Q = local APL signal at its APL→PN synapses per claw synapse. "boutons only" = all inhibition there; "KC + boutons" = both routes at the boutons' real share of APL's calyx output (hemibrain 0.097, male CNS 0.112).
- Null: KCs swap their APL synapse sites within class (100 shuffles), for the local λ with the best pattern r.
- Solver: exact active-set method (falls back to a damped iteration); it matches the damped iteration (gains to 4 decimals, responses within ~1e-6). The tree sums were checked against brute-force shortest paths (relative error ~1e-16).

## Answer
**ORN input.** Rank order, blocked vs intact (data 0.869, ≈ 0.92 noise-corrected): U 0.975, C 0.976, local on KCs 0.960–0.976, boutons only 0.938–0.961, KC + boutons 0.961–0.973. Pattern r with the data's change (ceiling ≈ 0.84): U 0.500, C 0.512, best local on KCs 0.596 (λ = 2 µm), best KC + boutons 0.580. Shuffled KC sites (λ = 2 µm): r 0.596 vs null 0.483 (5–95% 0.277–0.62), p = 0.158; rank 0.960 vs null 0.980, p = 0.01. Mean change in similarity: data 0.178, U 0.085, best local 0.062.

**GH146 input.** Rank order, blocked vs intact (data 0.873, ≈ 0.92 noise-corrected): U 0.982, C 0.982, local on KCs 0.954–0.981, boutons only 0.928–0.950, KC + boutons 0.963–0.981. Pattern r with the data's change (ceiling ≈ 0.84): U 0.342, C 0.388, best local on KCs 0.545 (λ = 2 µm), best KC + boutons 0.508. Shuffled KC sites (λ = 2 µm): r 0.545 vs null 0.286 (5–95% 0.04–0.466), p = 0.01; rank 0.954 vs null 0.981, p = 0.01. Mean change in similarity: data 0.201, U 0.057, best local 0.038.

In words: a local, non-spiking APL predicts **which** pairs change most better than a uniform or count-weighted APL, and the real placement of KCs on its arbor beats within-class shuffles for GH146 (p = 0.01) but not for ORN (p = 0.158). It does **not** reorder the pairs much more: the rank order stays near the uniform model's. Only inhibition placed entirely on the PN boutons comes near the data's reordering, and at the boutons' real share of APL output it does not. The model's mean change in similarity (U and local on KCs) is 33%–48% of the data's (ORN), 15%–28% of the data's (GH146).

## All versions
**ORN** | rank (mean of MBs) | pattern r
---|---|---
U | 0.975 | 0.500
C | 0.976 | 0.512
local 200 µm | 0.976 | 0.513
local 100 µm | 0.975 | 0.518
local 50 µm | 0.974 | 0.524
local 25 µm | 0.973 | 0.534
local 10 µm | 0.969 | 0.568
local 5 µm | 0.966 | 0.591
local 2 µm | 0.960 | 0.596
boutons inf | 0.961 | 0.475
KC + boutons inf | 0.973 | 0.502
boutons 50 µm | 0.957 | 0.510
KC + boutons 50 µm | 0.971 | 0.517
boutons 25 µm | 0.952 | 0.513
KC + boutons 25 µm | 0.969 | 0.526
boutons 10 µm | 0.942 | 0.520
KC + boutons 10 µm | 0.965 | 0.559
boutons 5 µm | 0.938 | 0.515
KC + boutons 5 µm | 0.961 | 0.580

**GH146** | rank (mean of MBs) | pattern r
---|---|---
U | 0.982 | 0.342
C | 0.982 | 0.388
local 200 µm | 0.981 | 0.362
local 100 µm | 0.980 | 0.361
local 50 µm | 0.978 | 0.380
local 25 µm | 0.976 | 0.413
local 10 µm | 0.973 | 0.481
local 5 µm | 0.967 | 0.515
local 2 µm | 0.954 | 0.545
boutons inf | 0.950 | 0.307
KC + boutons inf | 0.981 | 0.376
boutons 50 µm | 0.950 | 0.303
KC + boutons 50 µm | 0.977 | 0.371
boutons 25 µm | 0.947 | 0.339
KC + boutons 25 µm | 0.975 | 0.405
boutons 10 µm | 0.940 | 0.390
KC + boutons 10 µm | 0.970 | 0.477
boutons 5 µm | 0.928 | 0.409
KC + boutons 5 µm | 0.963 | 0.508

## Caveats
- λ = 2 µm is near the skeleton's resolution (synapses snap 0.4–0.8 µm to nodes).
- The λ for the shuffle test was picked first (best r), so p does not correct for trying several λ.
- Synapse counts stand in for strength; passive steady-state decay only; a strongly coupled threshold-linear network can have several steady states, and the solver finds the one reached from the blocked responses.

Files: `APL_local_model_summary.json`, `_fits.csv`, `_per_MB_{ORN,GH146}.csv`, `_odor_pairs.csv`, `_nulls.csv`; `APL_local_boutons_summary.json`, `_per_MB_{ORN,GH146}.csv`. Scripts: `scripts/apl_local_model.py`, `apl_local_boutons.py`, `apl_local_model_figure.py`, this file `apl_local_model_notes.py`.
