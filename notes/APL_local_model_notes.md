# Local, non-spiking APL in the KC model (official ROIs)

Yang asked (2026-10-23): does a model in which APL is non-spiking, so its inhibition stays local, and uses its real connectivity, reshape the KC representation (especially the Spearman rank order of odor-pair similarity between inhibition on and off) more like the data than a uniform APL or a simple connectivity APL? Hemibrain R, male CNS L and R. Step-by-step explanation with figures: `reports/11_local_APL_model.html` / `.pdf`. Figure: `APL_local_model.png`.

## Model in short
- Same pinned KC model as `tnt_compare.py`: each odor's input scaled so its fraction of KCs responding with APL blocked equals TNT; one APL gain per version sets the mean intact fraction to TNT3c's (6.83%). y = [x − 1 − a M y]+, solved per odor.
- U: M = 1/N. C: M_kj = u_k v_j / N (calyx APL→KC and KC→APL counts). local λ: M_kj = c Σ exp(−d(s,t)/λ) over APL→KC sites s on KC k and KC→APL sites t of KC j, d = distance along the APL skeleton; λ = ∞ gives C exactly. λ = 200 … 2 µm.
- Boutons: each bouton's output divided by (1 + β Q), Q = local APL signal at its APL→PN synapses per claw synapse. "boutons only" = all inhibition there; "KC + boutons" = both routes at the boutons' real share of APL's calyx output (hemibrain 0.097, male CNS 0.112).
- Null: KCs swap their APL synapse sites within class (100 shuffles), for the local λ with the best pattern r.
- Solver: exact active-set method (falls back to a damped iteration); it matches the damped iteration (gains to 4 decimals, responses within ~1e-6). The tree sums were checked against brute-force shortest paths (relative error ~1e-16).

## Answer
**ORN input.** Rank order, blocked vs intact (data 0.869, ≈ 0.92 noise-corrected): U 0.977, C 0.984, local on KCs 0.970–0.985, boutons only 0.942–0.961, KC + boutons 0.971–0.982. Pattern r with the data's change (ceiling ≈ 0.84): U 0.455, C 0.475, best local on KCs 0.581 (λ = 5 µm), best KC + boutons 0.584. Shuffled KC sites (λ = 5 µm): r 0.581 vs null 0.427 (5–95% 0.279–0.549), p = 0.02; rank 0.977 vs null 0.985, p = 0.01. Mean change in similarity: data 0.178, U 0.079, best local 0.044.

**GH146 input.** Rank order, blocked vs intact (data 0.873, ≈ 0.92 noise-corrected): U 0.983, C 0.984, local on KCs 0.963–0.984, boutons only 0.911–0.933, KC + boutons 0.962–0.981. Pattern r with the data's change (ceiling ≈ 0.84): U 0.316, C 0.319, best local on KCs 0.528 (λ = 2 µm), best KC + boutons 0.477. Shuffled KC sites (λ = 2 µm): r 0.528 vs null 0.186 (5–95% -0.069–0.408), p = 0.01; rank 0.963 vs null 0.977, p = 0.03. Mean change in similarity: data 0.201, U 0.061, best local 0.029.

In words: a local, non-spiking APL predicts **which** pairs change most better than a uniform or count-weighted APL, and the real placement of KCs on its arbor beats within-class shuffles for both inputs. It does **not** reorder the pairs much more: the rank order stays near the uniform model's. Only inhibition placed entirely on the PN boutons comes near the data's reordering, and at the boutons' real share of APL output it does not. The model's mean change in similarity (U and local on KCs) is 24%–44% of the data's (ORN), 14%–30% of the data's (GH146).

## All versions
**ORN** | rank (mean of MBs) | pattern r
---|---|---
U | 0.977 | 0.455
C | 0.984 | 0.475
local 200 µm | 0.984 | 0.469
local 100 µm | 0.985 | 0.468
local 50 µm | 0.984 | 0.477
local 25 µm | 0.982 | 0.503
local 10 µm | 0.979 | 0.558
local 5 µm | 0.977 | 0.581
local 2 µm | 0.970 | 0.567
boutons inf | 0.957 | 0.472
KC + boutons inf | 0.982 | 0.473
boutons 50 µm | 0.961 | 0.506
KC + boutons 50 µm | 0.981 | 0.480
boutons 25 µm | 0.956 | 0.522
KC + boutons 25 µm | 0.978 | 0.509
boutons 10 µm | 0.946 | 0.538
KC + boutons 10 µm | 0.973 | 0.566
boutons 5 µm | 0.942 | 0.544
KC + boutons 5 µm | 0.971 | 0.584

**GH146** | rank (mean of MBs) | pattern r
---|---|---
U | 0.983 | 0.316
C | 0.984 | 0.319
local 200 µm | 0.984 | 0.278
local 100 µm | 0.983 | 0.265
local 50 µm | 0.982 | 0.278
local 25 µm | 0.980 | 0.324
local 10 µm | 0.976 | 0.418
local 5 µm | 0.971 | 0.473
local 2 µm | 0.963 | 0.528
boutons inf | 0.931 | 0.347
KC + boutons inf | 0.981 | 0.330
boutons 50 µm | 0.933 | 0.337
KC + boutons 50 µm | 0.978 | 0.295
boutons 25 µm | 0.929 | 0.361
KC + boutons 25 µm | 0.975 | 0.337
boutons 10 µm | 0.921 | 0.389
KC + boutons 10 µm | 0.968 | 0.426
boutons 5 µm | 0.911 | 0.403
KC + boutons 5 µm | 0.962 | 0.477

## Caveats
- λ = 2 µm is near the skeleton's resolution (synapses snap 0.4–0.8 µm to nodes).
- The λ for the shuffle test was picked first (best r), so p does not correct for trying several λ.
- Synapse counts stand in for strength; passive steady-state decay only; a strongly coupled threshold-linear network can have several steady states, and the solver finds the one reached from the blocked responses.

Files: `APL_local_model_summary.json`, `_fits.csv`, `_per_MB_{ORN,GH146}.csv`, `_odor_pairs.csv`, `_nulls.csv`; `APL_local_boutons_summary.json`, `_per_MB_{ORN,GH146}.csv`. Scripts: `scripts/apl_local_model.py`, `apl_local_boutons.py`, `apl_local_model_figure.py`, this file `apl_local_model_notes.py`.
