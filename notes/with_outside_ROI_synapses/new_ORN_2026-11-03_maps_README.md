# Report 03 revision: maps and odor similarity with the new ORN data (2026-11-04)

These files use Yang's new ORN responses (`orn_inputs_2026-11-03/`) and the corrected calyx (synapses outside the official ROI added). They cover Yang's Report 3 list. The PN ↔ APL files came first (2026-11-04 00:35); the KC ↔ APL files were added after Yang chose the claws (2026-11-04 08:25, new claws `claw_v2m2` of `with_outside_ROI_synapses/claws_v2m2/`, hemibrain and male CNS only).

The scripts are in `scripts/odor_maps_2026_11/`: `common.py`, `maps2d.py`, `maps3d.py` and `similarity.py`.

**Two ORN datasets, kept separate.** The diagnostics of the two datasets failed the consistency check (`orn_inputs_2026-11-03/README.md`), so they are not merged:
- `ORN_megamat_diag`: dataset 1, 9 flies, 17 megamat odors + 25 diagnostics.
- `ORN_valid2_diag`: dataset 4, 6 flies, 22 validation2 odors + 25 diagnostics.

## Files
| File | What it shows |
|---|---|
| `<set>_map_{PN_to_APL,APL_to_PN}_posterior_view.png` | 2D maps, one row per odor. Columns are the calyces of all four datasets. The color scale is the same as the earlier maps (RdBu_r, white = 0, ends = min and max of both new tables). |
| `<set>_3D_maps_PN_APL.pdf`, `GH146_3D_maps_PN_APL.pdf` | One page per odor (details below). |
| `*_3D_example_*.png` | Single example pages. |
| `odor_similarity_PN_APL_weighted_vs_once.png/.csv` (+ `_summary.csv`) | The Fig 3-style comparison for PN ↔ APL synapses (details below). |
| `<set>_odor_distance_heatmaps_PN_APL.png`, `odor_distance_PN_APL_maps.csv` | Odor-by-odor heat maps (details below). |
| `<set>_map_{KC_to_APL,APL_to_KC,APL_to_KC_PN}_posterior_view.png` | 2D maps of KC ↔ APL synapses on the new claws, colored by the claw's glomerulus; columns are the model's three calyces (hemibrain R, male CNS L and R). `APL_to_KC_PN` = APL → KC on claws plus APL → PN. |
| `<set>_3D_maps_KC_APL.pdf`, `GH146_3D_maps_KC_APL.pdf`, `*_3D_example_KC_*.png` | As the PN ↔ APL 3D maps, for KC ↔ APL on claws and APL → KC + PN. |
| `odor_similarity_KC_APL_weighted_vs_once.png/.csv` (+ `_summary.csv`, `odor_similarity_KC_summary.json`) | The Fig 3-style comparison with glomeruli weighted by KC → APL, APL → KC or APL → KC + PN synapses on claws. |
| `<set>_odor_distance_heatmaps_KC_APL.png`, `odor_distance_KC_APL_maps.csv` | Odor heat maps for the three model calyces with the KC ↔ APL maps. |
| `kc_maps2d_summary.json` | Synapses per map and the share colored. |
| `*_summary.json` | Numbers. |

**3D maps.** Each page has:
- Rows: the five calyces of FlyWire, hemibrain and male CNS (BANC left out, as Yang asked).
- Columns: PN → APL and APL → PN, each from two angles. A = posterior-lateral (from behind, turned 35° toward the calyx's lateral side, 20° up); B = lateral (from the calyx's lateral side, 20° up).

The axes are anatomical. The anterior direction was set so that the peduncle lies anterior to the calyx.

**Weighted vs once (Fig 3-style).** Six panels: ORN and GH146, each for the model's three mushroom bodies (hemibrain R, male CNS L and R).
- x: the cosine similarity of two odors' glomerular responses, each glomerulus counted once.
- y: the same cosine with each glomerulus weighted by its PN ↔ APL calyx synapses.
- ORN uses the 16 TNT odors from `ORN_TNT16_flymean.csv`; GH146 uses the 13 of those it has.

**Odor heat maps.** One figure per ORN dataset.
- How each odor becomes a map: every PN ↔ APL synapse carries its glomerulus's response, and the map is smoothed with a 3D Gaussian of σ = 2 µm.
- What is compared: Euclidean distance (purple) and cosine similarity (blue) between the odors' maps.
- Layout: rows are a "responses only" reference (glomeruli once, no positions) plus five calyces × two directions. The CSV also has σ = 0 and 5 µm.

## Findings
- **Weighting by PN ↔ APL synapses barely changes odor similarity.** Weighted and unweighted similarities correlate at r 0.86–0.99 over odor pairs. Weighting by APL → PN synapses raises ORN similarities slightly, and weighting by PN → APL lowers them slightly (mean change −0.05 to +0.05).
- **Euclidean distances mostly follow response strength.** Strong odors are far from everything. The distance matrices of the calyces match the responses-only one at r 0.93–0.96 (σ = 2 µm, megamat set) and match each other at r ≥ 0.94.
- **Cosine similarities depend more on where the synapses sit.** They match the responses-only matrix at r 0.70–0.77 at σ = 2 µm, falling to 0.51–0.60 at 5 µm. Smoothing mixes neighbouring glomeruli's boutons.
- **KC ↔ APL on claws (new claws) gives the same picture.** Weighted and unweighted similarities correlate at r 0.96–0.99 (mean change −0.03 to +0.04). At σ = 2 µm the Euclidean matrices match the responses-only one at r 0.92–0.96 and the cosine matrices at r 0.76–0.89. 60–82% of the on-claw synapses get a color (fewer with GH146, which has fewer glomeruli).
