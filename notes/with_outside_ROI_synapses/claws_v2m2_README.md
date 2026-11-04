# Claws chosen by Yang (2026-11-04), as tables for the downstream scripts

These are the new claws in the format the models, maps and reports read. They cover hemibrain and male CNS only, the two datasets with KC skeletons. Built by `scripts/claws_geodesic/export_claws_v2.py` from the claw-site tables of `claws_v2.py`. All of it is the corrected calyx (synapses outside the official ROIs added).

**The rule** (`claw_v2m2`; details in `claws_geodesic/README.md`):
1. All of a KC's calyx synapse sites are clustered at 8 µm along its skeleton.
2. In each cluster, every PN with at least 2 PSDs gets its own claw.
3. Every other site joins the claw whose own PN's PSDs are nearest along the skeleton, if they are within 8 µm. Otherwise the site is not on a claw.

## Files
| File | Contents |
|---|---|
| `<ds>_KC_claws_derived.csv.gz` | One row per claw. Same columns as the old `<ds>_KC_claws_derived.csv.gz`, plus `calyx`, `n_sites`, `n_other_sites`, `cluster` and `shares_cluster_with_other_PN`. `n_PN_synapses` counts the PSDs of the claw's own PN. `main_bouton_id` is the bouton giving most of those PSDs. |
| `KC_APL_calyx_synapses_claws_v2m2.csv.gz` | Every KC ↔ APL synapse in the corrected calyx with its claw (empty = not on a claw), `claw_dist_nm` (along the skeleton), the 8 µm cluster and the old 2 µm claw. |
| `KC_APL_synapses_on_claws_by_glomerulus.csv` | The on-claw rows, same columns as `odor_weighted/KC_APL_synapses_on_claws_by_glomerulus.csv` of the old claws. |
| `export_summary.json` | Checks and counts. |

## Checks
- **Boutons:** rebuilt as `run.py` builds them; identical to the final bouton tables (898 hemibrain, 1,760 male CNS).
- **Matching:** each KC ↔ APL synapse is matched to its claw site by the KC-side position (the KC PSD for APL→KC, the KC T-bar for KC→APL). 98.2% match in hemibrain and 92.2% in male CNS.
- **Unmatched synapses:** almost all are male CNS posterior-calyx synapses (labelled CentralBrain-unspecified) more than 3 µm from any PN→KC PSD, so outside the segmentation's calyx sites. They got the same rule straight on the skeleton (5,354 male CNS, 129 hemibrain); 7–9% of them landed on a claw. They do not change the claws themselves.

## Share of KC ↔ APL calyx synapses on claws
| | KC→APL, new | KC→APL, old 2 µm | APL→KC, new | APL→KC, old 2 µm |
|---|---|---|---|---|
| hemibrain | 56% | 26% | 69% | 49% |
| male CNS | 58% | 21% | 66% | 38% |

Claws: 11,408 on 1,771 hemibrain KCs (6.4 per KC with claws) and 22,176 on 3,780 male CNS KCs (5.9 per KC).
