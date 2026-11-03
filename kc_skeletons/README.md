# KC skeletons with their KC↔APL synapses (male CNS, right mushroom body)

Yang asked (2026-10-28): show 10 KC skeletons with their KC↔APL synapses, to see where APL synapses sit outside the claws.
Built by `scripts/kc_skeleton_apl_synapses.py` (usage at the top of the script).

## Files
- `KC_skeletons_APL_synapses_calyx.png` / `.pdf`: only the calyx part of each KC, drawn on the corrected calyx, in a posterior view and a side view (same scale and limits in every panel), with claw centroids (dashed rings), the KC's PN→KC synapses (gray dots) and its calyx KC↔APL synapses. Redone 2026-10-29 at Yang's request (calyx part only, with the corrected calyx).
- `KC_skeletons_APL_synapses_overview.png` / `.pdf`: each whole KC in a posterior view and a side view, lobes and peduncle included, with the corrected calyx area drawn behind it.
- `KC_skeletons_APL_synapses_3D.html`: rotatable 3D view of the calyx part of each KC inside a faint cloud of the corrected calyx (gray inside the official ROI, orange outside it; a uniform sample of 12,000 calyx synapses). Pick the KC from the menu; hovering a synapse shows its location, its distance to the nearest claw and whether it lies outside the official ROIs. Opens offline (plotly is embedded).
- `KC_APL_synapses_by_location.csv`: one row per KC, synapse counts by direction and location.
- `KC_APL_calyx_synapses_on_skeleton.csv`: one row per calyx synapse, with its location class, claws downstream, distance along the KC to the nearest PN→KC synapse, and how far it was moved to snap onto the skeleton.
- `summary.json`: the shares quoted below.

## How the KCs were picked
Right-side KCs of male CNS v1.0 (APL_R) with at least 4 claws whose number of calyx KC↔APL synapses lies between the 40th and 90th percentile of their type, so each is a typical, well-connected KC. Picked at random (seed 1028), spread over classes: 3 KCg-m, 1 each of KCa'b'-ap1, KCa'b'-ap2, KCa'b'-m, 2 KCab-s, 1 KCab-c, 1 KCab-m.
Their calyx synapses are 32% on claws, close to all right-side KCs (29% with the added synapses, 32% official ROIs only), so they are representative on this point.

## The corrected calyx and the calyx part of a KC
- **Corrected calyx** (as in `with_outside_ROI_synapses/`): every PN→KC synapse assigned to CA(R) plus every calyx synapse of APL_R, inside the official CA(R) ROI or added from outside it (the official ROI misses part of the posterior calyx). The figures draw its projected area: pale gray where the official CA(R) ROI already covers it, pale orange where only the added synapses reach. The area is the outline of the synapse cloud, not the ROI mesh.
- **Calyx part of a KC**: skeleton nodes within 3 µm of any corrected-calyx synapse, keeping only the pieces that carry the KC's own calyx synapses (so bits of the soma neurite passing nearby drop out). Its cable length is in `KC_APL_synapses_by_location.csv` (`calyx_skeleton_length_um`, 123–268 µm).

## Where a synapse sits
- **On claw**: the KC side of the synapse is within 2 µm of a claw (claw_id set; the rule used everywhere in this project).
- For off-claw calyx synapses, the skeleton is rooted at its lobe tip, and each synapse's node gets the number of claws (with PN→KC input) beyond it, away from the axon:
  - **claw stalk** (1 claw beyond): on the branch leading to a single claw.
  - **dendritic trunk** (2 or more claws beyond): on the shared dendrite before it splits into claws.
  - **branch without claw** (no claw beyond): a side branch that ends without a claw. This also includes the soma neurite, which these skeletons do not label.
- **Peduncle** and **lobes**: by the synapse's MB region.
- Marker shape: circle = KC→APL (KC presynaptic), triangle (diamond in 3D) = APL→KC. A black outline in the 2D figures marks synapses outside the official ROIs that were added back (column `added_outside_official_ROI`).

## Results (185 calyx synapses on these 10 KCs)
- About two-thirds of calyx KC↔APL synapses are off claws: dendritic trunk 30%, claw stalk 21%, branch without claw 16%, on claw 32%.
- KC→APL is the more off-claw direction (28% on claws; 37% on the trunk). APL→KC is 37% on claws.
- Off-claw synapses sit a median 6.4 µm along the KC from the nearest PN→KC synapse (90% within 20 µm): about 6 µm on stalks and the trunk, 11–14 µm on branches without claws.
- Most KC↔APL synapses of these KCs are outside the calyx: lobes 751, peduncle 80, calyx 185 (of 1016).
- Official ROIs only: 168 calyx synapses, 33% on claws. The 17 added synapses split across all classes (4 on claws, 2 stalk, 7 trunk, 4 branch without claw).

## Caveats
- The public male CNS skeletons are coarse (median edge 0.7 µm). Synapses were snapped to the nearest node (median 0.55 µm, maximum 2 µm), so a synapse next to a branch point can land on the neighboring branch.
- Claws come from the project's derived claws, and PN→KC synapses were assigned to the nearest claw centroid. A claw with few PN synapses can be missed, which would turn its stalk into a "branch without claw".
- Only male CNS. Hemibrain KCs can be added the same way, but their skeletons come in one 1.8 GB archive.
