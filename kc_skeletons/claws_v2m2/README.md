# KC skeletons with APL synapses, new claws (2026-11-04)

The same figures and tables as `kc_skeletons/` (see its README for how synapses are placed on the skeleton), redone with the new claws of `with_outside_ROI_synapses/claws_v2m2/` (8 µm clusters along each KC's skeleton, one claw per PN with at least 2 PSDs) on the corrected calyx. Script: `scripts/kc_skeleton_apl_synapses_v2.py`. Shown in Report 02, Section 5.

- Ten example KCs, right calyx of male CNS (2 each of KCg-m, KCa'b'-ap2, KCab-s, KCab-c, KCab-m): `KC_skeletons_APL_synapses_calyx.png/.pdf`, `_overview.png/.pdf` (whole KC), `_3D.html`; tables `KC_APL_calyx_synapses_on_skeleton.csv`, `KC_APL_synapses_by_location.csv`.
- All 1,901 right-side KCs: `all_right_KCs_APL_calyx_synapses_on_skeleton.csv` (one row per calyx KC↔APL synapse) and `all_right_KCs_APL_synapses_by_location.csv` (one row per KC).
- `summary.json`: the shares below.

Over all right-side KCs, calyx KC↔APL synapses sit on claws (61%), on the dendritic trunk (16%), on branches without a claw (12%) and on claw stalks (10%). KC→APL is 57% on claws, APL→KC 65%. Off-claw synapses are a median 13 µm along the KC from the nearest PN→KC site (90% within 25 µm). With the old 2 µm claws, 29% of the right-side calyx synapses were on claws.
