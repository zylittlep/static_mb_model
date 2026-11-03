# APL synapses, version with the synapses outside the official ROIs

Built 2026-10-14. Same tables, plots and maps as the main folder (`../`), built by the same scripts, except that synapses lying
outside every official mushroom body ROI but belonging to the mushroom body are added, in all four datasets and in the calyx,
peduncle and lobes. Every added row is flagged. The main folder keeps the official ROIs only and is unchanged.

## Which synapses are added (`../scripts/offroi/assign.py`)
- **Anchors.** Calyx: the ALPN→KC synapses inside the official calyx ROI, extended through ALPN→KC synapses outside every MB ROI
  that join them in steps of ≤3 µm (claws the ROI leaves out). Peduncle and lobes: the APL↔KC synapses inside the official
  peduncle and lobe ROIs.
- **Growing through KC synapses.** APL↔KC synapses outside every MB ROI then join whichever anchor reaches them first, in steps of
  ≤3 µm, and take that anchor's region. KC dendrites and axons exist only in the mushroom body, so these chains stay inside it.
  This step matters most in the male CNS, whose calyx ROI misses the back of the calyx, where many KC↔APL synapses lie far from
  any claw.
- **Joining.** Any synapse outside every MB ROI joins the mushroom body when its postsynaptic site lies within 3 µm of an anchor
  point, with the region and side of the nearest one: CA, PED or a lobe (aL, a'L, bL, b'L, gL in hemibrain and male CNS; VL, ML
  in FlyWire and BANC). The calyx wins ties.
- **Accessory calyces stay out**, as in the official tables: no chain runs through synapses with γd or αβp KCs, whose dendrites
  form the accessory calyces next to the calyx and peduncle. (See the caveats for the few that still join.)
- **PN synapses near the calyx** are added when they are ALPN→KC, or when their T-bar belongs to a calyx bouton (joins a T-bar of
  the same PN that contacts a KC, in steps of ≤2 µm). PN synapses next to the calyx onto lateral horn and SLP neurons stay out.
- Applied to every table: APL synapses (all partners), the ALPN→KC synapses behind claws and per-KC PN input, PN boutons, ALPN
  synapses anywhere in the MB (whole-MB plots), and KC→MBON synapses (lobe plots). Same synapse filters as the main folder.

## How much is added
Synapses added, as a share of the official-ROI count:

| | FlyWire | hemibrain | male CNS | BANC |
|---|---|---|---|---|
| APL synapses, whole MB (delivered table) | +50,238 (+18.1%) | +3,809 (+1.6%) | +51,838 (+11.8%) | +12,655 (+18.8%) |
| … calyx | +3.6% | +2.6% | +30.9% | +2.0% |
| … peduncle | +23.9% | +5.5% | +11.9% | +17.3% |
| … lobes | +24.6% | +0.5% | +6.4% | +56.5% |
| KC↔APL, calyx | +4.0% | +2.2% | +32.0% | +2.2% |
| ALPN→KC, calyx | +0.8% | +2.8% | +14.0% | +0.7% |
| KC→MBON, lobes | +30.1% | +0.4% | +6.0% | +39.6% |

Where they come from (`roi_official` of the added rows): male CNS mostly `CentralBrain-unspecified` (the back of the calyx, and
gaps around the γ lobe and peduncle) and CRE (γ lobe); FlyWire and BANC mostly CRE, SIP and SMP, whose meshes take in parts of the
lobes (BANC ML +116%), and PLP, SCL and ICL next to the peduncle and calyx.

Mean per KC, official → this version:

| | FlyWire | hemibrain | male CNS | BANC |
|---|---|---|---|---|
| PN→KC, calyx | 72.5 → 73.0 | 93.5 → 96.1 | 84.3 → 96.1 | 37.4 → 37.6 |
| KC→APL, calyx | 5.8 → 6.1 | 9.7 → 9.9 | 6.7 → 9.1 | 2.0 → 2.0 |
| APL→KC, calyx | 7.8 → 8.0 | 13.2 → 13.5 | 7.5 → 9.7 | 2.1 → 2.1 |
| KC→APL, whole MB | 23.8 → 28.7 | 51.8 → 52.4 | 46.0 → 51.2 | 3.7 → 4.4 |
| APL→KC, whole MB | 24.1 → 28.2 | 48.6 → 49.3 | 43.1 → 47.6 | 4.2 → 5.0 |

With the added synapses, male CNS PN input per KC matches hemibrain (96 vs 96), and its calyx KC↔APL synapses per KC go from
38% below hemibrain to 20% below.

`added_synapses_summary.csv` has the counts per dataset, APL side, MB region, direction and partner group (official ROI vs
added). `where_synapses_were_added.png` shows them, in posterior view and in a side view of each MB: grey = inside the official ROIs,
colored = added (by region), dark = outside every MB ROI and not added.

## Columns that differ from the main tables
- `mb_region`: the region a synapse belongs to (its official ROI, or the region it joined).
- `mb_region_source`: `official ROI`, `added: outside the official ROIs, within 3 µm of the calyx`, or
  `added: outside the official ROIs, within 3 µm of the peduncle/lobes`. Keep `official ROI` rows to get the main-folder tables back.
- `outside_roi_dist_nm`: for added rows, the distance from the postsynaptic site to the nearest anchor point.
- `roi_official` keeps the dataset's own label (e.g. `CentralBrain-unspecified`, `SLP(R)`, `CRE_L`).
- `mb_compartment_source`: for added calyx and peduncle rows it repeats `mb_region_source`. Male CNS added lobe rows outside the
  official subcompartment volumes get their compartment from the DAN vote (`derived (DAN kNN)`), as the other datasets do.
- FlyWire's main table still applies the Codex ≥5-synapses-per-pair threshold, counted over the whole brain, so the added
  synapses don't change which pairs pass it.

## Caveats
- Where an accessory calyx touches the calyx or peduncle, synapses with γd and αβp KCs within 3 µm of an anchor still join:
  FlyWire 3,813, hemibrain 1,487 (1,033 of its 1,411 peduncle additions), male CNS 1,291, BANC 161. They are added rows with
  `partner_type` KCg-d or KCab-p in CA or PED, if you want to drop them. (γd synapses added in the γ lobe are axonal and belong
  there.)
- BANC: 1,539 APL synapses within 10 µm of the MB whose partners are unannotated fragments stay out (725 within 5 µm, mostly
  next to the medial lobe), because they lie more than 3 µm from every anchor point.

## Files
Same names and meaning as in the main folder: the four `<dataset>_APL_MB_synapses.csv.gz` tables, KC claws and PN boutons,
`summary_counts_by_compartment_and_partner_class.csv`, the per-PN / per-KC weight tables and scatter plots (calyx, whole MB,
lobes and compartments), `lobe_tables_summary.json`, and everything in `odor_weighted/` (odor, GH146, GH146 per-odor, ORN,
per-glomerulus and claw maps). Figure titles end with "[incl. synapses outside the official ROIs]".
Not repeated here: `calyx_shape_by_dataset.*` (it shows where the official ROIs leave synapses out, see `../odor_weighted/`) and
the compartment-method calibration files (`malecns_dan_type_compartment.csv`, `*_compartment_validation.json`).

## Scripts
`../scripts/variant.py` switches every pipeline script to this version (`APL_VARIANT=outside_roi`); without it the same scripts
rebuild the main folder (checked 2026-10-14: same tables and pixel-identical plots, except 5 hemibrain γ3–γ5 rows of
`summary_counts_by_compartment_and_partner_class.csv`, which was made from an earlier hemibrain table and is 1–4 synapses off). `../scripts/offroi/` holds the scans of
synapses outside the ROIs (`scan_flywire.py` after `dl_princeton.sh`, `scan_hemibrain.py`, `scan_malecns.py`,
`scan_malecns_pn.py`, `scan_banc.py`), the assignment (`assign.py`), the full rebuild (`build_variant.sh`), this folder's QC
figure and counts (`qc_added.py`), and the check that the main folder still rebuilds identically (`verify_official.sh`,
`compare_official.py`).
