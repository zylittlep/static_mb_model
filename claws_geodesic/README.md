# Claws by distance along the KC skeleton (2026-11-03, for Yang's review)

Yang asked (2026-11-03) for a new claw segmentation, a comparison with the old claws, and 40 KCs where the two differ, to review before choosing.

**Nothing downstream uses these claws yet.** Reports 02–12 still use the old claws.

**Update 2026-11-04 (later): Yang corrected the rule.** All calyx sites are clustered at 8 µm along the skeleton, every PN in a cluster gets its own claw, and other sites join a claw only if its nearest PN→KC PSD is ≤8 µm away. Results are in `Yang_rule_2026-11-04/`; see "Yang's corrected rule" at the end. The two versions below are superseded.

**Update 2026-11-04.** Yang's own version clustered only PN→KC synapses, so the PN-PSDs-only variant is the faithful one. Its review figures and tables are in `PN_PSDs_only/` (same file names as below; the 40 KCs there are picked by how much the PN-only claws differ from the old ones). See "PN PSDs only" at the end.

## The new method
Each KC is processed on its own:

1. **Sites.** Take all of the KC's synapse sites in the calyx, both T-bars (KC presynaptic) and PSDs (KC postsynaptic), whatever the partner.
2. **Calyx.** The calyx here is the corrected calyx. A site counts if it is labeled CA, or lies within 3 µm of a corrected-calyx PN→KC PSD (outside-ROI version).
3. **Geodesic distance.** Each site is placed on the nearest node of the KC's skeleton. The distance between two sites is the distance along the skeleton.
   - Skeletons are the public SWCs: hemibrain v1.2 from the skeleton archive, male CNS v1.0 per-body files.
   - Unconnected skeleton pieces are joined by their shortest links.
4. **Clusters.** Single linkage at 8 µm, which is the same as cutting the minimum spanning tree of the geodesic distances at 8 µm.
5. **Claws.** A cluster with at least one PN→KC PSD is a claw. Clusters without PN input (dendritic trunk sites) are not claws.
6. **Split.** A claw is split when it has more than 40 sites and its PN→KC PSDs come from two PNs, with the second PN giving at least 5 PSDs.
   - The PSDs of the two PNs seed the two parts. Every other site joins the nearer seed along the skeleton.
   - Each part's main PN, and so its glomerulus, is its own PN.
   - The rule is applied again to each part, because many 8 µm clusters hold three or more PNs' claws.

**Old method, for comparison.** Single linkage at 2 µm in straight-line distance on PN→KC PSDs only (`scripts/run.py`), recomputed here on the same corrected-calyx PSDs.

**An alternative is also computed:** the same steps on PN→KC PSDs only. Its columns and CSVs say "PN PSDs".

## Results
| | hemibrain R | male CNS L | male CNS R |
|---|---|---|---|
| KCs | 1,800 | 1,886 | 1,908 |
| claws per KC, old | 8.2 | 7.5 | 7.4 |
| claws per KC, new | 5.8 | 5.3 | 5.3 |

- **Effect of the split.** Before splitting, 25% of the new clusters have >40 sites and a second PN with ≥5 PSDs; 47% of the final claws come from a split.
- **Purity.** The main PN's share of a claw's PN PSDs (PSD-weighted) is 0.951 for the old claws and 0.958 for the new.
- **Agreement with Yang's hemibrain counts** (1,800 KCs in both; 83 of Yang's KCs have no calyx PN input here, 35 of them with 0 claws in Yang's file):

| method | vs `n_claws_corrected`: r | same count | within 1 | mean difference |
|---|---|---|---|---|
| old (2 µm) | 0.71 | 17% | 46% | +1.5 |
| new, all sites (main) | 0.86 | 29% | 69% | −0.9 |
| new, PN PSDs only | 0.83 | 26% | 70% | −0.4 |

- **Before splitting**, all-site clusters give 4.0 per KC against Yang's `n_claws` of 6.1 (corrected 6.7). Synapses on the dendritic trunk chain neighbouring claws together within 8 µm; only the PN split brings the count back up.
- **The PN-PSD-only clusters before splitting** match Yang's `n_claws` in mean (6.14 vs 6.13, r 0.79).
- **No setting tried reproduces Yang's counts closely.** The scan covered:
  - site set: all sites or PN PSDs; official CA only or the corrected calyx
  - cut: 3–10 µm
  - snap distance: added or not
  - minimum PN PSDs per claw: 1–5
  - split threshold: >20/30/40 sites, second PN ≥1–10

  The best match is 34% identical counts and r 0.85, so Yang's pipeline differs in some step.

## Files
- **Figures for review:**
  - `claw_disparity_KCs.pdf` and `claw_disparity_KCs_page1-5.png` show the 40 KCs where old and new differ most. There are 20 hemibrain R and 10 each from male CNS L and R, balanced across γ, αβ and α′β′. They are ranked by |Δ claws| + (1 − adjusted Rand index between the two labelings of the KC's PN PSDs).
  - `claw_disparity_KCs_3D.html` shows the same 40 KCs in 3D. Old and new are side by side and rotate together, and a dropdown picks the KC.
  - `claw_compare_summary.png` gives the counts against Yang's, the change per KC, claw sizes and the means per class.
- **Tables:**
  - `claws_per_KC.csv` has one row per KC: claws by every method, Yang's counts, the adjusted Rand index and Δ.
  - `claws_new_*.csv` has one row per new claw (main PN, glomerulus, sites, PN PSDs, centroid, from_split), one file per variant.
  - `disparity_KCs.csv` lists the 40 KCs.
  - `<ds>_claw_sites.parquet` has one row per KC calyx site with every labeling (`claw_old`, `claw_all`/`clawrec_all`, `claw_pn`/`clawrec_pn`, `cl0_*` before split).
  - `claw_compare_summary.json` and `<ds>_claw_run.json` hold the numbers and parameters.
- **Scripts** (`scripts/claws_geodesic/`):
  - `extract_kc_sites.py` gets every KC site near the calyx from the release synapse files.
  - `segment_claws.py` does the segmentation.
  - `compare_claws.py` does the comparison and figures. Set `CLAW_MAIN=all` (used here) or `pn`.

## PN PSDs only (`PN_PSDs_only/`, 2026-11-04)
Same steps as above, but step 1 keeps only the KC's PN→KC PSDs, so the split rule reads ">40 PN PSDs". Made with `CLAW_MAIN=pn compare_claws.py`.

| | hemibrain R | male CNS L | male CNS R |
|---|---|---|---|
| claws per KC, old | 8.2 | 7.5 | 7.4 |
| claws per KC, new (PN PSDs only) | 6.3 | 5.6 | 5.5 |

- **Agreement with Yang's hemibrain counts:** r 0.83, same 26%, within 1 70%, mean difference −0.4 against `n_claws_corrected`. Before the split: r 0.79, same 27%, mean difference +0.01 against `n_claws`.
- **The split changes little.** Only 3.5% of clusters have >40 PN PSDs and a second PN with ≥5 PSDs, and 7% of the final claws come from a split. Yang's correction changes the count in far more KCs, so their split rule probably counted something other than PN PSDs (asked Yang 2026-11-04).
- **Purity** (main PN's share of a claw's PN PSDs): 0.950, the same as the old claws.
- **By class** (hemibrain R, mean claws per KC, new vs Yang corrected): γ 8.2 vs 9.1, αβ 5.2 vs 5.5, α′β′ 5.2 vs 5.0.

## Yang's corrected rule (`Yang_rule_2026-11-04/`, 2026-11-04)
Yang's words: take all synapses on a KC, cluster by geodesic distance (minimum spanning tree cut at 8 µm); PN→KC PSDs in one cluster with different PN partners are split by partner; non-PN sites (e.g. APL) join a claw only if the nearest PN→KC synapse on that claw is within 8 µm, otherwise they are not on a claw.

Implementation (`scripts/claws_geodesic/claws_v2.py`, on the sites and 8 µm clusters of `segment_claws.py`):
1. The clusters are the all-site 8 µm clusters (`cl0_all`).
2. In each cluster, each PN with at least *m* PSDs there gets its own claw. *m* = 1 is the rule as written; *m* = 2, 3, 5 are kept for comparison. PSDs of PNs below *m* are treated like the other sites.
3. Every other site joins the claw whose nearest PN→KC PSD is closest along the skeleton, if that distance is ≤8 µm.

Columns `claw_v2m<m>` (−1 = not on a claw), `split_v2m<m>` (claw shares its cluster with another PN's claw) and `dist_v2m<m>` (µm to the nearest claw PSD) are in `<ds>_claw_sites.parquet`. The 40-KC figures and 3D view use *m* = 1.

| PN claw needs | claws per KC, hemibrain R | vs Yang `n_claws_corrected`: r | same | within 1 | mean diff |
|---|---|---|---|---|---|
| ≥1 PSD (as written) | 7.3 | 0.86 | 28% | 69% | +0.6 |
| ≥2 PSDs | 6.3 | 0.89 | 38% | 79% | −0.3 |
| ≥3 PSDs | 6.1 | 0.89 | 40% | 79% | −0.5 |
| ≥5 PSDs | 5.8 | 0.89 | 39% | 75% | −0.8 |
| old 2 µm | 8.2 | 0.71 | 17% | 46% | +1.5 |

Yang's file has 6.7 corrected claws per KC (6.1 before correction).
- **As written, the rule overcounts** because a single stray PSD from a PN makes its own claw. 13% of hemibrain claws (6% in male CNS) have one PN PSD. Requiring ≥2 PSDs gives the closest match of anything tried.
- **Claws per KC with *m* = 1:** male CNS L 6.3, male CNS R 6.2.
- **Synapses on claws (*m* = 1):** 85–89% of each KC's calyx synapse sites are on a claw, and about 70% of APL→KC PSDs (hemibrain 72%, male CNS 71%). The old 2 µm rule put 30–49% of APL→KC calyx synapses on claws.
- **Clusters with more than one PN:** 37% of the 8 µm clusters hold PSDs from two or more PNs, and 63% of the final claws share their cluster with another PN's claw.


### Chosen: ≥2 PSDs per PN (`Yang_rule_2026-11-04/chosen_min2PSD/`, Yang's choice 2026-11-04)
Yang picked *m* = 2 (column `claw_v2m2`). Reports 2–12 are rebuilt on these claws.

**Three-way comparison, hemibrain R** (`compare_three.py` → `three_way_claw_comparison.png`, `three_way_agreement.csv`, `three_way_by_subtype.csv`, `three_way_summary.json`). The comparison covers the old 2 µm claws, the new claws and Prat's segmentation (the uploaded `KC_Claw_Counts_Hemibrain` file, `n_claws_corrected`). Prat's file has claw counts per KC only, no claw positions, so per-KC examples can show Prat's count but not Prat's claws.

| pair | r | mean difference | same count | within 1 | within 2 |
|---|---|---|---|---|---|
| new vs Prat's | 0.89 | −0.33 | 38% | 79% | 95% |
| old vs Prat's | 0.71 | +1.54 | 17% | 46% | 67% |
| new vs old | 0.79 | −1.86 | 17% | 46% | 66% |

Means per KC: old 8.2, new 6.3, Prat's 6.7. 1,800 KCs are in all three. 83 of Prat's KCs have no corrected-calyx PN input in our tables and are left out (73 "KC part due to gap").

**40 example KCs** (`claw_disparity_KCs.pdf`, `_page1-5.png`, `_3D.html`): the KCs where old and new differ most, old claws on the left, new on the right, Prat's count in each title. PN→KC PSDs are colored and shaped by the glomerulus of their presynaptic PN, identical in both panels; the outlines show the claws.

**Tables for the downstream scripts:** `with_outside_ROI_synapses/claws_v2m2/` (`scripts/claws_geodesic/export_claws_v2.py`; README there).
