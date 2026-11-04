# static_mb_model

A static model of the fly mushroom body that asks whether APL's wiring can explain how blocking APL (TNT) changes odor responses in Kenyon cells (KCs). It is built from four adult connectomes (hemibrain v1.2, male CNS v1.0, FlyWire/FAFB v783, BANC v888) and from Yang's ORN, GH146 and TNT/TNT3c imaging.

This is a snapshot of the code and reports as of 2026-11-04. The reports were rebuilt that day for Yang's revision list of 2026-11-03: they use the corrected calyx (synapses just outside the official ROIs added back), new claws found along each KC's skeleton, the new ORN table, and new model versions (a PN-level step fitted to GH146, negative responses, inhibition on both PN boutons and KCs). Each report opens with what changed. The earlier versions are in the git history (commits of 2026-11-03).

## Start here
- `index.html` is the project overview, with the main findings and links to every report and the KC skeleton figures. Download the repository and open it in a browser, or turn on GitHub Pages for this repository.
- `reports/` holds the 13 step-by-step reports and a Q&A page answering the questions on Reports 05–11. The PDFs display directly on GitHub. Each HTML report is one self-contained file with its figures embedded, so download it to view it.

| # | Report |
|---|---|
| 00 | [Overview, roadmap and glossary](reports/00_overview_and_glossary.pdf) |
| 01 | [Connectome data: APL synapse tables and per-cell counts](reports/01_connectome_synapse_tables.pdf) |
| 02 | [Claws, boutons, calyx geometry and the outside-ROI version](reports/02_claws_boutons_calyx.pdf) |
| 03 | [Odor-weighted maps of APL synapses](reports/03_odor_weighted_maps.pdf) |
| 04 | [APL wiring per glomerulus: the connectome numbers the model uses](reports/04_glomerulus_wiring.pdf) |
| 05 | [The APL block model, built step by step](reports/05_APL_block_model.pdf) |
| 06 | [The TNT and TNT3c recordings: what is measured](reports/06_TNT_TNT3c_data.pdf) |
| 07 | [The model pinned to the recordings, and the comparison](reports/07_model_vs_TNT.pdf) |
| 08 | [Why APL feedback makes the effect of blocking uneven across odors](reports/08_why_feedback_is_uneven.pdf) |
| 09 | [Rank order of odor-pair similarity, with toy simulations](reports/09_rank_order.pdf) |
| 10 | [Distances along APL between glomeruli](reports/10_APL_geodesic_distances.pdf) |
| 11 | [A local, non-spiking APL in the KC model](reports/11_local_APL_model.pdf) |
| 12 | [APL confined to its calyx sub-arbors](reports/12_APL_sub_arbor_compartments.pdf) |
| Q&A | [Answers to the questions on Reports 05–11 (2026-11-03)](reports/QA_2026-11-03_reports_05-11.pdf) |

## Layout
| Folder | Contents |
|---|---|
| `scripts/` | All analysis code: synapse extraction per connectome, claws and boutons (the new skeleton-based claws in `claws_geodesic/`), the outside-ROI version (`offroi/`, `variant.py`), odor-weighted maps (`odor_maps_2026_11/` for the new ORN table), the APL models (block, local, compartment), the TNT comparison and toy simulations. |
| `scripts/reports/` | The scripts that build each report (`rNN_*.py`) and the Q&A page (`qa_2026_11_03.py`), the shared layout code (`report_lib.py`) and KaTeX for the math. The report scripts as they were before the revision are in `archive_2026-11-03/`. |
| `reports/` | Reports 00–12 as HTML and PDF, their figures as PNG (`figures/`) and the small tables some reports read (`data/`). |
| `kc_skeletons/` | 10 male CNS KC skeletons with their KC↔APL synapses: calyx and overview figures, an interactive 3D view, the tables and a README. `claws_v2m2/` holds the same with the new claws, plus the location of every calyx KC↔APL synapse of all 1,901 right-side KCs. |
| `claws_geodesic/` | The new claws (Yang's rule of 2026-11-04, at least 2 PSDs per PN): README with the rule, the comparison with the old 2 µm claws and Prat's hemibrain claw counts, and the 40 KCs where they disagree most (PDF, PNG pages and a 3D view). |
| `calyx_roi_check/` | Check of whether the male CNS calyx ROI is misaligned. It is not shifted; the left ROI is cut short at the back. Includes a README, figure, numbers and the ROI masks. |
| `notes/` | Plain-language notes for each model, the full file-by-file project log (`project_log_apl_synapses.md`), and READMEs for the new claw tables, the new ORN table and its maps. |

## Conventions
- Synapse weight is the count of T-bar→PSD pairs, the same as neuPrint "weight" and FlyWire syn_count.
- The models use hemibrain (right MB) and male CNS (left and right MB) only.
- Claws (since 2026-11-04): a KC's calyx sites are clustered at 8 µm along its skeleton, and in each cluster every PN with at least 2 PSDs gets its own claw (`APL_VARIANT=claws_v2m2`). Hemibrain and male CNS only; FlyWire and BANC keep the older 2 µm claws.
- TNT3c is the control (APL intact) and TNT has APL blocked. Each recording counts as an independent hemisphere. Responses are FC z-scores.
- Odor-pair similarity is the cosine of the KC population responses.
- Two versions of the calyx are kept. One uses the official ROIs only. The other adds back synapses outside the official MB ROIs (`with_outside_ROI_synapses`); the rule is in `scripts/offroi/assign.py` and `notes/with_outside_ROI_synapses/README.md`. The reports use the second (corrected) calyx since 2026-11-04.

## Running the code
- **Software.** Python 3.13 with numpy 2.5, pandas 2.3, scipy 1.18, matplotlib 3.11, pyarrow, plotly, networkx and tensorstore (the ROI lookup only). The report PDFs also need Node (KaTeX) and headless Chromium.
- **Paths.** The scripts use absolute paths under `/mnt/project-files/apl_synapses/` (outputs and connectome tables) and `/mnt/project-files/uploads/hearth/` (imaging data). To run elsewhere, replace those prefixes.
- **Threads.** Run the models with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, which is much faster here.
- **Not included.**
  - The connectome synapse tables (several hundred MB). The extraction scripts rebuild them from the public Google Cloud buckets named in `notes/project_log_apl_synapses.md`.
  - The ORN, GH146 and TNT/TNT3c imaging data.
