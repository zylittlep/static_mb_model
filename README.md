# static_mb_model

A static model of the fly mushroom body that asks whether APL's wiring can explain how blocking APL (TNT) changes odor responses in Kenyon cells (KCs). It is built from four adult connectomes (hemibrain v1.2, male CNS v1.0, FlyWire/FAFB v783, BANC v888) and from Yang's ORN, GH146 and TNT/TNT3c imaging.

This is a snapshot of the code and reports as of 2026-11-03, before the revisions requested that day (new claw segmentation, corrected-calyx rebuild, new model versions).

## Start here
- `index.html` is the project overview, with the main findings and links to every report and the KC skeleton figures. Download the repository and open it in a browser, or turn on GitHub Pages for this repository.
- `reports/` holds the 13 step-by-step reports. The PDFs display directly on GitHub. Each HTML report is one self-contained file with its figures embedded, so download it to view it.

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

## Layout
| Folder | Contents |
|---|---|
| `scripts/` | All analysis code: synapse extraction per connectome, claws and boutons, the outside-ROI version (`offroi/`, `variant.py`), odor-weighted maps, the APL models (block, local, compartment), the TNT comparison and toy simulations. |
| `scripts/reports/` | The scripts that build each report (`rNN_*.py`), the shared layout code (`report_lib.py`) and KaTeX for the math. |
| `reports/` | Reports 00–12 as HTML and PDF, their figures as PNG (`figures/`) and the small tables some reports read (`data/`). |
| `kc_skeletons/` | 10 male CNS KC skeletons with their KC↔APL synapses: calyx and overview figures, an interactive 3D view, the tables and a README. |
| `calyx_roi_check/` | Check of whether the male CNS calyx ROI is misaligned. It is not shifted; the left ROI is cut short at the back. Includes a README, figure, numbers and the ROI masks. |
| `notes/` | Plain-language notes for each model and the full file-by-file project log (`project_log_apl_synapses.md`). |

## Conventions
- Synapse weight is the count of T-bar→PSD pairs, the same as neuPrint "weight" and FlyWire syn_count.
- The models use hemibrain (right MB) and male CNS (left and right MB) only.
- TNT3c is the control (APL intact) and TNT has APL blocked. Each recording counts as an independent hemisphere.
- Odor-pair similarity is the cosine of the KC population responses.
- Two versions of the calyx are kept. One uses the official ROIs only. The other adds back synapses outside the official MB ROIs (`with_outside_ROI_synapses`); the rule is in `scripts/offroi/assign.py` and `notes/with_outside_ROI_synapses/README.md`.

## Running the code
- **Software.** Python 3.13 with numpy 2.5, pandas 2.3, scipy 1.18, matplotlib 3.11, pyarrow, plotly, networkx and tensorstore (the ROI lookup only). The report PDFs also need Node (KaTeX) and headless Chromium.
- **Paths.** The scripts use absolute paths under `/mnt/project-files/apl_synapses/` (outputs and connectome tables) and `/mnt/project-files/uploads/hearth/` (imaging data). To run elsewhere, replace those prefixes.
- **Threads.** Run the models with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, which is much faster here.
- **Not included.**
  - The connectome synapse tables (several hundred MB). The extraction scripts rebuild them from the public Google Cloud buckets named in `notes/project_log_apl_synapses.md`.
  - The ORN, GH146 and TNT/TNT3c imaging data.
