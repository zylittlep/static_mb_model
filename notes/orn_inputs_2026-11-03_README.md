# New ORN inputs (Yang's responses_wide.csv, 2026-11-03)

Built by `scripts/orn_inputs_2026_11_03.py` from `uploads/hearth/780d103e-a7b8-4cc2-ac34-39d15f700dab`. Each fly's trials are averaged first, then the flies (a glomerulus counts only in the flies where it was imaged).

## Tables
| File | What it holds |
|---|---|
| `ORN_megamat_diagnostics_flymean.csv` | Glomeruli × odors for dataset 1 (9 flies): megamat (17 odors) + diagnostics. |
| `ORN_validation2_diagnostics_flymean.csv` | Glomeruli × odors for dataset 4 (6 flies): validation2 (22 odors) + diagnostics. |
| `ORN_*_nflies.csv` | How many flies are behind each value. |
| `ORN_TNT16_flymean.csv` | The 16 TNT odors the model uses: 12 from megamat and 4 from validation2 (paa @ −4, -aPine, euc, B-myr), as in the old `ORN_ovlp16` table. |

- **Duplicate odors.** 2h, ms and va @ −3 were given in both megamat and diagnostics. The megamat presentation is used.
- **Same pattern as before, smaller values.** Compared with the old `ORN_ovlp16` table, each odor's glomerular pattern is almost the same (r 0.96–0.997). The values are smaller, though: about 0.8× for the megamat odors and 0.7× for the validation2 odors.
- **Four more glomeruli** are in the new table: DL4, VA4, VL1 and VM4.

## Are the two diagnostic datasets the same? No.
The diagnostics panel was given to the dataset 1 flies (megamat) and the dataset 4 flies (validation2). The two were compared on the 23 diagnostic odors and 38 glomeruli that both have; see `diagnostics_consistency.png` and `.json`.

**The criterion** (my default, since Yang said "if it passes certain criteria"):
- Compare 3 flies of dataset 1 with 3 flies of dataset 4, and 3 flies with 3 other flies of the same dataset.
- Do this both for the response pattern and for the odor-by-odor similarity matrix.
- The two datasets count as the same only if both of these hold:
  1. The between-dataset correlation is not below the 5th percentile of the within-dataset ones.
  2. A fly-label permutation test finds no difference between the dataset means.

**Results**

| | within dataset 1 | within dataset 4 | between datasets |
|---|---|---|---|
| response pattern r (median) | 0.87 | 0.86 | 0.76 |
| odor-similarity r (median) | 0.82 | 0.82 | 0.53 |

- The fly-label permutation gives p = 0.0002. The difference between the datasets is 2.9 times what reshuffling flies gives.
- Dataset 4's responses are about 15% larger.

**Where they differ** (panels b, c of the figure):
- **HCl @ −1** drives DC4 strongly in dataset 1 (1.24) but hardly in dataset 4 (0.17).
- **DM5** responds to a-terp and 3mtp only in dataset 4.
- **DC1** responds more to fench and paa in dataset 4.
- **Acid-sensing glomeruli.** DC4, DP1m and DP1l have different odor tuning in the two datasets.

Most glomeruli agree well (r > 0.9). Leaving HCl out does not rescue the check: the odor-similarity r is still 0.60 between datasets, against 0.80–0.84 within.

**So the diagnostics are not merged.** Following Yang's request, the maps and correlation heat maps are made separately for megamat + diagnostics and validation2 + diagnostics.
