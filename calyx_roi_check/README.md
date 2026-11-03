# Is the male CNS calyx ROI misaligned? (2026-11-03)

Yang asked whether the roughly one third of left KC↔APL calyx synapses that fall outside the official CA(L) ROI in male CNS v1.0 could come from a misaligned calyx ROI.

**Answer:** the ROI is not shifted. The left calyx ROI is cut short at the back of the calyx. Its front edge matches the calyx exactly, but its back edge stops about 6 µm before the calyx ends. The left calyx itself is as big as the right one, but its ROI is 22% smaller.

## How it was checked
- **Synapse labels come from the ROI volume.** The release labels synapses by looking them up in the ROI volume `gs://flyem-male-cns/rois/fullbrain-roi-v4` at 256 nm, and reproduces those labels 100% on a 40,000-synapse sample. Versions v4.1 and v5 give the same labels (`roi_volume_vs_release_labels.json`).
- **Synapses tested.** The calyx synapses are the corrected calyx: ALPN→KC (postsynaptic site) and KC↔APL calyx synapses, inside the official ROI or added back from outside it by the 3 µm rule.
- **Scripts.** `scripts/calyx_roi_check_prep.py` (volume vs labels) and `scripts/calyx_roi_check.py` (everything else).

## Results (`calyx_roi_check.json`)

| | CA(L) | CA(R) |
|---|---|---|
| ALPN→KC inside the ROI | 81.4% | 93.9% |
| KC↔APL inside the ROI | 66.4% | 86.9% |
| Best shift of the ROI (µm, x y z) | −3.0, +1.75, +2.25 | +1.5, +0.5, −1.0 |
| Inside after that shift (ALPN→KC / KC↔APL) | 86.2% / 76.2% | 94.7% / 87.3% |
| Best small rotation (±8°) + shift, all calyx synapses | 85% | 94% |
| ROI volume left empty (>3 µm from any calyx synapse), as is / shifted | 2% / 9% | 3% / 5% |
| Front-back extent along the calyx axis (µm), synapses | −16.2 to +16.9 | −15.2 to +15.9 |
| Front-back extent, ROI | −16.2 to +10.6 | −14.9 to +15.2 |
| Volume (µm³): synapse cloud / ROI | 59,062 / 49,031 | 59,031 / 63,004 |
| Outside synapses: distance to ROI, median / 90% / max (µm) | 2.1 / 5.5 / 18.7 | 1.0 / 2.9 / 12.1 |
| Share of outside synapses in the back third of the calyx | 71% | 31% |

- **No rigid fix exists.** A rigid shift or small rotation can't fix the left ROI. Moving it back to cover the posterior calyx uncovers its front, so the best shift only reaches 86% and leaves more of the ROI empty.
- **Why KC↔APL synapses are hit harder than ALPN→KC.** They sit slightly further back in the left calyx (median +1.3 vs −0.8 µm along the calyx axis). So 15.5% of them lie behind the ROI's back edge, against 9.2% of ALPN→KC.
- **The right calyx ROI fits well.** Its misses are thin, about 1 µm at the edges all round.

## Figure
`calyx_roi_check.png` shows both calyces in posterior, side and top views. It has the projected ROI outline (blue), synapses inside it (gray) and synapses outside it (orange). In the side and top views, the left calyx's orange band sits behind the ROI's back edge.

Other files: `calyx_roi_shift_scan.csv` (share inside for every 1 µm shift of the ROI within ±14 µm) and `calyx_roi_masks_256nm.npz` (the two ROI masks with their voxel offsets).
