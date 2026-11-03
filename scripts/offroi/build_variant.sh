#!/bin/bash
# Build the version with the synapses outside the official ROIs into apl_synapses/with_outside_ROI_synapses/.
# Run from the working folder (with dl/, out/, scripts/, odor/) after offroi/assign.py <dataset> for all four datasets.
# The default build (official ROIs only) is the same chain without APL_VARIANT.
set -e
export APL_VARIANT=outside_roi
V=/mnt/project-files/apl_synapses/with_outside_ROI_synapses/
L=offroi/logs_variant
mkdir -p "$V/odor_weighted" "$L"
step() { echo "=== $(date +%H:%M:%S) $1"; }
for ds in flywire hemibrain malecns banc; do
  step "run $ds"; python3 -I scripts/run.py $ds > $L/run_$ds.txt 2>&1
done
step export;          python3 -I scripts/export.py flywire hemibrain malecns banc > $L/export.txt 2>&1
step summary_counts;  python3 -I scripts/summary_counts.py > $L/summary_counts.txt 2>&1
step scatter;         python3 -I scripts/scatter.py > $L/scatter.txt 2>&1
step plots3;          python3 -I scripts/plots3.py > $L/plots3.txt 2>&1
rm -f "$V/scatter3_KC_APL_reciprocal.png"     # plots3 draft; plots_draw.py writes the calyx version
step plots_draw;      python3 -I scripts/plots_draw.py > $L/plots_draw.txt 2>&1
step mb_plots;        python3 -I scripts/mb_plots.py > $L/mb_plots.txt 2>&1
step lobe_tables;     python3 -I scripts/lobe_tables.py > $L/lobe_tables.txt 2>&1
cp out/outside_roi/lobe_tables_summary.json "$V"
step lobe_plots;      python3 -I scripts/lobe_plots.py > $L/lobe_plots.txt 2>&1
step odor_maps;       python3 -I odor/odor_maps.py > $L/odor_maps.txt 2>&1
step gh146_maps;      python3 -I odor/gh146_maps.py > $L/gh146_maps.txt 2>&1
step glom_maps;       python3 -I odor/glom_maps.py > $L/glom_maps.txt 2>&1
step orn_maps;        python3 -I odor/orn_maps.py > $L/orn_maps.txt 2>&1
step claw_glom_maps;  python3 -I odor/claw_glom_maps.py > $L/claw_glom_maps.txt 2>&1
step gh146_per_odor;  python3 -I odor/gh146_per_odor_maps.py > $L/gh146_per_odor_maps.txt 2>&1
step DONE
