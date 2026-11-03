#!/bin/bash
# Rebuild the default version (official ROIs only) into scratch folders, to check that the variant-aware scripts still reproduce
# the delivered files. Nothing is written to /mnt/project-files.
set -e
unset APL_VARIANT
R=$PWD/verify_official
export APL_OUT=$R/out_dir/ APL_FINAL=$R/final/
rm -rf "$R/out_dir" "$R/final"; mkdir -p "$R/out_dir/odor_weighted" "$R/final" "$R/logs"
L=$R/logs
step() { echo "=== $(date +%H:%M:%S) $1"; }
for ds in flywire hemibrain malecns banc; do
  step "run $ds"; python3 -I scripts/run.py $ds > $L/run_$ds.txt 2>&1
done
step export;          python3 -I scripts/export.py flywire hemibrain malecns banc > $L/export.txt 2>&1
step summary_counts;  python3 -I scripts/summary_counts.py > $L/summary_counts.txt 2>&1
step scatter;         python3 -I scripts/scatter.py > $L/scatter.txt 2>&1
step plots3;          python3 -I scripts/plots3.py > $L/plots3.txt 2>&1
rm -f "$APL_OUT/scatter3_KC_APL_reciprocal.png"
step plots_draw;      python3 -I scripts/plots_draw.py > $L/plots_draw.txt 2>&1
step mb_plots;        python3 -I scripts/mb_plots.py > $L/mb_plots.txt 2>&1
step lobe_tables;     python3 -I scripts/lobe_tables.py > $L/lobe_tables.txt 2>&1
step lobe_plots;      python3 -I scripts/lobe_plots.py > $L/lobe_plots.txt 2>&1
step odor_maps;       python3 -I odor/odor_maps.py > $L/odor_maps.txt 2>&1
step gh146_maps;      python3 -I odor/gh146_maps.py > $L/gh146_maps.txt 2>&1
step glom_maps;       python3 -I odor/glom_maps.py > $L/glom_maps.txt 2>&1
step orn_maps;        python3 -I odor/orn_maps.py > $L/orn_maps.txt 2>&1
step claw_glom_maps;  python3 -I odor/claw_glom_maps.py > $L/claw_glom_maps.txt 2>&1
step gh146_per_odor;  python3 -I odor/gh146_per_odor_maps.py > $L/gh146_per_odor_maps.txt 2>&1
step odor_similarity; python3 -I odor/odor_similarity.py > $L/odor_similarity.txt 2>&1   # official ROIs only
step DONE
