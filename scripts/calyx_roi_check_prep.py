"""Male CNS calyx ROI check (Yang, 2026-11-03): which official ROI volume matches the v1.0 release's synapse ROI labels.
Looks up a sample of left and right calyx synapses (official CA plus those added outside the ROI) in each candidate ROI volume
(gs://flyem-male-cns/rois/fullbrain-roi-v4, v4.1, v5; 256 nm) and compares with the release label (primary ROI of the PSD).
Usage: python3 scripts/calyx_roi_check_prep.py <raw PN->KC calyx parquet>"""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
from roilookup import lookup, names
pn = pd.read_parquet(sys.argv[1], columns=['x_post', 'y_post', 'z_post', 'primary_post', 'roi_assigned'])
rng = np.random.default_rng(0)
smp = pn.iloc[rng.choice(len(pn), 40000, replace=False)]
P = smp[['x_post', 'y_post', 'z_post']].values * 8.0
out = {}
for v in ['fullbrain-roi-v4', 'fullbrain-roi-v4.1', 'fullbrain-roi-v5']:
    path = f'gs://flyem-male-cns/rois/{v}/'
    lab = lookup(path, P, 0, 256.0)
    rel = smp.primary_post.values
    inca = np.isin(rel, ['CA(L)', 'CA(R)'])
    agree = (lab[inca] == rel[inca]).mean()
    un = rel == 'CentralBrain-unspecified'
    out[v] = {'official CA labels reproduced': round(float(agree), 4),
              'CentralBrain-unspecified points: volume label': pd.Series(lab[un]).replace('', '<none>').value_counts().head(5).to_dict(),
              'all points: agreement with release label': round(float((pd.Series(lab).replace('', 'CentralBrain-unspecified').values == rel).mean()), 4)}
    print(v, json.dumps(out[v], ensure_ascii=False), flush=True)
json.dump(out, open('/mnt/project-files/apl_synapses/calyx_roi_check/roi_volume_vs_release_labels.json', 'w'), indent=1, ensure_ascii=False)
