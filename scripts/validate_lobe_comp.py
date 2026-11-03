"""How well does the in-lobe DAN vote recover male CNS official compartments for APL<->KC lobe synapses?
Run twice: DAN T-bars labelled with the fine lobe (hemibrain situation) and with only VL / ML (FlyWire / BANC situation)."""
import sys, json; sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from prep import malecns
from meta_util import kc_class
from core_lobe import class_lobe, famdist_from_malecns, derive_in_lobe, COARSE
d = malecns(); fam = famdist_from_malecns()
a = pd.read_parquet('out/final_malecns_apl_mb.parquet')
a = a[a.mb_region.isin(list(COARSE)) & a.partner_type.astype(str).str.startswith('KC')].copy()
a['cls'] = a.partner_type.map(kc_class)
a['lobe'] = class_lobe(a.cls.values, a.mb_region.map(COARSE).values)
ok = a.lobe.notna() & (a.lobe == a.mb_region)
print('APL<->KC lobe rows', len(a), 'class-lobe consistent with ROI', round(ok.mean(), 4))
a = a[ok & a.mb_compartment_official.notna()]
a = a[a.mb_compartment_official.str[:-1] + 'L' == a.lobe]  # official compartment inside that lobe
dan = d['dan'].dropna(subset=['lobe', 'family'])
res = {}
for name, reg in [('fine_lobe_DANs', dan.lobe.values), ('VL_ML_DANs', dan.lobe.map(COARSE).values)]:
    comp, share = derive_in_lobe(a[['post_x', 'post_y', 'post_z']].values, a.lobe.values, dan[['x', 'y', 'z']].values, reg, dan.family.values, fam)
    hit = pd.Series(comp == a.mb_compartment_official.values, index=a.index)
    res[name] = {'n': int(len(a)), 'agreement': round(float(hit.mean()), 4),
                 'per_compartment': hit.groupby(a.mb_compartment_official).mean().round(3).to_dict()}
    print(name, res[name], flush=True)
json.dump(res, open('out/lobe_compartment_validation.json', 'w'), indent=1)
