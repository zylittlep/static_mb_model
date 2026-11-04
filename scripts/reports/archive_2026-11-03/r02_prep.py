"""Prep for Report 02 (run once from the working directory that holds out/, like run.py).
Re-derives the claws of hemibrain and male CNS from the raw PN->KC calyx synapses with the same rule as run.py
(single linkage at 2 um per KC), checks that the claw count matches the delivered claw tables, and saves to reports/data/:
  r02_claw_purity.csv   one row per claw: dataset, claw_id, n_PN_synapses, n_PNs, main_PN_share
  r02_example_points.csv  PN->KC sites of one example PN (its T-bars, with bouton) and one example KC (its PSDs, with claw)."""
import sys, os
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
from core import cluster_by_owner

B = '/mnt/project-files/apl_synapses/'
OD = B + 'reports/data/'; os.makedirs(OD, exist_ok=True)
rows, ex = [], []
for ds in ['hemibrain', 'malecns']:
    p = pd.read_parquet(f'out/{ds}_pnca_raw.parquet')
    claws = pd.read_csv(f'{B}{ds}_KC_claws_derived.csv.gz', dtype={'kc_id': str, 'main_PN_id': str})
    bout = pd.read_csv(f'{B}{ds}_PN_boutons_derived.csv.gz', dtype={'pn_id': str})
    t = pd.DataFrame({'pre_id': p.body_pre.astype('int64').astype(str), 'post_id': p.body_post.astype('int64').astype(str)})
    for s in ['pre', 'post']:
        for c in 'xyz': t[f'{s}_{c}'] = p[f'{c}_{s}'].astype(float) * 8
    tb = t.drop_duplicates(['pre_id', 'pre_x', 'pre_y', 'pre_z']).reset_index(drop=True)
    tb['bouton_id'] = cluster_by_owner(tb.pre_id.values, tb[['pre_x', 'pre_y', 'pre_z']].values, 2000, 'bouton')
    assert tb.bouton_id.nunique() == len(bout), (ds, tb.bouton_id.nunique(), len(bout))
    kc = t[t.post_id.isin(set(claws.kc_id))].reset_index(drop=True)
    kc['claw_id'] = cluster_by_owner(kc.post_id.values, kc[['post_x', 'post_y', 'post_z']].values, 2000, 'claw')
    assert kc.claw_id.nunique() == len(claws), (ds, kc.claw_id.nunique(), len(claws))
    g = kc.groupby('claw_id').pre_id
    pur = pd.DataFrame({'n_PN_synapses': g.size(), 'n_PNs': g.nunique(), 'main_PN_share': g.agg(lambda s: s.value_counts().iloc[0] / len(s))})
    rows.append(pur.reset_index().assign(dataset=ds))
    if ds == 'hemibrain':
        # example PN: the median-sized DM1 PN by T-bar count; example KC: an alpha/beta KC with the median claw count
        bt = bout.assign(n=1).groupby(['pn_id', 'pn_type']).n_tbars.sum().reset_index()
        dm1 = bt[bt.pn_type.astype(str).str.startswith('DM1_')].sort_values('n_tbars')
        pn = dm1.pn_id.iloc[len(dm1) // 2]
        e1 = tb[tb.pre_id == pn][['pre_x', 'pre_y', 'pre_z', 'bouton_id']].rename(columns=lambda c: c.replace('pre_', '')).rename(columns={'bouton_id': 'unit'})
        ex.append(e1.assign(dataset=ds, cell=pn, kind='PN T-bars by bouton', cell_type=dm1.pn_type.iloc[len(dm1) // 2]))
        cpk = claws.groupby(['kc_id', 'kc_type']).size().rename('n').reset_index()
        cand = cpk[(cpk.kc_type == 'KCab-m') & (cpk.n == 8)].kc_id.sort_values()
        k = cand.iloc[len(cand) // 2]
        e2 = kc[kc.post_id == k][['post_x', 'post_y', 'post_z', 'claw_id', 'pre_id']].rename(columns=lambda c: c.replace('post_', '')).rename(columns={'claw_id': 'unit'})
        e2['pre_type'] = e2.pre_id.map(bout.drop_duplicates('pn_id').set_index('pn_id').pn_type)
        ex.append(e2.assign(dataset=ds, cell=k, kind='KC PN-input sites by claw', cell_type='KCab-m'))
    print(ds, 'boutons', len(bout), 'claws', len(claws), 'main-PN share median', pur.main_PN_share.median(), '>=0.9:', (pur.main_PN_share >= 0.9).mean())
pd.concat(rows).to_csv(OD + 'r02_claw_purity.csv', index=False)
pd.concat(ex).to_csv(OD + 'r02_example_points.csv', index=False)
