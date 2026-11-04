"""Prep for the rebuilt Report 04 (2026-11-04): the bouton weight p_g per calyx, for every connectome (Yang: "a figure with
p_g per connectome: row 1 FlyWire, row 2 hemibrain, row 3 male CNS L, ...").

Same counts as scripts/scatter.py (PN_calyx_weights_APL_vs_KC.csv, corrected calyx), split by calyx:
- PN -> KC synapses: the calyx side of each synapse is that of the nearest APL calyx synapse (APL innervates its own calyx);
- APL <-> PN synapses: the side of the APL.
Checks that the two sides add up to PN_calyx_weights_APL_vs_KC.csv for every PN.

Usage (from the working folder with out/, as scatter.py): APL_VARIANT=outside_roi python3 r04_prep_v2.py
Writes reports/data/r04v2_PN_calyx_weights_per_side.csv and r04v2_pg_per_calyx.csv."""
import sys, re
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from prep import malecns, hemibrain, flywire, banc
from variant import OUT, FINAL, fw_apl_raw, OUTSIDE
assert OUTSIDE, 'run with APL_VARIANT=outside_roi (corrected calyx)'

OD = '/mnt/project-files/apl_synapses/reports/data/'
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'


def uni(t, ds):
    t = str(t)
    if t.startswith('M_') or '_' not in t: return ''
    g = [HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in t.split('_')[0].split('+') if re.fullmatch(GPAT, n)]
    return g[0] if len(g) == 1 else ''


def sid(s):
    return pd.Series(s).astype('int64').astype(str).values if pd.Series(s).dtype != object else pd.Series(s).astype(float).astype('int64').astype(str).values \
        if pd.Series(s).astype(str).str.contains(r'\.').any() else pd.Series(s).astype(str).values


ref = pd.read_csv(OUT + 'PN_calyx_weights_APL_vs_KC.csv', dtype={'pn_id': str})
rows, check = [], {}
for ds, fn in [('flywire', flywire), ('hemibrain', hemibrain), ('malecns', malecns), ('banc', banc)]:
    d = fn(); meta = d['meta'].drop_duplicates('id').set_index('id')
    p = d['pnca'].copy()
    if ds == 'flywire':
        proof = set(meta.index); p = p[(p.pre_id != p.post_id) & p.post_id.isin(proof)]
    p = p[p.post_id.map(meta['type']).astype(str).str.startswith('KC')].copy()
    p['pre_id'] = sid(p.pre_id)
    fa = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet'); fa = fa[fa.mb_region == 'CA'].copy()
    if ds == 'hemibrain': fa['apl_side'] = fa.apl_side.fillna('right')        # one APL (the right one) in hemibrain
    fa['apl_side'] = fa.apl_side.replace({'L': 'left', 'R': 'right'})
    assert fa.apl_side.isin(['left', 'right']).all(), (ds, fa.apl_side.unique())
    # side of each PN -> KC synapse = side of the nearest APL calyx synapse
    tree = cKDTree(fa[['post_x', 'post_y', 'post_z']].values.astype(float))
    dist, ix = tree.query(p[['post_x', 'post_y', 'post_z']].values.astype(float))
    p['side'] = fa.apl_side.values[ix]
    check[ds] = {'PN->KC nearest APL synapse, median nm': round(float(np.median(dist)), 1), '90th pct nm': round(float(np.percentile(dist, 90)), 1)}
    kc = p.groupby(['pre_id', 'side']).size().rename('PN_to_KC')
    a = fa
    if ds == 'flywire':
        r = fw_apl_raw(); P = 720575940000000000
        r['pre_id'] = (r.pre_root_id_720575940.astype('int64') + P).astype(str); r['post_id'] = (r.post_root_id_720575940.astype('int64') + P).astype(str)
        r = r[r.neuropil.fillna('').str.startswith('MB_CA') & (r.pre_id != r.post_id) & r.pre_id.isin(proof) & r.post_id.isin(proof)]
        A = {'720575940613583001': 'right', '720575940624547622': 'left'}
        assert fa.groupby('apl_id').apl_side.first().to_dict() == A
        r['apl_is_pre'] = r.pre_id.isin(A); r['partner_id'] = r.post_id.where(r.apl_is_pre, r.pre_id)
        r['apl_side'] = r.pre_id.where(r.apl_is_pre, r.post_id).map(A); a = r
    pns = set(d['pnca'].pre_id.astype(str))
    a = a.assign(partner_id=sid(a.partner_id)); a = a[a.partner_id.isin(set(sid(pd.Series(list(pns)))))]
    ain = a[~a.apl_is_pre].groupby(['partner_id', 'apl_side']).size().rename('PN_to_APL')
    aout = a[a.apl_is_pre].groupby(['partner_id', 'apl_side']).size().rename('APL_to_PN')
    kc.index = kc.index.set_names(['pn_id', 'side']); ain.index = ain.index.set_names(['pn_id', 'side']); aout.index = aout.index.set_names(['pn_id', 'side'])
    t = pd.concat([kc, ain, aout], axis=1).fillna(0).astype(int).reset_index()
    t['pn_type'] = t.pn_id.map(dict(zip(sid(pd.Series(meta.index)), meta['type']))); t.insert(0, 'dataset', ds)
    # the two sides add up to the published per-PN table
    s = t.groupby('pn_id')[['PN_to_KC', 'PN_to_APL', 'APL_to_PN']].sum()
    rr = ref[ref.dataset == ds].set_index('pn_id')[['PN_to_KC', 'PN_to_APL', 'APL_to_PN']]
    assert set(s.index) == set(rr.index), (ds, len(set(s.index) ^ set(rr.index)))
    assert (s.reindex(rr.index) == rr).all().all(), ds
    t['glomerulus'] = [uni(x, ds) for x in t.pn_type]
    rows.append(t)
    check[ds]['PN->KC by side'] = t.groupby('side').PN_to_KC.sum().to_dict()
    check[ds]['APL->PN by side'] = t.groupby('side').APL_to_PN.sum().to_dict()
    print(ds, check[ds], flush=True)
T = pd.concat(rows, ignore_index=True)
T.to_csv(OD + 'r04v2_PN_calyx_weights_per_side.csv', index=False)
G = T[T.glomerulus != ''].groupby(['dataset', 'side', 'glomerulus']).agg(
    n_PNs=('pn_id', 'nunique'), PN_to_KC=('PN_to_KC', 'sum'), PN_to_APL=('PN_to_APL', 'sum'), APL_to_PN=('APL_to_PN', 'sum')).reset_index()
with np.errstate(invalid='ignore', divide='ignore'):
    G['p_g'] = G.APL_to_PN / G.PN_to_KC
    G['p_in_g'] = G.PN_to_APL / G.PN_to_KC
G.to_csv(OD + 'r04v2_pg_per_calyx.csv', index=False)
import json; json.dump(check, open(OD + 'r04v2_prep_check.json', 'w'), indent=1, default=int)
