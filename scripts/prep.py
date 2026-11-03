"""Normalise each dataset into common frames (coordinates in nm).
apl:  syn_id, pre_id, post_id, pre_x..z, post_x..z, roi_official, conf, (extra)
pnca: pre_id (PN), post_id, pre_x..z, post_x..z
dan:  pre_id (DAN), x,y,z, lobe (LOBE_COMPS key), family
meta: id -> type, class, side/instance
"""
import numpy as np, pandas as pd, re, sys
sys.path.insert(0, 'scripts')
from core import dan_family
from variant import OUTSIDE, ADDED
O = 'out/'; D = 'dl/'
XYZ = ['x', 'y', 'z']
MB_PREFIX = ('CA(', 'PED(', 'aL(', "a'L(", 'bL(', "b'L(", 'gL(')
LOBE_FLYEM = {'aL': 'aL', "a'L": "a'L", 'bL': 'bL', "b'L": "b'L", 'gL': 'gL'}


def outside_roi(df, ds):
    """Outside-ROI version: MB label, source and distance of every raw APL synapse (offroi/assign.py; row-aligned)."""
    g = pd.read_parquet(ADDED + f'{ds}_apl_assigned.parquet')
    for c in g.columns:
        df[c] = g[c].values
    return df


def flyem_lobe(roi):
    m = re.match(r"^([a-z]'?L)\(", str(roi))
    return m.group(1) if m else None


def malecns():
    m = pd.read_feather(D + 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    meta = pd.DataFrame({'id': m.bodyId.astype('int64').astype(str), 'type': m.type, 'class': m['class'],
                         'superclass': m.superclass, 'instance': m.instance, 'side': m.somaSide})
    a = pd.read_parquet(O + 'malecns_apl_raw.parquet')
    apl = pd.DataFrame({'pre_id': a.body_pre.astype(str), 'post_id': a.body_post.astype(str)})
    for s in ['pre', 'post']:
        for c in XYZ:
            apl[f'{s}_{c}'] = a[f'{c}_{s}'].astype(float) * 8
    apl['roi_official'] = a.primary_post.values
    apl['conf_pre'] = a.conf_pre.values; apl['conf_post'] = a.conf_post.values
    apl['syn_id'] = ['mcns_%d_%d_%d__%d_%d_%d' % t for t in zip(a.x_pre, a.y_pre, a.z_pre, a.x_post, a.y_post, a.z_post)]
    if OUTSIDE: outside_roi(apl, 'malecns')
    p = pd.read_parquet(ADDED + 'malecns_pnca_raw.parquet' if OUTSIDE else O + 'malecns_pnca_raw.parquet')
    pn = pd.DataFrame({'pre_id': p.body_pre.astype(str), 'post_id': p.body_post.astype(str)})
    for s in ['pre', 'post']:
        for c in XYZ:
            pn[f'{s}_{c}'] = p[f'{c}_{s}'].astype(float) * 8
    d = pd.read_parquet(O + 'malecns_dan_comp.parquet')
    dan = pd.DataFrame({'pre_id': d.body_pre.astype(str), 'x': d.x_pre * 8., 'y': d.y_pre * 8., 'z': d.z_pre * 8.,
                        'lobe': d.primary_post.map(flyem_lobe), 'family': d['type'].map(dan_family),
                        'comp_official': d.comp})
    return dict(apl=apl, pnca=pn, dan=dan, meta=meta)


def hemibrain():
    h = pd.read_feather(D + 'hemibrain_121_meta.feather')
    meta = pd.DataFrame({'id': h.hemibrain_121_id.astype(str), 'type': h.cell_type, 'class': h.cell_class,
                         'superclass': h.super_class, 'instance': h.instance, 'side': None})
    # The compiled table uses FlyWire-harmonised names and mislabels ~200 a'b' KCs as KCg-m;
    # use the official neuPrint v1.2 type (from the synapse-points release file) as `type`.
    off = pd.read_parquet(O + 'hemibrain_body_types.parquet')
    off = off.assign(id=off.body.astype('int64').astype(str)).drop_duplicates('id').set_index('id')['type']
    meta['type_harmonized'] = meta['type']
    meta['type'] = meta.id.map(off).fillna(meta['type'])
    extra = off[~off.index.isin(meta.id)]
    meta = pd.concat([meta, pd.DataFrame({'id': extra.index, 'type': extra.values})], ignore_index=True)
    a = pd.read_parquet(O + 'hemibrain_apl_raw.parquet')
    apl = pd.DataFrame({'pre_id': a.body_pre.astype('int64').astype(str), 'post_id': a.body_post.astype('int64').astype(str)})
    for s in ['pre', 'post']:
        for c in XYZ:
            apl[f'{s}_{c}'] = a[f'{c}_{s}'].astype(float) * 8
    apl['roi_official'] = a.roi_post.values
    apl['roi_pre'] = a.roi_pre.values
    apl['conf_pre'] = a.conf_pre.values; apl['conf_post'] = a.conf_post.values
    apl['syn_id'] = ['hb_%d_%d_%d__%d_%d_%d' % t for t in zip(a.x_pre, a.y_pre, a.z_pre, a.x_post, a.y_post, a.z_post)]
    if OUTSIDE: outside_roi(apl, 'hemibrain')
    p = pd.read_parquet(ADDED + 'hemibrain_pnca_raw.parquet' if OUTSIDE else O + 'hemibrain_pnca_raw.parquet')
    pn = pd.DataFrame({'pre_id': p.body_pre.astype('int64').astype(str), 'post_id': p.body_post.astype('int64').astype(str)})
    for s in ['pre', 'post']:
        for c in XYZ:
            pn[f'{s}_{c}'] = p[f'{c}_{s}'].astype(float) * 8
    d = pd.read_parquet(O + 'hemibrain_dan_raw.parquet')
    tmap = meta.set_index('id')['type']
    dan = pd.DataFrame({'pre_id': d.body.astype('int64').astype(str), 'x': d.x * 8., 'y': d.y * 8., 'z': d.z * 8.,
                        'lobe': d.roi.map(flyem_lobe)})
    dan['family'] = dan.pre_id.map(tmap).map(dan_family)
    return dict(apl=apl, pnca=pn, dan=dan, meta=meta)


def fw_lobe(npil):
    s = str(npil)
    if s.startswith('MB_VL'): return 'VL'
    if s.startswith('MB_ML'): return 'ML'
    return None


def flywire():
    PFX = 720575940000000000
    c = pd.read_csv(D + 'classification.csv.gz', dtype={'root_id': str})
    t = pd.read_csv(D + 'consolidated_cell_types.csv.gz', dtype={'root_id': str})
    meta = c.merge(t[['root_id', 'primary_type']], on='root_id', how='left')
    meta = pd.DataFrame({'id': meta.root_id, 'type': meta.primary_type, 'class': meta['class'],
                         'superclass': meta.super_class, 'instance': None, 'side': meta.side})
    proof = set(meta.id)

    def fix(df):
        df = df.rename(columns={'pre_root_id_720575940': 'pre_id', 'post_root_id_720575940': 'post_id'})
        df['pre_id'] = (df.pre_id.astype('int64') + PFX).astype(str)
        df['post_id'] = (df.post_id.astype('int64') + PFX).astype(str)
        for s in ['pre', 'ctr', 'post']:
            for k in XYZ:
                df[f'{s}_{k}'] = df[f'{s}_{k}'].astype(float)
        return df
    a = fix(pd.read_parquet(O + 'fafbP_apl_raw.parquet'))
    if OUTSIDE: outside_roi(a, 'flywire')
    # Codex default: proofread neurons only, no autapses, >=5 synapses per neuron pair
    a = a[(a.pre_id != a.post_id) & a.pre_id.isin(proof) & a.post_id.isin(proof)]
    n = a.groupby(['pre_id', 'post_id']).size().rename('pair_n').reset_index()
    a = a.merge(n, on=['pre_id', 'post_id']); a = a[a.pair_n >= 5]
    a['roi_official'] = a.neuropil.fillna('')
    a['syn_id'] = ['fwP_%d_%d_%d' % t for t in zip(a.ctr_x, a.ctr_y, a.ctr_z)]
    a = a.rename(columns={'size': 'syn_size'})
    p = fix(pd.read_parquet(ADDED + 'flywire_pnca_raw.parquet' if OUTSIDE else O + 'fafbP_pnca_raw.parquet'))
    d = pd.read_parquet(O + 'fafbP_dan_raw.parquet')
    d = fix(d)
    tmap = meta.set_index('id')['type']
    dan = pd.DataFrame({'pre_id': d.pre_id, 'x': d.pre_x, 'y': d.pre_y, 'z': d.pre_z, 'lobe': d.neuropil.map(fw_lobe)})
    dan['family'] = dan.pre_id.map(tmap).map(dan_family)
    return dict(apl=a.reset_index(drop=True), pnca=p, dan=dan, meta=meta)


def banc():
    b = pd.read_feather(D + 'banc_888_meta.feather')
    meta = pd.DataFrame({'id': b.banc_888_id.astype(str), 'type': b.cell_type, 'class': b.cell_class,
                         'superclass': b.super_class, 'instance': None, 'side': b.side})
    cx = pd.read_csv(D + 'neurons.csv.gz', dtype={'Root ID': str})  # codex BANC 888 (has APL_L typed)
    cx = cx.set_index('Root ID')
    miss = meta.type.isna() & meta.id.isin(cx.index)
    meta.loc[miss, 'type'] = cx.loc[meta.id[miss], 'Primary Cell Type'].values
    a = pd.read_parquet(O + 'banc_v2_apl_raw.parquet')
    if OUTSIDE: outside_roi(a, 'banc')
    a = a.rename(columns={'pre_root_id': 'pre_id', 'post_root_id': 'post_id', 'size': 'syn_size'})
    a['roi_official'] = a.neuropil.fillna('') + '_' + a.side.fillna('').str[0].str.upper()
    a['syn_id'] = 'banc_' + a['id'].astype(str)
    p = pd.read_parquet(ADDED + 'banc_pnca_raw.parquet' if OUTSIDE else O + 'banc_pnca_raw.parquet').rename(columns={'pre_root_id': 'pre_id', 'post_root_id': 'post_id'})
    d = pd.read_parquet(O + 'banc_dan_raw.parquet').rename(columns={'pre_root_id': 'pre_id', 'post_root_id': 'post_id'})
    tmap = meta.set_index('id')['type']
    dan = pd.DataFrame({'pre_id': d.pre_id, 'x': d.pre_x, 'y': d.pre_y, 'z': d.pre_z, 'lobe': d.neuropil.map(fw_lobe)})
    dan['family'] = dan.pre_id.map(tmap).map(dan_family)
    return dict(apl=a, pnca=p, dan=dan, meta=meta)
