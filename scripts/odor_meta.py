"""Shared loaders for the odor-weighted PN<->APL maps (types, PN sets, APL rows with coordinates)."""
import re, sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from variant import FINAL, fw_apl_raw
D = 'dl/'; O = 'out/'
FW_APL = {'720575940613583001', '720575940624547622'}
P = 720575940000000000


def meta_types(ds):
    """id(str) -> type, set of antennal-lobe PN ids, set of proofread ids (FlyWire only)."""
    if ds == 'flywire':
        c = pd.read_csv(D + 'classification.csv.gz', dtype={'root_id': str})
        t = pd.read_csv(D + 'consolidated_cell_types.csv.gz', dtype={'root_id': str}).drop_duplicates('root_id').set_index('root_id')['primary_type']
        return c.root_id.map(t).set_axis(c.root_id), set(c.root_id[c['class'] == 'ALPN']), set(c.root_id), c.set_index('root_id')['side']
    if ds == 'hemibrain':
        h = pd.read_feather(D + 'hemibrain_121_meta.feather')
        typ = pd.Series(h.cell_type.values, index=h.hemibrain_121_id.astype(str))
        off = pd.read_parquet(O + 'hemibrain_body_types.parquet')
        off = pd.Series(off['type'].values, index=off.body.astype('int64').astype(str))
        off = off[~off.index.duplicated()]
        typ = off.combine_first(typ[~typ.index.duplicated()])
        return typ, set(h.hemibrain_121_id[h.cell_class == 'antennal_lobe_projection_neuron'].astype(str)), None, None
    if ds == 'malecns':
        m = pd.read_feather(D + 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
        ids = m.bodyId.astype('int64').astype(str)
        return pd.Series(m.type.values, index=ids), set(ids[m['class'] == 'ALPN']), None, None
    b = pd.read_feather(D + 'banc_888_meta.feather', columns=['banc_888_id', 'cell_type', 'cell_class'])
    ids = b.banc_888_id.astype(str)
    typ = pd.Series(b.cell_type.values, index=ids)
    cx = pd.read_csv(D + 'neurons.csv.gz', dtype={'Root ID': str}).drop_duplicates('Root ID').set_index('Root ID')['Primary Cell Type']
    typ = typ.where(typ.notna(), typ.index.to_series().map(cx))
    return typ, set(ids[b.cell_class == 'antennal_lobe_projection_neuron']), None, None


def apl_rows(ds, proof=None):
    """Every APL synapse in the MB ROIs: apl_id, apl_is_pre, partner_id, region, pre/post xyz in nm (final tables are already nm)."""
    if ds == 'flywire':
        r = fw_apl_raw()
        r['pre_id'] = (r.pre_root_id_720575940.astype('int64') + P).astype(str)
        r['post_id'] = (r.post_root_id_720575940.astype('int64') + P).astype(str)
        n = r.neuropil.fillna('')
        r = r[n.str.startswith('MB_') & (r.pre_id != r.post_id) & r.pre_id.isin(proof) & r.post_id.isin(proof)].copy()
        r['apl_is_pre'] = r.pre_id.isin(FW_APL)
        t = pd.DataFrame({'apl_id': np.where(r.apl_is_pre, r.pre_id, r.post_id), 'apl_is_pre': r.apl_is_pre.values,
                          'partner_id': np.where(r.apl_is_pre, r.post_id, r.pre_id),
                          'region': r.neuropil.str.replace(r'^MB_', '', regex=True).str.replace(r'_[LR]$', '', regex=True).values})
        for c in ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z']: t[c] = r[c].astype(float).values
        return t
    a = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet')
    t = pd.DataFrame({'apl_id': a.apl_id.astype(str).values, 'apl_is_pre': a.apl_is_pre.values,
                      'partner_id': a.partner_id.astype(str).values, 'region': a.mb_region.values})
    for c in ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z']: t[c] = a[c].astype(float).values
    return t
