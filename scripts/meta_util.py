"""Cell types and KC / MBON id sets per dataset (same type sources as the per-KC tables)."""
import pandas as pd
D = 'dl/'; O = 'out/'


def kc_class(t):
    t = str(t)
    if t.startswith('KCg'): return 'γ'
    if t.startswith("KCa'b'") or t.startswith('KCapbp'): return 'α′β′'
    if t.startswith('KCab'): return 'αβ'
    if t.startswith('KC'): return 'no subtype'
    return None


def types(ds):
    """Series id(str) -> type, and the set of MBON ids."""
    if ds == 'flywire':
        c = pd.read_csv(D + 'classification.csv.gz', dtype={'root_id': str})
        t = pd.read_csv(D + 'consolidated_cell_types.csv.gz', dtype={'root_id': str}).drop_duplicates('root_id').set_index('root_id')['primary_type']
        return pd.Series(c.root_id.map(t).values, index=c.root_id), set(c.root_id[c['class'] == 'MBON'])
    if ds == 'hemibrain':
        h = pd.read_feather(D + 'hemibrain_121_meta.feather')
        typ = pd.Series(h.cell_type.values, index=h.hemibrain_121_id.astype(str))
        off = pd.read_parquet(O + 'hemibrain_body_types.parquet')
        off = pd.Series(off['type'].values, index=off.body.astype('int64').astype(str))
        typ = off[~off.index.duplicated()].combine_first(typ[~typ.index.duplicated()])
        return typ, set(typ.index[typ.astype(str).str.startswith('MBON')])
    if ds == 'malecns':
        m = pd.read_feather(D + 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
        ids = m.bodyId.astype('int64').astype(str)
        return pd.Series(m.type.values, index=ids), set(ids[m['class'] == 'MBON'])
    b = pd.read_feather(D + 'banc_888_meta.feather', columns=['banc_888_id', 'cell_type', 'cell_class'])
    ids = b.banc_888_id.astype(str)
    typ = pd.Series(b.cell_type.values, index=ids)
    cx = pd.read_csv(D + 'neurons.csv.gz', dtype={'Root ID': str}).drop_duplicates('Root ID').set_index('Root ID')['Primary Cell Type']
    typ = typ.where(typ.notna(), typ.index.to_series().map(cx))
    return typ, set(ids[b.cell_class == 'mushroom_body_output_neuron'])


def kc_ids(typ):
    return set(typ.index[typ.astype(str).str.startswith('KC')])
