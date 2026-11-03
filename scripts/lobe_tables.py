"""Per KC, per lobe and per lobe compartment: KC->MBON, KC->APL and APL->KC synapse counts (MB lobes only).
Lobe = KC class + vertical/medial position (core_lobe.class_lobe); synapses that cannot sit in the KC's own lobes
(e.g. a gamma KC in the vertical lobe) are dropped and counted.
Compartment: male CNS official subcompartment ROI where it lies in that lobe, otherwise (and for the other datasets)
the in-lobe DAN vote (core_lobe.derive_in_lobe; 97.8% agreement with male CNS official compartments).
Counts are T-bar->PSD pairs; FlyWire uses all Princeton synapses between proofread neurons (no per-pair threshold).
In the outside-ROI version (variant.py) the APL and KC->MBON rows also include the lobe synapses outside the official ROIs
that offroi/assign.py adds."""
import sys, json; sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from prep import malecns, hemibrain, flywire, banc
from meta_util import types, kc_class
from core_lobe import class_lobe, famdist_from_malecns, derive_in_lobe, COARSE
from variant import OUTSIDE, ADDED, FINAL, OUT, fw_apl_raw

O = 'out/'
FW_APL = {'720575940613583001', '720575940624547622'}
P = 720575940000000000
fam = famdist_from_malecns()
XYZ = ['post_x', 'post_y', 'post_z']


def apl_rows(ds, kcs, proof):
    """APL<->KC rows in the lobes: kc_id, direction, coarse (VL/ML), roi (fine lobe ROI or ''), post xyz (nm), official comp."""
    if ds == 'flywire':
        r = fw_apl_raw()
        r['pre_id'] = (r.pre_root_id_720575940.astype('int64') + P).astype(str)
        r['post_id'] = (r.post_root_id_720575940.astype('int64') + P).astype(str)
        n = r.neuropil.fillna('')
        r = r[(n.str.startswith('MB_VL') | n.str.startswith('MB_ML')) & (r.pre_id != r.post_id) & r.pre_id.isin(proof) & r.post_id.isin(proof)]
        pre_apl = r.pre_id.isin(FW_APL)
        t = pd.DataFrame({'kc_id': np.where(pre_apl, r.post_id, r.pre_id), 'direction': np.where(pre_apl, 'APL_to_KC', 'KC_to_APL'),
                          'coarse': r.neuropil.str[3:5].values, 'roi': ''})
        for c in XYZ: t[c] = r[c].astype(float).values
        t['comp_official'] = None
    else:
        a = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet')
        a = a[a.mb_region.isin(list(COARSE) + ['VL', 'ML'])]
        t = pd.DataFrame({'kc_id': a.partner_id.astype(str).values, 'direction': np.where(a.apl_is_pre, 'APL_to_KC', 'KC_to_APL'),
                          'coarse': a.mb_region.map(lambda s: COARSE.get(s, s)).values,
                          'roi': a.mb_region.where(a.mb_region.isin(list(COARSE)), '').values})
        for c in XYZ: t[c] = a[c].astype(float).values
        t['comp_official'] = a['mb_compartment_official'].values if 'mb_compartment_official' in a else None
    return t[t.kc_id.isin(kcs)]


def kcmbon(ds):
    """KC -> MBON synapses in the official lobe ROIs, plus (outside-ROI version) those added to the lobes."""
    r = pd.read_parquet(O + f'{ds}_kcmbon_lobes.parquet')
    if OUTSIDE:
        r = pd.concat([r, pd.read_parquet(ADDED + f'{ds}_kcmbon_added.parquet', columns=list(r.columns))], ignore_index=True)
    return r


def mbon_rows(ds, kcs, mbons):
    if ds == 'flywire':
        r = kcmbon('flywire')
        t = pd.DataFrame({'kc_id': (r.pre_root_id_720575940.astype('int64') + P).astype(str),
                          'mbon_id': (r.post_root_id_720575940.astype('int64') + P).astype(str), 'coarse': r.neuropil.str[3:5], 'roi': ''})
        for c in XYZ: t[c] = r[c].astype(float).values
    elif ds == 'hemibrain':
        r = kcmbon('hemibrain')
        lobe = r.roi_post.str.extract(r"^([a-z]'?L)\(")[0]
        t = pd.DataFrame({'kc_id': r.body_pre.astype('int64').astype(str), 'mbon_id': r.body_post.astype('int64').astype(str),
                          'coarse': lobe.map(COARSE), 'roi': lobe})
        for c, s in zip(XYZ, ['x_post', 'y_post', 'z_post']): t[c] = r[s].astype(float).values * 8
    elif ds == 'malecns':
        r = kcmbon('malecns')
        lobe = r.roi.str.extract(r"^([a-z]'?L)\(")[0]
        t = pd.DataFrame({'kc_id': r.body_pre.astype('int64').astype(str), 'mbon_id': r.body_post.astype('int64').astype(str),
                          'coarse': lobe.map(COARSE), 'roi': lobe, 'side': r.roi.str[-2]})
        for c, s in zip(XYZ, ['x_post', 'y_post', 'z_post']): t[c] = r[s].astype(float).values * 8
    else:
        r = kcmbon('banc')
        t = pd.DataFrame({'kc_id': r.pre_root_id.astype(str), 'mbon_id': r.post_root_id.astype(str), 'coarse': r.neuropil.str[3:5], 'roi': ''})
        for c in XYZ: t[c] = r[c].astype(float).values
    t['direction'] = 'KC_to_MBON'
    return t[t.kc_id.isin(kcs) & t.mbon_id.isin(mbons)]


def official_lookup(t):
    """Male CNS official subcompartment for each post point, looked up per lobe ROI and side to keep the read box small."""
    from roilookup import lookup
    out = pd.Series(None, index=t.index, dtype=object)
    for key, g in t.groupby(['roi', 'side']):
        oc = pd.Series(lookup('gs://flyem-male-cns/rois/malecns-subcompartments-v3/', g[XYZ].values, 0, 256), index=g.index)
        out.loc[g.index] = oc.str.replace(r'\((L|R)\)', '', regex=True).where(lambda s: s.str.match(r"^(a|a'|b|b'|g)\d$"), None)
    return out


summary, long = {}, []
DSS = sys.argv[1:] or ['flywire', 'hemibrain', 'malecns', 'banc']
FN = {'flywire': flywire, 'hemibrain': hemibrain, 'malecns': malecns, 'banc': banc}
for ds, fn in [(k, FN[k]) for k in DSS]:
    typ, mbons = types(ds)
    kcs = set(typ.index[typ.astype(str).str.startswith('KC')])
    proof = set(typ.index) if ds == 'flywire' else None
    a = apl_rows(ds, kcs, proof)
    m = mbon_rows(ds, kcs, mbons)
    if ds == 'malecns':
        m['comp_official'] = official_lookup(m)
    else:
        m['comp_official'] = None
    t = pd.concat([a, m.drop(columns=[c for c in ['mbon_id', 'side'] if c in m.columns])], ignore_index=True)
    t['kc_type'] = t.kc_id.map(typ); t['kc_class'] = t.kc_type.map(kc_class)
    t['lobe'] = class_lobe(t.kc_class.values, t.coarse.values)
    s = {'rows': int(len(t)), 'dropped_no_lobe': int(t.lobe.isna().sum()),
         'roi_disagrees_with_class_lobe': float((t.lobe.notna() & (t.roi != t.lobe))[t.roi != ''].mean()) if (t.roi != '').any() else None}
    t = t[t.lobe.notna()].reset_index(drop=True)
    d = fn(); dan = d['dan'].dropna(subset=['lobe', 'family'])
    comp, share = derive_in_lobe(t[XYZ].values, t.lobe.values, dan[['x', 'y', 'z']].values, dan.lobe.values, dan.family.values, fam)
    t['comp_derived'] = comp
    off_ok = t.comp_official.notna() & (t.comp_official.astype(str).str[:-1] + 'L' == t.lobe)
    t['compartment'] = np.where(off_ok, t.comp_official, t.comp_derived)
    t['compartment_source'] = np.where(off_ok, 'official', np.where(t.comp_derived.notna(), 'derived', None))
    if ds == 'malecns':
        both = off_ok & t.comp_derived.notna()
        s['derived_vs_official_agreement'] = float((t.comp_derived[both] == t.comp_official[both]).mean())
    s['compartment_source'] = t.compartment_source.value_counts(dropna=False).to_dict()
    s['by_direction'] = t.direction.value_counts().to_dict()
    summary[ds] = s
    print(ds, json.dumps(s, default=str), flush=True)
    t.to_parquet(FINAL + f'lobe_rows_{ds}.parquet')
    for level, col in [('lobe', 'lobe'), ('compartment', 'compartment')]:
        g = t.dropna(subset=[col]).groupby(['kc_id', col, 'direction']).size().unstack('direction', fill_value=0)
        for c in ['KC_to_MBON', 'KC_to_APL', 'APL_to_KC']:
            if c not in g: g[c] = 0
        g = g[['KC_to_MBON', 'KC_to_APL', 'APL_to_KC']].reset_index().rename(columns={col: 'region'})
        g.insert(0, 'level', level); g.insert(0, 'dataset', ds)
        g['kc_type'] = g.kc_id.map(typ); g['kc_class'] = g.kc_type.map(kc_class)
        long.append(g)
L = pd.concat(long, ignore_index=True)
L = L[['dataset', 'kc_id', 'kc_type', 'kc_class', 'level', 'region', 'KC_to_MBON', 'KC_to_APL', 'APL_to_KC']]
if len(DSS) == 4:
    L.to_csv(OUT + 'KC_lobe_compartment_weights_MBON_APL.csv', index=False)
    json.dump(summary, open(FINAL + 'lobe_tables_summary.json', 'w'), indent=1, default=str)
else:
    L.to_csv(FINAL + 'test_lobe_weights.csv', index=False)
print('rows', len(L))
