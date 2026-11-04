"""Export the claws Yang chose (2026-11-04: 8 um geodesic clusters of all KC calyx sites, split by PN partner, a PN needs
>= 2 PSDs to get its own claw; column claw_v2m2 of the claw-site tables from claws_v2.py) in the format the downstream
scripts read, for hemibrain and male CNS (the datasets with KC skeletons).

Writes to apl_synapses/with_outside_ROI_synapses/claws_v2m2/:
  <ds>_KC_claws_derived.csv.gz   one row per claw, same columns as the old table (+ calyx, n_sites, cluster, split)
  KC_APL_calyx_synapses_claws_v2m2.csv.gz   every KC<->APL synapse in the corrected calyx, with its claw (or none)
  KC_APL_synapses_on_claws_by_glomerulus.csv   the on-claw rows, same columns as odor_weighted/ of the old claws
  export_summary.json

PN boutons are rebuilt exactly as run.py builds them (2 um single linkage of each PN's calyx T-bars) and checked
against final_<ds>_boutons.parquet. A claw's main bouton = the bouton giving most of its own PN's PSDs.
KC<->APL synapses are matched to claw sites by the KC-side position (APL->KC: the KC PSD; KC->APL: the KC T-bar).
Corrected-calyx KC<->APL synapses that are not claw sites (the segmentation's calyx = label CA or within 3 um of a PN->KC
PSD; mostly male CNS posterior-calyx synapses labelled CentralBrain-unspecified) get the same rule directly: on the claw
whose own PN's PSDs are nearest along the KC skeleton, if <= 8 um. They do not change the claws themselves.

Usage (APL_VARIANT=outside_roi, OMP_NUM_THREADS=1, run from the working folder with out/ and dl/):
  python3 export_claws_v2.py <dir with <ds>_claw_sites.parquet (claws_v2.py output)> <out dir> <swc dir with swc_<ds>/>"""
import sys, os, re, json
SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, SCR)
import numpy as np, pandas as pd
from prep import malecns, hemibrain
from core import cluster_by_owner
from variant import OUTSIDE, FINAL
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from multiprocessing import Pool

assert OUTSIDE, 'run with APL_VARIANT=outside_roi'
SITES, OUT, SWCROOT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
M = 2
COL, DCOL, SCOL = f'claw_v2m{M}', f'dist_v2m{M}', f'split_v2m{M}'
BOUTON_D = 2000
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'
VIEW = {'hemibrain': lambda x, y, z: (-x, -z), 'malecns': lambda x, y, z: (-x, -y)}


def gloms_of(t, ds):
    t = str(t)
    if t.startswith('M_') or '_' not in t: return []
    names = [n for n in t.split('_')[0].split('+') if re.fullmatch(GPAT, n)]
    return [HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in names]


CUT = 8000.


def load_tree(path):
    """Same skeleton graph as claws_v2.py / segment_claws.py."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
    s = pd.read_csv(path, comment='#', sep=r'\s+', header=None, names=['id', 't', 'x', 'y', 'z', 'r', 'p'])
    xyz = s[['x', 'y', 'z']].values.astype(float) * 8
    n = len(s); pos = pd.Series(np.arange(n), index=s.id.values)
    ch = np.where(s.p.values != -1)[0]; par = pos.reindex(s.p.values[ch]).values.astype(int)
    ei, ej = list(ch), list(par)
    ncomp, lab = connected_components(coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n)), directed=False)
    if ncomp > 1:
        comps = [np.where(lab == c)[0] for c in range(ncomp)]; trees = [cKDTree(xyz[c]) for c in comps]
        Dm = np.zeros((ncomp, ncomp)); link = {}
        for a in range(ncomp):
            for b in range(a + 1, ncomp):
                sm, bg = (a, b) if len(comps[a]) <= len(comps[b]) else (b, a)
                dd, kk = trees[bg].query(xyz[comps[sm]]); m = int(np.argmin(dd))
                Dm[a, b] = dd[m] + 1e-6; link[(a, b)] = (comps[sm][m], comps[bg][kk[m]])
        T = minimum_spanning_tree(Dm).tocoo()
        for a, b in zip(T.row, T.col):
            i, k = link[(min(a, b), max(a, b))]; ei.append(i); ej.append(k)
    ei, ej = np.array(ei), np.array(ej)
    w = np.linalg.norm(xyz[ei] - xyz[ej], axis=1) + 1e-3
    return xyz, coo_matrix((np.r_[w, w], (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n)).tocsr()


def extra_one(args):
    """Sites of one KC that are not claw sites -> (claw id or None, distance along the skeleton to its own-PN PSDs)."""
    path, q, own = args          # q: (n, 3) query xyz; own: DataFrame claw_id, x, y, z of the claws' own-PN PSDs
    if not os.path.exists(path): return [None] * len(q), [np.nan] * len(q)
    xyz, G = load_tree(path)
    tr = cKDTree(xyz); nq = tr.query(q)[1]; no = tr.query(own[['x', 'y', 'z']].values)[1]
    dmin = np.full(len(q), np.inf); best = np.full(len(q), None, dtype=object)
    for c, ii in own.groupby('claw_id').indices.items():
        d = dijkstra(G, indices=np.unique(no[ii]), min_only=True, limit=CUT + 1)[nq]
        b = d < dmin; dmin[b] = d[b]; best[b] = c
    return [c if dd <= CUT else None for c, dd in zip(best, dmin)], list(np.where(np.isfinite(dmin), dmin, np.nan))


def key(kc, x, y, z):
    return pd.MultiIndex.from_arrays([np.asarray(kc).astype('int64'), np.round(np.asarray(x, float)).astype('int64'),
                                      np.round(np.asarray(y, float)).astype('int64'), np.round(np.asarray(z, float)).astype('int64')])


summary, on_rows, all_rows = {}, [], []
for ds, fn in [('hemibrain', hemibrain), ('malecns', malecns)]:
    d = fn()
    meta = d['meta'].drop_duplicates('id').set_index('id')
    typ = meta['type']
    # ---- PN boutons, as run.py ----
    p = d['pnca'].copy()
    tb = p.drop_duplicates(['pre_id', 'pre_x', 'pre_y', 'pre_z']).reset_index(drop=True)
    tb['bouton_id'] = cluster_by_owner(tb.pre_id.values, tb[['pre_x', 'pre_y', 'pre_z']].values, BOUTON_D, 'bouton')
    fb = pd.read_parquet(FINAL + f'final_{ds}_boutons.parquet').set_index('bouton_id')
    nb = tb.groupby('bouton_id').size()
    same_boutons = bool(len(nb) == len(fb) and (nb.reindex(fb.index) == fb.n_tbars).all())
    p = p.merge(tb[['pre_id', 'pre_x', 'pre_y', 'pre_z', 'bouton_id']], on=['pre_id', 'pre_x', 'pre_y', 'pre_z'], how='left')
    # ---- claw sites ----
    S = pd.read_parquet(os.path.join(SITES, f'{ds}_claw_sites.parquet'))
    S['site_key'] = key(S.kc_id, S.x, S.y, S.z)
    # PN PSD -> its PN T-bar's bouton (PSD position + KC + PN identify the pnca row)
    pk = p.assign(_k=key(p.post_id, p.post_x, p.post_y, p.post_z))
    pk = pk[~pk._k.duplicated()].set_index('_k')
    ps = S.is_PN.values
    S['bouton_id'] = None
    S.loc[ps, 'bouton_id'] = pk.bouton_id.reindex(S.site_key[ps]).values
    psd_bouton_found = float(pd.Series(S.loc[ps, 'bouton_id']).notna().mean())
    S['claw_id'] = np.where(S[COL] >= 0, S.kc_id.astype(str) + '_v2claw' + (S[COL] + 1).astype(str), None)
    # ---- claw table ----
    on = S[S.claw_id.notna()]
    own = on[on.is_PN].copy()
    owner = own.groupby('claw_id').partner_id.agg(lambda s: s.value_counts().index[0])
    own = own[own.partner_id.values == owner.reindex(own.claw_id).values]
    C = own.groupby('claw_id').agg(kc_id=('kc_id', 'first'), kc_type=('kc_type', 'first'), calyx=('calyx', 'first'),
                                   n_PN_synapses=('partner_id', 'size'), main_PN_id=('partner_id', 'first'),
                                   main_bouton_id=('bouton_id', lambda s: s.value_counts().index[0] if s.notna().any() else None),
                                   centroid_x_nm=('x', 'mean'), centroid_y_nm=('y', 'mean'), centroid_z_nm=('z', 'mean'),
                                   cluster=('cl0_all', 'first'), shares_cluster_with_other_PN=(SCOL, 'first'))
    C['n_sites'] = on.groupby('claw_id').size().reindex(C.index)
    C['n_other_sites'] = C.n_sites - C.n_PN_synapses
    C['main_PN_type'] = C.main_PN_id.astype(str).map(typ)
    C = C.reset_index()
    C.insert(0, 'dataset', ds)
    C = C[['dataset', 'claw_id', 'kc_id', 'kc_type', 'n_PN_synapses', 'main_PN_id', 'main_bouton_id', 'centroid_x_nm', 'centroid_y_nm',
           'centroid_z_nm', 'main_PN_type', 'calyx', 'n_sites', 'n_other_sites', 'cluster', 'shares_cluster_with_other_PN']]
    C.to_csv(os.path.join(OUT, f'{ds}_KC_claws_derived.csv.gz'), index=False)
    # ---- KC <-> APL in the corrected calyx ----
    f = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet')
    kcs = set(typ.index[typ.astype(str).str.startswith('KC')])
    f = f[(f.mb_region == 'CA') & f.partner_id.astype(str).isin(kcs)].reset_index(drop=True)
    sd = f.apl_side.where(f.apl_side.notna(), f.apl_instance.str[-1])
    a = pd.DataFrame({'dataset': ds, 'apl_id': f.apl_id.astype(str), 'apl_side': sd.astype(str).str[0].str.upper().values,
                      'direction': np.where(f.apl_is_pre, 'APL_to_KC', 'KC_to_APL'), 'kc_id': f.partner_id.astype(str),
                      'kc_type': f.partner_id.astype(str).map(typ).values,
                      'x_nm': f.post_x.values, 'y_nm': f.post_y.values, 'z_nm': f.post_z.values})
    kk = key(f.partner_id.astype('int64'), f.partner_side_x, f.partner_side_y, f.partner_side_z)
    kind = np.where(f.apl_is_pre, 'PSD', 'Tbar')
    Si = S[~S.site_key.duplicated()].set_index('site_key')
    found = kk.isin(Si.index)
    a['in_claw_site_table'] = found
    r = Si.reindex(kk)
    a['site_kind_matches'] = (r.kind.values == kind) | ~found
    a['claw_id'] = r.claw_id.values
    a['claw_dist_nm'] = np.round(r[DCOL].values)          # along the skeleton, to the nearest PSD of the claw's PN
    # corrected-calyx synapses that are not claw sites: same rule, straight on the skeleton
    a['claw_site_source'] = np.where(found, 'claw site table', None)
    miss = np.flatnonzero(~found & a.kc_id.isin(set(C.kc_id.astype(str))).values)
    ownp = own[['kc_id', 'claw_id', 'x', 'y', 'z']].assign(kc_id=lambda t: t.kc_id.astype(str))
    og = ownp.groupby('kc_id')
    kx = f[['partner_side_x', 'partner_side_y', 'partner_side_z']].values.astype(float)
    jobs, idx = [], []
    for kid, ii in pd.Series(miss).groupby(a.kc_id.values[miss]).apply(lambda t: t.values).items():
        jobs.append((os.path.join(SWCROOT, f'swc_{ds}', f'{kid}.swc'), kx[ii], og.get_group(kid).reset_index(drop=True))); idx.append(ii)
    with Pool(int(os.environ.get('NPROC', '4'))) as pool:
        res = pool.map(extra_one, jobs, chunksize=4)
    for ii, (cl, dd) in zip(idx, res):
        a.loc[ii, 'claw_id'] = cl; a.loc[ii, 'claw_dist_nm'] = np.round(np.array(dd, float))
        a.loc[ii, 'claw_site_source'] = 'not a claw site: assigned on the skeleton'
    a['cluster_8um'] = r.cl0_all.values
    a['old_claw_id_2um'] = f.claw_id.values
    cm = C.set_index('claw_id')
    a['claw_main_PN_id'] = a.claw_id.map(cm.main_PN_id)
    a['claw_main_PN_type'] = a.claw_id.map(cm.main_PN_type)
    a['claw_main_bouton_id'] = a.claw_id.map(cm.main_bouton_id)
    a['glomeruli'] = ['+'.join(gloms_of(t, ds)) if isinstance(t, str) else '' for t in a.claw_main_PN_type]
    u, v = VIEW[ds](a.x_nm.values / 1e3, a.y_nm.values / 1e3, a.z_nm.values / 1e3)
    a['view_right_um'], a['view_dorsal_um'] = np.round(u, 3), np.round(v, 3)
    all_rows.append(a)
    on_rows.append(a[a.claw_id.notna()])
    sm = {'boutons rebuilt': int(len(nb)), 'boutons identical to final table': same_boutons,
          'PN PSDs with a bouton': round(psd_bouton_found, 4), 'claws': int(len(C)), 'KCs with claws': int(C.kc_id.nunique()),
          'claws per KC mean': round(float(C.groupby('kc_id').size().mean()), 2),
          'claws without main bouton': int(C.main_bouton_id.isna().sum()),
          'KC<->APL calyx synapses': int(len(a)), 'matched to a claw site': round(float(found.mean()), 4),
          'site kind matches': bool(a.site_kind_matches.all()),
          'not claw sites, assigned on the skeleton': int(len(miss)),
          'of those on a claw': round(float(pd.Series(a.claw_id.values[miss]).notna().mean()), 3) if len(miss) else None}
    for dr in ['KC_to_APL', 'APL_to_KC']:
        x = a[a.direction == dr]
        sm[f'{dr}: on a claw, share of all'] = round(float(x.claw_id.notna().mean()), 3)
        sm[f'{dr}: on a claw, share of matched'] = round(float(x.claw_id[x.in_claw_site_table].notna().mean()), 3)
        sm[f'{dr}: on an old 2 um claw'] = round(float(x.old_claw_id_2um.notna().mean()), 3)
    summary[ds] = sm
    print(ds, json.dumps(sm, indent=1), flush=True)

A = pd.concat(all_rows, ignore_index=True)
A.to_csv(os.path.join(OUT, 'KC_APL_calyx_synapses_claws_v2m2.csv.gz'), index=False)
cols = ['dataset', 'apl_id', 'apl_side', 'direction', 'kc_id', 'kc_type', 'claw_id', 'claw_dist_nm', 'claw_main_PN_id',
        'claw_main_PN_type', 'glomeruli', 'x_nm', 'y_nm', 'z_nm', 'view_right_um', 'view_dorsal_um']
pd.concat(on_rows, ignore_index=True)[cols].to_csv(os.path.join(OUT, 'KC_APL_synapses_on_claws_by_glomerulus.csv'), index=False)
json.dump(summary, open(os.path.join(OUT, 'export_summary.json'), 'w'), indent=1)
