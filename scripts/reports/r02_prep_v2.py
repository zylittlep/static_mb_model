"""Prep for the rebuilt Report 02 (2026-11-04, new claws). Needs the scratch working data, so it is run once; its outputs in
reports/data/ let the report rebuild without them.

Usage: python3 -I r02_prep_v2.py <claw-site dir with <ds>_claw_sites.parquet (claws_v2.py output)> <hemibrain KC SWC dir>
                                 <hemibrain PN SWC dir> <Prat's claw-count csv> <out/outside_roi/hemibrain_pnca_raw.parquet>
Writes reports/data/:
  r02v2_claw_composition.csv   one row per new claw (hemibrain, male CNS): own-PN PSDs, PSDs of other PNs that joined it,
                               non-PN sites that joined it
  r02v2_site_shares.csv        per dataset: share of calyx sites / PN PSDs on a claw, by kind
  r02v2_fig1_segments.csv      calyx part of the skeletons of 3 random PNs and 10 random KCs (hemibrain R), µm, posterior view
  r02v2_fig1_points.csv        their calyx T-bars (PNs, with bouton) and calyx synapse sites (KCs, with claw and glomerulus)
  r02v2_fig1_cells.csv         the 13 cells, with old, new and Prat's claw counts for the KCs"""
import sys, os, re
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SITES, SWC_KC, SWC_PN, PRAT, RAW = sys.argv[1:6]
B = '/mnt/project-files/apl_synapses/'
CV = B + 'with_outside_ROI_synapses/claws_v2m2/'
OD = B + 'reports/data/'
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'


def glom(t, ds):
    t = str(t)
    if t.startswith('M_') or '_' not in t: return ''
    g = [n for n in t.split('_')[0].split('+') if re.fullmatch(GPAT, n)]
    return '+'.join(HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in g)


# ---------------------------------------------------------------- claw composition
comp, shares = [], []
for ds in ['hemibrain', 'malecns']:
    S = pd.read_parquet(os.path.join(SITES, f'{ds}_claw_sites.parquet'),
                        columns=['kc_id', 'kind', 'is_PN', 'partner_id', 'claw_v2m2', 'split_v2m2', 'cl0_all', 'calyx', 'kc_type'])
    on = S[S.claw_v2m2 >= 0].copy()
    on['claw'] = on.kc_id.astype(str) + '_' + on.claw_v2m2.astype(str)
    own_pn = on[on.is_PN].groupby('claw').partner_id.agg(lambda s: s.value_counts().index[0])
    on['own'] = on.is_PN & (on.partner_id.values == own_pn.reindex(on.claw).values)
    g = on.groupby('claw')
    c = pd.DataFrame({'own_PN_PSDs': g.own.sum(), 'other_PN_PSDs': g.apply(lambda d: int((d.is_PN & ~d.own).sum())),
                      'non_PN_sites': g.apply(lambda d: int((~d.is_PN).sum())), 'shares_cluster': g.split_v2m2.first(),
                      'kc_type': g.kc_type.first(), 'calyx': g.calyx.first()})
    c['dataset'] = ds
    comp.append(c.reset_index())
    for kind, m in [('all calyx sites', np.ones(len(S), bool)), ('PN->KC PSDs', S.is_PN.values),
                    ('other PSDs (non-PN partner)', ((S.kind == 'PSD') & ~S.is_PN).values), ('KC T-bars', (S.kind == 'Tbar').values)]:
        shares.append({'dataset': ds, 'sites': kind, 'n': int(m.sum()), 'share on a claw': round(float((S.claw_v2m2.values[m] >= 0).mean()), 4)})
    print(ds, 'claws', len(c), flush=True)
pd.concat(comp, ignore_index=True).to_csv(OD + 'r02v2_claw_composition.csv', index=False)
pd.DataFrame(shares).to_csv(OD + 'r02v2_site_shares.csv', index=False)


# ---------------------------------------------------------------- Figure 1 cells (hemibrain R)
def swc(path):
    s = pd.read_csv(path, comment='#', sep=r'\s+', header=None, names=['id', 't', 'x', 'y', 'z', 'r', 'p'])
    xyz = s[['x', 'y', 'z']].values.astype(float) * 8 / 1e3
    pos = pd.Series(np.arange(len(s)), index=s.id.values)
    ch = np.where(s.p.values != -1)[0]; par = pos.reindex(s.p.values[ch]).values.astype(int)
    return xyz, ch, par


def calyx_part(xyz, ch, par, pts, d=3.0):
    """Skeleton segments whose two nodes lie within d µm of the calyx cloud, keeping only connected pieces that carry the
    cell's own calyx points."""
    inn = CAL_TREE.query(xyz, distance_upper_bound=d)[0] <= d
    e = inn[ch] & inn[par]
    _, lab = connected_components(coo_matrix((np.ones(e.sum()), (ch[e], par[e])), shape=(len(xyz), len(xyz))), directed=False)
    own = np.unique(lab[cKDTree(xyz).query(pts)[1]])
    keep = e & np.isin(lab[ch], own)
    return np.c_[xyz[ch[keep]], xyz[par[keep]]]


VIEW = lambda P: np.c_[-P[:, 0], -P[:, 2], P[:, 1]]       # hemibrain native -> (right, dorsal, anterior-posterior) µm
S = pd.read_parquet(os.path.join(SITES, 'hemibrain_claw_sites.parquet'))
claws = pd.read_csv(CV + 'hemibrain_KC_claws_derived.csv.gz')
bout = pd.read_csv(CV + 'hemibrain_PN_boutons_derived.csv.gz')
apl = pd.read_csv(CV + 'hemibrain_APL_MB_synapses.csv.gz', low_memory=False,
                  usecols=['mb_region', 'partner_id', 'partner_type', 'pre_x_nm', 'pre_y_nm', 'pre_z_nm', 'post_x_nm', 'post_y_nm', 'post_z_nm',
                           'bouton_id', 'direction'])
CAL = S[['x', 'y', 'z']].values / 1e3          # every KC calyx site (corrected calyx)
CAL_TREE = cKDTree(CAL)
rng = np.random.default_rng(1103)
# 3 random uniglomerular PNs with calyx T-bars and a skeleton
have_pn = {int(f[:-4]) for f in os.listdir(SWC_PN) if f.endswith('.swc')}
bp = bout.assign(g=[glom(t, 'hemibrain') for t in bout.pn_type])
bp = bp[(bp.g != '') & ~bp.g.str.contains(r'\+') & bp.pn_id.isin(have_pn)]
pns = list(rng.choice(sorted(bp.pn_id.unique()), 3, replace=False))
# 10 random KCs with claws and a skeleton
have_kc = {int(f[:-4]) for f in os.listdir(SWC_KC) if f.endswith('.swc')}
kcs = list(rng.choice(sorted(set(claws.kc_id) & have_kc), 10, replace=False))
prat = pd.read_csv(PRAT).set_index('bodyId')
old = S[S.is_PN].groupby('kc_id').claw_old.nunique()
segs, pts, cells = [], [], []
raw = pd.read_parquet(RAW)          # corrected-calyx PN -> KC synapses (8 nm voxels), as run.py reads them
sys.path.insert(0, B + 'scripts')
from core import cluster_by_owner
t = pd.DataFrame({'pre_id': raw.body_pre.astype('int64'), 'x': raw.x_pre * 8., 'y': raw.y_pre * 8., 'z': raw.z_pre * 8.})
tb = t.drop_duplicates().reset_index(drop=True)
tb['bouton_id'] = cluster_by_owner(tb.pre_id.astype(str).values, tb[['x', 'y', 'z']].values, 2000, 'bouton')
assert tb.bouton_id.nunique() == len(bout), (tb.bouton_id.nunique(), len(bout))
for pid in pns:
    xyz, ch, par = swc(os.path.join(SWC_PN, f'{pid}.swc'))
    q = tb[tb.pre_id == pid]
    P = q[['x', 'y', 'z']].values / 1e3
    sg = calyx_part(xyz, ch, par, P, d=4.0)
    segs.append(pd.DataFrame(np.c_[VIEW(sg[:, :3]), VIEW(sg[:, 3:])], columns=['r1', 'd1', 'a1', 'r2', 'd2', 'a2']).assign(cell=pid))
    V = VIEW(P)
    pts.append(pd.DataFrame({'cell': pid, 'kind': 'PN T-bar', 'right': V[:, 0], 'dorsal': V[:, 1], 'ap': V[:, 2], 'unit': q.bouton_id.values,
                             'glomerulus': bp[bp.pn_id == pid].g.iloc[0], 'on_unit': True}))
    cells.append({'cell': pid, 'kind': 'PN', 'type': bp[bp.pn_id == pid].pn_type.iloc[0], 'glomerulus': bp[bp.pn_id == pid].g.iloc[0],
                  'calyx_tbars': len(q), 'boutons': int(q.bouton_id.nunique())})
pn_type = bout.drop_duplicates('pn_id').set_index('pn_id').pn_type
for kid in kcs:
    xyz, ch, par = swc(os.path.join(SWC_KC, f'{kid}.swc'))
    s = S[S.kc_id == kid]
    P = s[['x', 'y', 'z']].values / 1e3
    sg = calyx_part(xyz, ch, par, P)
    segs.append(pd.DataFrame(np.c_[VIEW(sg[:, :3]), VIEW(sg[:, 3:])], columns=['r1', 'd1', 'a1', 'r2', 'd2', 'a2']).assign(cell=kid))
    V = VIEW(P)
    g_site = np.where(s.is_PN, [glom(pn_type.get(p, ''), 'hemibrain') for p in s.partner_id], '')
    cm = claws[claws.kc_id == kid].copy()
    cm['k'] = cm.claw_id.str.extract(r'_v2claw(\d+)$')[0].astype(int) - 1
    cg = cm.set_index('k').main_PN_type.map(lambda x: glom(x, 'hemibrain'))
    unit = s.claw_v2m2.values
    pts.append(pd.DataFrame({'cell': kid, 'kind': np.where(s.is_PN, 'PN->KC PSD', np.where(s.kind == 'Tbar', 'KC T-bar', 'other PSD')),
                             'right': V[:, 0], 'dorsal': V[:, 1], 'ap': V[:, 2], 'unit': unit, 'glomerulus': g_site,
                             'claw_glomerulus': [cg.get(u, '') if u >= 0 else '' for u in unit], 'on_unit': unit >= 0}))
    cells.append({'cell': kid, 'kind': 'KC', 'type': s.kc_type.iloc[0], 'new_claws': int(len(cm)), 'old_claws': int(old.get(kid, 0)),
                  'Prat_claws': (int(prat.loc[kid, 'n_claws_corrected']) if kid in prat.index else np.nan),
                  'calyx_sites': len(s), 'PN_PSDs': int(s.is_PN.sum())})
pd.concat(segs, ignore_index=True).round(3).to_csv(OD + 'r02v2_fig1_segments.csv', index=False)
pd.concat(pts, ignore_index=True).round(3).to_csv(OD + 'r02v2_fig1_points.csv', index=False)
pd.DataFrame(cells).to_csv(OD + 'r02v2_fig1_cells.csv', index=False)
print(pd.DataFrame(cells))
