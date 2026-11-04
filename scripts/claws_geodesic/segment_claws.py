"""Claws by geodesic single linkage on each KC's skeleton (Yang's algorithm, 2026-11-03), next to the old 2 um claws.

For every KC: its synapse sites in the calyx are placed on the nearest node of its skeleton; two sites are linked when
their distance along the skeleton is <= CUT (8 um); the connected groups are the clusters (= cutting the minimum
spanning tree of the geodesic distances at 8 um). A cluster with at least one PN->KC PSD is a claw. A claw with more
than BIG (40) sites whose PN PSDs come from two PNs (second PN >= MIN2 PSDs) is split in two: the PSDs of each of the two
PNs seed one part and every other site joins the nearer seed along the skeleton; each part's main PN is then its own PN.

Two site sets ("variants"):
  all : every KC synapse site in the calyx, T-bars and PSDs, whatever the partner (Yang's wording).
  pn  : PN->KC PSDs only.
Calyx sites: label CA(L/R), or within 3 um of a corrected-calyx PN->KC PSD (outside-ROI version, roi_assigned CA).
Old claws: the 2 um Euclidean single linkage on PN->KC PSDs (scripts/run.py), recomputed on the same corrected-calyx PSDs.

Usage: python3 segment_claws.py <ds> <working dir with out/claws/<ds>_kc_sites.parquet, out/outside_roi/> <swc dir> <out dir>
Run with OMP_NUM_THREADS=1."""
import sys, os, re, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from multiprocessing import Pool

ds, WD, SWC, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CUT, BIG, MIN2, IN_CAL, OLD_D = 8000., 40, 5, 3000., 2000.
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'


def glom(t):
    t = str(t)
    if t.startswith('M_') or '_' not in t: return ''
    g = [n for n in t.split('_')[0].split('+') if re.fullmatch(GPAT, n)]
    g = [HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in g]
    return '+'.join(g)


# ---------------------------------------------------------------- inputs
os.chdir(WD); sys.path.insert(0, 'scripts')
from meta_util import types
typ, _ = types(ds)
pn = pd.read_parquet(f'out/outside_roi/{ds}_pnca_raw.parquet')
pn = pn[pn.roi_assigned.astype(str).str.startswith('CA')]
PNS = set(pn.body_pre.astype('int64'))
anc = pn[['x_post', 'y_post', 'z_post']].values * 8.
anc_side = pn.roi_assigned.astype(str).str[3].values          # 'L' / 'R'
S = pd.read_parquet(f'out/claws/{ds}_kc_sites.parquet')
S['kc_type'] = S.kc_id.astype(str).map(typ).astype(str)
d, j = cKDTree(anc).query(S[['x', 'y', 'z']].values, distance_upper_bound=IN_CAL)
near = np.isfinite(d)
side = np.where(near, anc_side[np.minimum(j, len(anc) - 1)], '')
lab = S.roi.astype(str).str.extract(r'^CA\((L|R)\)')[0].fillna('').values
S['calyx'] = np.where(lab != '', lab, side)
S = S[S.calyx != ''].reset_index(drop=True)
S['is_PN'] = (S.kind == 'PSD') & S.partner_id.isin(PNS)
# KC side = side of most of its calyx PN PSDs; keep that calyx only
ks = S[S.is_PN].groupby('kc_id').calyx.agg(lambda s: s.value_counts().index[0])
S = S[S.kc_id.map(ks) == S.calyx].reset_index(drop=True)
print(ds, 'calyx sites', len(S), S.kind.value_counts().to_dict(), 'PN PSDs', int(S.is_PN.sum()), 'KCs', S.kc_id.nunique(), flush=True)


def load_tree(path):
    """SWC (8 nm units) -> node xyz (nm) and an undirected sparse graph; unconnected pieces joined by shortest links."""
    s = pd.read_csv(path, comment='#', sep=r'\s+', header=None, names=['id', 't', 'x', 'y', 'z', 'r', 'p'])
    xyz = s[['x', 'y', 'z']].values.astype(float) * 8
    n = len(s); pos = pd.Series(np.arange(n), index=s.id.values)
    ch = np.where(s.p.values != -1)[0]; par = pos.reindex(s.p.values[ch]).values.astype(int)
    ei, ej = list(ch), list(par)
    ncomp, lab = connected_components(coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n)), directed=False)
    nb = 0
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
            i, k = link[(min(a, b), max(a, b))]; ei.append(i); ej.append(k); nb += 1
    ei, ej = np.array(ei), np.array(ej)
    w = np.linalg.norm(xyz[ei] - xyz[ej], axis=1) + 1e-3
    G = coo_matrix((np.r_[w, w], (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n)).tocsr()
    return xyz, G, ncomp


def geo_clusters(G, node, cut):
    """Single linkage at `cut` along the skeleton for sites at skeleton nodes `node`."""
    u, inv = np.unique(node, return_inverse=True)
    D = dijkstra(G, indices=u, limit=cut + 1)[:, u]
    a, b = np.nonzero(np.isfinite(D) & (D <= cut))
    n = len(u)
    _, l = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    return l[inv]


def split(G, node, pnid, idx, rec):
    """Split one claw (site indices idx) by its two main PNs; returns list of (indices, main PN). Recursive if rec."""
    p = pd.Series(pnid[idx]); p = p[p >= 0].value_counts()
    if len(idx) <= BIG or len(p) < 2 or p.iloc[1] < MIN2:
        return [(idx, int(p.index[0]) if len(p) else -1, False)]
    p1, p2 = int(p.index[0]), int(p.index[1])
    s1 = np.unique(node[idx][pnid[idx] == p1]); s2 = np.unique(node[idx][pnid[idx] == p2])
    d1 = dijkstra(G, indices=s1, min_only=True); d2 = dijkstra(G, indices=s2, min_only=True)
    to2 = d2[node[idx]] < d1[node[idx]]
    to2[pnid[idx] == p1] = False; to2[pnid[idx] == p2] = True
    parts = [(idx[~to2], p1), (idx[to2], p2)]
    out = []
    for ii, pp in parts:
        if rec:
            out += [(a, b, True) for a, b, _ in split(G, node, pnid, ii, rec)]
        else:
            out.append((ii, pp, True))
    return out


def old_claws(xyz):
    if len(xyz) == 1: return np.zeros(1, int)
    pr = cKDTree(xyz).query_pairs(OLD_D, output_type='ndarray'); n = len(xyz)
    return connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)), directed=False)[1]


def one(args):
    kid, g = args
    path = os.path.join(SWC, f'{kid}.swc')
    if not os.path.exists(path): return None
    xyz, G, npieces = load_tree(path)
    P = g[['x', 'y', 'z']].values
    snap, node = cKDTree(xyz).query(P)
    pnid = np.where(g.is_PN.values, g.partner_id.values, -1).astype('int64')
    res = {'kc_id': kid, 'sites': g.index.values, 'snap_nm': snap, 'pieces': npieces}
    for var, m in [('all', np.ones(len(g), bool)), ('pn', g.is_PN.values)]:
        idx_all = np.flatnonzero(m)
        cl = np.full(len(g), -1); clr = np.full(len(g), -1); cl0 = np.full(len(g), -1)
        flag = np.zeros(len(g), bool); flagr = np.zeros(len(g), bool)
        if len(idx_all):
            lab = geo_clusters(G, node[idx_all], CUT)
            cl0[idx_all] = lab
            k = 0; kr = 0
            for c in np.unique(lab):
                ii = idx_all[lab == c]
                if not (pnid[ii] >= 0).any(): continue                 # no PN input: not a claw
                for jj, _, f in split(G, node, pnid, ii, False):
                    cl[jj] = k; flag[jj] = f; k += 1
                for jj, _, f in split(G, node, pnid, ii, True):
                    clr[jj] = kr; flagr[jj] = f; kr += 1
        res[f'cl0_{var}'] = cl0; res[f'claw_{var}'] = cl; res[f'split_{var}'] = flag
        res[f'clawrec_{var}'] = clr; res[f'splitrec_{var}'] = flagr
    old = np.full(len(g), -1); ip = np.flatnonzero(g.is_PN.values)
    if len(ip): old[ip] = old_claws(P[ip])
    res['claw_old'] = old
    return res


if __name__ == '__main__':
    groups = [(k, g) for k, g in S.groupby('kc_id')]
    with Pool(4) as pool:
        R = [r for r in pool.imap_unordered(one, groups, chunksize=4) if r is not None]
    print('KCs done', len(R), 'of', len(groups), flush=True)
    cols = ['cl0_all', 'claw_all', 'split_all', 'clawrec_all', 'splitrec_all', 'cl0_pn', 'claw_pn', 'split_pn',
            'clawrec_pn', 'splitrec_pn', 'claw_old', 'snap_nm']
    A = {c: np.full(len(S), -1, dtype=float) for c in cols}
    for r in R:
        for c in cols: A[c][r['sites']] = r[c]
    for c in cols: S[c] = A[c].astype(int) if c != 'snap_nm' else A[c]
    S = S[S.kc_id.isin([r['kc_id'] for r in R])].reset_index(drop=True)
    S['pn_type'] = S.partner_id.astype(str).map(typ).where(S.is_PN)
    S.to_parquet(os.path.join(OUT, f'{ds}_claw_sites.parquet'))
    json.dump({'skeleton_pieces_joined_median': float(np.median([r['pieces'] for r in R])),
               'snap_um_median_p95': [float(np.median(S.snap_nm) / 1e3), float(np.quantile(S.snap_nm, 0.95) / 1e3)],
               'params': {'cut_um': CUT / 1e3, 'big_sites': BIG, 'min_second_pn_psds': MIN2, 'in_calyx_um': IN_CAL / 1e3,
                          'old_claw_um': OLD_D / 1e3}}, open(os.path.join(OUT, f'{ds}_claw_run.json'), 'w'), indent=1)
    print('saved', flush=True)
