"""Claws, Yang's corrected rule (2026-11-04), applied to the calyx sites already placed and clustered by segment_claws.py.

1. All of the KC's calyx synapse sites (T-bars and PSDs, any partner) are clustered by single linkage at 8 µm along the KC's
   skeleton (= cutting the minimum spanning tree of the geodesic distances at 8 µm). These are the cl0_all clusters.
2. Within a cluster, PN->KC PSDs are split by their presynaptic PN: each PN gives its own claw (its PSDs in that cluster).
   Variant m = k: only a PN with >= k PSDs in the cluster makes a claw; PSDs of smaller PNs are treated like non-PN sites.
   m = 1 is the rule as written.
3. Every other site (non-PN partner, e.g. APL, or a T-bar) joins the claw whose nearest PN->KC PSD is closest along the
   skeleton, if that distance is <= 8 µm; otherwise it is not on a claw.

Usage: python3 claws_v2.py <ds> <in dir with <ds>_claw_sites.parquet> <swc dir> <out dir>   (OMP_NUM_THREADS=1)
Adds columns claw_v2m<k> (claw index per KC, -1 = not on a claw) and split_v2m<k> (claw shares its 8 µm cluster with
another PN's claw) for k in 1, 2, 3, 5; also writes the old columns through unchanged."""
import sys, os, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from multiprocessing import Pool

ds, IN, SWC, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
CUT = 8000.
MS = [1, 2, 3, 5]
S = pd.read_parquet(os.path.join(IN, f'{ds}_claw_sites.parquet'))
assert (S.cl0_all >= 0).all()


def load_tree(path):
    """SWC (8 nm units) -> node xyz (nm) and an undirected sparse graph; unconnected pieces joined by shortest links
    (same as segment_claws.py)."""
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


def one(args):
    kid, g = args
    path = os.path.join(SWC, f'{kid}.swc')
    if not os.path.exists(path): return None
    xyz, G = load_tree(path)
    node = cKDTree(xyz).query(g[['x', 'y', 'z']].values)[1]
    pnid = np.where(g.is_PN.values, g.partner_id.values, -1).astype('int64')
    cl0 = g.cl0_all.values
    out = {'sites': g.index.values}
    for m in MS:
        claw = np.full(len(g), -1); split = np.zeros(len(g), bool); dist = np.full(len(g), np.nan)
        k = 0
        for c in np.unique(cl0):
            ii = np.flatnonzero(cl0 == c)
            cnt = pd.Series(pnid[ii][pnid[ii] >= 0]).value_counts()
            pns = list(cnt.index[cnt >= m])
            if not pns: continue
            dmin = np.full(len(ii), np.inf); best = np.full(len(ii), -1)
            for q, p in enumerate(pns):
                own = ii[pnid[ii] == p]
                claw[own] = k + q
                d = dijkstra(G, indices=np.unique(node[own]), min_only=True, limit=CUT + 1)[node[ii]]
                better = d < dmin; dmin[better] = d[better]; best[better] = k + q
            rest = ~np.isin(pnid[ii], pns)
            ok = rest & (dmin <= CUT)
            claw[ii[ok]] = best[ok]
            dist[ii[rest]] = dmin[rest]
            if len(pns) > 1: split[ii[claw[ii] >= 0]] = True
            k += len(pns)
        out[f'claw_v2m{m}'] = claw; out[f'split_v2m{m}'] = split; out[f'dist_v2m{m}'] = dist
    return out


if __name__ == '__main__':
    groups = [(k, g) for k, g in S.groupby('kc_id')]
    with Pool(4) as pool:
        R = [r for r in pool.imap_unordered(one, groups, chunksize=4) if r is not None]
    print(ds, 'KCs done', len(R), 'of', len(groups), flush=True)
    for m in MS:
        for pre, dt in [('claw', int), ('split', bool), ('dist', float)]:
            col = f'{pre}_v2m{m}'
            a = np.full(len(S), -1 if dt is int else (False if dt is bool else np.nan), dtype=dt)
            for r in R: a[r['sites']] = r[col]
            S[col] = a
    S.to_parquet(os.path.join(OUT, f'{ds}_claw_sites.parquet'))
    on = {m: round(float((S[f'claw_v2m{m}'] >= 0).mean()), 3) for m in MS}
    json.dump({'share_of_calyx_sites_on_claws': on, 'cut_um': CUT / 1e3, 'min_PSDs_per_PN_claw': MS},
              open(os.path.join(OUT, f'{ds}_claws_v2_run.json'), 'w'), indent=1)
    print('saved', on, flush=True)
