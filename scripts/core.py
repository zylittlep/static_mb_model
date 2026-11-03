"""Shared claw / bouton / compartment logic. All coordinates in nm."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

LOBE_COMPS = {
    'VL': ['a1', 'a2', 'a3', "a'1", "a'2", "a'3"],
    'ML': ['b1', 'b2', "b'1", "b'2", 'g1', 'g2', 'g3', 'g4', 'g5'],
    'aL': ['a1', 'a2', 'a3'], "a'L": ["a'1", "a'2", "a'3"], 'bL': ['b1', 'b2'],
    "b'L": ["b'1", "b'2"], 'gL': ['g1', 'g2', 'g3', 'g4', 'g5'],
}


def single_linkage(xyz, d):
    """Connected components of points closer than d (nm). Returns labels."""
    if len(xyz) == 1:
        return np.zeros(1, int)
    pairs = cKDTree(xyz).query_pairs(d, output_type='ndarray')
    from scipy.sparse import coo_matrix
    n = len(xyz)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def cluster_by_owner(owner, xyz, d, prefix):
    """Cluster points per owner neuron; returns array of string ids '<owner>_<prefix><k>'
    with k ordered by cluster size (largest = 1)."""
    owner = np.asarray(owner).astype(str)
    out = np.empty(len(owner), dtype=object)
    order = np.argsort(owner, kind='stable')
    so = owner[order]
    starts = np.flatnonzero(np.r_[True, so[1:] != so[:-1]])
    ends = np.r_[starts[1:], len(so)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        lab = single_linkage(xyz[idx], d)
        cnt = np.bincount(lab)
        rank = np.empty_like(cnt); rank[np.argsort(-cnt, kind='stable')] = np.arange(1, len(cnt) + 1)
        out[idx] = [f"{so[s]}_{prefix}{r}" for r in rank[lab]]
    return out


def assign_nearest(q_owner, q_xyz, ref_owner, ref_xyz, ref_label, dmax):
    """For each query point, nearest reference point of the same owner within dmax.
    Returns (label or None, distance)."""
    q_owner = np.asarray(q_owner).astype(str); ref_owner = np.asarray(ref_owner).astype(str)
    lab = np.full(len(q_owner), None, dtype=object); dist = np.full(len(q_owner), np.nan)
    ref = pd.Series(np.arange(len(ref_owner))).groupby(ref_owner).apply(lambda s: s.values)
    for o in np.unique(q_owner):
        qi = np.flatnonzero(q_owner == o)
        if o not in ref.index:
            continue
        ri = ref[o]
        d, j = cKDTree(ref_xyz[ri]).query(q_xyz[qi])
        dist[qi] = d
        ok = d <= dmax
        lab[qi[ok]] = np.asarray(ref_label)[ri[j[ok]]]
    return lab, dist


def dan_family(t):
    import re
    m = re.match(r'^(PAM\d\d|PPL\d\d\d)', str(t))
    return m.group(1) if m else None


def derive_compartment(q_xyz, q_region, dan_xyz, dan_region, dan_family_arr, famdist, k=15, dmax=8000):
    """Soft kNN vote: each DAN T-bar votes with its family's compartment distribution
    (restricted to the lobe region). q_region / dan_region are lobe keys of LOBE_COMPS.
    famdist: DataFrame index=family, columns=compartments, rows sum to 1.
    Returns (compartment, vote share, nearest DAN distance)."""
    comp = np.full(len(q_xyz), None, dtype=object); share = np.full(len(q_xyz), np.nan); nd = np.full(len(q_xyz), np.nan)
    for reg in np.unique(q_region):
        if reg not in LOBE_COMPS:
            continue
        cols = [c for c in LOBE_COMPS[reg] if c in famdist.columns]
        qi = np.flatnonzero(q_region == reg)
        di = np.flatnonzero((dan_region == reg) & pd.Series(dan_family_arr).isin(famdist.index).values)
        if len(di) == 0:
            continue
        W = famdist.reindex(dan_family_arr[di])[cols].fillna(0).values
        W = W / np.clip(W.sum(1, keepdims=True), 1e-9, None)
        kk = min(k, len(di))
        d, j = cKDTree(dan_xyz[di]).query(q_xyz[qi], k=kk)
        if kk == 1:
            d = d[:, None]; j = j[:, None]
        V = W[j].sum(1)
        best = V.argmax(1)
        comp[qi] = np.array(cols, dtype=object)[best]
        share[qi] = V[np.arange(len(qi)), best] / np.clip(V.sum(1), 1e-9, None)
        nd[qi] = d[:, 0]
        far = d[:, 0] > dmax
        comp[qi[far]] = None
    return comp, share, nd
