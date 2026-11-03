"""Calyx / MB geometry per dataset: native axes -> anatomical (R, D, A) in µm, landmarks, rigid alignment."""
import numpy as np, pandas as pd

O = 'out/'   # run from the working directory that holds out/ and dl/, like run.py
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}


def to_rda(ds, x, y, z):
    """Native nm -> (fly's right, dorsal, A-P with unknown sign) in µm, same axes as the posterior-view maps."""
    x, y, z = x / 1e3, y / 1e3, z / 1e3
    return {'flywire': np.c_[x, -y, z], 'hemibrain': np.c_[-x, -z, y],
            'malecns': np.c_[-x, -y, z], 'banc': np.c_[-x, z, y]}[ds]


def load_mbs():
    """dict (ds, side) -> DataFrame of APL MB synapses (post site) with R, D, A columns (left MBs NOT mirrored),
    region in {CA, PED, VL, ML}; A sign set so the lobes are anterior of the calyx."""
    out = {}
    for ds in DS:
        a = pd.read_parquet(O + f'final_{ds}_apl_mb.parquet',
                            columns=['apl_id', 'apl_side', 'roi_official', 'mb_region', 'post_x', 'post_y', 'post_z'])
        side = a.apl_side.fillna('R').astype(str).str[0].str.upper()
        reg = a.mb_region.replace({'aL': 'VL', "a'L": 'VL', 'bL': 'ML', "b'L": 'ML', 'gL': 'ML'})
        roi_side = a.roi_official.astype(str).str.extract(r'(?:_|\()([LR])\)?$')[0]
        P = to_rda(ds, a.post_x.values.astype(float), a.post_y.values.astype(float), a.post_z.values.astype(float))
        t = pd.DataFrame(P, columns=['R', 'D', 'A']).assign(side=side.values, region=reg.values, roi_side=roi_side.values)
        t = t[(t.roi_side == t.side) | t.roi_side.isna()]
        lob = t[t.region.isin(['VL', 'ML'])].A.mean(); ca = t[t.region == 'CA'].A.mean()
        if lob < ca: t['A'] = -t.A
        for s_, g in t.groupby('side'):
            out[(ds, s_)] = g.drop(columns=['roi_side']).reset_index(drop=True)
    return out


def landmarks(g, mirror):
    """7 landmarks defined without reference to any axis (so they are rotation invariant)."""
    X = g[['R', 'D', 'A']].values.copy()
    if mirror: X[:, 0] = -X[:, 0]
    reg = g.region.values
    c = {r: X[reg == r].mean(0) for r in ['CA', 'PED', 'VL', 'ML']}
    def far(r, q, frm):
        P = X[reg == r]; d = np.linalg.norm(P - frm, axis=1)
        return P[d >= np.quantile(d, q)].mean(0)
    L = np.array([c['CA'], c['PED'], c['VL'], c['ML'], far('VL', .98, c['PED']), far('ML', .98, c['PED']), far('CA', .95, c['PED'])])
    return L, X


def kabsch(P, Q):
    """Rotation Rm (det +1) minimising |(P - Pc) Rm - (Q - Qc)|; returns Rm, Pc, Qc, size ratio |Q|/|P| (RMS about centroid)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1, 1, d]) @ Vt
    s = np.sqrt(((Q - Qc) ** 2).sum() / ((P - Pc) ** 2).sum())
    return Rm, Pc, Qc, s
