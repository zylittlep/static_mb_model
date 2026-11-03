"""Lobe and compartment assignment for KC-side synapses in the MB lobes.
Lobe: a KC's axon runs in fixed lobes (ab -> a + b, a'b' -> a' + b', g -> g), so a KC synapse's lobe is set by the KC class
plus vertical (VL) vs medial (ML) position. Compartment: soft kNN vote (k=15) of DAN T-bars whose family mainly innervates
that lobe, each voting with its family's compartment distribution measured in male CNS (same idea as the delivered tables,
but restricted to the KC's own lobe)."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from core import LOBE_COMPS

COARSE = {'aL': 'VL', "a'L": 'VL', 'bL': 'ML', "b'L": 'ML', 'gL': 'ML'}
LOBE_NAME = {'aL': 'α', "a'L": 'α′', 'bL': 'β', "b'L": 'β′', 'gL': 'γ'}
COMP_NAME = {f'{k}{i}': f'{v}{i}' for k, v in [('a', 'α'), ("a'", 'α′'), ('b', 'β'), ("b'", 'β′'), ('g', 'γ')] for i in range(1, 6)}


def class_lobe(kc_cls, coarse):
    """kc_cls: γ / α′β′ / αβ; coarse: VL / ML. Returns aL, a'L, bL, b'L, gL or None."""
    out = np.full(len(kc_cls), None, dtype=object)
    kc_cls = np.asarray(kc_cls, dtype=object); coarse = np.asarray(coarse, dtype=object)
    out[(kc_cls == 'αβ') & (coarse == 'VL')] = 'aL'; out[(kc_cls == 'αβ') & (coarse == 'ML')] = 'bL'
    out[(kc_cls == 'α′β′') & (coarse == 'VL')] = "a'L"; out[(kc_cls == 'α′β′') & (coarse == 'ML')] = "b'L"
    out[(kc_cls == 'γ') & (coarse == 'ML')] = 'gL'
    return out


def famdist_from_malecns(path='out/malecns_dan_comp.parquet'):
    f = pd.read_parquet(path)
    f['family'] = f['type'].str.extract(r'^(PAM\d\d|PPL\d\d\d)')[0]
    f = f[f.comp.str.match(r"^(a|a'|b|b'|g)\d$")]
    return pd.crosstab(f.family, f.comp, normalize='index')


def derive_in_lobe(q_xyz, q_lobe, dan_xyz, dan_region, dan_family, famdist, k=15, dmax=8000, min_mass=0.15):
    """dan_region may be the fine lobe (hemibrain / male CNS) or VL / ML (FlyWire / BANC)."""
    comp = np.full(len(q_xyz), None, dtype=object); share = np.full(len(q_xyz), np.nan)
    dan_family = np.asarray(dan_family, dtype=object); dan_region = np.asarray(dan_region, dtype=object)
    q_lobe = np.asarray(q_lobe, dtype=object)
    for lobe in [l for l in pd.unique(q_lobe) if l in COARSE]:
        cols = [c for c in LOBE_COMPS[lobe] if c in famdist.columns]
        mass = famdist[cols].sum(1)
        fams = set(mass.index[mass >= min_mass])
        qi = np.flatnonzero(q_lobe == lobe)
        di = np.flatnonzero(np.isin(dan_region, [lobe, COARSE[lobe]]) & np.isin(dan_family, list(fams)))
        if len(di) == 0 or len(qi) == 0:
            continue
        W = famdist.reindex(dan_family[di])[cols].values
        W = W / W.sum(1, keepdims=True)
        kk = min(k, len(di))
        d, j = cKDTree(dan_xyz[di]).query(q_xyz[qi], k=kk)
        if kk == 1:
            d = d[:, None]; j = j[:, None]
        V = W[j].sum(1); best = V.argmax(1)
        c = np.array(cols, dtype=object)[best]
        c[d[:, 0] > dmax] = None
        comp[qi] = c; share[qi] = V[np.arange(len(qi)), best] / V.sum(1)
    return comp, share
