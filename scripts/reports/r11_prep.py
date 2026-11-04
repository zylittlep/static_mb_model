"""Prep for Report 11: properties of the local coupling matrix M per mushroom body and lambda, a brute-force check of the
two-pass tree computation, and the distribution of geodesic distances between calyx KC<->APL sites.
APL_VARIANT=outside_roi uses the second version. Writes reports/data/r11_M_properties[_outside_roi].csv and
r11_checks[_outside_roi].json."""
import os, sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import apl_local_model as lm
tc = lm.tc
TAG = {'outside_roi': '_outside_roi', 'claws_v2m2': '_claws_v2m2'}.get(os.environ.get('APL_VARIANT', ''), '')
OD = '/mnt/project-files/apl_synapses/reports/data/'
LAMS = [500, 200, 100, 50, 25, 10, 5, 2, 1]
rng = np.random.default_rng(11)
rows, checks = [], {}
for k, mb in tc.MB.items():
    g = lm.geometry(k, mb)
    N = len(mb['W']); u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean(); Mc = np.outer(u, v) / N
    n = len(g['cpar'])
    G = csr_matrix((g['clen'][1:] / 1e3, (np.arange(1, n), g['cpar'][1:])), shape=(n, n))   # µm; BFS order: index 0 is the root
    assert (g['cpar'][1:] >= 0).all()
    # distances between random pairs of APL input sites (KC->APL) and APL output sites (APL->KC)
    nin = g['Win'].tocoo().row
    nout = g['Wout'].tocoo().row
    src = rng.choice(nin, 300, replace=False)
    D = dijkstra(G, directed=False, indices=src)
    dpair = D[:, rng.choice(nout, 300, replace=False)].ravel()
    checks[k] = {'geodesic distance KC->APL site to APL->KC site, random pairs (µm): median, 10%, 90%':
                 [round(float(np.median(dpair)), 1), round(float(np.quantile(dpair, .1)), 1), round(float(np.quantile(dpair, .9)), 1)],
                 'nodes of the compressed tree': int(n), 'snap distance synapse to node, median (µm)': round(g['snap_um'], 2)}
    np.save(OD + f'r11_pair_distances_{k.replace(" ", "_")}{TAG}.npy', dpair.astype(np.float32))
    # brute-force check of the two-pass kernel sum at lambda = 10 µm: phi(v, j) = sum over KC->APL sites t of KC j of exp(-d(v, t)/lambda)
    lam = 10.0
    kk = np.exp(-g['clen'] / (lam * 1e3)).astype(np.float64)
    js = rng.choice(N, 40, replace=False); Wsub = g['Win'][:, js]
    up = Wsub.toarray().astype(np.float64)
    for vv in range(n - 1, 0, -1): up[g['cpar'][vv]] += up[vv] * kk[vv]
    for vv in range(1, n): up[vv] = up[vv] * (1 - kk[vv] ** 2) + up[g['cpar'][vv]] * kk[vv]
    vs = rng.choice(n, 60, replace=False)
    Dv = dijkstra(G, directed=False, indices=vs)
    brute = np.exp(-Dv / lam) @ Wsub.toarray()
    rel = np.abs(up[vs] - brute).max() / max(brute.max(), 1e-12)
    checks[k]['two-pass vs brute force at 10 µm: max abs difference / max value'] = float(f'{rel:.2e}')
    for lam in LAMS:
        Mk = lm.kernel_M(g['cpar'], g['clen'], g['Win'], g['Wout'], lam); Mk = Mk * (Mc.mean() / Mk.mean())
        rs_ = Mk.sum(1); ok = rs_ > 0
        pr = rs_[ok] ** 2 / (Mk[ok] ** 2).sum(1)
        cls = mb['cls']; same = (cls[:, None] == cls[None])
        rows.append(dict(mushroom_body=k, lam_um=lam, self_share=float(np.mean(np.diag(Mk)[ok] / rs_[ok])),
                         effective_inhibitors_frac_median=float(np.median(pr) / N), row_sum_cv=float(rs_.std() / rs_.mean()),
                         r_with_C=float(np.corrcoef(Mk.ravel()[::53], Mc.ravel()[::53])[0, 1]),
                         same_class_share=float((Mk * same).sum() / Mk.sum())))
        print(k, lam, rows[-1], flush=True)
    rows.append(dict(mushroom_body=k, lam_um=np.inf, self_share=float(np.mean(np.diag(Mc) / Mc.sum(1))),
                     effective_inhibitors_frac_median=float(np.median(Mc.sum(1) ** 2 / (Mc ** 2).sum(1)) / N), row_sum_cv=float(Mc.sum(1).std() / Mc.sum(1).mean()),
                     r_with_C=1.0, same_class_share=float((Mc * (mb['cls'][:, None] == mb['cls'][None])).sum() / Mc.sum())))
    print(k, checks[k], flush=True)
pd.DataFrame(rows).to_csv(OD + f'r11_M_properties{TAG}.csv', index=False)
json.dump(checks, open(OD + f'r11_checks{TAG}.json', 'w'), indent=1)
