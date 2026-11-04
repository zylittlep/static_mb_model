"""Prep for Report 12 (APL confined to calyx sub-arbors): per-KC compartment spread, coupling-matrix properties of the compartment
versions, and synapse positions colored by compartment with Report 10's sub-arbor names. APL_VARIANT=outside_roi uses the
second version. Writes reports/data/r12_kc_spread[_outside_roi].csv, r12_M_properties[_outside_roi].csv,
r12_positions[_outside_roi].csv, r12_names[_outside_roi].json."""
import os, sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import apl_compartment_model as acm
lm, tc, ag = acm.lm, acm.tc, acm.ag
TAG = '_outside_roi' if os.environ.get('APL_VARIANT', '') == 'outside_roi' else ''
OD = '/mnt/project-files/apl_synapses/reports/data/'
VIEW = {'hemibrain': lambda P: (-P[:, 0], -P[:, 2]), 'malecns': lambda P: (-P[:, 0], -P[:, 1])}     # posterior view, as Report 10
rng = np.random.default_rng(12)
spread, props, pos, names = [], [], [], {}
for k, mb in tc.MB.items():
    ds, side = mb['ds'], k.split()[1]
    g = acm.geometry_c(k, mb)
    T = ag.load_tree(ag.SK + ag.APLS[lm.APL_OF[k]][2]); lab = ag.subarbors(T, ds, f'APL_{side}')
    # Report 10's names: pieces holding >= 2% of APL synapses on microglomeruli of uniglomerular PNs, S1, S2, ... by that share
    d, _, _ = ag.synapses(ds, f'APL_{side}')
    pc = lab[cKDTree(T['xyz']).query(d[['x', 'y', 'z']].values)[1]]
    vc = pd.Series(pc[(d.glom != '').values]).value_counts(normalize=True)
    big = [p for p in vc.index if p >= 0 and vc[p] >= 0.02]
    nm = {int(p): f'S{i + 1}' for i, p in enumerate(big)}
    names[k] = {str(p): n for p, n in nm.items()}
    # per KC: how its APL->KC and KC->APL synapses spread over compartments
    Oc, Ic = g['Oc'].toarray(), g['Ic'].toarray()
    for j in range(Oc.shape[1]):
        o, i_ = Oc[:, j], Ic[:, j]; tot = o + i_
        if tot.sum() == 0: continue
        main = int(np.argmax(tot))
        spread.append(dict(mushroom_body=k, kc_class=mb['cls'][j], n_syn=int(tot.sum()), n_compartments=int((tot > 0).sum()),
                           main_share=float(tot[main] / tot.sum()), main_compartment=nm.get(int(g['comp_labels'][main]), 'small piece')))
    # coupling matrices
    N = len(mb['W']); u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean(); Mc = np.outer(u, v) / N
    same_cls = mb['cls'][:, None] == mb['cls'][None]
    Fs = {l: acm.phi_cut(g['cpar'], g['clen'], g['same'], g['Win'], l) for l in [np.inf, 100, 25, 10]}
    MM = {'C': Mc, 'compartments inf': np.asarray(g['Wout'].T @ Fs[np.inf]),
          'compartments inf, per length': (Oc / g['Lc'][:, None]).T @ Ic}
    for l in [100, 25, 10]:
        MM[f'compartments {l} µm'] = np.asarray(g['Wout'].T @ Fs[l])
        MM[f'local {l} µm'] = lm.kernel_M(g['cpar'], g['clen'], g['Win'], g['Wout'], l).astype(float)
    for vv, M in MM.items():
        M = M * (Mc.mean() / M.mean()); rs_ = M.sum(1); ok = rs_ > 0
        props.append(dict(mushroom_body=k, version=vv,
                          effective_inhibitors_frac_median=float(np.median(rs_[ok] ** 2 / (M[ok] ** 2).sum(1)) / N),
                          zero_coupling_pairs=float((M == 0).mean()), r_with_C=float(np.corrcoef(M.ravel()[::53], Mc.ravel()[::53])[0, 1]),
                          same_class_share=float((M * same_cls).sum() / M.sum()), self_share=float(np.mean(np.diag(M)[ok] / rs_[ok])),
                          row_sum_cv=float(rs_.std() / rs_.mean())))
    # positions of KC<->APL synapse nodes (subsample) with compartment names, posterior view
    rows = np.r_[g['Win'].tocoo().row, g['Wout'].tocoo().row]
    take = rng.choice(len(rows), min(8000, len(rows)), replace=False); rr = rows[take]
    X, Y = VIEW[ds](g['node_xyz'][rr] / 1e3)
    pos.append(pd.DataFrame(dict(mushroom_body=k, x_um=X, y_um=Y, compartment=[nm.get(int(c), 'small piece') for c in g['comp'][rr]])))
    print(k, 'compartments', len(g['comp_labels']), 'named', nm, flush=True)
pd.concat(pos, ignore_index=True).round(2).to_csv(OD + f'r12_positions{TAG}.csv', index=False)
json.dump(names, open(OD + f'r12_names{TAG}.json', 'w'), indent=1)
pd.DataFrame(spread).to_csv(OD + f'r12_kc_spread{TAG}.csv', index=False)
pd.DataFrame(props).round(5).to_csv(OD + f'r12_M_properties{TAG}.csv', index=False)
