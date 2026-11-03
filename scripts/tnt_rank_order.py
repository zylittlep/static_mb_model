"""Does the rank order of odor-pair similarity change between APL blocked and intact? (Yang, 2026-10-16)
Data: Spearman between TNT and TNT3c similarity over the 120 odor pairs (cosine per recording, averaged), raw and 0/1 (> 1)
responses; noise check with half-samples (5 vs 5 TNT3c recordings, 4 vs 4 TNT): within-condition vs between-condition.
Model (official ROIs, pinned as in tnt_compare.py, threshold 1): Spearman between blocked and intact similarity per
mushroom body, for uniform feedback (U, from the TNT_CACHE npz written by tnt_compare.py) and for a fixed threshold raise
that is the same for every odor and KC. Output: odor_weighted/TNT_rank_order.csv."""
import sys
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from orn_load import load
import apl_block_model as abm
UP = '/mnt/project-files/uploads/hearth/'
D = {'TNT': load(UP + '2e7fb552-f4ec-49d7-b06f-14ee778f6433'), 'TNT3c': load(UP + '76debf61-257e-4375-b4d8-a02b48c4f62a')}
O = list(next(iter(D['TNT'].values())).columns); IU = np.triu_indices(16, 1)
def cos(X):
    n = np.linalg.norm(X, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (X.T @ X) / np.outer(n, n)
SR = {c: np.array([cos(v.values)[IU] for v in d.values()]) for c, d in D.items()}
SB = {c: np.array([cos((v.values > 1).astype(float))[IU] for v in d.values()]) for c, d in D.items()}
rs = np.random.default_rng(3)
OUT = []
for nm, S in [('raw responses', SR), ('responses > 1 (0/1)', SB)]:
    full = spearmanr(S['TNT'].mean(0), S['TNT3c'].mean(0))[0]
    w3, wt, bt = [], [], []
    for _ in range(1000):
        a, b = rs.permutation(len(S['TNT3c'])), rs.permutation(len(S['TNT']))
        A3, B3 = S['TNT3c'][a[:5]].mean(0), S['TNT3c'][a[5:10]].mean(0); AT, BT = S['TNT'][b[:4]].mean(0), S['TNT'][b[4:8]].mean(0)
        w3.append(spearmanr(A3, B3)[0]); wt.append(spearmanr(AT, BT)[0]); bt.append(np.mean([spearmanr(AT, B3)[0], spearmanr(BT, A3)[0]]))
    OUT += [dict(source='data', responses=nm, comparison='TNT vs TNT3c, all recordings', spearman=full),
            dict(source='data', responses=nm, comparison='half vs half, within TNT3c (mean of 1000 splits)', spearman=np.mean(w3)),
            dict(source='data', responses=nm, comparison='half vs half, within TNT', spearman=np.mean(wt)),
            dict(source='data', responses=nm, comparison='half vs half, TNT vs TNT3c', spearman=np.mean(bt)),
            dict(source='data', responses=nm, comparison='fraction of splits with between < both withins', spearman=np.mean((np.array(bt) < np.array(w3)) & (np.array(bt) < np.array(wt))))]
    print(f'DATA {nm}: spearman(TNT, TNT3c) over 120 pairs = {full:.3f}; with half-samples: within TNT3c {np.mean(w3):.3f}, within TNT {np.mean(wt):.3f}, '
          f'between {np.mean(bt):.3f}; between < both withins in {np.mean((np.array(bt) < np.array(w3)) & (np.array(bt) < np.array(wt))):.2f} of splits')
# model: uniform feedback (U) and a fixed, odor-independent threshold raise (same pinning)
z = np.load(sys.argv[1])   # npz written by tnt_compare.py with TNT_CACHE
CL = {c: pd.Series(np.mean([(v.values > 1).mean(0) for v in d.values()], 0), index=O) for c, d in D.items()}
RESP = abm.responses(); MB, _ = abm.mushroom_bodies()
for rn, R in RESP.items():
    cl_b = CL['TNT'].reindex(R.columns).values; target = float(CL['TNT3c'].reindex(R.columns).mean())
    iu = np.triu_indices(R.shape[1], 1); rows = []
    for k, mb in MB.items():
        W = mb['W']; gl = [g for g in W.columns if g in R.index]
        Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None); H = Wm @ Rp
        g = np.array([1.0 / np.quantile(H[:, j], 1 - cl_b[j]) for j in range(H.shape[1])]); X = Wm @ (Rp * g[None])
        c = abm.calib(lambda c: (X - 1 - c > 0).mean(0).mean(), target)
        Sb, Sf, Sin = cos(np.maximum(X - 1, 0))[iu], cos(np.maximum(X - 1 - c, 0))[iu], cos(X)[iu]
        Sbz, Su = z[f'1.0|{rn}|{k}|S|blocked'], z[f'1.0|{rn}|{k}|S|U']
        assert np.allclose(Sb, Sbz, atol=1e-8, equal_nan=True)
        ok = ~np.isnan(Su) & ~np.isnan(Sf)
        rows.append(dict(mb=k, blocked_vs_U=spearmanr(Sb[ok], Su[ok])[0], blocked_vs_fixed=spearmanr(Sb[ok], Sf[ok])[0],
                         input_vs_blocked=spearmanr(Sin[ok], Sb[ok])[0], input_vs_U=spearmanr(Sin[ok], Su[ok])[0], input_vs_fixed=spearmanr(Sin[ok], Sf[ok])[0],
                         cl_spread_blocked=np.std((np.maximum(X - 1, 0) > 0).mean(0)), cl_spread_fixed=np.std((X - 1 - c > 0).mean(0))))
    T = pd.DataFrame(rows)
    print(f'\nMODEL {rn} (spearman over odor pairs, per mushroom body):'); print(T.round(3).to_string(index=False))
    print('  median:', T.drop(columns='mb').median().round(3).to_dict())
    for _, r in T.iterrows():
        OUT += [dict(source='model', responses=rn, mushroom_body=r.mb, comparison='blocked vs intact, uniform feedback (U)', spearman=r.blocked_vs_U),
                dict(source='model', responses=rn, mushroom_body=r.mb, comparison='blocked vs intact, fixed threshold raise', spearman=r.blocked_vs_fixed)]
pd.DataFrame(OUT).round(4).to_csv('/mnt/project-files/apl_synapses/odor_weighted/TNT_rank_order.csv', index=False)
