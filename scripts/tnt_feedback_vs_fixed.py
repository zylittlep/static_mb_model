"""Why does APL's feedback make the effect of blocking APL uneven across odors? (Yang, 2026-10-16)
Model pinned to the TNT / TNT3c data as in tnt_compare.py (official ROIs, threshold 1). With a uniform APL, inhibition is a
threshold raise that is the same for every KC but can differ between odors. Four rules for that raise are compared, each with
the blocked responses of tnt_compare.py:
  feedback      uniform APL feedback (U), read from the TNT_CACHE npz written by tnt_compare.py;
  fixed         the same raise for every odor, set so the mean fraction responding matches TNT3c (as in tnt_rank_order.py);
  same share    a raise per odor, set so every odor keeps the same share of its responding KCs (mean matches TNT3c);
  TNT3c per odor  a raise per odor, set so each odor's fraction responding equals its TNT3c fraction.
Per odor: gain in fraction responding when APL is blocked (blocked / intact), spread across odors (coefficient of variation),
and the additive per-odor part of the similarity change (dS_ab = mu + f_a + f_b). Per pair: model similarity change vs the
data's (r over pairs, per mushroom body, bootstrap over recordings, each an independent hemisphere).
Data only: the change in similarity vs the change in chance overlap sqrt(p_a p_b) (cosine of two unrelated 0/1 codes, p =
fraction responding, per recording); and whether odors that drive more units gain more, with odor strength (fraction > 1,
both conditions) and gain taken from separate halves of the recordings (4 + 5 for strength, 4 + 6 for gain).
Usage: python3 -I tnt_feedback_vs_fixed.py <cache.npz>. Outputs: odor_weighted/TNT_feedback_vs_fixed.csv (per odor),
odor_weighted/TNT_feedback_vs_fixed.json (summary)."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from orn_load import load
import apl_block_model as abm

OUT = abm.B + 'odor_weighted/'          # official: odor_weighted/; APL_VARIANT picks the variant folder
UP = '/mnt/project-files/uploads/hearth/'
D = {'TNT': load(UP + '2e7fb552-f4ec-49d7-b06f-14ee778f6433'), 'TNT3c': load(UP + '76debf61-257e-4375-b4d8-a02b48c4f62a')}
O = list(next(iter(D['TNT'].values())).columns)
z = np.load(sys.argv[1])
rs = np.random.default_rng(5)
RULES = ['feedback', 'fixed', 'same share', 'TNT3c per odor']
cv = lambda x: round(float(np.std(x) / np.mean(x)), 3)


def cos(X):
    n = np.linalg.norm(X, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (X.T @ X) / np.outer(n, n)


def per_odor(dS, a, b, od):
    """Least-squares fit dS_ab = mu + f_a + f_b; returns f per odor and the share of variance over pairs it explains."""
    A = np.zeros((len(dS), len(od) + 1)); A[:, -1] = 1
    for i, (x, y) in enumerate(zip(a, b)): A[i, od.index(x)] = 1; A[i, od.index(y)] = 1
    f = np.linalg.lstsq(A, dS, rcond=None)[0]
    return pd.Series(f[:-1], index=od), float(1 - np.var(dS - A @ f) / np.var(dS))


# data, one entry per recording: cosine similarity (raw, 0/1), fraction responding (> 1) and response size per odor
SR = {c: np.array([cos(v.values) for v in d.values()]) for c, d in D.items()}
SB = {c: np.array([cos((v.values > 1).astype(float)) for v in d.values()]) for c, d in D.items()}
SIZE = {'fraction > 1': lambda X: (X > 1).mean(0), 'mean positive response': lambda X: np.clip(X, 0, None).mean(0),
        'root mean square response': lambda X: np.sqrt((X ** 2).mean(0)), 'mean response above 1': lambda X: np.clip(X - 1, 0, None).mean(0)}
ST = {k: {c: np.array([f(v.values) for v in d.values()]) for c, d in D.items()} for k, f in SIZE.items()}
PR = ST['fraction > 1']
CL = {c: pd.Series(PR[c].mean(0), index=O) for c in D}
iu16 = np.triu_indices(16, 1)
chance = {c: np.mean([np.sqrt(np.outer(p, p))[iu16] for p in PR[c]], 0) for c in D}
d_chance = chance['TNT'] - chance['TNT3c']
d_raw = (SR['TNT'].mean(0) - SR['TNT3c'].mean(0))[iu16]
d_bin = (np.nanmean(SB['TNT'], 0) - np.nanmean(SB['TNT3c'], 0))[iu16]
split = {k: [] for k in SIZE}
for _ in range(2000):
    t, n = rs.permutation(len(PR['TNT'])), rs.permutation(len(PR['TNT3c']))
    strength = (PR['TNT'][t[:4]].mean(0) + PR['TNT3c'][n[:5]].mean(0)) / 2
    for k in SIZE:
        split[k].append(spearmanr(strength, ST[k]['TNT'][t[4:]].mean(0) / ST[k]['TNT3c'][n[5:]].mean(0))[0])
summary = {'data, 16 odors': {
    'r over pairs, similarity change vs change in chance overlap sqrt(pa pb)': {
        'raw responses': round(float(np.corrcoef(d_raw, d_chance)[0, 1]), 3), 'responses > 1 (0/1)': round(float(np.corrcoef(d_bin, d_chance)[0, 1]), 3)},
    'mean 0/1 similarity, TNT3c and TNT': [round(float(np.nanmean(SB[c], 0)[iu16].mean()), 3) for c in ['TNT3c', 'TNT']],
    'mean chance overlap sqrt(pa pb), TNT3c and TNT': [round(float(chance[c].mean()), 3) for c in ['TNT3c', 'TNT']],
    'spread across odors (coefficient of variation), TNT and TNT3c': {k: [cv(ST[k]['TNT'].mean(0)), cv(ST[k]['TNT3c'].mean(0))] for k in SIZE},
    'spearman, odor strength (half of the recordings) vs gain TNT / TNT3c (other half), 2000 splits: mean, 5%, 95%, fraction > 0': {
        k: [round(float(np.mean(v)), 3), round(float(np.quantile(v, 0.05)), 3), round(float(np.quantile(v, 0.95)), 3), round(float(np.mean(np.array(v) > 0)), 3)]
        for k, v in split.items()}}}
draws = [{c: rs.choice(len(SR[c]), len(SR[c])) for c in D} for _ in range(1000)]

P = pd.read_csv(OUT + 'TNT_vs_model_odor_pairs.csv').query('threshold == 1.0')
RESP = abm.responses(); MB, _ = abm.mushroom_bodies()
rows = []
for rn in ['ORN', 'GH146']:
    R = RESP[rn]; od = list(R.columns); iu = np.triu_indices(len(od), 1); ix = np.ix_([O.index(o) for o in od], [O.index(o) for o in od])
    cl_b = CL['TNT'].reindex(od).values; cl_i = CL['TNT3c'].reindex(od).values; target = float(cl_i.mean())
    frac = {r: [] for r in RULES + ['blocked']}; dS = {r: [] for r in RULES}
    for k, mb in MB.items():
        W = mb['W']; gl = [g for g in W.columns if g in R.index]
        Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None); H = Wm @ Rp
        g = np.array([1.0 / np.quantile(H[:, j], 1 - cl_b[j]) for j in range(H.shape[1])]); X = Wm @ (Rp * g[None])
        Sb = cos(np.maximum(X - 1, 0))[iu]; fb = (X > 1).mean(0)
        assert np.allclose(Sb, z[f'1.0|{rn}|{k}|S|blocked'], atol=1e-8, equal_nan=True)
        per = lambda tgt: np.array([abm.calib(lambda c, j=j: (X[:, j] - 1 - c > 0).mean(), tgt[j]) for j in range(X.shape[1])])
        raise_ = {'fixed': np.full(len(od), abm.calib(lambda c: (X - 1 - c > 0).mean(0).mean(), target)),
                  'same share': per(fb * target / fb.mean()), 'TNT3c per odor': per(cl_i)}
        frac['blocked'].append(fb); frac['feedback'].append(z[f'1.0|{rn}|{k}|cl|U']); dS['feedback'].append(Sb - z[f'1.0|{rn}|{k}|S|U'])
        for r, c in raise_.items():
            frac[r].append((X - 1 - c[None] > 0).mean(0)); dS[r].append(Sb - cos(np.maximum(X - 1 - c[None], 0))[iu])
    frac = {r: np.mean(v, 0) for r, v in frac.items()}
    per_mb = {r: np.array(v) for r, v in dS.items()}; dS = {r: np.nanmean(v, 0) for r, v in dS.items()}
    dd = (SR['TNT'].mean(0) - SR['TNT3c'].mean(0))[ix][iu]
    a, b = [od[i] for i in iu[0]], [od[j] for j in iu[1]]
    Q = P.query('responses == @rn').set_index(['odor_a', 'odor_b'])
    q = pd.DataFrame([Q.loc[(x, y)] if (x, y) in Q.index else Q.loc[(y, x)] for x, y in zip(a, b)])
    assert np.allclose(dd, q.dS_data, atol=1e-3) and np.allclose(dS['feedback'], q.dS_model_U, atol=1e-3)
    boot = np.array([[np.corrcoef((SR['TNT'][dr['TNT']].mean(0) - SR['TNT3c'][dr['TNT3c']].mean(0))[ix][iu], dS[r])[0, 1] for r in RULES]
                     for dr in draws])
    fd, sd = per_odor(dd, a, b, od); pf = {r: per_odor(dS[r], a, b, od) for r in RULES}
    gain_d = (CL['TNT'] / CL['TNT3c']).reindex(od).values; gain = {r: frac['blocked'] / frac[r] for r in RULES}
    summary[rn] = {
        'spread of fraction responding across odors (coefficient of variation)': dict(
            {'model blocked (= TNT)': cv(frac['blocked'])}, **{f'model intact, {r}': cv(frac[r]) for r in RULES[:3]},
            **{'data TNT3c': cv(cl_i)}),
        'gain per odor (blocked / intact fraction responding), range': dict(
            {r: [round(float(gain[r].min()), 2), round(float(gain[r].max()), 2)] for r in RULES[:3]}, data=[round(float(gain_d.min()), 2), round(float(gain_d.max()), 2)]),
        'spearman over odors, gain vs fraction responding with APL blocked': dict(
            {r: round(float(spearmanr(frac['blocked'], gain[r])[0]), 2) for r in RULES[:3]}, data=round(float(spearmanr(cl_b, gain_d)[0]), 2)),
        'spearman over odors, model gain vs data gain': {r: round(float(spearmanr(gain_d, gain[r])[0]), 2) for r in RULES[:3]},
        'mean similarity change': dict(data=round(float(dd.mean()), 3), **{r: round(float(dS[r].mean()), 3) for r in RULES}),
        'r over pairs with the data': {r: round(float(np.corrcoef(dd, dS[r])[0, 1]), 3) for r in RULES},
        'r over pairs, feedback vs fixed model change': round(float(np.corrcoef(dS['feedback'], dS['fixed'])[0, 1]), 3),
        'r over pairs per mushroom body': {r: dict(zip(MB, [round(float(np.corrcoef(dd, x)[0, 1]), 3) for x in per_mb[r]])) for r in RULES},
        'bootstrap over recordings, r 5-95%': {r: np.round(np.quantile(boot[:, i], [0.05, 0.95]), 3).tolist() for i, r in enumerate(RULES)},
        'bootstrap over recordings, fraction of draws with feedback r higher than': {r: round(float(np.mean(boot[:, 0] > boot[:, i])), 3) for i, r in enumerate(RULES) if i},
        'share of the change over pairs that is per odor': dict(data=round(sd, 2), **{r: round(pf[r][1], 2) for r in RULES}),
        'spearman of per-odor parts with the data': {r: round(float(spearmanr(fd, pf[r][0])[0]), 2) for r in RULES}}
    for j, o in enumerate(od):
        rows.append(dict(responses=rn, odor=o, frac_TNT=CL['TNT'][o], frac_TNT3c=CL['TNT3c'][o], gain_data=gain_d[j], model_frac_blocked=frac['blocked'][j],
                         **{f'model_frac_intact_{r.replace(" ", "_")}': frac[r][j] for r in RULES[:3]},
                         **{f'gain_model_{r.replace(" ", "_")}': gain[r][j] for r in RULES[:3]},
                         per_odor_dS_data=fd[o], **{f'per_odor_dS_{r.replace(" ", "_")}': pf[r][0][o] for r in RULES}))
pd.DataFrame(rows).round(4).to_csv(OUT + 'TNT_feedback_vs_fixed.csv', index=False)
json.dump(summary, open(OUT + 'TNT_feedback_vs_fixed.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
