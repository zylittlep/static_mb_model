"""New ORN inputs from Yang's responses_wide.csv (2026-11-03) and the consistency check of the two diagnostic datasets.

Input: /mnt/project-files/uploads/hearth/780d103e-a7b8-4cc2-ac34-39d15f700dab (responses_wide.csv): per trial, per fly, per
glomerulus; panels megamat (dataset 1), validation2 (dataset 4), kmix/cmix (dataset 5), diagnostics (all three).
Per fly: mean over trials. Per dataset: mean over flies (NaN-skipping; a glomerulus counts only in the flies where it was imaged).

Outputs (folder orn_inputs_2026-11-03/):
  ORN_megamat_diagnostics_flymean.csv   glomeruli x odors, dataset 1 flies (megamat + diagnostics). 2h, ms and va @ -3 are in both
                                        panels; the megamat presentation is used (as in the earlier tables).
  ORN_validation2_diagnostics_flymean.csv  glomeruli x odors, dataset 4 flies.
  ORN_*_nflies.csv                      number of flies behind each value.
  ORN_TNT16_flymean.csv                 the 16 TNT odors for the model: 12 from megamat, -aPine/euc/B-myr/paa from validation2
                                        (as in the old ORN_ovlp16 table), plus a comparison with that old table.
  diagnostics_consistency.json/.png     are the diagnostics of datasets 1 and 4 the same? (see check() below)
Run: python3 -I orn_inputs_2026_11_03.py"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SRC = '/mnt/project-files/uploads/hearth/780d103e-a7b8-4cc2-ac34-39d15f700dab'
OLD = '/mnt/project-files/uploads/hearth/5d477ed3-345f-4332-8b48-1ad3e2e0a0c1'
OUT = '/mnt/project-files/apl_synapses/orn_inputs_2026-11-03/'
os.makedirs(OUT, exist_ok=True)
rs = np.random.default_rng(0)
BLUE, ORANGE, GRAY, INK = '#2a78d6', '#eb6834', '#898781', '#0b0b0b'

X = pd.read_csv(SRC, header=[0, 1, 2, 3], index_col=[0, 1, 2, 3])
T = X.groupby(level=['panel', 'odor1', 'odor2']).mean()                    # mean over trials (repeats)


def flies(ds, panels):
    """{fly: DataFrame odors x glomeruli} for one dataset; odor labels 'name @ conc' (mixtures keep both parts)."""
    s = T.loc[T.index.get_level_values('panel').isin(panels), T.columns.get_level_values(0) == ds]
    s = s[s.notna().any(axis=1)]
    lab = [o1 if o2 == 'solvent' else f'{o1} + {o2}' for _, o1, o2 in s.index]
    pan = s.index.get_level_values('panel')
    out = {}
    for fly in s.columns.droplevel(3).unique():
        m = s.loc[:, [c for c in s.columns if c[:3] == fly]]
        m.columns = m.columns.get_level_values(3); m.index = lab
        m = m[~((pan == 'diagnostics') & pd.Index(lab).isin(pd.Index(lab)[pan != 'diagnostics']))]   # same odor in two panels: keep the main panel's
        out[f'{fly[1]}_fly{fly[2]}'] = m.loc[:, m.notna().any(axis=0)]
    return out


def dup_r(ds, main):
    """r over flies x glomeruli between the main-panel and diagnostics presentations of the same odor at the same concentration."""
    s = T.loc[:, T.columns.get_level_values(0) == ds]
    a = s.xs(main, level='panel').droplevel('odor2'); b = s.xs('diagnostics', level='panel').droplevel('odor2')
    out = {}
    for o in a.index.intersection(b.index):
        x, y = a.loc[o].values, b.loc[o].values; ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 10: out[o] = round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 3)
    return out


def flymean(fl):
    A = pd.concat(fl, names=['fly', 'odor'])
    return A.groupby(level='odor').mean().T, A.notna().groupby(level='odor').sum().T     # glomeruli x odors


FL = {'1': flies('1', ['megamat', 'diagnostics']), '4': flies('4', ['validation2', 'diagnostics'])}
NAME = {'1': 'megamat_diagnostics', '4': 'validation2_diagnostics'}
MEAN = {}
for ds, fl in FL.items():
    m, n = flymean(fl); MEAN[ds] = m
    m.round(5).to_csv(OUT + f'ORN_{NAME[ds]}_flymean.csv', index_label='glomerulus')
    n.to_csv(OUT + f'ORN_{NAME[ds]}_nflies.csv', index_label='glomerulus')

# ---------------------------------------------------------------- 16 TNT odors for the model
OD16 = {'2h': ('1', '2h @ -3'), 'IaA': ('1', 'IaA @ -3'), 'pa': ('1', 'pa @ -3'), '2-but': ('1', '2-but @ -3'), 'eb': ('1', 'eb @ -3'),
        'ep': ('1', 'ep @ -3'), '6al': ('1', '6al @ -3'), 't2h': ('1', 't2h @ -3'), '1-8ol': ('1', '1-8ol @ -3'), '1-5ol': ('1', '1-5ol @ -3'),
        '1-6ol': ('1', '1-6ol @ -3'), 'ms': ('1', 'ms @ -3'), 'paa': ('4', 'paa @ -4'), '-aPine': ('4', '-aPine @ -1.5'),
        'euc': ('4', 'euc @ -1.75'), 'B-myr': ('4', 'B-myr @ -2')}
old = pd.read_csv(OLD, index_col=0)
new16 = pd.DataFrame({o: MEAN[ds][c] for o, (ds, c) in OD16.items()})
new16 = new16[new16.notna().any(axis=1)]
new16.round(5).to_csv(OUT + 'ORN_TNT16_flymean.csv', index_label='glomerulus')
g = old.index.intersection(new16.index)
cmp16 = {'glomeruli: old table / new': [len(old), len(new16)], 'shared glomeruli': len(g),
         'only in new': sorted(set(new16.index) - set(old.index)), 'only in old': sorted(set(old.index) - set(new16.index)),
         'per odor r (new vs old, shared glomeruli)': {o: round(float(new16.loc[g, o].corr(old.loc[g, o])), 3) for o in old.columns},
         'per odor max |new - old|': {o: round(float((new16.loc[g, o] - old.loc[g, o]).abs().max()), 3) for o in old.columns}}

# ---------------------------------------------------------------- diagnostics: dataset 1 vs dataset 4
D = {ds: {f: m.loc[[o for o in m.index if o in set(T.xs('diagnostics', level='panel').index.get_level_values(0))]] for f, m in FL[ds].items()}
     for ds in FL}
ALLO = sorted(set().union(*[m.index for d in D.values() for m in d.values()]))
ALLG = sorted(set().union(*[m.columns for d in D.values() for m in d.values()]))
ARR = {ds: np.array([m.reindex(index=ALLO, columns=ALLG).values for m in d.values()]) for ds, d in D.items()}
cov = {ds: np.isfinite(a).sum(0) for ds, a in ARR.items()}
o_ok = [i for i in range(len(ALLO)) if all(cov[ds][i].max() >= len(ARR[ds]) / 2 for ds in ARR)]
g_ok = [j for j in range(len(ALLG)) if all(cov[ds][:, j].max() >= len(ARR[ds]) / 2 for ds in ARR)]
ODO, GLO = [ALLO[i] for i in o_ok], [ALLG[j] for j in g_ok]
A1, A4 = ARR['1'][:, o_ok][:, :, g_ok], ARR['4'][:, o_ok][:, :, g_ok]


def cosm(M):
    M = np.nan_to_num(np.clip(M, 0, None)); n = np.linalg.norm(M, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (M @ M.T) / np.outer(n, n)


def check(a1, a4, nd=1000, k=3):
    """Matched-size comparison: mean of k flies vs mean of k other flies, within dataset 1, within dataset 4, and between.
    'pattern r': Pearson r over all glomerulus x odor values. 'similarity r': Pearson r between the two odor-by-odor similarity matrices
    (cosine over glomeruli, negatives set to 0). Plus a fly-label permutation test of the mean squared difference between datasets."""
    iu = np.triu_indices(a1.shape[1], 1)

    def st(x, y):
        a, b = np.nanmean(x, 0), np.nanmean(y, 0); ok = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[ok], b[ok])[0, 1], np.corrcoef(np.nan_to_num(cosm(a)[iu]), np.nan_to_num(cosm(b)[iu]))[0, 1]
    W1, W4, BT = [], [], []
    for _ in range(nd):
        p, q = rs.permutation(len(a1)), rs.permutation(len(a4))
        W1.append(st(a1[p[:k]], a1[p[k:2 * k]])); W4.append(st(a4[q[:k]], a4[q[k:2 * k]])); BT.append(st(a1[p[:k]], a4[q[:k]]))
    W1, W4, BT = map(np.array, (W1, W4, BT))
    allf = np.concatenate([a1, a4]); n1 = len(a1)
    msd = lambda idx: np.nanmean((np.nanmean(allf[idx[:n1]], 0) - np.nanmean(allf[idx[n1:]], 0)) ** 2)
    obs = msd(np.arange(len(allf))); null = np.array([msd(rs.permutation(len(allf))) for _ in range(5000)])
    summ = lambda v: {'median': round(float(np.median(v)), 3), '5%': round(float(np.percentile(v, 5)), 3), '95%': round(float(np.percentile(v, 95)), 3)}
    res = {f'{nm}: {key}': summ(v[:, c]) for nm, v in [('within dataset 1', W1), ('within dataset 4', W4), ('between', BT)]
           for c, key in enumerate(['pattern r', 'similarity r'])}
    res['between below both within 5th percentiles'] = {key: bool(np.median(BT[:, c]) < min(np.percentile(W1[:, c], 5), np.percentile(W4[:, c], 5)))
                                                        for c, key in enumerate(['pattern r', 'similarity r'])}
    res['fly-label permutation: observed / null mean squared difference, p'] = [round(float(obs / null.mean()), 2),
                                                                                 round(float((1 + np.sum(null >= obs)) / (1 + len(null))), 4)]
    return res, (W1, W4, BT)


res_all, dist_all = check(A1, A4)
noHCl = [i for i, o in enumerate(ODO) if not o.startswith('HCl')]
res_nohcl, _ = check(A1[:, noHCl], A4[:, noHCl])
M1, M4 = np.nanmean(A1, 0), np.nanmean(A4, 0)
per_odor = pd.Series({o: np.corrcoef(*[v[np.isfinite(M1[i]) & np.isfinite(M4[i])] for v in (M1[i], M4[i])])[0, 1] for i, o in enumerate(ODO)}).sort_values()
per_glom = []
for j, gname in enumerate(GLO):
    ok = np.isfinite(M1[:, j]) & np.isfinite(M4[:, j]); w = []
    for _ in range(200):
        p = rs.permutation(len(A1)); x, y = np.nanmean(A1[p[:4], :, j], 0), np.nanmean(A1[p[4:8], :, j], 0); o = np.isfinite(x) & np.isfinite(y)
        if o.sum() > 5: w.append(np.corrcoef(x[o], y[o])[0, 1])
    per_glom.append((gname, np.corrcoef(M1[ok, j], M4[ok, j])[0, 1], np.nanmedian(w)))
per_glom = pd.DataFrame(per_glom, columns=['glomerulus', 'r between datasets', 'r within dataset 1 (4 vs 4 flies)']).set_index('glomerulus')
ok = np.isfinite(M1) & np.isfinite(M4)
passed = not (res_all['between below both within 5th percentiles']['pattern r'] or res_all['between below both within 5th percentiles']['similarity r']
              or res_all['fly-label permutation: observed / null mean squared difference, p'][1] < 0.05)
summary = {
    'criterion (chosen 2026-11-03)': 'merge only if (1) for both the response pattern and the odor-similarity matrix, the correlation between '
    '3 flies of dataset 1 and 3 flies of dataset 4 is not below the 5th percentile of the same correlation between two sets of 3 flies of '
    'one dataset, and (2) a fly-label permutation test finds no difference between the dataset means (p >= 0.05)',
    'passed': passed,
    'flies: dataset 1 / 4': [len(A1), len(A4)], 'shared diagnostic odors': ODO, 'shared glomeruli': GLO,
    'left out (odor imaged in fewer than half the flies of one dataset)': [o for o in ALLO if o not in ODO],
    'all shared odors': res_all, 'without HCl': res_nohcl,
    'response size: dataset 4 / dataset 1 (least squares through 0)': round(float((M1[ok] * M4[ok]).sum() / (M1[ok] ** 2).sum()), 3),
    'per odor: r between datasets over glomeruli (lowest 5)': per_odor.head(5).round(2).to_dict(),
    'per glomerulus: r between datasets over odors (lowest 8)': per_glom.sort_values('r between datasets').head(8).round(2).to_dict('index'),
    'HCl @ -1 in DC4: dataset 1 / 4': [round(float(M1[ODO.index('HCl @ -1'), GLO.index('DC4')]), 3), round(float(M4[ODO.index('HCl @ -1'), GLO.index('DC4')]), 3)]
    if 'HCl @ -1' in ODO and 'DC4' in GLO else None,
    'new 16-odor model input vs old ORN_ovlp16 table': cmp16,
    'same odor and concentration in megamat and diagnostics (dataset 1): r over flies x glomeruli': dup_r('1', 'megamat')}
json.dump(summary, open(OUT + 'diagnostics_consistency.json', 'w'), indent=1)

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={'width_ratios': [1, 1.15, 1]})
W1, W4, BT = dist_all
a = axs[0]
for c, key in enumerate(['pattern r', 'similarity r']):
    for k, (v, col, lab) in enumerate([(W1, BLUE, 'within dataset 1'), (W4, '#1f9e6e', 'within dataset 4'), (BT, ORANGE, 'between datasets')]):
        x = c * 4 + k
        bp = a.boxplot(v[:, c], positions=[x], widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
        bp['boxes'][0].set_facecolor(col); bp['boxes'][0].set_alpha(0.75)
        if c == 0: a.plot([], [], color=col, lw=6, alpha=0.75, label=lab)
a.set_xticks([1, 5], ['response pattern\n(glomerulus × odor values)', 'odor-by-odor similarity\n(cosine over glomeruli)'])
a.set_ylabel('Pearson r, mean of 3 flies vs mean of 3 other flies'); a.legend(fontsize=7.5, loc='lower left')
a.set_title('a  Between datasets vs within', fontsize=10, loc='left')
b = axs[1]
pg = per_glom.sort_values('r between datasets')
b.scatter(pg['r within dataset 1 (4 vs 4 flies)'], pg['r between datasets'], s=24, color=BLUE, zorder=3)
for gname, r in pg.head(8).iterrows():
    b.annotate(gname, (r.iloc[1], r.iloc[0]), xytext=(4, -3), textcoords='offset points', fontsize=7.5, color=INK)
b.plot([-0.3, 1], [-0.3, 1], color=GRAY, lw=0.8, ls='--')
b.set_xlabel('r over odors, two halves of dataset 1 (median)'); b.set_ylabel('r over odors, dataset 1 vs dataset 4')
b.set_title('b  Per glomerulus: odor tuning', fontsize=10, loc='left')
c = axs[2]
po = per_odor
c.barh(range(len(po)), po.values, color=[ORANGE if v < 0.7 else BLUE for v in po.values], height=0.7)
c.set_yticks(range(len(po)), po.index, fontsize=7.5); c.set_xlabel('r over glomeruli, dataset 1 vs dataset 4')
c.set_title('c  Per odor: glomerular pattern', fontsize=10, loc='left'); c.set_xlim(0, 1)
fig.tight_layout(); fig.savefig(OUT + 'diagnostics_consistency.png', dpi=200, facecolor='white'); plt.close(fig)
print(json.dumps({k: summary[k] for k in ['passed', 'flies: dataset 1 / 4', 'response size: dataset 4 / dataset 1 (least squares through 0)']}, indent=0))
print(json.dumps(res_all, indent=0)); print(json.dumps(cmp16, indent=0)[:1500])
