"""Odor-odor similarity of KC<->APL synapses on claws, each weighted by its claw glomerulus's response (official ROIs only).

Odors: the 16 odors of Yang's ORN_ovlp16_resp_flyave_merge_dropna.csv, taken as the APL16 set of the TNT / TNT3c data, in
that table's order: 2h, IaA, pa, 2-but, eb, ep, 6al, t2h, 1-8ol, 1-5ol, 1-6ol, ms (@ -3, megamat panel), paa (@ -4),
-aPine (@ -1.5), euc (@ -1.75), B-myr (@ -2) (validation2 panel).
Responses: ORN = that table (36 glomeruli, fly means). GH146 = mean peak dF/F over 7 flies x 3 repeats (29 glomeruli,
GH146_mean_by_glomerulus.csv); it has 13 of the 16 odors: the 12 megamat ones @ -3 and paa @ -5 (glomeruli_diagnostics);
-aPine, euc and B-myr were not presented (blank rows and columns).
Synapses: KC<->APL synapses on KC claws in the official calyx ROIs (KC_APL_synapses_on_claws_by_glomerulus.csv: KC-side
site within 2 µm of a claw), both APLs of a dataset together, KC->APL and APL->KC separately. Each synapse gets the response
of the glomerulus of its claw's main PN. Synapses whose claw glomerulus has no response in that table (or whose main PN is
bi-/multiglomerular or untyped) are left out, not set to 0.
Similarity: Pearson correlation between two odors' synapse vectors. Every synapse on a claw of glomerulus g carries g's
response, so this equals the correlation across glomeruli weighted by each glomerulus's number of on-claw KC<->APL synapses.
Reference ("responses only"): the same correlation with each glomerulus of the response table counted once."""
import json, sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts')
from variant import OUT as VOUT, TAG

OUT = VOUT + 'odor_weighted/'
ORN_TABLE = '/mnt/project-files/uploads/hearth/5d477ed3-345f-4332-8b48-1ad3e2e0a0c1'   # ORN_ovlp16_resp_flyave_merge_dropna.csv
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
DIRS = [('KC_to_APL', 'KC → APL'), ('APL_to_KC', 'APL → KC')]
INK, MUTED, NAN_C = '#0b0b0b', '#52514e', '#d6d6d4'
CMAP = matplotlib.colormaps['RdBu_r'].copy(); CMAP.set_bad(NAN_C)

orn = pd.read_csv(ORN_TABLE, index_col=0)
ODORS = list(orn.columns)
assert len(ODORS) == 16
gh = pd.read_csv(OUT + 'GH146_mean_by_glomerulus.csv', index_col=0).drop(columns='n_flies')
GH_COL = {o: f'megamat: {o} @ -3' for o in ODORS} | {'paa': 'glomeruli_diagnostics: paa @ -5'}
gh16 = pd.DataFrame({o: gh[c] for o, c in GH_COL.items() if c in gh.columns}).reindex(columns=ODORS)
RESP = {'ORN': orn, 'GH146': gh16}
CONC = {'ORN': '2h … ms @ −3 (megamat), paa @ −4, −aPine @ −1.5, euc @ −1.75, B-myr @ −2 (validation2); 36 glomeruli, fly means',
        'GH146': '2h … ms @ −3 (megamat), paa @ −5 (diagnostics); −aPine, euc, B-myr not in the GH146 panels; 29 glomeruli, '
                 'mean over 7 flies × 3 repeats'}

K = pd.read_csv(OUT + 'KC_APL_synapses_on_claws_by_glomerulus.csv', usecols=['dataset', 'apl_id', 'direction', 'glomeruli'])
K['glomerulus'] = K.glomeruli.fillna('').where(~K.glomeruli.fillna('').str.contains(r'\+'), '')   # bi-/multiglomerular or untyped main PN: ''
N = K.groupby(['dataset', 'direction', 'glomerulus']).size()


def wcorr(R, w):
    """Weighted Pearson correlation between the columns of R (glomeruli x odors), weights w per glomerulus; NaN columns stay NaN."""
    X = R.values.astype(float); w = np.asarray(w, float)
    ok = ~np.isnan(X).all(0)
    Xo = X[:, ok]
    mu = (w[:, None] * Xo).sum(0) / w.sum()
    Xc = Xo - mu
    C = (Xc.T * w) @ Xc / w.sum()
    d = np.sqrt(np.diag(C))
    out = np.full((X.shape[1], X.shape[1]), np.nan)
    out[np.ix_(ok, ok)] = C / np.outer(d, d)
    return pd.DataFrame(out, index=R.columns, columns=R.columns)


rows, info, mats = [], {}, {}
for rn, R in RESP.items():
    mats[(rn, 'responses only', '')] = wcorr(R, np.ones(len(R)))
    for d, _ in DIRS:
        for ds in DS:
            n = N.loc[(ds, d)]
            gl = [g for g in R.index if g in n.index]
            mats[(rn, ds, d)] = wcorr(R.loc[gl], n[gl].values)
            info[f'{rn} {ds} {d}'] = {'synapses_used': int(n[gl].sum()), 'synapses_on_claws': int(n.sum()),
                                      'share_used': round(float(n[gl].sum() / n.sum()), 3), 'glomeruli_used': len(gl)}
for (rn, ds, d), Mx in mats.items():
    t = Mx.stack(future_stack=True).rename('pearson_r').reset_index().rename(columns={'level_0': 'odor_a', 'level_1': 'odor_b'})
    rows.append(t.assign(responses=rn, dataset=ds or 'responses only', direction=d))
L = pd.concat(rows, ignore_index=True)[['responses', 'direction', 'dataset', 'odor_a', 'odor_b', 'pearson_r']]
L.round(4).to_csv(OUT + 'odor_similarity_KC_APL_on_claws.csv', index=False)

# weights actually used: on-claw KC<->APL synapses per glomerulus
Wt = N.unstack(['dataset', 'direction'], fill_value=0)
Wt.columns = [f'{ds} {d}' for ds, d in Wt.columns]
Wt = Wt.drop(index='', errors='ignore')
Wt.insert(0, 'in_GH146', Wt.index.isin(gh16.index)); Wt.insert(0, 'in_ORN', Wt.index.isin(orn.index))
Wt.to_csv(OUT + 'odor_similarity_synapses_per_glomerulus.csv')


def heat(a, Mx, title):
    im = a.imshow(Mx.values, cmap=CMAP, vmin=-1, vmax=1, interpolation='nearest')
    a.set_xticks(range(16)); a.set_yticks(range(16))
    a.set_xticklabels(ODORS, rotation=90, fontsize=7.5, color=INK); a.set_yticklabels(ODORS, fontsize=7.5, color=INK)
    a.tick_params(length=0)
    for s in a.spines.values(): s.set_visible(False)
    a.set_title(title, fontsize=9, color=INK, loc='left')
    return im


def figure(rn, fname):
    fig, ax = plt.subplots(2, 5, figsize=(17.5, 8.6), constrained_layout=True)
    for i, (d, arrow) in enumerate(DIRS):
        for j, ds in enumerate([''] + DS):
            a = ax[i, j]
            if ds == '':
                if i == 0:
                    im = heat(a, mats[(rn, 'responses only', '')],
                              f'{rn} responses only\n{len(RESP[rn])} glomeruli, each counted once')
                else:
                    a.axis('off')
                    a.text(0, .95, 'Pearson r between two odors\' vectors of\nKC↔APL synapses on KC claws, each weighted\n'
                                   f'by the {rn} response of its claw\'s\nglomerulus (main PN). Same as a correlation\n'
                                   'across glomeruli weighted by each\nglomerulus\'s on-claw synapse count.\n\n'
                                   'Synapses on claws of glomeruli without\na response are left out. Grey = odor\nnot in the table.',
                           transform=a.transAxes, fontsize=8.5, color=MUTED, va='top', linespacing=1.35)
                continue
            s = info[f'{rn} {ds} {d}']
            heat(a, mats[(rn, ds, d)], f'{LABEL[ds]} · {arrow} on claws\n{s["synapses_used"]:,} synapses '
                                       f'({s["share_used"]:.0%} of on-claw), {s["glomeruli_used"]} glomeruli')
    cb = fig.colorbar(im, ax=ax, shrink=.5, aspect=25, pad=.01)
    cb.set_label('Pearson r', fontsize=9, color=MUTED); cb.ax.tick_params(labelsize=8, colors=MUTED); cb.outline.set_visible(False)
    fig.suptitle(f'Odor-pair similarity of KC↔APL synapses on claws weighted by {rn} glomerulus responses, APL16 odors · '
                 f'official ROIs, both APLs of a dataset together{TAG}\n{CONC[rn]}', fontsize=11, color=INK, x=0.005, ha='left')
    fig.savefig(OUT + fname, dpi=160, bbox_inches='tight'); plt.close(fig)


figure('ORN', 'odor_similarity_KC_APL_on_claws_ORN.png')
figure('GH146', 'odor_similarity_KC_APL_on_claws_GH146.png')

# ---------------- numbers for the reply ----------------
iu = np.triu_indices(16, 1)
def up(Mx): return Mx.values[iu]
cmp = {}
for rn in RESP:
    base = up(mats[(rn, 'responses only', '')]); ok = ~np.isnan(base)
    for d, _ in DIRS:
        for ds in DS:
            v = up(mats[(rn, ds, d)])
            cmp[f'{rn} {ds} {d}'] = {'max_abs_diff_vs_responses_only': round(float(np.nanmax(np.abs(v - base))), 3),
                                     'mean_abs_diff_vs_responses_only': round(float(np.nanmean(np.abs(v - base))), 3),
                                     'r_with_responses_only': round(float(np.corrcoef(v[ok], base[ok])[0, 1]), 3)}
    for d, _ in DIRS:
        V = np.array([up(mats[(rn, ds, d)])[ok] for ds in DS])
        cmp[f'{rn} {d} between datasets: min r of upper triangles'] = round(float(np.corrcoef(V)[np.triu_indices(4, 1)].min()), 3)
    v1 = np.array([up(mats[(rn, ds, 'KC_to_APL')])[ok] for ds in DS]); v2 = np.array([up(mats[(rn, ds, 'APL_to_KC')])[ok] for ds in DS])
    cmp[f'{rn} KC_to_APL vs APL_to_KC max abs diff'] = round(float(np.abs(v1 - v2).max()), 3)
    b = mats[(rn, 'responses only', '')].where(~np.eye(16, dtype=bool))
    cmp[f'{rn} responses only: most similar pairs'] = b.stack().sort_values(ascending=False).iloc[:6:2].round(2).to_dict().__repr__()
    cmp[f'{rn} responses only: least similar pairs'] = b.stack().sort_values().iloc[:6:2].round(2).to_dict().__repr__()
json.dump({'synapses': info, 'comparisons': cmp, 'odors': ODORS, 'GH146_columns': GH_COL}, open(OUT + 'odor_similarity_summary.json', 'w'), indent=1)
print(json.dumps(cmp, indent=1))
print(json.dumps({k: v for k, v in info.items() if 'flywire' in k or 'banc' in k}, indent=1))
