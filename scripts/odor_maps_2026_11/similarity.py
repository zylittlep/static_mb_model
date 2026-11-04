"""Odor-pair similarity seen through APL's PN synapses (Report 03 revision, 2026-11). Two analyses, PN <-> APL only (the
KC <-> APL versions wait for the claw choice):

1. Weighted vs unweighted cosine similarity (Yang: "another figure, similar to Fig 3, with PN-APL synapses; y weighted by
   PN-APL synapses, x glomeruli counted once"). Odors a, b with glomerular responses r_ga, r_gb:
       x = sum_g r_ga r_gb / sqrt(sum_g r_ga^2 sum_g r_gb^2)                      (every glomerulus of the table once)
       y = sum_g n_g r_ga r_gb / sqrt(sum_g n_g r_ga^2 sum_g n_g r_gb^2)          (n_g = PN<->APL calyx synapses of g)
   y is the cosine over synapses, each synapse carrying its glomerulus's response. ORN: the 16 TNT odors of the new
   table (ORN_TNT16_flymean.csv); GH146: the 13 of them that GH146 has (12 megamat @ -3, paa @ -5 from diagnostics).

2. Odor-by-odor distance between response-weighted synapse maps (Yang: "correlation (euclidean) between odors by weighing
   the APL<->KC/PN synaptic coordinates by responses"). Each odor o gives a map over the calyx: every PN<->APL synapse e at
   position x_e carries the weight w_o(e) = response of its glomerulus, smoothed with a 3D Gaussian of width sigma. With
   G_ee' = exp(-|x_e - x_e'|^2 / (4 sigma^2)) (the overlap of two such Gaussians, normalized to 1 at distance 0):
       K_ab = (1/N) sum_e sum_e' w_a(e) G_ee' w_b(e')
       Euclidean distance d_ab = sqrt(K_aa + K_bb - 2 K_ab);   cosine c_ab = K_ab / sqrt(K_aa K_bb)
   sigma = 0 gives the synapse-level version (G = identity: d = root of the summed squared response differences / sqrt N).
   Reference: the same with glomeruli counted once and no positions ("responses only").
   Done for each new ORN dataset (megamat + diagnostics; validation2 + diagnostics; not merged, they failed the
   consistency check), each calyx of FlyWire, hemibrain and male CNS, and each direction."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import common as c

S = c.pn_synapses()
RS = c.response_sets()
SIGMAS = [0.0, 2.0, 5.0]
MAIN_SIGMA = 2.0
DIRS = [('PN_to_APL', 'PN → APL'), ('APL_to_PN', 'APL → PN')]
summ = {}


def wcos(R, n):
    """cosine between columns of R (glomeruli x odors) with glomerulus weights n"""
    X = R.values * np.sqrt(n)[:, None]
    G = X.T @ X; d = np.sqrt(np.diag(G))
    return pd.DataFrame(G / np.outer(d, d), index=R.columns, columns=R.columns)


# ---------------------------------------------------------------- 1. weighted vs once
orn16 = pd.read_csv(c.NEW + 'ORN_TNT16_flymean.csv', index_col=0)
gh = pd.read_csv(c.OW + 'GH146_mean_by_glomerulus.csv', index_col=0).drop(columns='n_flies')
gh13 = pd.DataFrame({o: gh[f'megamat: {o} @ -3'] for o in orn16.columns if f'megamat: {o} @ -3' in gh.columns})
gh13['paa'] = gh['glomeruli_diagnostics: paa @ -5']
gh13 = gh13[[o for o in orn16.columns if o in gh13.columns]]
rows = []
for resp, R in [('ORN', orn16), ('GH146', gh13)]:
    R = R.dropna()                                    # glomeruli with a value for every odor (ORN: DL4 and VM4 lack the validation2 odors)
    once = wcos(R, np.ones(len(R)))
    for ds in ['flywire', 'hemibrain', 'malecns', 'banc']:
        for side in ['L', 'R']:
            for d, _ in DIRS:
                t = S[(S.dataset == ds) & (S.apl_side == side) & (S.direction == d)]
                if not len(t): continue
                n = t.glomerulus.value_counts().reindex(R.index).fillna(0).values
                w = wcos(R, n)
                for i, a in enumerate(R.columns):
                    for b in R.columns[i + 1:]:
                        rows.append([resp, ds, side, d, a, b, once.loc[a, b], w.loc[a, b], int(n.sum())])
SIM = pd.DataFrame(rows, columns=['responses', 'dataset', 'apl_side', 'direction', 'odor_a', 'odor_b', 'cosine_glomeruli_once',
                                  'cosine_weighted_by_PN_APL_synapses', 'synapses_used'])
SIM.round(5).to_csv(c.OUT + 'odor_similarity_PN_APL_weighted_vs_once.csv', index=False)
agree = SIM.groupby(['responses', 'dataset', 'apl_side', 'direction']).apply(
    lambda g: pd.Series({'r': np.corrcoef(g.cosine_glomeruli_once, g.cosine_weighted_by_PN_APL_synapses)[0, 1],
                         'mean_change': (g.cosine_weighted_by_PN_APL_synapses - g.cosine_glomeruli_once).mean(),
                         'max_abs_change': (g.cosine_weighted_by_PN_APL_synapses - g.cosine_glomeruli_once).abs().max()}), include_groups=False)
agree.round(4).to_csv(c.OUT + 'odor_similarity_PN_APL_weighted_vs_once_summary.csv')
summ['weighted_vs_once'] = {'r_range': [round(float(agree.r.min()), 3), round(float(agree.r.max()), 3)],
                            'mean_change_range': [round(float(agree.mean_change.min()), 3), round(float(agree.mean_change.max()), 3)],
                            'max_abs_change': round(float(agree.max_abs_change.max()), 3)}

DCOL = {'PN_to_APL': '#3b3a37', 'APL_to_PN': '#a8a7a1'}
fig, axs = plt.subplots(2, 3, figsize=(10.4, 7.2))
for i, resp in enumerate(['ORN', 'GH146']):
    for j, (ds, side) in enumerate(c.MODEL_MBS):
        ax = axs[i, j]
        lo = SIM[SIM.responses == resp][['cosine_glomeruli_once', 'cosine_weighted_by_PN_APL_synapses']].min().min()
        lo = np.floor(lo * 10) / 10
        txt = []
        for d, dl in DIRS:
            g = SIM[(SIM.responses == resp) & (SIM.dataset == ds) & (SIM.apl_side == side) & (SIM.direction == d)]
            ax.scatter(g.cosine_glomeruli_once, g.cosine_weighted_by_PN_APL_synapses, s=13, lw=0.6,
                       facecolor=DCOL[d] if d == 'PN_to_APL' else 'none', edgecolor=DCOL[d], label=dl, zorder=3 if d == 'PN_to_APL' else 2)
            txt.append(f'{dl}: r = {agree.loc[(resp, ds, side, d), "r"]:.3f}')
        ax.plot([lo, 1], [lo, 1], color=c.MUTED, lw=0.8, ls='--', zorder=1)
        ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0); ax.set_aspect('equal')
        npair = len(g)
        ax.set_title(f'{resp}, {c.DSN[ds]} {"left" if side == "L" else "right"}\n' + '; '.join(txt), fontsize=8.3, loc='left')
        if i == 1: ax.set_xlabel('cosine similarity, glomeruli counted once')
        if j == 0: ax.set_ylabel(f'cosine weighted by PN ↔ APL\ncalyx synapses ({npair} pairs)')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        if i == 0 and j == 0: ax.legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(c.OUT + 'odor_similarity_PN_APL_weighted_vs_once.png', dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 2. distances between response-weighted maps
def gram(P, sigma):
    if sigma == 0: return None
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (4 * sigma ** 2))


def kmat(W, G):
    N = W.shape[0]
    return (W.T @ W) / N if G is None else (W.T @ G @ W) / N


def dist_cos(K):
    dg = np.diag(K)
    D = np.sqrt(np.clip(dg[:, None] + dg[None, :] - 2 * K, 0, None))
    C = K / np.sqrt(np.outer(dg, dg))
    return D, C


def upper(A):
    i, j = np.triu_indices(len(A), 1); return A[i, j]


long, mats = [], {}
for rs in ['ORN_megamat_diag', 'ORN_valid2_diag']:
    M = RS[rs][0].dropna()                                    # validation2 table: VM4 lacks one odor -> left out
    odors = list(M.columns)
    Dref, Cref = dist_cos(kmat(M.values, None) * len(M))      # responses only: glomeruli once, no positions (unnormalized)
    mats[(rs, 'responses only', '', 'euclid', 0.0)] = Dref; mats[(rs, 'responses only', '', 'cosine', 0.0)] = Cref
    for ds, side in c.CALYCES:
        for d, _ in DIRS:
            t = S[(S.dataset == ds) & (S.apl_side == side) & (S.direction == d) & S.glomerulus.isin(M.index)]
            P = t[['right_um', 'anterior_um', 'dorsal_um']].values
            W = M.loc[t.glomerulus].values
            for sg in SIGMAS:
                D, C = dist_cos(kmat(W, gram(P, sg)))
                mats[(rs, f'{ds} {side}', d, 'euclid', sg)] = D; mats[(rs, f'{ds} {side}', d, 'cosine', sg)] = C
            summ.setdefault('synapses_used', {})[f'{rs} {ds} {side} {d}'] = [len(t), round(len(t) / ((S.dataset == ds) & (S.apl_side == side) & (S.direction == d)).sum(), 3)]
    ia, ib = np.triu_indices(len(odors), 1)
    for k, A in mats.items():
        if k[0] != rs: continue
        for a, b, v in zip(ia, ib, A[ia, ib]):
            long.append(list(k) + [odors[a], odors[b], v])
    # agreement of each calyx's matrix with the responses-only one and between calyces
    for meas in ['euclid', 'cosine']:
        ref = upper(mats[(rs, 'responses only', '', meas, 0.0)])
        for sg in SIGMAS:
            for ds, side in c.CALYCES:
                for d, _ in DIRS:
                    v = upper(mats[(rs, f'{ds} {side}', d, meas, sg)])
                    summ.setdefault('vs_responses_only_r', {})[f'{rs} {meas} sigma{sg:g} {ds} {side} {d}'] = round(float(np.corrcoef(v, ref)[0, 1]), 4)
        cal = [upper(mats[(rs, f'{ds} {side}', d, meas, MAIN_SIGMA)]) for ds, side in c.CALYCES for d, _ in DIRS]
        R_ = np.corrcoef(cal); summ.setdefault('between_calyces_r_min', {})[f'{rs} {meas}'] = round(float(R_[np.triu_indices(len(cal), 1)].min()), 4)
L_ = pd.DataFrame(long, columns=['odor_set', 'calyx', 'direction', 'measure', 'sigma_um', 'odor_a', 'odor_b', 'value'])
L_.round(6).to_csv(c.OUT + 'odor_distance_PN_APL_maps.csv', index=False)

# heat maps: one PNG per odor set; rows = responses only + 5 calyces, columns = (PN -> APL, APL -> PN) x (distance, cosine)
for rs in ['ORN_megamat_diag', 'ORN_valid2_diag']:
    M, panel, title = RS[rs]; M = M.dropna(); odors = list(M.columns); no = len(odors)
    nmeg = sum(p != 'diagnostics' for p in panel)
    rowsk = [('responses only', '')] + [(f'{ds} {side}', d) for ds, side in c.CALYCES for d, _ in DIRS]
    fig, axs = plt.subplots(len(c.CALYCES) + 1, 4, figsize=(17, 4.3 * (len(c.CALYCES) + 1)))
    for r in range(len(c.CALYCES) + 1):
        for k in range(4):
            ax = axs[r, k]; meas = 'euclid' if k % 2 == 0 else 'cosine'
            if r == 0:
                if k >= 2: ax.axis('off'); continue
                A = mats[(rs, 'responses only', '', meas, 0.0)]; lab = 'responses only (glomeruli once, no positions)'
            else:
                ds, side = c.CALYCES[r - 1]; d = DIRS[k // 2][0]
                A = mats[(rs, f'{ds} {side}', d, meas, MAIN_SIGMA)]
                lab = f'{c.DSN[ds]} {"left" if side == "L" else "right"} · {DIRS[k // 2][1]}'
            cm = 'Purples' if meas == 'euclid' else 'Blues'
            vmax = np.nanmax(A); vmin = 0 if meas == 'euclid' else min(0, np.nanmin(A))
            im = ax.imshow(A, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.axhline(nmeg - 0.5, color=c.INK, lw=0.6); ax.axvline(nmeg - 0.5, color=c.INK, lw=0.6)
            ax.set_xticks(range(no), [o.replace(' -', ' −') for o in odors], rotation=90, fontsize=4.6)
            ax.set_yticks(range(no), [o.replace(' -', ' −') for o in odors], fontsize=4.6)
            ax.set_title(f'{lab}\n{"Euclidean distance" if meas == "euclid" else "cosine similarity"}', fontsize=9, loc='left')
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    fig.suptitle(f'{title}: odor-by-odor distance (purple) and cosine similarity (blue) between response-weighted maps of PN ↔ APL calyx '
                 f'synapses\nEach synapse carries its glomerulus\'s response to the odor; maps smoothed with a 3D Gaussian of σ = {MAIN_SIGMA:g} µm '
                 f'(corrected calyx). Lines separate the {RS[rs][1][0]} odors (top left) from the diagnostics. Each panel has its own color scale.',
                 fontsize=11, x=0.01, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(c.OUT + f'{rs}_odor_distance_heatmaps_PN_APL.png', dpi=110); plt.close(fig)
    print(rs, 'heat maps', flush=True)

json.dump(summ, open(c.OUT + 'odor_similarity_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != 'vs_responses_only_r' and k != 'synapses_used'}, indent=1))
v = summ['vs_responses_only_r']
for rs in ['ORN_megamat_diag', 'ORN_valid2_diag']:
    for meas in ['euclid', 'cosine']:
        for sg in SIGMAS:
            vals = [x for k, x in v.items() if k.startswith(f'{rs} {meas} sigma{sg:g} ')]
            print(rs, meas, sg, 'r vs responses only', min(vals), max(vals))
