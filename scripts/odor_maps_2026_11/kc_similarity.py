"""KC <-> APL versions of similarity.py (Report 03 revision, 2026-11-04), on the new claws (claw_v2m2), hemibrain R and
male CNS L/R. A KC <-> APL synapse on a claw carries the response of its claw's PN glomerulus; synapses off claws, or on a
claw whose PN type names two glomeruli or none, are left out.

1. Weighted vs unweighted cosine similarity, as Fig 3 of Report 03 (Yang: "for Figure 3, ... 6 panels: 3 ORN + 3 GH146"):
       x = sum_g r_ga r_gb / sqrt(sum_g r_ga^2 sum_g r_gb^2)                      (every glomerulus once)
       y = sum_g n_g r_ga r_gb / sqrt(sum_g n_g r_ga^2 sum_g n_g r_gb^2)          (n_g = KC <-> APL synapses on claws of g)
   ORN: the 16 TNT odors of the new table; GH146: the 13 of them that GH146 has. Directions KC -> APL and APL -> KC, and
   in the table also APL -> KC + PN (APL -> KC on claws plus APL -> PN on boutons).

2. Distances between response-weighted synapse maps (Yang: "correlation (euclidean) between odors by weighing the
   APL<->KC/PN synaptic coordinates by responses"), as in similarity.py:
       K_ab = (1/N) sum_e sum_e' w_a(e) G_ee' w_b(e'),  G_ee' = exp(-|x_e - x_e'|^2 / (4 sigma^2))
       d_ab = sqrt(K_aa + K_bb - 2 K_ab),  c_ab = K_ab / sqrt(K_aa K_bb)
   Kinds KC -> APL, APL -> KC, APL -> KC + PN; the two new ORN datasets (not merged). With up to ~21,000 synapses per
   calyx the Gaussian matrix is built in row blocks (float32)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import common as c

RS = c.response_sets()
K_ = c.kc_synapses()
P_ = c.pn_synapses()
P_ = P_[P_.dataset.isin(['hemibrain', 'malecns'])]
cols = ['dataset', 'apl_side', 'direction', 'glomerulus', 'right_um', 'anterior_um', 'dorsal_um']
S = pd.concat([K_[cols].assign(partner='KC'), P_[cols].assign(partner='PN')], ignore_index=True)
KINDS = [('KC_to_APL', 'KC → APL', lambda t: t[(t.partner == 'KC') & (t.direction == 'KC_to_APL')]),
         ('APL_to_KC', 'APL → KC', lambda t: t[(t.partner == 'KC') & (t.direction == 'APL_to_KC')]),
         ('APL_to_KC_PN', 'APL → KC + PN', lambda t: t[t.direction.isin(['APL_to_KC', 'APL_to_PN'])])]
SIGMAS = [0.0, 2.0, 5.0]
MAIN_SIGMA = 2.0
summ = {}


def wcos(R, n):
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
    R = R.dropna()
    once = wcos(R, np.ones(len(R)))
    for ds, side in c.MODEL_MBS:
        for kk, _, sel in KINDS:
            t = sel(S); t = t[(t.dataset == ds) & (t.apl_side == side)]
            n = t.glomerulus.value_counts().reindex(R.index).fillna(0).values
            w = wcos(R, n)
            for i, a in enumerate(R.columns):
                for b in R.columns[i + 1:]:
                    rows.append([resp, ds, side, kk, a, b, once.loc[a, b], w.loc[a, b], int(n.sum())])
SIM = pd.DataFrame(rows, columns=['responses', 'dataset', 'apl_side', 'kind', 'odor_a', 'odor_b', 'cosine_glomeruli_once',
                                  'cosine_weighted_by_synapses', 'synapses_used'])
SIM.round(5).to_csv(c.OUT + 'odor_similarity_KC_APL_weighted_vs_once.csv', index=False)
agree = SIM.groupby(['responses', 'dataset', 'apl_side', 'kind']).apply(
    lambda g: pd.Series({'r': np.corrcoef(g.cosine_glomeruli_once, g.cosine_weighted_by_synapses)[0, 1],
                         'mean_change': (g.cosine_weighted_by_synapses - g.cosine_glomeruli_once).mean(),
                         'max_abs_change': (g.cosine_weighted_by_synapses - g.cosine_glomeruli_once).abs().max(),
                         'synapses_used': g.synapses_used.iloc[0]}), include_groups=False)
agree.round(4).to_csv(c.OUT + 'odor_similarity_KC_APL_weighted_vs_once_summary.csv')
summ['weighted_vs_once'] = agree.round(4).reset_index().to_dict('records')

DCOL = {'KC_to_APL': '#3b3a37', 'APL_to_KC': '#a8a7a1'}
fig, axs = plt.subplots(2, 3, figsize=(10.4, 7.2))
for i, resp in enumerate(['ORN', 'GH146']):
    lo = np.floor(SIM[SIM.responses == resp][['cosine_glomeruli_once', 'cosine_weighted_by_synapses']].min().min() * 10) / 10
    for j, (ds, side) in enumerate(c.MODEL_MBS):
        ax = axs[i, j]; txt = []
        for kk, kl, _ in KINDS[:2]:
            g = SIM[(SIM.responses == resp) & (SIM.dataset == ds) & (SIM.apl_side == side) & (SIM.kind == kk)]
            ax.scatter(g.cosine_glomeruli_once, g.cosine_weighted_by_synapses, s=13, lw=0.6,
                       facecolor=DCOL[kk] if kk == 'KC_to_APL' else 'none', edgecolor=DCOL[kk], label=kl, zorder=3 if kk == 'KC_to_APL' else 2)
            txt.append(f'{kl}: r = {agree.loc[(resp, ds, side, kk), "r"]:.3f}')
        ax.plot([lo, 1], [lo, 1], color=c.MUTED, lw=0.8, ls='--', zorder=1)
        ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0); ax.set_aspect('equal')
        ax.set_title(f'{resp}, {c.DSN[ds]} {"left" if side == "L" else "right"}\n' + '; '.join(txt), fontsize=8.3, loc='left')
        if i == 1: ax.set_xlabel('cosine similarity, glomeruli counted once')
        if j == 0: ax.set_ylabel(f'cosine weighted by KC ↔ APL synapses\non claws ({len(g)} pairs)')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        if i == 0 and j == 0: ax.legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(c.OUT + 'odor_similarity_KC_APL_weighted_vs_once.png', dpi=150); plt.close(fig)
print('weighted vs once done', flush=True)


# ---------------------------------------------------------------- 2. distances between response-weighted maps
def kmat(P, W, sigma, block=1500):
    """(1/N) W' G W with G = exp(-d^2 / 4 sigma^2), built block by block"""
    N = len(P)
    if sigma == 0: return (W.T @ W) / N
    P = P.astype(np.float32); W32 = W.astype(np.float32); out = np.zeros((W.shape[1], W.shape[1]))
    sq = (P ** 2).sum(1)
    for s in range(0, N, block):
        Pb = P[s:s + block]
        d2 = np.maximum(sq[s:s + block, None] + sq[None, :] - 2 * Pb @ P.T, 0)
        G = np.exp(-d2 / np.float32(4 * sigma ** 2))
        out += W32[s:s + block].T.astype(np.float64) @ (G @ W32).astype(np.float64)
    return out / N


def dist_cos(K):
    dg = np.diag(K)
    return np.sqrt(np.clip(dg[:, None] + dg[None, :] - 2 * K, 0, None)), K / np.sqrt(np.outer(dg, dg))


def upper(A):
    i, j = np.triu_indices(len(A), 1); return A[i, j]


SETS = ['ORN_megamat_diag', 'ORN_valid2_diag']
long, mats = [], {}
Ms = {rs: RS[rs][0].dropna() for rs in SETS}
for rs, M in Ms.items():
    Dref, Cref = dist_cos(M.values.T @ M.values)
    mats[(rs, 'responses only', '', 'euclid', 0.0)] = Dref; mats[(rs, 'responses only', '', 'cosine', 0.0)] = Cref
for ds, side in c.MODEL_MBS:
    for kk, _, sel in KINDS:
        t0 = sel(S); t0 = t0[(t0.dataset == ds) & (t0.apl_side == side)]
        for rs, M in Ms.items():
            t = t0[t0.glomerulus.isin(M.index)]
            P = t[['right_um', 'anterior_um', 'dorsal_um']].values
            W = M.loc[t.glomerulus].values
            for sg in SIGMAS:
                D, C = dist_cos(kmat(P, W, sg))
                mats[(rs, f'{ds} {side}', kk, 'euclid', sg)] = D; mats[(rs, f'{ds} {side}', kk, 'cosine', sg)] = C
            summ.setdefault('synapses_used', {})[f'{rs} {ds} {side} {kk}'] = [len(t), round(len(t) / len(t0), 3)]
            print(ds, side, kk, rs, len(t), flush=True)
for rs, M in Ms.items():
    odors = list(M.columns); ia, ib = np.triu_indices(len(odors), 1)
    for k, A in mats.items():
        if k[0] != rs: continue
        for a, b, v in zip(ia, ib, A[ia, ib]):
            long.append(list(k) + [odors[a], odors[b], v])
    for meas in ['euclid', 'cosine']:
        ref = upper(mats[(rs, 'responses only', '', meas, 0.0)])
        for sg in SIGMAS:
            for ds, side in c.MODEL_MBS:
                for kk, _, _ in KINDS:
                    v = upper(mats[(rs, f'{ds} {side}', kk, meas, sg)])
                    summ.setdefault('vs_responses_only_r', {})[f'{rs} {meas} sigma{sg:g} {ds} {side} {kk}'] = round(float(np.corrcoef(v, ref)[0, 1]), 4)
        cal = [upper(mats[(rs, f'{ds} {side}', kk, meas, MAIN_SIGMA)]) for ds, side in c.MODEL_MBS for kk, _, _ in KINDS]
        R_ = np.corrcoef(cal); summ.setdefault('between_calyces_and_kinds_r_min', {})[f'{rs} {meas}'] = round(float(R_[np.triu_indices(len(cal), 1)].min()), 4)
pd.DataFrame(long, columns=['odor_set', 'calyx', 'kind', 'measure', 'sigma_um', 'odor_a', 'odor_b', 'value']).round(6).to_csv(
    c.OUT + 'odor_distance_KC_APL_maps.csv', index=False)

for rs in SETS:
    M, panel, title = RS[rs]; M = M.dropna(); odors = list(M.columns); no = len(odors)
    nmeg = sum(p != 'diagnostics' for p in panel)
    fig, axs = plt.subplots(len(c.MODEL_MBS) + 1, 6, figsize=(25, 4.3 * (len(c.MODEL_MBS) + 1)))
    for r in range(len(c.MODEL_MBS) + 1):
        for k in range(6):
            ax = axs[r, k]; meas = 'euclid' if k % 2 == 0 else 'cosine'
            if r == 0:
                if k >= 2: ax.axis('off'); continue
                A = mats[(rs, 'responses only', '', meas, 0.0)]; lab = 'responses only (glomeruli once, no positions)'
            else:
                ds, side = c.MODEL_MBS[r - 1]; kk, kl, _ = KINDS[k // 2]
                A = mats[(rs, f'{ds} {side}', kk, meas, MAIN_SIGMA)]
                lab = f'{c.DSN[ds]} {"left" if side == "L" else "right"} · {kl}'
            cm = 'Purples' if meas == 'euclid' else 'Blues'
            vmax = np.nanmax(A); vmin = 0 if meas == 'euclid' else min(0, np.nanmin(A))
            im = ax.imshow(A, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.axhline(nmeg - 0.5, color=c.INK, lw=0.6); ax.axvline(nmeg - 0.5, color=c.INK, lw=0.6)
            ax.set_xticks(range(no), [o.replace(' -', ' −') for o in odors], rotation=90, fontsize=4.6)
            ax.set_yticks(range(no), [o.replace(' -', ' −') for o in odors], fontsize=4.6)
            ax.set_title(f'{lab}\n{"Euclidean distance" if meas == "euclid" else "cosine similarity"}', fontsize=9, loc='left')
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    fig.suptitle(f'{title}: odor-by-odor distance (purple) and cosine similarity (blue) between response-weighted maps of KC ↔ APL '
                 f'synapses on claws, and of APL → KC + PN synapses\nEach synapse carries the response of its claw\'s (or bouton\'s) '
                 f'glomerulus; maps smoothed with a 3D Gaussian of σ = {MAIN_SIGMA:g} µm (new claws, corrected calyx). Lines separate the '
                 f'{panel[0]} odors (top left) from the diagnostics. Each panel has its own color scale.', fontsize=11, x=0.01, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(c.OUT + f'{rs}_odor_distance_heatmaps_KC_APL.png', dpi=100); plt.close(fig)
    print(rs, 'heat maps', flush=True)

json.dump(summ, open(c.OUT + 'odor_similarity_KC_summary.json', 'w'), indent=1)
v = summ['vs_responses_only_r']
for rs in SETS:
    for meas in ['euclid', 'cosine']:
        for sg in SIGMAS:
            vals = [x for k, x in v.items() if k.startswith(f'{rs} {meas} sigma{sg:g} ')]
            print(rs, meas, sg, 'r vs responses only', min(vals), max(vals))
print(summ['between_calyces_and_kinds_r_min'])
