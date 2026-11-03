"""Can APL's wiring make the effect of blocking APL (TNT) nonuniform across odors? Official ROIs only.

1. APL wiring per glomerulus (APL_wiring_by_glomerulus.csv). Claws (2 µm clusters of a KC's PN synapses; glomerulus = the
   claw's main PN's): on-claw KC->APL and APL->KC synapses per claw and per PN->KC synapse on those claws. PN boutons in the
   calyx: APL->PN and PN->APL synapses per PN->KC synapse of the glomerulus's PNs. Uniglomerular PN types only.
2. Model, one per mushroom body (hemibrain R, male CNS L/R; KC side from its claw positions). FlyWire and BANC are
   left out since 2026-10-21 (Yang).
   KC input = sum over its claws of (PN synapses on the claw x response of the claw's glomerulus) / all its claw PN synapses.
   Responses: ORN table (16 odors) or GH146 means (13 odors), negative values set to 0; claws of glomeruli without a response
   give 0. KC response y = [input - threshold - APL inhibition]+; APL activity A = mean over KCs of (drive weight x y),
   solved self-consistently per odor.
   Blocked (TNT) = no APL; the threshold makes 20% of KCs respond on average over odors. Intact: APL gain set so that 10%
   respond on average. APL versions, all calibrated the same way:
     U    uniform: inhibition A on every KC, every KC drives APL alike
     C    per KC from the connectome: inhibition ~ the KC's APL->KC synapses in the calyx, drive ~ its KC->APL synapses in the
          calyx (each / its mean; calyx only since 2026-10-21, Yang; the whole-MB version is in odor_weighted/archive_2026-10-21_KC_to_APL_whole_MB/)
     Cs   C with the KCs' (inhibition, drive) pairs shuffled among KCs of the same class;  Cc  class means of C
     P0   on PN boutons, divisive, all glomeruli alike: glomerulus g's input / (1 + b A)
     P    on PN boutons, per glomerulus: / (1 + b p_g A), p_g = APL->PN per PN->KC synapse of g (claw-input-weighted mean 1)
     L    on claws, per glomerulus: / (1 + b q_g A), q_g = on-claw APL->KC per PN->KC synapse on g's claws (mean 1)
     M0/M U plus P0 / P, with the bouton part at its synapse share (APL->PN / (APL->PN + APL->KC) in the calyx, ~11%):
          KC gain (1 - f) x U's, bouton gain f x P0's, then scaled together to 10%
   Odor similarity S = cosine similarity between two odors' KC population responses, as in Yang's RDMs and the TNT comparison
   (Pearson r until 2026-10-21; still in the pair table as the r_* columns); dS = S(blocked) - S(intact).
   Odors with < 0.5% of KCs responding (intact, any MB) are left out of the pair results.
   Settings: 20% -> 10% (figure, main numbers), 10% -> 5% and 30% -> 10% (robustness, in the tables and summary)."""
import os, re, json
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.stats import spearmanr

# APL_VARIANT=outside_roi reads the tables with synapses outside the official ROIs added (with_outside_ROI_synapses/);
# the default (official ROIs only) is unchanged.
B = '/mnt/project-files/apl_synapses/' + ('with_outside_ROI_synapses/' if os.environ.get('APL_VARIANT', '') == 'outside_roi' else '')
OUT = os.environ.get('APL_BLOCK_OUT', B + 'odor_weighted/')
ORN_TABLE = '/mnt/project-files/uploads/hearth/5d477ed3-345f-4332-8b48-1ad3e2e0a0c1'   # ORN_ovlp16_resp_flyave_merge_dropna.csv
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'
# Yang, 2026-10-21: hemibrain and male CNS only; FlyWire and BANC are discarded. The four-connectome outputs are in
# odor_weighted/archive_2026-10-21_four_connectomes/ (DS = ['flywire', 'hemibrain', 'malecns', 'banc'] reproduces them).
DS = ['hemibrain', 'malecns']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
DS_COLOR = {'flywire': '#2a78d6', 'hemibrain': '#eb6834', 'malecns': '#1f9e6e', 'banc': '#8a5cc2'}
INK, MUTED, GRID, NAN_C = '#0b0b0b', '#52514e', '#e1e0d9', '#d6d6d4'
SETTINGS = {'20% -> 10%': (0.2, 0.1), '10% -> 5%': (0.1, 0.05), '30% -> 10%': (0.3, 0.1)}
MAIN = '20% -> 10%'
VARS = ['U', 'C', 'Cs', 'Cc', 'P0', 'P', 'L', 'M0', 'M']
MIN_CL = 0.005
SIM = 'cos'          # similarity used for the summary and figure: 'cos' (cosine) or 'r' (Pearson)
rng = np.random.default_rng(0)


def gloms_of(t, ds):
    t = str(t)
    if t.startswith('M_') or '_' not in t: return []
    names = [n for n in t.split('_')[0].split('+') if re.fullmatch(GPAT, n)]
    return [HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in names]


def uni(t, ds):
    g = gloms_of(t, ds); return g[0] if len(g) == 1 else ''


# ---------------- 1. APL wiring per glomerulus ----------------
kw = pd.read_csv(B + 'KC_weights_APL_PN_per_KC.csv', dtype={'kc_id': str})
PNW = pd.read_csv(B + 'PN_calyx_weights_APL_vs_KC.csv')
K = pd.read_csv(B + 'odor_weighted/KC_APL_synapses_on_claws_by_glomerulus.csv',
                usecols=['dataset', 'apl_side', 'direction', 'claw_id', 'kc_id'], dtype={'kc_id': str})
CLAWS = {ds: pd.read_csv(B + f'{ds}_KC_claws_derived.csv.gz', dtype={'kc_id': str}) for ds in DS}
on = K.groupby(['claw_id', 'direction']).size().unstack(fill_value=0)
rows = []
for ds in DS:
    c = CLAWS[ds].join(on, on='claw_id').fillna({'KC_to_APL': 0, 'APL_to_KC': 0})
    c['glomerulus'] = [uni(t, ds) for t in c.main_PN_type]
    g = c[c.glomerulus != ''].groupby('glomerulus').agg(claws=('claw_id', 'size'), PN_synapses_on_claws=('n_PN_synapses', 'sum'),
                                                         KC_to_APL_on_claws=('KC_to_APL', 'sum'), APL_to_KC_on_claws=('APL_to_KC', 'sum'))
    p = PNW[PNW.dataset == ds].assign(glomerulus=lambda d: [uni(t, ds) for t in d.pn_type])
    p = p[p.glomerulus != ''].groupby('glomerulus').agg(n_PNs=('pn_id', 'size'), PN_to_KC=('PN_to_KC', 'sum'),
                                                         PN_to_APL=('PN_to_APL', 'sum'), APL_to_PN=('APL_to_PN', 'sum'))
    rows.append(g.join(p, how='outer').fillna(0).reset_index().assign(dataset=ds))
GW = pd.concat(rows, ignore_index=True)
with np.errstate(invalid='ignore', divide='ignore'):
    GW['KC_to_APL_per_claw'] = GW.KC_to_APL_on_claws / GW.claws
    GW['APL_to_KC_per_claw'] = GW.APL_to_KC_on_claws / GW.claws
    GW['KC_to_APL_per_claw_PN_synapse'] = GW.KC_to_APL_on_claws / GW.PN_synapses_on_claws
    GW['APL_to_KC_per_claw_PN_synapse'] = GW.APL_to_KC_on_claws / GW.PN_synapses_on_claws
    GW['APL_to_PN_per_PN_to_KC'] = GW.APL_to_PN / GW.PN_to_KC
    GW['PN_to_APL_per_PN_to_KC'] = GW.PN_to_APL / GW.PN_to_KC
GW = GW[['dataset', 'glomerulus'] + [c for c in GW.columns if c not in ('dataset', 'glomerulus')]]
GWi = GW.set_index(['dataset', 'glomerulus'])
ok = (GWi.claws.unstack('dataset').reindex(columns=DS).fillna(0) >= 40).all(1) & \
     (GWi.PN_to_KC.unstack('dataset').reindex(columns=DS).fillna(0) >= 500).all(1)
GL_OK = ok[ok].index.tolist()          # glomeruli with >= 40 claws and >= 500 PN->KC synapses in every dataset
wiring = {}
for col in ['APL_to_PN_per_PN_to_KC', 'PN_to_APL_per_PN_to_KC', 'APL_to_KC_per_claw_PN_synapse', 'KC_to_APL_per_claw_PN_synapse',
            'APL_to_KC_per_claw', 'KC_to_APL_per_claw']:
    X = GWi[col].unstack('dataset').reindex(index=GL_OK, columns=DS)
    rel = X / X.median()
    S = X.corr(method='spearman').values[np.triu_indices(len(DS), 1)]
    wiring[col] = {'glomeruli': len(GL_OK), '10th-90th percentile, x median': {ds: [round(float(rel[ds].quantile(.1)), 2), round(float(rel[ds].quantile(.9)), 2)] for ds in DS},
                   'spearman between datasets: min-max': [round(float(S.min()), 2), round(float(S.max()), 2)],
                   'lowest (mean x median)': rel.mean(1).sort_values().head(5).round(2).to_dict(),
                   'highest (mean x median)': rel.mean(1).sort_values().tail(5).round(2).to_dict()}
F_SHARE = (PNW.groupby('dataset').APL_to_PN.sum() / (PNW.groupby('dataset').APL_to_PN.sum() + kw.groupby('dataset').APL_to_KC_calyx.sum())).to_dict()


# ---------------- 2. model ----------------
def responses():
    orn = pd.read_csv(ORN_TABLE, index_col=0)
    gh = pd.read_csv(B + 'odor_weighted/GH146_mean_by_glomerulus.csv', index_col=0).drop(columns='n_flies')
    col = {o: f'megamat: {o} @ -3' for o in orn.columns} | {'paa': 'glomeruli_diagnostics: paa @ -5'}
    gh13 = pd.DataFrame({o: gh[c] for o, c in col.items() if c in gh.columns})[[o for o in orn.columns if col[o] in gh.columns]]
    return {'ORN': orn, 'GH146': gh13}


def mushroom_bodies():
    out, sides = {}, {}
    for ds in DS:
        cl = CLAWS[ds].copy()
        cl['glom'] = [uni(t, ds) for t in cl.main_PN_type]
        xyz = cl.groupby('kc_id')[['centroid_x_nm', 'centroid_y_nm', 'centroid_z_nm']].mean()
        if ds == 'hemibrain':
            side = pd.Series('R', index=xyz.index)
        else:   # two calyces: 2-means on the KCs' claw positions, named by the side of the APL their on-claw synapses go to
            _, lab = kmeans2(xyz.values / 1e3, 2, seed=0, minit='++')
            vote = K[K.dataset == ds].groupby('kc_id').apl_side.agg(lambda s: s.mode().iloc[0])
            name = {k: vote.reindex(xyz.index[lab == k]).mode().iloc[0] for k in (0, 1)}
            assert name[0] != name[1], name
            side = pd.Series([name[k] for k in lab], index=xyz.index)
        S = cl.groupby('kc_id').n_PN_synapses.sum()
        w = kw[kw.dataset == ds].set_index('kc_id')
        gw = GWi.loc[ds]
        for sd in sorted(side.unique()):
            ids = side.index[side == sd]
            c = cl[cl.kc_id.isin(ids) & (cl.glom != '')]
            W = c.pivot_table(index='kc_id', columns='glom', values='n_PN_synapses', aggfunc='sum', fill_value=0)
            W = W.reindex(ids, fill_value=0).div(S[ids], axis=0)
            out[f'{ds} {sd}'] = dict(ds=ds, W=W, u=w.APL_to_KC_calyx.reindex(ids).fillna(0).values.astype(float),
                                     v=w.KC_to_APL_calyx.reindex(ids).fillna(0).values.astype(float),
                                     cls=w.kc_class.reindex(ids).fillna('?').values,
                                     p=gw.APL_to_PN_per_PN_to_KC, q=gw.APL_to_KC_per_claw_PN_synapse)
            sides[f'{ds} {sd}'] = int(len(ids))
    return out, sides


def cl_of(Y): return (Y > 0).mean(0)


def fixed_point(Yfun, n_odor):
    """A = mean_i v_i y_i(A) per odor; the right side falls as A grows, so bisect."""
    F = lambda A: Yfun(A)[1]
    lo, hi = np.zeros(n_odor), F(np.zeros(n_odor)) + 1e-12
    for _ in range(60):
        m = (lo + hi) / 2; g = m - F(m); hi = np.where(g > 0, m, hi); lo = np.where(g > 0, lo, m)
    return Yfun((lo + hi) / 2)[0]


def model_y(Wm, Rp, th, a, b, u, v, p):
    """KC responses with subtractive inhibition a u_i A on the KC and divisive inhibition 1/(1 + b p_g A) on glomerulus g's input."""
    def Yfun(A):
        X = Wm @ (Rp / (1 + b * p[:, None] * A[None])) if b else Wm @ Rp
        Y = np.maximum(X - th - a * u[:, None] * A[None], 0)
        return Y, (v[:, None] * Y).mean(0)
    return fixed_point(Yfun, Rp.shape[1])


def calib(f, target, hi=1.0):
    while f(hi) > target: hi *= 2
    lo = 0.0
    for _ in range(50):
        m = (lo + hi) / 2
        if f(m) > target: lo = m
        else: hi = m
    return (lo + hi) / 2


def corr(Y):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.corrcoef(Y.T)


def cosm(Y):
    """cosine similarity between the columns (odors) of Y (KCs x odors)."""
    n = np.linalg.norm(Y, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (Y.T @ Y) / np.outer(n, n)


def run_mb(mb, R, cl_blocked, cl_intact):
    W = mb['W']; gl = [g for g in W.columns if g in R.index]
    Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None)
    H = Wm @ Rp
    lo, hi = 0.0, H.max()
    for _ in range(60):
        m = (lo + hi) / 2
        if (H > m).mean(0).mean() > cl_blocked: lo = m
        else: hi = m
    th = (lo + hi) / 2
    n, one, oneg = H.shape[0], np.ones(H.shape[0]), np.ones(len(gl))
    u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean()
    us, vs, uc, vc = u.copy(), v.copy(), u.copy(), v.copy()
    for c in np.unique(mb['cls']):
        i = np.where(mb['cls'] == c)[0]; j = rng.permutation(i)
        us[i], vs[i] = u[j], v[j]; uc[i], vc[i] = u[i].mean(), v[i].mean()

    def norm(x):
        x = x.reindex(gl)
        return (x / (np.nansum(Wm.sum(0) * x.values) / Wm[:, ~x.isna().values].sum())).fillna(1.0).values
    p, q = norm(mb['p']), norm(mb['q'])
    Y = {'blocked': np.maximum(H - th, 0)}
    gain = {}
    for name, (uu, vv) in {'U': (one, one), 'C': (u, v), 'Cs': (us, vs), 'Cc': (uc, vc)}.items():
        gain[name] = calib(lambda a: cl_of(model_y(Wm, Rp, th, a, 0, uu, vv, oneg)).mean(), cl_intact)
        Y[name] = model_y(Wm, Rp, th, gain[name], 0, uu, vv, oneg)
    for name, pp in {'P0': oneg, 'P': p, 'L': q}.items():
        gain[name] = calib(lambda b: cl_of(model_y(Wm, Rp, th, 0, b, one, one, pp)).mean(), cl_intact)
        Y[name] = model_y(Wm, Rp, th, 0, gain[name], one, one, pp)
    f = F_SHARE[mb['ds']]
    for name, pp in {'M0': oneg, 'M': p}.items():
        s = calib(lambda s: cl_of(model_y(Wm, Rp, th, s * (1 - f) * gain['U'], s * f * gain['P0'], one, one, pp)).mean(), cl_intact)
        Y[name] = model_y(Wm, Rp, th, s * (1 - f) * gain['U'], s * f * gain['P0'], one, one, pp)
    return {'H': H, 'Y': Y, 'n_kc': n}


if __name__ == '__main__':
    GW.round(5).to_csv(OUT + 'APL_wiring_by_glomerulus.csv', index=False)
    RESP = responses()
    MB, SIDES = mushroom_bodies()
    MBS = list(MB)
    if os.environ.get('APL_BLOCK_REUSE') and os.path.exists(OUT + 'APL_block_model_odor_pairs.csv'):   # redraw from the saved tables
        OD = pd.read_csv(OUT + 'APL_block_model_per_odor.csv'); PR = pd.read_csv(OUT + 'APL_block_model_odor_pairs.csv')
    else:
        od_rows, pr_rows = [], []
        for st, (cb, ci) in SETTINGS.items():
            for rn, R in RESP.items():
                for k, mb in MB.items():
                    r = run_mb(mb, R, cb, ci)
                    O = list(R.columns); iu = np.triu_indices(len(O), 1)
                    rc = {c: corr(y) for c, y in r['Y'].items()}
                    cc = {c: cosm(y) for c, y in r['Y'].items()}
                    od_rows.append(pd.DataFrame({'responses': rn, 'mushroom_body': k, 'coding_levels': st, 'odor': O,
                                                 'frac_KCs_responding_blocked': cl_of(r['Y']['blocked']),
                                                 **{f'frac_KCs_responding_intact_{c}': cl_of(r['Y'][c]) for c in VARS}}))
                    pr_rows.append(pd.DataFrame({'responses': rn, 'mushroom_body': k, 'coding_levels': st, 'odor_a': [O[i] for i in iu[0]],
                                                 'odor_b': [O[j] for j in iu[1]], 'r_KC_input': corr(r['H'])[iu], 'r_blocked': rc['blocked'][iu],
                                                 **{f'r_intact_{c}': rc[c][iu] for c in VARS},
                                                 'cos_KC_input': cosm(r['H'])[iu], 'cos_blocked': cc['blocked'][iu],
                                                 **{f'cos_intact_{c}': cc[c][iu] for c in VARS}}))
                    print(st, rn, k, flush=True)
        OD = pd.concat(od_rows, ignore_index=True); PR = pd.concat(pr_rows, ignore_index=True)
    low = OD[[f'frac_KCs_responding_intact_{c}' for c in VARS]].min(1) < MIN_CL
    bad = set(map(tuple, OD.loc[low, ['responses', 'mushroom_body', 'coding_levels', 'odor']].values))
    DROP = {(rn, st): sorted({o for (rn_, k, st_, o) in bad if rn_ == rn and st_ == st}) for rn in RESP for st in SETTINGS}
    PR['odor_dropped'] = [(o1 in DROP[(rn, st)]) or (o2 in DROP[(rn, st)]) for rn, st, o1, o2 in zip(PR.responses, PR.coding_levels, PR.odor_a, PR.odor_b)]
    OD.round(5).to_csv(OUT + 'APL_block_model_per_odor.csv', index=False)
    PR.round(5).to_csv(OUT + 'APL_block_model_odor_pairs.csv', index=False)


    # ---------------- summary numbers ----------------
    def dR(st, rn, c, keep=True):
        P = PR[(PR.coding_levels == st) & (PR.responses == rn)]
        if keep: P = P[~P.odor_dropped]
        M = P.pivot_table(index=['odor_a', 'odor_b'], columns='mushroom_body', values=[f'{SIM}_blocked', f'{SIM}_intact_{c}'], sort=False)
        return (M[f'{SIM}_blocked'] - M[f'{SIM}_intact_{c}'])[MBS]


    tri = np.triu_indices(len(MBS), 1)
    summary = {'mushroom_bodies_KCs': SIDES, 'bouton_share_of_APL_calyx_output_synapses': {k: round(v, 3) for k, v in F_SHARE.items() if k in DS},
               'wiring_by_glomerulus': wiring, 'odors_left_out_of_pair_results': {f'{rn} {st}': v for (rn, st), v in DROP.items()}, 'model': {}}
    for st in SETTINGS:
        for rn in RESP:
            o = OD[(OD.coding_levels == st) & (OD.responses == rn)]
            m = o.groupby('odor', sort=False).mean(numeric_only=True)
            gain = {c: m.frac_KCs_responding_blocked - m[f'frac_KCs_responding_intact_{c}'] for c in VARS}
            gU = (o.frac_KCs_responding_blocked - o.frac_KCs_responding_intact_U).groupby(o.mushroom_body).apply(lambda s: s.values)
            G = np.array([gU[k] for k in MBS])
            d = {c: dR(st, rn, c) for c in VARS}
            s = {'mean frac of KCs responding, blocked / intact U': {od: [round(float(m.loc[od, 'frac_KCs_responding_blocked']), 3), round(float(m.loc[od, 'frac_KCs_responding_intact_U']), 3)] for od in m.index},
                 'per-odor gain (blocked - intact): r between versions (mean over MBs)': {c: round(float(np.corrcoef(gain['U'], gain[c])[0, 1]), 3) for c in VARS if c != 'U'},
                 'per-odor gain U vs blocked fraction: r': round(float(np.corrcoef(gain['U'], m.frac_KCs_responding_blocked)[0, 1]), 3),
                 'per-odor gain U, spearman between MBs: min': round(float(pd.DataFrame(G.T).corr('spearman').values[tri].min()), 2),
                 'pairs kept': int(len(d['U'])),
                 'dS U: mean, sd, min, max over pairs (mean over MBs)': [round(float(x), 3) for x in (d['U'].mean(1).mean(), d['U'].mean(1).std(), d['U'].mean(1).min(), d['U'].mean(1).max())],
                 'dS U: r between MBs (median)': round(float(np.median(d['U'].corr().values[tri])), 2)}
            for a, b in [('C', 'U'), ('Cs', 'U'), ('C', 'Cs'), ('P', 'P0'), ('L', 'P0'), ('M', 'M0')]:
                X = d[a] - d[b]
                s[f'dS {a} - dS {b}'] = {'mean |diff| over pairs and MBs': round(float(X.abs().values.mean()), 3),
                                        'mean |dS {}|'.format(b): round(float(d[b].abs().values.mean()), 3),
                                        'r between MBs (median)': round(float(np.median(X.corr().values[tri])), 2),
                                        'largest (mean over MBs)': {f'{p1}-{p2}': round(float(x), 3) for (p1, p2), x in
                                                                    X.mean(1).reindex(X.mean(1).abs().sort_values(ascending=False).index[:6]).items()}}
            s['dS C-U vs Cs-U maps: r (mean over MBs)'] = round(float(np.corrcoef((d['C'] - d['U']).mean(1), (d['Cs'] - d['U']).mean(1))[0, 1]), 2)
            summary['model'][f'{rn} {st}'] = s
    json.dump(summary, open(OUT + 'APL_block_model_summary.json', 'w'), indent=1, default=str)


    # ---------------- figure ----------------
    ODORS = list(RESP['ORN'].columns)
    fig = plt.figure(figsize=(21, 17))
    gs = fig.add_gridspec(4, 5, height_ratios=[0.5, 0.5, 1, 1], width_ratios=[1.3, 1, 1, 1, 1], hspace=0.62, wspace=0.34,
                          top=0.925, bottom=0.04, left=0.045, right=0.95)
    FMT = matplotlib.ticker.FuncFormatter(lambda y, _: f'{y:g}')


    def spearman_range(col, a, b):
        X = GWi[col].unstack('dataset').reindex(index=GL_OK, columns=DS)
        v = [X[i].corr(X[j], method='spearman') for i in a for j in b if i != j and (i < j or j not in a)]
        return min(v), max(v)


    def strip(ax, col, title):
        X = GWi[col].unstack('dataset').reindex(index=GL_OK, columns=DS)
        rel = X / X.median()
        order = rel.mean(1).sort_values().index
        for i, ds in enumerate(DS):
            y = rel.loc[order, ds].values; xx = np.arange(len(order)) + (i - (len(DS) - 1) / 2) * 0.18; z = y <= 0
            ax.scatter(xx[~z], y[~z], s=16, color=DS_COLOR[ds], label=LABEL[ds], zorder=3, lw=0)
            ax.scatter(xx[z], np.full(z.sum(), 0.03), s=16, facecolor='none', edgecolor=DS_COLOR[ds], lw=0.9, zorder=3)
        ax.set_yscale('log'); ax.axhline(1, color=GRID, lw=1, zorder=1)
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.03, 0.1, 0.3, 1, 3]))
        ax.yaxis.set_major_formatter(FMT); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_ylim(0.025, 4)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7.5, color=INK)
        ax.set_xlim(-0.8, len(order) - 0.2)
        ax.set_ylabel('× dataset median', fontsize=8.5, color=MUTED); ax.tick_params(axis='y', labelsize=8, colors=MUTED)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=10, color=INK, loc='left')


    a4 = spearman_range('APL_to_PN_per_PN_to_KC', DS, DS)
    ax = fig.add_subplot(gs[0, :])
    strip(ax, 'APL_to_PN_per_PN_to_KC', f'a  APL→PN synapses on each glomerulus\'s PN boutons per PN→KC synapse of those PNs (calyx); {len(GL_OK)} glomeruli with '
          f'≥ 40 claws and ≥ 500 PN→KC synapses in both datasets; open circle = none. Same order in hemibrain and male CNS (Spearman {a4[0]:.2f})')
    ax.legend(fontsize=8, frameon=False, ncol=4, loc='upper left')
    b3 = spearman_range('APL_to_KC_per_claw_PN_synapse', DS, DS)
    ax = fig.add_subplot(gs[1, :])
    strip(ax, 'APL_to_KC_per_claw_PN_synapse', f'b  APL→KC synapses on the claws of each glomerulus per PN→KC synapse on those claws. Spearman {b3[0]:.2f} '
          f'between hemibrain and male CNS')

    CMAP = matplotlib.colormaps['RdBu_r'].copy(); CMAP.set_bad(NAN_C)
    VM = 0.2
    MK = {'U': ('o', '#2a78d6', 'intact, uniform APL'), 'C': ('s', '#eb6834', 'intact, APL per KC (connectome)'),
          'P': ('^', '#1f9e6e', 'intact, APL on PN boutons (connectome)'), 'M': ('D', '#8a5cc2', 'intact, KC + boutons at synapse share')}


    def mat(rn, a, b=None):
        """mean over MBs of dR_a (or dR_a - dR_b) as a 16 x 16 frame in ODORS order."""
        X = dR(MAIN, rn, a) if b is None else dR(MAIN, rn, a) - dR(MAIN, rn, b)
        M = pd.DataFrame(np.nan, index=ODORS, columns=ODORS)
        for (o1, o2), x in X.mean(1).items(): M.loc[o1, o2] = M.loc[o2, o1] = x
        return M


    def heat(ax, M, title):
        im = ax.imshow(M.values, cmap=CMAP, vmin=-VM, vmax=VM, interpolation='nearest')
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(ODORS, rotation=90, fontsize=7, color=INK); ax.set_yticklabels(ODORS, fontsize=7, color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(title, fontsize=8.6, color=INK, loc='left')
        return im


    last = []
    for row, rn in [(2, 'ORN'), (3, 'GH146')]:
        s = summary['model'][f'{rn} {MAIN}']
        L0 = 'cdefg' if row == 2 else 'hijkl'
        o = OD[(OD.coding_levels == MAIN) & (OD.responses == rn)].groupby('odor', sort=False).mean(numeric_only=True)
        o = o.sort_values('frac_KCs_responding_blocked')
        ax = fig.add_subplot(gs[row, 0])
        x = np.arange(len(o))
        ax.bar(x, o.frac_KCs_responding_blocked * 100, color='#d3d2cc', width=0.75, label='blocked (TNT)', zorder=2)
        for i, (c, (mk, col, lab)) in enumerate(MK.items()):
            ax.scatter(x + (i - 1.5) * 0.13, o[f'frac_KCs_responding_intact_{c}'] * 100, marker=mk, s=18, color=col, label=lab, zorder=3, lw=0)
        ax.set_xticks(x); ax.set_xticklabels(o.index, rotation=90, fontsize=7.5, color=INK)
        ax.set_ylabel(f'% of KCs responding (mean of {len(MBS)} MBs)', fontsize=8.5, color=MUTED); ax.tick_params(axis='y', labelsize=8, colors=MUTED)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        rmin = np.floor(min(s['per-odor gain (blocked - intact): r between versions (mean over MBs)'].values()) * 100) / 100
        ax.set_title(f'{L0[0]}  {rn} responses, per odor ({len(o)} odors)\nKCs gained when APL is blocked track odor strength\n'
                     f'(r {s["per-odor gain U vs blocked fraction: r"]:.2f}), the same for every APL version\n(r ≥ {rmin:.2f} between versions)',
                     fontsize=8.6, color=INK, loc='left')
        if row == 2: ax.legend(fontsize=7, frameon=False, loc='upper left')
        d = s['dS U: mean, sd, min, max over pairs (mean over MBs)']
        heat(fig.add_subplot(gs[row, 1]), mat(rn, 'U'), f'{L0[1]}  Δcos = cos(blocked) − cos(intact)\nuniform APL: mean {d[0]:+.3f}, {d[2]:+.2f} to {d[3]:+.2f}\n'
                                                           f'same pattern in every MB (median r {s["dS U: r between MBs (median)"]:.2f})')
        heat(fig.add_subplot(gs[row, 2]), mat(rn, 'C', 'U'), f'{L0[2]}  APL synapses per KC (connectome)\nminus uniform APL; the same counts shuffled\n'
                                                                f'within KC class give a similar map (r {s["dS C-U vs Cs-U maps: r (mean over MBs)"]:.2f})')
        heat(fig.add_subplot(gs[row, 3]), mat(rn, 'P', 'P0'), f'{L0[3]}  APL on PN boutons, per glomerulus,\nminus all glomeruli alike\n'
                                                                 f'all inhibition on boutons (upper bound)\nsame in every MB (median r {s["dS P - dS P0"]["r between MBs (median)"]:.2f})')
        axl = fig.add_subplot(gs[row, 4])
        im = heat(axl, mat(rn, 'M', 'M0'), f'{L0[4]}  as {L0[3]}, boutons at their synapse share\n(11% of APL\'s calyx output) plus\nuniform inhibition on the KCs\n'
                                             f'mean |change| {s["dS M - dS M0"]["mean |diff| over pairs and MBs"]:.3f} ({s["dS P - dS P0"]["mean |diff| over pairs and MBs"]:.3f} in {L0[3]})')
        last.append(axl)
    drop = DROP[('ORN', MAIN)]
    fig.suptitle('Can APL\'s wiring make the effect of blocking APL nonuniform across odors?  Official ROIs · KC model on each connectome\'s PN→KC claws '
                 f'({len(MBS)} mushroom bodies: hemibrain R, male CNS L/R) · 20% of KCs respond with APL blocked, 10% with it intact\n'
                 f'cos = cosine similarity of two odors\' KC population responses; Δcos = cos(blocked) − cos(intact), mean of the {len(MBS)} MBs. Blocked responses are '
                 'the same for every APL version, so maps e–g and j–l show only how the intact responses differ.\n'
                 'Grey: odor not in the response table' + (f', or < 0.5% of KCs respond ({", ".join(drop)} in ORN)' if drop else '') + '.',
                 fontsize=10.5, color=INK, x=0.01, ha='left', y=0.99)
    fig.canvas.draw()
    for axl in last:
        p = axl.get_position()
        cax = fig.add_axes([p.x1 + 0.008, p.y0, 0.007, p.height])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Δcos', fontsize=9, color=MUTED); cb.ax.tick_params(labelsize=8, colors=MUTED); cb.outline.set_visible(False)
    fig.savefig(OUT + 'APL_block_model.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'model'}, indent=1, default=str)[:1500])
