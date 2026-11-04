"""APL with its signals confined to calyx sub-arbors (compartments) in the KC model (Yang, 2026-10-24). Extends
apl_local_model.py and apl_local_boutons.py. Hemibrain R, male CNS L and R. Official ROIs by default; APL_VARIANT=outside_roi
uses with_outside_ROI_synapses/.

Compartments = APL's calyx sub-arbors of Report 10 (apl_geodesic.subarbors): skeleton nodes within 5 µm of a calyx APL synapse,
split into the connected pieces of the skeleton restricted to them. Pieces connect to each other only through the peduncle or
lobes; here no signal crosses between pieces. Every piece is its own compartment, the small ones too.
KC model and pinning as in apl_local_model.py: y_ko = [x_ko - 1 - a sum_j M_kj y_jo]+, input gains pinned to TNT's blocked
fractions per odor, one APL gain per version pinned to TNT3c's mean intact fraction (6.83%).
Versions of M (KC route):
  compartments inf             isopotential compartments: KC j's KC->APL synapses in compartment c raise c's potential, which
                               inhibits every APL->KC synapse in c:  M_kj = sum_c O_ck I_cj   (O, I: per-compartment counts)
  compartments inf, per length the same, potential divided by the compartment's cable length L_c (input resistance ~ 1 / membrane):
                               M_kj = sum_c O_ck I_cj / L_c
  compartments lambda          passive decay exp(-d / lambda) inside a compartment, nothing across (lambda = 200, 100, 50, 25, 10 µm):
                               the local model of Report 11 with every skeleton edge between compartments cut
  U, C                         reference, as in apl_local_model.py (C = one compartment, no decay)
Bouton route (as apl_local_boutons.py, potential from the same cut kernel): 'boutons, compartments lambda' (all inhibition on
PN boutons) and 'KC + boutons, compartments lambda' (bouton share f of APL's calyx output: hemibrain 0.097, male CNS 0.112),
lambda = inf and 50 µm.
Each M is scaled to the mean of C (the gain calibration removes the scale anyway).
Null for the isopotential compartment version (inf or inf, per length) with the better pattern r: KC sites swapped within class
(M -> M[pi][:, pi]), 100x: does it matter which KC sits in which compartment?
Checks: the cut kernel at lambda = inf equals sum_c O_c^T I_c; compartments with one piece would equal C.
Outputs (odor_weighted/): APL_compartment_model_per_MB_<input>.csv, _fits.csv, _odor_pairs.csv, _nulls.csv, _summary.json,
APL_compartment_model_compartments.csv (per compartment: synapses, KCs, cable length). Figure: apl_compartment_model_figure.py."""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
import apl_local_model as lm
import apl_local_boutons as lb
tc, ag, abm = lm.tc, lm.ag, lm.tc.abm

OUT = abm.B + 'odor_weighted/'
KLAMS = [np.inf, 200, 100, 50, 25, 10]
BLAMS = [np.inf, 50]
NNULL = int(os.environ.get('COMP_NULLS', 100))
LN = lambda l: 'inf' if np.isinf(l) else f'{l:g} µm'
rs = np.random.default_rng(24)


def geometry_c(k, mb):
    """Compressed APL tree with KC<->APL and APL->PN (bouton) synapse nodes, compartment label of every compressed node."""
    ds, side = mb['ds'], k.split()[1]
    T = ag.load_tree(ag.SK + ag.APLS[lm.APL_OF[k]][2])
    lab = ag.subarbors(T, ds, f'APL_{side}')                 # node -> connected calyx piece, -1 outside the calyx
    t = pd.read_csv(abm.B + f'{ds}_APL_MB_synapses.csv.gz', low_memory=False,
                    usecols=['direction', 'apl_instance', 'mb_region', 'partner_id', 'bouton_id', 'pre_x_nm', 'pre_y_nm', 'pre_z_nm',
                             'post_x_nm', 'post_y_nm', 'post_z_nm'])
    t = t[(t.apl_instance == f'APL_{side}') & (t.mb_region == 'CA')]
    ids = list(mb['W'].index); kix = {s: i for i, s in enumerate(ids)}
    cl = abm.CLAWS[ds]; cl = cl[cl.kc_id.isin(kix)].copy()
    cl['glom'] = [abm.uni(x, ds) for x in cl.main_PN_type]
    S = cl.groupby('kc_id').n_PN_synapses.sum()
    cu = cl[(cl.glom != '') & cl.main_bouton_id.notna()]       # new claws (claws_v2m2): 8 of 33,584 claws have no bouton; left out here
    bouts = sorted(cu.main_bouton_id.unique()); bix = {b: i for i, b in enumerate(bouts)}
    B = csr_matrix((cu.n_PN_synapses.values / S.reindex(cu.kc_id).values, (cu.kc_id.map(kix).values, cu.main_bouton_id.map(bix).values)),
                   shape=(len(ids), len(bouts))).toarray()
    bglom = cu.groupby('main_bouton_id').glom.first().reindex(bouts).values
    nb = cu.groupby('main_bouton_id').n_PN_synapses.sum().reindex(bouts).values
    kc = t[t.partner_id.astype(str).isin(kix)]; kout = kc.direction.str.startswith('APL_output').values
    pn = t[t.bouton_id.isin(bix) & t.direction.str.startswith('APL_output')]
    P_kc = np.where(kout[:, None], kc[['pre_x_nm', 'pre_y_nm', 'pre_z_nm']].values, kc[['post_x_nm', 'post_y_nm', 'post_z_nm']].values)
    P_pn = pn[['pre_x_nm', 'pre_y_nm', 'pre_z_nm']].values
    tr = cKDTree(T['xyz']); s_kc, n_kc = tr.query(P_kc); n_pn = tr.query(P_pn)[1]
    cpar, clen, idx = lm.compress(T, np.unique(np.r_[n_kc, n_pn]))
    n = len(cpar); kk = kc.partner_id.astype(str).map(kix).values
    nodes = np.empty(n, int); nodes[idx[idx >= 0]] = np.where(idx >= 0)[0]
    clab = lab[nodes]
    par_lab = np.where(cpar >= 0, clab[np.maximum(cpar, 0)], -2)
    same = (clab >= 0) & (clab == par_lab)                     # edge to the parent stays inside one compartment
    Win = csr_matrix((np.ones((~kout).sum()), (idx[n_kc[~kout]], kk[~kout])), shape=(n, len(ids)))
    Wout = csr_matrix((np.ones(kout.sum()), (idx[n_kc[kout]], kk[kout])), shape=(n, len(ids)))
    Bout = csr_matrix((1.0 / nb[pn.bouton_id.map(bix).values], (idx[n_pn], pn.bouton_id.map(bix).values)), shape=(n, len(bouts)))
    # compartments: per-compartment synapse counts (I, O) and cable length L_c of each piece (edges with both ends in it)
    comp = clab.copy(); nolab = comp < 0
    comp[nolab] = lab.max() + 1 + np.arange(nolab.sum())       # a synapse node outside every piece is its own compartment
    cu_, ci = np.unique(comp, return_inverse=True)
    agg = csr_matrix((np.ones(n), (ci, np.arange(n))), shape=(len(cu_), n))        # node -> its compartment
    Ic, Oc, Bc = agg @ Win, agg @ Wout, agg @ Bout
    par = T['parent']; ok = par >= 0
    e_in = ok & (lab >= 0) & (lab == np.where(ok, lab[np.maximum(par, 0)], -3))
    Lpiece = np.bincount(lab[e_in], weights=T['L'][e_in], minlength=lab.max() + 2)
    Lc = np.array([Lpiece[c] if c <= lab.max() else 0.0 for c in cu_]) / 1e3          # µm
    Lc = np.maximum(Lc, 1.0)                                                           # a piece without edges counts as 1 µm
    return dict(cpar=cpar, clen=clen, same=same, Win=Win, Wout=Wout, Bout=Bout, B=B, bglom=bglom, Ic=Ic, Oc=Oc, Bc=Bc, Lc=Lc,
                comp_labels=cu_, comp=comp, node_xyz=T['xyz'][nodes], n_nodes=n, snap_um=float(np.median(s_kc) / 1e3),
                frac_kc_syn_unlabeled=float(nolab[idx[n_kc]].mean()))


def phi_cut(cpar, clen, same, Win, lam_um):
    """Phi[v, j] = sum over KC->APL sites t of KC j in v's compartment of exp(-d(v, t) / lambda); two passes on the tree with every
    edge between compartments cut (kappa = 0)."""
    k = (np.exp(-clen / (lam_um * 1e3)) if np.isfinite(lam_um) else np.ones(len(clen))).astype(np.float64) * same
    up = Win.toarray().astype(np.float64)
    for v in range(len(cpar) - 1, 0, -1):
        p = cpar[v]
        if p >= 0 and k[v] > 0: up[p] += up[v] * k[v]
    for v in range(1, len(cpar)):
        p = cpar[v]
        if p >= 0 and k[v] > 0: up[v] = up[v] * (1 - k[v] * k[v]) + up[p] * k[v]
    return up


if __name__ == '__main__':
    t0 = time.time()
    G, MS, MQ, CROWS, checks = {}, {}, {}, [], {}
    KV = ['U', 'C'] + [f'compartments {LN(l)}' for l in KLAMS[:1]] + ['compartments inf, per length'] + [f'compartments {LN(l)}' for l in KLAMS[1:]]
    for k, mb in tc.MB.items():
        G[k] = g = geometry_c(k, mb)
        N = len(mb['W']); u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean(); Mc = np.outer(u, v) / N
        MS[k] = {'U': np.full((N, N), 1.0 / N), 'C': Mc}; MQ[k] = {}
        sc = lambda M: M * (Mc.mean() / M.mean())
        Fs = {}
        for l in sorted(set(KLAMS) | set(BLAMS), key=lambda x: -x):
            Fs[l] = phi_cut(g['cpar'], g['clen'], g['same'], g['Win'], l)
        for l in KLAMS:
            MS[k][f'compartments {LN(l)}'] = sc(np.asarray(g['Wout'].T @ Fs[l]))
        Oc, Ic = g['Oc'].toarray(), g['Ic'].toarray()
        Mper = sc((Oc / g['Lc'][:, None]).T @ Ic)
        MS[k]['compartments inf, per length'] = Mper
        Msum = Oc.T @ Ic
        checks[k] = {'cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference':
                     float(f"{np.abs(np.asarray(g['Wout'].T @ Fs[np.inf]) - Msum).max() / Msum.max():.2e}"),
                     'r of compartments inf with C': round(float(np.corrcoef(MS[k]['compartments inf'].ravel()[::53], Mc.ravel()[::53])[0, 1]), 4),
                     'share of M (compartments inf) within KC class': round(float((MS[k]['compartments inf'] * (mb['cls'][:, None] == mb['cls'][None])).sum() / MS[k]['compartments inf'].sum()), 4),
                     'compressed nodes': g['n_nodes'], 'compartments holding KC<->APL synapses': int(((Oc.sum(1) + Ic.sum(1)) > 0).sum()),
                     'KC<->APL synapse nodes outside every piece (share)': round(g['frac_kc_syn_unlabeled'], 5)}
        for l in BLAMS:
            Q = np.asarray(g['Bout'].T @ Fs[l]); M = MS[k][f'compartments {LN(l)}']
            MQ[k][l] = (M, Q / Q.mean() / N)
        # per-compartment table
        tot_in, tot_out = Ic.sum(), Oc.sum()
        for c, lab_ in enumerate(g['comp_labels']):
            if Ic[c].sum() + Oc[c].sum() == 0: continue
            CROWS.append(dict(mushroom_body=k, compartment=int(lab_), KC_to_APL=int(Ic[c].sum()), APL_to_KC=int(Oc[c].sum()),
                              share_KC_to_APL=Ic[c].sum() / tot_in, share_APL_to_KC=Oc[c].sum() / tot_out,
                              KCs_with_synapses=int(((Ic[c] + Oc[c]) > 0).sum()), cable_length_um=g['Lc'][c],
                              APL_to_PN_boutons=float(g['Bc'].toarray()[c].sum() > 0)))
        print(k, checks[k], f'{time.time() - t0:.0f} s', flush=True)
    CT = pd.DataFrame(CROWS)
    # effective number of compartments seen by a KC's inhibition (participation ratio over compartments, weighted by APL->KC)
    for k in tc.MB:
        x = CT[CT.mushroom_body == k].share_APL_to_KC.values
        checks[k]['effective number of compartments (1 / sum of squared APL->KC shares)'] = round(float(1 / np.sum(x ** 2)), 2)
    CT.round(5).to_csv(OUT + 'APL_compartment_model_compartments.csv', index=False)
    rows_per, rows_fit, rows_pair, rows_null, summary = [], [], [], [], {'checks': checks}
    for rn, R in tc.RESP.items():
        od = list(R.columns); iu = np.triu_indices(len(od), 1); pi = tc.pair_index(od)
        cl_b = tc.CLD[tc.TH_MAIN]['TNT'].reindex(od).values; target = float(tc.CLD[tc.TH_MAIN]['TNT3c'].reindex(od).mean())
        y = tc.data_dS({c: np.arange(len(tc.SREC[c])) for c in tc.COND}, pi)
        S3c, ST = tc.SREC['TNT3c'].mean(0)[pi], tc.SREC['TNT'].mean(0)[pi]
        SI, SB, GA, XX = {}, {}, {}, {}
        for k, mb in tc.MB.items():
            g = G[k]; f = abm.F_SHARE[mb['ds']]
            W = mb['W']; gl = [x for x in W.columns if x in R.index]
            Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None); H = Wm @ Rp
            gain = np.array([1.0 / np.quantile(H[:, j], 1 - cl_b[j]) for j in range(H.shape[1])])
            Rpg = pd.DataFrame(Rp * gain[None], index=gl, columns=od)
            X = Wm @ Rpg.values; XX[k] = X
            ok = np.array([x in Rpg.index for x in g['bglom']])
            Rb = np.zeros((len(g['bglom']), len(od))); Rb[ok] = Rpg.reindex(g['bglom'][ok]).values
            assert np.allclose(g['B'] @ Rb, X, atol=1e-9)
            Sb = lm.cos(np.maximum(X - 1, 0))[iu]; SB[k] = Sb
            out = {}
            for vv in KV:
                a, Y, it = lm.calibrate(X, 1.0, MS[k][vv], target); out[vv] = (Y, a)
            for l in BLAMS:
                M, Q = MQ[k][l]
                a0 = out[f'compartments {LN(l)}'][1]
                b0, Yb = lb.calib(lambda b, Y0: lb.solve(g['B'], Rb, 1.0, 0, M, b, Q, Y0)[0], target)
                s, Ym = lb.calib(lambda s, Y0: lb.solve(g['B'], Rb, 1.0, s * (1 - f) * a0, M, s * f * b0, Q, Y0)[0], target)
                out[f'boutons, compartments {LN(l)}'] = (Yb, b0); out[f'KC + boutons, compartments {LN(l)}'] = (Ym, s)
            for vv, (Y, a) in out.items():
                Si = lm.cos(Y)[iu]; SI.setdefault(vv, {})[k] = Si; GA.setdefault(vv, {})[k] = a
                rows_per.append(dict(responses=rn, mushroom_body=k, version=vv, gain=a, frac_intact=float((Y > 0).mean()),
                                     spearman_blocked_vs_intact=spearmanr(Sb, Si, nan_policy='omit')[0],
                                     r_dS_with_data=np.corrcoef(y, Sb - Si)[0, 1], mean_dS=float(np.nanmean(Sb - Si))))
                print(rn, k, vv, f"a {a:.3g} rank {rows_per[-1]['spearman_blocked_vs_intact']:.3f} r {rows_per[-1]['r_dS_with_data']:.3f}",
                      f'{time.time() - t0:.0f} s', flush=True)
        PER = pd.DataFrame([r for r in rows_per if r['responses'] == rn])
        VERS = list(SI)
        Sbm = np.nanmean([SB[k] for k in tc.MB], 0)
        PAIR = pd.DataFrame(dict(responses=rn, odor_a=np.array(od)[iu[0]], odor_b=np.array(od)[iu[1]], S_TNT3c=S3c, S_TNT=ST, dS_data=y,
                                 S_model_blocked=Sbm))
        fits = {}
        for vv in VERS:
            Si = np.nanmean([SI[vv][k] for k in tc.MB], 0); PAIR[f'S_model_intact_{vv}'] = Si
            fits[vv] = dict(responses=rn, version=vv,
                            rank_spearman_blocked_vs_intact_mean_of_MBs=float(PER[PER.version == vv].spearman_blocked_vs_intact.mean()),
                            r_dS_with_data=float(np.corrcoef(y, Sbm - Si)[0, 1]), mean_dS_model=float(np.nanmean(Sbm - Si)), mean_dS_data=float(y.mean()))
            rows_fit.append(fits[vv])
        rows_pair.append(PAIR)
        kcv = ['compartments inf', 'compartments inf, per length']       # the pure compartment model: no decay inside a compartment
        best = max(kcv, key=lambda vv: fits[vv]['r_dS_with_data'])
        nr, nrank = [], []
        for s_ in range(NNULL):
            Sis = []
            for k, mb in tc.MB.items():
                perm = np.arange(len(mb['W']))
                for c in np.unique(mb['cls']):
                    i = np.where(mb['cls'] == c)[0]; perm[i] = rs.permutation(i)
                a, Y, _ = lm.calibrate(XX[k], 1.0, MS[k][best][np.ix_(perm, perm)], target, a0=GA[best][k])
                Sis.append(lm.cos(Y)[iu])
            nr.append(np.corrcoef(y, Sbm - np.nanmean(Sis, 0))[0, 1])
            nrank.append(np.mean([spearmanr(SB[k], x, nan_policy='omit')[0] for k, x in zip(tc.MB, Sis)]))
            rows_null.append(dict(responses=rn, version=best, shuffle=s_ + 1, r_dS_with_data=nr[-1], rank_spearman_mean_of_MBs=nrank[-1]))
            if s_ % 10 == 9: print(rn, 'null', s_ + 1, f'{time.time() - t0:.0f} s', flush=True)
        fb = fits[best]
        summary[rn] = {
            'data: Spearman TNT vs TNT3c similarity over odor pairs': round(float(spearmanr(ST, S3c)[0]), 3),
            'model: Spearman blocked vs intact (mean over MBs) by version': {vv: round(fits[vv]['rank_spearman_blocked_vs_intact_mean_of_MBs'], 3) for vv in VERS},
            'model: Spearman blocked vs intact per MB': {vv: {k: round(float(x), 3) for k, x in PER[PER.version == vv].set_index('mushroom_body').spearman_blocked_vs_intact.items()} for vv in VERS},
            'r over odor pairs, model change (MB-averaged) vs data change': {vv: round(fits[vv]['r_dS_with_data'], 3) for vv in VERS},
            'r per MB': {vv: {k: round(float(x), 3) for k, x in PER[PER.version == vv].set_index('mushroom_body').r_dS_with_data.items()} for vv in VERS},
            'mean change in similarity: data, model by version': [round(float(y.mean()), 3), {vv: round(fits[vv]['mean_dS_model'], 3) for vv in VERS}],
            f'null ({NNULL} within-class swaps of KC sites on APL), version {best}': {
                'r real, null mean, null 5-95%, p': [round(fb['r_dS_with_data'], 3), round(float(np.mean(nr)), 3),
                                                     [round(float(np.quantile(nr, .05)), 3), round(float(np.quantile(nr, .95)), 3)],
                                                     round(float((1 + np.sum(np.array(nr) >= fb['r_dS_with_data'])) / (1 + NNULL)), 3)],
                'rank Spearman real, null mean, null 5-95%, p (lower = more reordering)': [
                    round(fb['rank_spearman_blocked_vs_intact_mean_of_MBs'], 3), round(float(np.mean(nrank)), 3),
                    [round(float(np.quantile(nrank, .05)), 3), round(float(np.quantile(nrank, .95)), 3)],
                    round(float((1 + np.sum(np.array(nrank) <= fb['rank_spearman_blocked_vs_intact_mean_of_MBs'])) / (1 + NNULL)), 3)]},
            'gains per MB': {vv: {k: float(f'{GA[vv][k]:.4g}') for k in tc.MB} for vv in VERS}}
        PER.round(4).to_csv(OUT + f'APL_compartment_model_per_MB_{rn}.csv', index=False)
    pd.DataFrame(rows_fit).round(4).to_csv(OUT + 'APL_compartment_model_fits.csv', index=False)
    pd.concat(rows_pair, ignore_index=True).round(4).to_csv(OUT + 'APL_compartment_model_odor_pairs.csv', index=False)
    pd.DataFrame(rows_null).round(4).to_csv(OUT + 'APL_compartment_model_nulls.csv', index=False)
    json.dump(summary, open(OUT + 'APL_compartment_model_summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'checks'}, indent=1)[:6000])
    print(f'done {time.time() - t0:.0f} s')
