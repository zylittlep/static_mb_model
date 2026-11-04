"""Local (non-spiking) APL with its PN-bouton synapses too (Yang, 2026-10-23; extends apl_local_model.py). Hemibrain R,
male CNS L and R. Official ROIs by default; APL_VARIANT=outside_roi uses with_outside_ROI_synapses/.

apl_local_model.py lets APL inhibit KCs locally (exp(-d / lambda) along the skeleton). Here APL also inhibits PN boutons
locally: each claw's input is divided by (1 + beta Q_bo), Q_bo = the APL potential at the APL->PN synapses on the claw's main
bouton b, per PN->KC synapse of that bouton (the per-bouton, local form of version P of apl_block_model.py):
  x_ko = sum_b B_kb g_o r_g(b),o / (1 + beta Q_bo),   B_kb = PN synapses on KC k's claws whose main bouton is b / all of k's
         claw PN synapses (claws of uniglomerular PNs; the same input as W when beta = 0)
  Q_bo = sum_j Qm_bj y_jo,   Qm_bj = sum over (s in APL->PN synapses on b, t in KC->APL synapses from j) exp(-d(s, t) / lambda)
         / (PN synapses on the claws of b)
  y_ko = [x_ko - 1 - a sum_j M_kj y_jo]+          (M as in apl_local_model.py)
Versions, each pinned like tnt_compare.py (TNT blocked fractions per odor; one gain sets TNT3c's mean intact fraction):
  boutons lambda      all inhibition on boutons, locally (a = 0): upper bound for the bouton route.
  KC + boutons lambda   both routes at the boutons' real share f of APL's calyx output synapses (hemibrain 0.097, male CNS
                        0.112): KC gain (1 - f) x a_local, bouton gain f x beta_boutons (each from its own calibration at
                        the same lambda), then scaled together.
  lambda = inf, 50, 25, 10, 5 µm (inf = per-bouton counts, APL as one unit).
Outputs: odor_weighted/APL_local_boutons_per_MB_<input>.csv, APL_local_boutons_summary.json. The combined figure of both
scripts, APL_local_model.png, is drawn by apl_local_model_figure.py."""
import os, sys, json, time
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
import apl_local_model as lm
tc, ag, abm = lm.tc, lm.ag, lm.tc.abm

OUT = abm.B + 'odor_weighted/'
LAMS = [np.inf, 50, 25, 10, 5]
LNAME = lambda l: 'inf' if np.isinf(l) else f'{l:g} µm'


def geometry_b(k, mb):
    """Compressed APL tree holding KC->APL, APL->KC and APL->PN (bouton) synapse nodes; input matrices."""
    ds, side = mb['ds'], k.split()[1]
    T = ag.load_tree(ag.SK + ag.APLS[lm.APL_OF[k]][2])
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
    nb = cu.groupby('main_bouton_id').n_PN_synapses.sum().reindex(bouts).values      # PN synapses on the claws of each bouton
    kc = t[t.partner_id.astype(str).isin(kix)]; kout = kc.direction.str.startswith('APL_output').values
    pn = t[t.bouton_id.isin(bix) & t.direction.str.startswith('APL_output')]           # APL -> PN on these boutons
    P_kc = np.where(kout[:, None], kc[['pre_x_nm', 'pre_y_nm', 'pre_z_nm']].values, kc[['post_x_nm', 'post_y_nm', 'post_z_nm']].values)
    P_pn = pn[['pre_x_nm', 'pre_y_nm', 'pre_z_nm']].values
    tr = cKDTree(T['xyz']); n_kc = tr.query(P_kc)[1]; n_pn = tr.query(P_pn)[1]
    cpar, clen, idx = lm.compress(T, np.unique(np.r_[n_kc, n_pn]))
    n = len(cpar); kk = kc.partner_id.astype(str).map(kix).values
    Win = csr_matrix((np.ones((~kout).sum()), (idx[n_kc[~kout]], kk[~kout])), shape=(n, len(ids)))
    Wout = csr_matrix((np.ones(kout.sum()), (idx[n_kc[kout]], kk[kout])), shape=(n, len(ids)))
    Bout = csr_matrix((1.0 / nb[pn.bouton_id.map(bix).values], (idx[n_pn], pn.bouton_id.map(bix).values)), shape=(n, len(bouts)))
    return dict(cpar=cpar, clen=clen, Win=Win, Wout=Wout, Bout=Bout, B=B, bglom=bglom, n_APL_to_PN=len(pn), boutons=len(bouts))


def phi(cpar, clen, Win, lam_um):
    k = np.exp(-clen / (lam_um * 1e3)).astype(np.float32) if np.isfinite(lam_um) else np.ones(len(clen), np.float32)
    up = Win.toarray().astype(np.float32)
    for v in range(len(cpar) - 1, 0, -1):
        p = cpar[v]
        if p >= 0: up[p] += up[v] * k[v]
    for v in range(1, len(cpar)):
        p = cpar[v]
        if p >= 0: up[v] = up[v] * (1 - k[v] * k[v]) + up[p] * k[v]
    return up


def solve(B, Rb, th, a, M, beta, Q, Y0=None, eta=0.3, tol=1e-6, maxit=6000):
    Y = np.maximum(B @ Rb - th, 0) if Y0 is None else Y0.copy()
    best, stall = np.inf, 0
    for it in range(maxit):
        Xn = B @ (Rb / (1 + beta * (Q @ Y))) if beta else B @ Rb
        Yn = np.maximum(Xn - th - (a * (M @ Y) if a else 0), 0)
        res = float(np.abs(Yn - Y).max())
        if res < tol: return Yn, it
        if res > 0.999 * best:
            stall += 1
            if stall > 20: eta *= 0.5; stall = 0
        else: best, stall = res, 0
        Y += eta * (Yn - Y)
    return Y, maxit


def calib(f, target, x0=1.0):
    lo, hi = 0.0, x0; Y = None
    while True:
        Y = f(hi, Y)
        if (Y > 0).mean(0).mean() <= target: break
        lo, hi = hi, hi * 2
    for _ in range(32):
        m = (lo + hi) / 2; Y = f(m, Y)
        if (Y > 0).mean(0).mean() > target: lo = m
        else: hi = m
    x = (lo + hi) / 2
    return x, f(x, Y)


if __name__ == '__main__':
    t0 = time.time()
    G, MQ = {}, {}
    for k, mb in tc.MB.items():
        G[k] = g = geometry_b(k, mb)
        N = len(mb['W']); u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean(); Mc = np.outer(u, v) / N
        MQ[k] = {}
        for l in LAMS:
            F = phi(g['cpar'], g['clen'], g['Win'], 1e12 if np.isinf(l) else l)
            M = np.asarray(g['Wout'].T @ F); Q = np.asarray(g['Bout'].T @ F)
            MQ[k][l] = (M * (Mc.mean() / M.mean()), Q / Q.mean() / N)
        print(k, {x: g[x] for x in ('n_APL_to_PN', 'boutons')}, f'{time.time() - t0:.0f} s', flush=True)
    rows, SI, SB = [], {}, {}
    summary = {'mushroom bodies': {k: {x: G[k][x] for x in ('n_APL_to_PN', 'boutons')} for k in G}}
    for rn, R in tc.RESP.items():
        od = list(R.columns); iu = np.triu_indices(len(od), 1); pi = tc.pair_index(od)
        cl_b = tc.CLD[tc.TH_MAIN]['TNT'].reindex(od).values; target = float(tc.CLD[tc.TH_MAIN]['TNT3c'].reindex(od).mean())
        y = tc.data_dS({c: np.arange(len(tc.SREC[c])) for c in tc.COND}, pi)
        for k, mb in tc.MB.items():
            g = G[k]; f = abm.F_SHARE[mb['ds']]
            W = mb['W']; gl = [x for x in W.columns if x in R.index]
            Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None); H = Wm @ Rp
            gain = np.array([1.0 / np.quantile(H[:, j], 1 - cl_b[j]) for j in range(H.shape[1])])
            Rpg = pd.DataFrame(Rp * gain[None], index=gl, columns=od)
            ok = np.array([x in Rpg.index for x in g['bglom']])
            Rb = np.zeros((len(g['bglom']), len(od))); Rb[ok] = Rpg.reindex(g['bglom'][ok]).values
            X = Wm @ Rpg.values
            assert np.allclose(g['B'] @ Rb, X, atol=1e-9), 'bouton input differs from W'
            Sb = lm.cos(np.maximum(X - 1, 0))[iu]; SB[(rn, k)] = Sb
            for l in LAMS:
                M, Q = MQ[k][l]
                a0, Ya = calib(lambda a, Y0: solve(g['B'], Rb, 1.0, a, M, 0, Q, Y0)[0], target)
                b0, Yb = calib(lambda b, Y0: solve(g['B'], Rb, 1.0, 0, M, b, Q, Y0)[0], target)
                s, Ym = calib(lambda s, Y0: solve(g['B'], Rb, 1.0, s * (1 - f) * a0, M, s * f * b0, Q, Y0)[0], target)
                for nm, Y, gn in [(f'boutons {LNAME(l)}', Yb, b0), (f'KC + boutons {LNAME(l)}', Ym, s)]:
                    Si = lm.cos(Y)[iu]; SI.setdefault((rn, nm), {})[k] = Si
                    rows.append(dict(responses=rn, mushroom_body=k, version=nm, gain=gn, spearman_blocked_vs_intact=spearmanr(Sb, Si, nan_policy='omit')[0],
                                     r_dS_with_data=np.corrcoef(y, Sb - Si)[0, 1], mean_dS=float(np.nanmean(Sb - Si)),
                                     mean_frac_intact=float((Y > 0).mean())))
                    print(rn, k, nm, f"rank {rows[-1]['spearman_blocked_vs_intact']:.3f} r {rows[-1]['r_dS_with_data']:.3f}", f'{time.time() - t0:.0f} s', flush=True)
                if np.isinf(l):   # KC-only local at lambda = inf must equal C
                    rows.append(dict(responses=rn, mushroom_body=k, version='check: KC only, lambda inf (= C)', gain=a0,
                                     spearman_blocked_vs_intact=spearmanr(Sb, lm.cos(Ya)[iu], nan_policy='omit')[0],
                                     r_dS_with_data=np.corrcoef(y, Sb - lm.cos(Ya)[iu])[0, 1], mean_dS=float(np.nanmean(Sb - lm.cos(Ya)[iu])),
                                     mean_frac_intact=float((Ya > 0).mean())))
    D = pd.DataFrame(rows)
    for rn in tc.RESP:
        D[D.responses == rn].round(4).to_csv(OUT + f'APL_local_boutons_per_MB_{rn}.csv', index=False)
    for rn, R in tc.RESP.items():
        x = D[D.responses == rn]; od = list(R.columns)
        y = tc.data_dS({c: np.arange(len(tc.SREC[c])) for c in tc.COND}, tc.pair_index(od))
        Sb = np.nanmean([SB[(rn, k)] for k in tc.MB], 0)
        summary.setdefault(rn, {})['r with the data change, model change averaged over MBs (as tnt_compare.py)'] = {
            nm: round(float(np.corrcoef(y, Sb - np.nanmean([d[k] for k in tc.MB], 0))[0, 1]), 3) for (r_, nm), d in SI.items() if r_ == rn}
        summary[rn] |= {'Spearman blocked vs intact, mean over MBs': x.groupby('version', sort=False).spearman_blocked_vs_intact.mean().round(3).to_dict(),
                       'r with the data change, per MB then mean': x.groupby('version', sort=False).r_dS_with_data.mean().round(3).to_dict()}
    json.dump(summary, open(OUT + 'APL_local_boutons_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1)); print(f'done {time.time() - t0:.0f} s')
