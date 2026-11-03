"""A non-spiking, local APL in the KC model: does it reshape the KC representation like TNT vs TNT3c better than a uniform or
per-KC-count APL? (Yang, 2026-10-23.) Hemibrain R, male CNS L and R (Yang, 2026-10-21).
Official ROIs by default; APL_VARIANT=outside_roi uses with_outside_ROI_synapses/ (inputs and outputs).

KC model and pinning as in tnt_compare.py: x_ko = sum_g W_kg g_o r_go (ORN 16 odors or GH146 13 odors), threshold 1, each
odor's gain g_o set so that with APL blocked its fraction of KCs responding equals its TNT fraction > 1; one APL gain per
version sets the mean intact fraction over odors to TNT3c's (6.8%).
  y_ko = [x_ko - 1 - a I_ko]+,   I_ko = sum_j M_kj y_jo   (solved per odor; damped fixed-point iteration)
Versions (M):
  U        M_kj = 1 / N: one APL activity per odor, the same on every KC (= U of tnt_compare.py).
  C        M_kj = u_k v_j / N: u_k = KC k's APL->KC synapses in the calyx, v_j = KC j's KC->APL synapses in the calyx
           (each / its mean). APL is still one unit (= C of tnt_compare.py) = the local model with lambda = infinity.
  local    non-spiking APL: KC j's output enters APL at its KC->APL synapses and spreads passively along APL's skeleton,
           falling off as exp(-d / lambda) with geodesic distance d; KC k is inhibited at its APL->KC synapses by the APL
           potential there:  M_kj = c x sum over (s in APL->KC synapses on k, t in KC->APL synapses from j) exp(-d(s, t) / lambda),
           c making the mean of M equal to C's. lambda = 200, 100, 50, 25, 10, 5, 2 µm (env LOCAL_LAMS). Distances along the skeletons of
           apl_geodesic.py (calyx KC<->APL synapses snapped to the nearest node); computed exactly on the tree, two passes.
Null for the best local lambda: KCs' APL synapse sites swapped among KCs of the same class (KC k gets KC pi(k)'s input and
output sites, M -> M[pi][:, pi]), 100 shuffles per mushroom body: does the real placement of each KC on APL matter?
Readouts, as in tnt_compare.py and tnt_rank_order.py, model values averaged over the 3 mushroom bodies:
  rank order  Spearman over odor pairs between blocked and intact cosine similarity (data TNT vs TNT3c: 0.87);
  pattern     r over odor pairs between the model's change in similarity (blocked - intact) and the data's;
  suppression Spearman over odors between model and data blocked / intact fraction responding;
  size        mean change in similarity.
Outputs: odor_weighted/APL_local_model_fits.csv, APL_local_model_odor_pairs.csv, APL_local_model_nulls.csv,
APL_local_model_summary.json. The figure APL_local_model.png is drawn by apl_local_model_figure.py."""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
import tnt_compare as tc
import apl_geodesic as ag

OUT = tc.abm.B + 'odor_weighted/'
LAMS = [float(x) if '.' in x else int(x) for x in os.environ.get('LOCAL_LAMS', '200,100,50,25,10,5,2').split(',')]
NNULL = int(os.environ.get('LOCAL_NULLS', 100))
APL_OF = {'hemibrain R': 'hemibrain R', 'malecns L': 'male CNS L', 'malecns R': 'male CNS R'}
rs = np.random.default_rng(23)


# ---------------------------------------------------------------- APL geometry per mushroom body
def compress(T, marked):
    """Smallest subtree joining the marked nodes, with chains of unmarked, unbranched nodes contracted.
    Returns parent (-1 for the root), edge length (nm) and BFS order of the compressed tree, and the map node -> new index."""
    par, order, L = T['parent'], T['order'], T['L']
    cnt = np.zeros(len(par)); cnt[marked] = 1
    for v in order[::-1]:
        if par[v] >= 0: cnt[par[v]] += cnt[v]
    tot = cnt[T['root']]; keep = cnt > 0
    nkids = np.zeros(len(par), int)
    kv = np.where(keep & (par >= 0))[0]; np.add.at(nkids, par[kv], 1)
    # Steiner root: deepest node whose subtree holds every marked node
    r = T['root']
    is_m = np.zeros(len(par), bool); is_m[marked] = True
    kids = {}
    for v in kv: kids.setdefault(par[v], []).append(v)
    while not is_m[r] and nkids[r] == 1: r = kids[r][0]
    ess = keep & (is_m | (nkids >= 2)); ess[r] = True
    anc = np.full(len(par), -1); dist = np.zeros(len(par))
    inside = np.zeros(len(par), bool); inside[r] = True
    for v in order:
        p = par[v]
        if p < 0 or not keep[v] or v == r or not inside[p]: continue
        inside[v] = True
        if ess[p]: anc[v], dist[v] = p, L[v]
        else: anc[v], dist[v] = anc[p], dist[p] + L[v]
    nodes = np.array([v for v in order if inside[v] and ess[v]])
    idx = np.full(len(par), -1); idx[nodes] = np.arange(len(nodes))
    cpar = np.where(nodes == r, -1, idx[np.maximum(anc[nodes], 0)]); cpar[nodes == r] = -1
    clen = np.where(nodes == r, 0.0, dist[nodes])
    return cpar, clen, idx


def kernel_M(cpar, clen, Win, Wout, lam_um):
    """M = Wout^T K Win with K_st = exp(-d(s, t) / lambda): two passes over the compressed tree (BFS order = index order)."""
    k = np.exp(-clen / (lam_um * 1e3)).astype(np.float32)
    up = Win.toarray().astype(np.float32)
    n = len(cpar)
    for v in range(n - 1, 0, -1):
        p = cpar[v]
        if p >= 0: up[p] += up[v] * k[v]
    phi = up                                   # in place: phi[v] = up[v] + (phi[p] - up[v] k_v) k_v, parents first
    for v in range(1, n):
        p = cpar[v]
        if p >= 0: phi[v] = up[v] * (1 - k[v] * k[v]) + phi[p] * k[v]
    return np.asarray(Wout.T @ phi)


def geometry(k, mb):
    ds, side = mb['ds'], k.split()[1]
    T = ag.load_tree(ag.SK + ag.APLS[APL_OF[k]][2])
    t = pd.read_csv(tc.abm.B + f'{ds}_APL_MB_synapses.csv.gz', low_memory=False,
                    usecols=['direction', 'apl_instance', 'mb_region', 'partner_id', 'pre_x_nm', 'pre_y_nm', 'pre_z_nm', 'post_x_nm', 'post_y_nm', 'post_z_nm'])
    ids = list(mb['W'].index); kix = {s: i for i, s in enumerate(ids)}
    t = t[(t.apl_instance == f'APL_{side}') & (t.mb_region == 'CA') & t.partner_id.astype(str).isin(kix)]
    out = t.direction.str.startswith('APL_output').values
    P = np.where(out[:, None], t[['pre_x_nm', 'pre_y_nm', 'pre_z_nm']].values, t[['post_x_nm', 'post_y_nm', 'post_z_nm']].values)
    snap, node = cKDTree(T['xyz']).query(P)
    cpar, clen, idx = compress(T, np.unique(node))
    kc = t.partner_id.astype(str).map(kix).values; cn = idx[node]; n = len(cpar)
    Win = csr_matrix((np.ones((~out).sum()), (cn[~out], kc[~out])), shape=(n, len(ids)))
    Wout = csr_matrix((np.ones(out.sum()), (cn[out], kc[out])), shape=(n, len(ids)))
    return dict(cpar=cpar, clen=clen, Win=Win, Wout=Wout, n_in=int((~out).sum()), n_out=int(out.sum()),
                snap_um=float(np.median(snap) / 1e3), nodes=n)


# ---------------------------------------------------------------- KC model with a coupling matrix
def solve(X, th, a, M, Y0=None, tol=1e-6):
    """y = [X - th - a M y]+ per odor (columns). Active-set method: guess which KCs respond (from Y0, else from X > th),
    solve the linear system for them, update the set, repeat until it stops changing. Falls back to the damped iteration
    (solve_damped) if the set cycles or the result fails the residual check. Returns (Y, iterations of the last odor)."""
    Xt = X - th; Y = np.zeros_like(X, dtype=float); S0 = (Y0 > 0) if Y0 is not None else (Xt > 0); its = 0
    for o in range(X.shape[1]):
        S = S0[:, o].copy(); seen = set()
        for its in range(100):
            idx = np.flatnonzero(S); y = np.zeros(len(X))
            if len(idx):
                A = a * M[np.ix_(idx, idx)]; A[np.diag_indices(len(idx))] += 1.0
                y[idx] = np.linalg.solve(A, Xt[idx, o])
            r = Xt[:, o] - a * (M @ y)
            Sn = np.where(S, y > 0, r > 0)
            if (Sn == S).all(): break
            key = Sn.tobytes()
            if key in seen: return solve_damped(X, th, a, M, Y0)
            seen.add(key); S = Sn
        else:
            return solve_damped(X, th, a, M, Y0)
        Y[:, o] = np.maximum(y, 0)
    if np.abs(Y - np.maximum(Xt - a * (M @ Y), 0)).max() > tol * max(1.0, float(np.abs(Xt).max())):
        return solve_damped(X, th, a, M, Y0)
    return Y, its


def solve_damped(X, th, a, M, Y0=None, eta=0.3, tol=1e-6, maxit=4000):
    """y = [X - th - a M y]+ per odor (columns), damped iteration; eta halves if the residual stops falling."""
    Y = np.maximum(X - th, 0) if Y0 is None else Y0.copy()
    best = np.inf; stall = 0
    for it in range(maxit):
        Yn = np.maximum(X - th - a * (M @ Y), 0)
        res = float(np.abs(Yn - Y).max())
        if res < tol: return Yn, it
        if res > 0.999 * best:
            stall += 1
            if stall > 20: eta *= 0.5; stall = 0
        else: best = res; stall = 0
        Y += eta * (Yn - Y)
    return Y, maxit


def calibrate(X, th, M, target, a0=1.0):
    cl = lambda a, Y0=None: solve(X, th, a, M, Y0)[0]
    lo, hi = 0.0, a0
    while (cl(hi) > 0).mean(0).mean() > target: lo, hi = hi, hi * 2
    Y = None
    for _ in range(32):
        m = (lo + hi) / 2; Y = cl(m, Y)
        if (Y > 0).mean(0).mean() > target: lo = m
        else: hi = m
    a = (lo + hi) / 2; Y, it = solve(X, th, a, M, Y)
    return a, Y, it


def cos(Y): return tc.cos(Y)


if __name__ == '__main__':
    t0 = time.time()
    VERS = ['U', 'C'] + [f'local {l} µm' for l in LAMS]
    GEO = {}
    for k, mb in tc.MB.items():
        GEO[k] = geometry(k, mb)
        print(k, {x: GEO[k][x] for x in ('n_in', 'n_out', 'snap_um', 'nodes')}, f'{time.time() - t0:.0f} s', flush=True)
    MS = {}
    for k, mb in tc.MB.items():
        N = len(mb['W']); u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean()
        Mc = np.outer(u, v) / N
        MS[k] = {'U': np.full((N, N), 1.0 / N), 'C': Mc}
        g = GEO[k]
        for l in LAMS:
            Mk = kernel_M(g['cpar'], g['clen'], g['Win'], g['Wout'], l)
            MS[k][f'local {l} µm'] = Mk * (Mc.mean() / Mk.mean())
        # check: lambda -> infinity gives C (the synapse counts of this table vs the per-KC table)
        Minf = np.asarray(g['Wout'].sum(0)).ravel()[:, None] * np.asarray(g['Win'].sum(0)).ravel()[None]
        MS[k]['_check_C_vs_counts_r'] = float(np.corrcoef(Minf.ravel()[::97], Mc.ravel()[::97])[0, 1])
        print(k, 'M ready', f'{time.time() - t0:.0f} s', 'C vs counts r', round(MS[k]['_check_C_vs_counts_r'], 4), flush=True)
    rows_fit, rows_pair, rows_null, summary = [], [], [], {'mushroom bodies': {k: {x: GEO[k][x] for x in ('n_in', 'n_out', 'snap_um', 'nodes')} for k in GEO}}
    th_data = tc.TH_MAIN
    for rn, R in tc.RESP.items():
        od = list(R.columns); iu = np.triu_indices(len(od), 1); pi = tc.pair_index(od); oi = np.array([tc.O.index(o) for o in od])
        cl_b = tc.CLD[th_data]['TNT'].reindex(od).values; target = float(tc.CLD[th_data]['TNT3c'].reindex(od).mean())
        y = tc.data_dS({c: np.arange(len(tc.SREC[c])) for c in tc.COND}, pi)
        S3c, ST = tc.SREC['TNT3c'].mean(0)[pi], tc.SREC['TNT'].mean(0)[pi]
        dr = tc.data_ratio({c: np.arange(len(tc.SREC[c])) for c in tc.COND}, th_data, oi)
        res = {vv: {} for vv in VERS}; Sb_all = {}; per = []
        for k, mb in tc.MB.items():
            W = mb['W']; gl = [g_ for g_ in W.columns if g_ in R.index]
            Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None); H = Wm @ Rp
            gain = np.array([1.0 / np.quantile(H[:, j], 1 - cl_b[j]) for j in range(H.shape[1])])
            X = Wm @ (Rp * gain[None]); Yb = np.maximum(X - 1, 0); Sb = cos(Yb)[iu]; Sb_all[k] = Sb
            for vv in VERS:
                a, Y, it = calibrate(X, 1.0, MS[k][vv], target)
                Si = cos(Y)[iu]
                res[vv][k] = dict(a=a, Si=Si, cl=(Y > 0).mean(0), it=it, Y=Y)
                per.append(dict(responses=rn, mushroom_body=k, version=vv, gain=a, iterations=it,
                                spearman_blocked_vs_intact=spearmanr(Sb, Si, nan_policy='omit')[0],
                                r_dS_with_data=np.corrcoef(y, Sb - Si)[0, 1], mean_dS=float(np.nanmean(Sb - Si))))
                print(rn, k, vv, f'a {a:.3g}, it {it}, rank {per[-1]["spearman_blocked_vs_intact"]:.3f}, r {per[-1]["r_dS_with_data"]:.3f}', f'{time.time() - t0:.0f} s', flush=True)
            # check U against tnt_compare's solution
        PER = pd.DataFrame(per)
        Sb = np.nanmean([Sb_all[k] for k in tc.MB], 0)
        PAIR = pd.DataFrame(dict(responses=rn, odor_a=np.array(od)[iu[0]], odor_b=np.array(od)[iu[1]], S_TNT3c=S3c, S_TNT=ST, dS_data=y, S_model_blocked=Sb))
        fits = {}
        for vv in VERS:
            Si = np.nanmean([res[vv][k]['Si'] for k in tc.MB], 0); dSm = Sb - Si
            fits[vv] = dict(responses=rn, version=vv,
                            rank_spearman_blocked_vs_intact_mean_of_MBs=float(PER[(PER.version == vv)].spearman_blocked_vs_intact.mean()),
                            rank_spearman_blocked_vs_intact_of_MB_mean=float(spearmanr(Sb, Si, nan_policy='omit')[0]),
                            r_dS_with_data=float(np.corrcoef(y, dSm)[0, 1]),
                            mean_dS_model=float(np.nanmean(dSm)), mean_dS_data=float(y.mean()))
            rows_fit.append(fits[vv])
            PAIR[f'S_model_intact_{vv}'] = Si
        rows_pair.append(PAIR)
        # suppression per odor (blocked / intact fraction), mean over MBs
        Xb = {}
        for k, mb in tc.MB.items():
            W = mb['W']; gl = [g_ for g_ in W.columns if g_ in R.index]
            Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None); H = Wm @ Rp
            gain = np.array([1.0 / np.quantile(H[:, j], 1 - cl_b[j]) for j in range(H.shape[1])])
            Xb[k] = ((Wm @ (Rp * gain[None])) > 1).mean(0)
        clbm = np.mean([Xb[k] for k in tc.MB], 0)
        for vv in VERS:
            clm = np.mean([res[vv][k]['cl'] for k in tc.MB], 0)
            fits[vv]['rho_suppression_with_data'] = float(spearmanr(dr, clbm / clm)[0])
        # data rank order, as in tnt_rank_order.py
        fits_data = dict(rank_spearman_TNT_vs_TNT3c=float(spearmanr(ST, S3c)[0]))
        # null for the best local lambda (highest r with the data's change)
        loc = [vv for vv in VERS if vv.startswith('local')]
        best = max(loc, key=lambda vv: fits[vv]['r_dS_with_data'])
        nr, nrank = [], []
        for s in range(NNULL):
            Sis = []
            for k, mb in tc.MB.items():
                perm = np.arange(len(mb['W']))
                for c in np.unique(mb['cls']):
                    i = np.where(mb['cls'] == c)[0]; perm[i] = rs.permutation(i)
                Mp = MS[k][best][np.ix_(perm, perm)]
                W = mb['W']; gl = [g_ for g_ in W.columns if g_ in R.index]
                Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None); H = Wm @ Rp
                gain = np.array([1.0 / np.quantile(H[:, j], 1 - cl_b[j]) for j in range(H.shape[1])])
                X = Wm @ (Rp * gain[None])
                a, Y, _ = calibrate(X, 1.0, Mp, target, a0=res[best][k]['a'])
                Sis.append(cos(Y)[iu])
            Si = np.nanmean(Sis, 0)
            nr.append(np.corrcoef(y, Sb - Si)[0, 1]); nrank.append(np.mean([spearmanr(Sb_all[k], x, nan_policy='omit')[0] for k, x in zip(tc.MB, Sis)]))
            rows_null.append(dict(responses=rn, version=best, shuffle=s + 1, r_dS_with_data=nr[-1], rank_spearman_mean_of_MBs=nrank[-1]))
            if s % 10 == 9: print(rn, 'null', s + 1, f'{time.time() - t0:.0f} s', flush=True)
        summary[rn] = {
            'data: Spearman TNT vs TNT3c similarity over odor pairs': round(fits_data['rank_spearman_TNT_vs_TNT3c'], 3),
            'model: Spearman blocked vs intact (mean over MBs) by version': {vv: round(fits[vv]['rank_spearman_blocked_vs_intact_mean_of_MBs'], 3) for vv in VERS},
            'model: Spearman blocked vs intact per MB': {vv: {k: round(float(x), 3) for k, x in PER[PER.version == vv].set_index('mushroom_body').spearman_blocked_vs_intact.items()} for vv in VERS},
            'r over odor pairs, model change vs data change': {vv: round(fits[vv]['r_dS_with_data'], 3) for vv in VERS},
            'r per MB': {vv: {k: round(float(x), 3) for k, x in PER[PER.version == vv].set_index('mushroom_body').r_dS_with_data.items()} for vv in VERS},
            'Spearman over odors, suppression model vs data': {vv: round(fits[vv]['rho_suppression_with_data'], 3) for vv in VERS},
            'mean change in similarity: data, model by version': [round(float(y.mean()), 3), {vv: round(fits[vv]['mean_dS_model'], 3) for vv in VERS}],
            f'null ({NNULL} within-class swaps of KC sites on APL), version {best}': {
                'r real, null mean, null 5-95%, p': [round(fits[best]['r_dS_with_data'], 3), round(float(np.mean(nr)), 3),
                                                     [round(float(np.quantile(nr, .05)), 3), round(float(np.quantile(nr, .95)), 3)],
                                                     round(float((1 + np.sum(np.array(nr) >= fits[best]['r_dS_with_data'])) / (1 + NNULL)), 3)],
                'rank Spearman real, null mean, null 5-95%, p (lower = more reordering)': [
                    round(fits[best]['rank_spearman_blocked_vs_intact_mean_of_MBs'], 3), round(float(np.mean(nrank)), 3),
                    [round(float(np.quantile(nrank, .05)), 3), round(float(np.quantile(nrank, .95)), 3)],
                    round(float((1 + np.sum(np.array(nrank) <= fits[best]['rank_spearman_blocked_vs_intact_mean_of_MBs'])) / (1 + NNULL)), 3)]},
            'gains a per MB': {vv: {k: float(f'{res[vv][k]["a"]:.4g}') for k in tc.MB} for vv in VERS}}
        PER.round(4).to_csv(OUT + f'APL_local_model_per_MB_{rn}.csv', index=False)
    pd.DataFrame(rows_fit).round(4).to_csv(OUT + 'APL_local_model_fits.csv', index=False)
    pd.concat(rows_pair, ignore_index=True).round(4).to_csv(OUT + 'APL_local_model_odor_pairs.csv', index=False)
    pd.DataFrame(rows_null).round(4).to_csv(OUT + 'APL_local_model_nulls.csv', index=False)
    json.dump(summary, open(OUT + 'APL_local_model_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    print(f'done {time.time() - t0:.0f} s')
