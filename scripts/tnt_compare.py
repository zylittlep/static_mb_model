"""TNT vs TNT3c KC responses compared with the connectome APL model of apl_block_model.py. Official ROIs only.

Data (Yang, 2026-10-15): per-recording KC responses to the 16 ORN_ovlp16 odors, 8 TNT recordings (7 flies) and 11 TNT3c
recordings (8 flies). TNT3c is the control (APL intact), TNT has APL blocked. Odor similarity S = cosine similarity of two odors'
responses over all units of a recording, averaged over recordings (= 1 - Yang's RDMs); dS = S(TNT) - S(TNT3c).
Responding = response > 1; per-odor suppression = fraction responding in TNT / fraction in TNT3c.

Model: the KC model of apl_block_model.py, one per mushroom body (hemibrain R, male CNS L/R; FlyWire and BANC discarded
2026-10-21, Yang), with
ORN (16 odors) or GH146 (13 odors) inputs, pinned to the data: each odor's input is scaled so that with APL blocked the
fraction of KCs responding equals that odor's fraction in TNT, and one APL gain per version sets the mean fraction over odors
to TNT3c's. Model similarity = cosine of KC responses, averaged over the 3 mushroom bodies.
Versions U, C, P0, P, L, M0, M as in apl_block_model.py, plus feed-forward APL drive from PNs:
  F0  APL activity per odor = the odor's mean KC input, subtractive on every KC (no feedback)
  F   same, each glomerulus's input weighted by its PN->APL synapses per PN->KC synapse (claw-input-weighted mean 1)
Nulls (200 shuffles per mushroom body, the real wiring solved alongside): C with the KCs' (APL->KC, KC->APL) pairs shuffled
among KCs of the same class; P and M with the bouton weights p_g shuffled across glomeruli; F with the PN->APL weights
shuffled across glomeruli. Bootstrap (2,000 draws): recordings resampled within each condition, each recording treated as an
independent hemisphere (Yang, 2026-10-15). Threshold 1 is the main setting; threshold 0.5 repeats the versions without nulls.
Env: TNT_OUT (output folder, default odor_weighted/), TNT_SHUFFLES (default 200), TNT_CACHE (npz path; reused if present)."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, time, multiprocessing as mp
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import apl_block_model as abm
from orn_load import load

OUT = os.environ.get('TNT_OUT', abm.B + 'odor_weighted/')
CACHE = os.environ.get('TNT_CACHE')
UP = '/mnt/project-files/uploads/hearth/'
FILES = {'TNT': (UP + '2e7fb552-f4ec-49d7-b06f-14ee778f6433', UP + '148a9ca4-5d84-4c34-b071-0de79e706a4b'),     # resp_perfly.pkl, RDM.csv
         'TNT3c': (UP + '76debf61-257e-4375-b4d8-a02b48c4f62a', UP + '67eaa6de-5589-4f3b-8a0c-09ebd68e01c2')}
COND = ['TNT3c', 'TNT']
TH_MAIN, TH_ALT = 1.0, 0.5
NS = int(os.environ.get('TNT_SHUFFLES', 200))
NB = 2000
VERS = ['U', 'C', 'P0', 'P', 'L', 'M0', 'M', 'F0', 'F'] + (['MC'] if abm.REVISION else [])   # MC: revision runs only (2026-11-04)
NULLS = ['C', 'P', 'M', 'F']
VLABEL = {'U': 'uniform', 'C': 'per-KC counts', 'P0': 'boutons', 'P': 'boutons per glom.', 'L': 'claws per glom.',
          'M0': 'U + boutons', 'M': 'U + boutons per glom.', 'F0': 'PN drive', 'F': 'PN drive per glom.', 'MC': 'C + boutons per glom.'}
NLABEL = {'C': 'C: per-KC\ncounts', 'P': 'P: all on\nboutons', 'M': 'M: U + boutons\nat real share', 'F': 'F: PN->APL\ndrive'}
INK, MUTED, GRID, NAN_C = abm.INK, abm.MUTED, abm.GRID, abm.NAN_C
CC = {'TNT': '#c8553d', 'TNT3c': '#2a78d6'}
RC = {'ORN': '#1f2a44', 'GH146': '#8a5cc2'}

# ---------------------------------------------------------------- data
D = {c: load(FILES[c][0]) for c in COND}
O = list(next(iter(D['TNT'].values())).columns)
NO, IU = len(O), np.triu_indices(len(O), 1)
assert all(list(v.columns) == O for d in D.values() for v in d.values())
FLY = {c: np.array([r.split('F')[0] for r in D[c]]) for c in COND}       # date_flynumber; only counted in the summary


def cos(Y):
    n = np.linalg.norm(Y, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (Y.T @ Y) / np.outer(n, n)


def cos3(Y):
    """(KCs, sets, odors) -> (sets, odors, odors)"""
    n = np.linalg.norm(Y, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.einsum('kso,ksp->sop', Y, Y, optimize=True) / (n[:, :, None] * n[:, None, :])


SREC = {c: np.array([cos(v.values)[IU] for v in D[c].values()]) for c in COND}               # recordings x odor pairs
for c in COND:
    rdm = pd.read_csv(FILES[c][1], index_col=0).loc[O, O].values
    assert np.abs(1 - rdm[IU] - SREC[c].mean(0)).max() < 1e-6, c                               # = Yang's RDM
FRAC = {th: {c: np.array([(v.values > th).mean(0) for v in D[c].values()]) for c in COND} for th in (TH_MAIN, TH_ALT)}
CLD = {th: {c: pd.Series(FRAC[th][c].mean(0), index=O) for c in COND} for th in FRAC}

# ---------------------------------------------------------------- model
RESP = abm.responses()
MB, SIDES = abm.mushroom_bodies()


def norm_w(x, w):
    """Glomerulus weights scaled to a claw-input-weighted mean of 1; glomeruli without a value get 1."""
    x = np.asarray(x, float); ok = ~np.isnan(x)
    return np.where(ok, x / (np.sum(w[ok] * x[ok]) / w[ok].sum()), 1.0)


def bisect_fixed(F, lo, hi, it=26):
    """A = F(A) per entry; F falls as A grows."""
    hi = hi + 1e-12
    for _ in range(it):
        m = (lo + hi) / 2; up = m > F(m); hi = np.where(up, m, hi); lo = np.where(up, lo, m)
    return (lo + hi) / 2


def calib_vec(cl, target, x0, it=26):
    """Per set, the gain x where the mean fraction responding cl(x) (falling in x) reaches target."""
    lo, hi = np.zeros_like(x0), 2 * x0
    for _ in range(40):
        over = cl(hi) > target
        if not over.any(): break
        hi = np.where(over, hi * 2, hi)
    for _ in range(it):
        m = (lo + hi) / 2; over = cl(m) > target; lo = np.where(over, m, lo); hi = np.where(over, hi, m)
    return (lo + hi) / 2


def solve_sub(X, th, Uu, Vv, target, a0):
    """y = [X - th - a u A]+, A = mean(v y); columns of Uu / Vv = sets of per-KC weights, a calibrated per set."""
    K, S = Uu.shape; Xt = X - th; buf = np.empty((K, S, X.shape[1]))
    def Y(a, A):
        np.multiply(Uu[:, :, None], (a[:, None] * A)[None], out=buf); np.subtract(Xt[:, None, :], buf, out=buf)
        return np.maximum(buf, 0, out=buf)
    hi0 = np.einsum('ks,ko->so', Vv, np.maximum(Xt, 0)) / K
    fixed = lambda a: Y(a, bisect_fixed(lambda A: np.einsum('ks,kso->so', Vv, Y(a, A)) / K, np.zeros_like(hi0), hi0))
    a = calib_vec(lambda a: (fixed(a) > 0).mean(0).mean(1), target, np.full(S, a0))
    return fixed(a).copy()


def solve_mix(Wm, Rg, th, aU, bP, f, P, target, s0):
    """y = [Wm (Rg / (1 + s f bP p_g A)) - th - s (1 - f) aU A]+, A = mean(y); columns of P = sets of glomerulus weights."""
    S = P.shape[1]
    def Y(s, A):
        y = np.tensordot(Wm, Rg[:, None, :] / (1 + (s * f * bP)[None, :, None] * P[:, :, None] * A[None]), axes=(1, 0))
        y -= th + (s * (1 - f) * aU)[None, :, None] * A[None]
        return np.maximum(y, 0, out=y)
    hi0 = np.repeat(np.maximum(Wm @ Rg - th, 0).mean(0)[None], S, 0)
    fixed = lambda s: Y(s, bisect_fixed(lambda A: Y(s, A).mean(0), np.zeros_like(hi0), hi0))
    s = calib_vec(lambda s: (fixed(s) > 0).mean(0).mean(1), target, np.full(S, s0))
    return fixed(s)


def solve_ff(X, th, Af, target, a0):
    """y = [X - th - a A_o]+ with fixed feed-forward APL activity per odor (rows of Af = sets)."""
    Xt = X - th
    a = calib_vec(lambda a: (Xt[:, None, :] - a[None, :, None] * Af[None] > 0).mean(0).mean(1), target, np.full(len(Af), a0))
    return np.maximum(Xt[:, None, :] - a[None, :, None] * Af[None], 0)


def run_task(task):
    rn, k, th_data, seed = task
    t0 = time.time(); rs = np.random.default_rng(seed)
    R, mb = RESP[rn], MB[k]
    cl_b = CLD[th_data]['TNT'].reindex(R.columns).values
    target = float(CLD[th_data]['TNT3c'].reindex(R.columns).mean())
    W = mb['W']; gl = [g for g in W.columns if g in R.index]
    Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None)
    H = Wm @ Rp; th = 1.0
    g = np.array([th / np.quantile(H[:, j], 1 - cl_b[j]) for j in range(H.shape[1])])     # per-odor input gain
    Rg = Rp * g[None]; X = Wm @ Rg; w = Wm.sum(0)
    iu = np.triu_indices(R.shape[1], 1)
    one, oneg = np.ones(len(X)), np.ones(len(gl))
    u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean()
    praw = mb['p'].reindex(gl).values
    fraw = abm.GWi.loc[mb['ds']].PN_to_APL_per_PN_to_KC.reindex(gl).values
    p, q, pf = norm_w(praw, w), norm_w(mb['q'].reindex(gl).values, w), norm_w(fraw, w)
    f = abm.F_SHARE[mb['ds']]
    Y, gain = {'blocked': np.maximum(X - th, 0)}, {}
    for nm, (uu, vv) in {'U': (one, one), 'C': (u, v)}.items():
        gain[nm] = abm.calib(lambda a: abm.cl_of(abm.model_y(Wm, Rg, th, a, 0, uu, vv, oneg)).mean(), target)
        Y[nm] = abm.model_y(Wm, Rg, th, gain[nm], 0, uu, vv, oneg)
    for nm, pp in {'P0': oneg, 'P': p, 'L': q}.items():
        gain[nm] = abm.calib(lambda b: abm.cl_of(abm.model_y(Wm, Rg, th, 0, b, one, one, pp)).mean(), target)
        Y[nm] = abm.model_y(Wm, Rg, th, 0, gain[nm], one, one, pp)
    for nm, pp in {'M0': oneg, 'M': p}.items():
        mk = lambda s: abm.model_y(Wm, Rg, th, s * (1 - f) * gain['U'], s * f * gain['P0'], one, one, pp)
        gain[nm] = abm.calib(lambda s: abm.cl_of(mk(s)).mean(), target)
        Y[nm] = mk(gain[nm])
    if 'MC' in VERS:     # inhibition on KCs per KC (C) and on boutons per glomerulus (P), at the boutons' synapse share f
        mkc = lambda s: abm.model_y(Wm, Rg, th, s * (1 - f) * gain['C'], s * f * gain['P'], u, v, p)
        gain['MC'] = abm.calib(lambda s: abm.cl_of(mkc(s)).mean(), target)
        Y['MC'] = mkc(gain['MC'])
    for nm, pp in {'F0': oneg, 'F': pf}.items():
        Af = (w * pp) @ Rg / w.sum()
        gain[nm] = abm.calib(lambda a: (X - th - a * Af[None] > 0).mean(0).mean(), target)
        Y[nm] = np.maximum(X - th - gain[nm] * Af[None], 0)
    out = {'cl': {nm: abm.cl_of(y) for nm, y in Y.items()}, 'S': {nm: cos(y)[iu] for nm, y in Y.items()}}
    if seed is not None:                     # nulls; set 0 = the real wiring, solved the same way as the shuffles
        Uu, Vv = [u], [v]
        for _ in range(NS):
            uu, vv = u.copy(), v.copy()
            for c in np.unique(mb['cls']):
                i = np.where(mb['cls'] == c)[0]; j = rs.permutation(i); uu[i], vv[i] = u[j], v[j]
            Uu.append(uu); Vv.append(vv)
        def gshuf(x):
            ok = ~np.isnan(x); cols = [norm_w(x, w)]
            for _ in range(NS):
                y = x.copy(); y[ok] = rs.permutation(x[ok]); cols.append(norm_w(y, w))
            return np.array(cols).T
        Pm, Pf = gshuf(praw), gshuf(fraw)
        null = {'C': solve_sub(X, th, np.array(Uu).T, np.array(Vv).T, target, gain['C']),
                'P': solve_mix(Wm, Rg, th, 0.0, gain['P0'], 1.0, Pm, target, gain['P'] / gain['P0']),
                'M': solve_mix(Wm, Rg, th, gain['U'], gain['P0'], f, Pm, target, gain['M']),
                'F': solve_ff(X, th, ((w[:, None] * Pf).T @ Rg) / w.sum(), target, gain['F'])}
        out['null'] = {}
        for nm, Yn in null.items():
            Sn = cos3(Yn)[:, iu[0], iu[1]]
            out['null'][nm] = {'cl': (Yn > 0).mean(0), 'S': Sn}
            out.setdefault('check', {})[nm] = float(np.nanmax(np.abs(Sn[0] - out['S'][nm])))
    print(f'{rn} {k} threshold {th_data}: {time.time() - t0:.0f} s', flush=True)
    return out


def model_runs():
    tasks = [(rn, k, TH_MAIN, 1000 * i + j) for i, rn in enumerate(RESP) for j, k in enumerate(MB)]
    tasks += [(rn, k, TH_ALT, None) for rn in RESP for k in MB]
    with mp.get_context('fork').Pool(min(4, os.cpu_count())) as pool:
        res = pool.map(run_task, tasks, chunksize=1)
    flat = {}
    for (rn, k, th, _), r in zip(tasks, res):
        for nm in r['cl']:
            flat[f'{th}|{rn}|{k}|cl|{nm}'] = r['cl'][nm]; flat[f'{th}|{rn}|{k}|S|{nm}'] = r['S'][nm]
        for nm, d in r.get('null', {}).items():
            flat[f'{th}|{rn}|{k}|nullcl|{nm}'] = d['cl']; flat[f'{th}|{rn}|{k}|nullS|{nm}'] = d['S']
        for nm, x in r.get('check', {}).items():
            flat[f'{th}|{rn}|{k}|check|{nm}'] = np.array(x)
    return flat


# ---------------------------------------------------------------- comparison helpers
def pair_index(odors):
    """Positions of the upper-triangle pairs of `odors` (a subsequence of O) in the full 16-odor pair list."""
    pos = {(a, b): n for n, (a, b) in enumerate(zip(np.array(O)[IU[0]], np.array(O)[IU[1]]))}
    i, j = np.triu_indices(len(odors), 1)
    return np.array([pos[(odors[a], odors[b])] for a, b in zip(i, j)])


def additive(y, n):
    """dS_ab = mu + f_a + f_b: per-odor effects (mean 0), residuals, R^2."""
    i, j = np.triu_indices(n, 1); Xd = np.zeros((len(i), n + 1)); Xd[:, 0] = 1
    Xd[np.arange(len(i)), i + 1] = 1; Xd[np.arange(len(i)), j + 1] = 1
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None); fit = Xd @ b
    return b[1:] - b[1:].mean(), y - fit, 1 - np.var(y - fit) / np.var(y)


def mantel_p(y, x, n, rs, nperm=5000):
    """p of corr(y, x) between odor-pair vectors when x's odor labels are permuted."""
    M = np.zeros((n, n)); i, j = np.triu_indices(n, 1); M[i, j] = x; M = M + M.T
    r0 = np.corrcoef(y, x)[0, 1]; cnt = 0
    for _ in range(nperm):
        pm = rs.permutation(n); cnt += np.corrcoef(y, M[np.ix_(pm, pm)][i, j])[0, 1] >= r0
    return (cnt + 1) / (nperm + 1)


def boot_draws(rs, nb):
    """Recordings resampled with replacement within each condition (each recording = an independent hemisphere)."""
    return [{c: rs.choice(len(SREC[c]), len(SREC[c])) for c in COND} for _ in range(nb)]


def split_halves(rs, nh):
    """Random splits of each condition's recordings into two halves."""
    out = []
    for _ in range(nh):
        h = [{}, {}]
        for c in COND:
            p = rs.permutation(len(SREC[c])); m = len(p) // 2; h[0][c], h[1][c] = p[:m], p[m:]
        out.append(h)
    return out


def data_dS(idx, pi):
    return SREC['TNT'][idx['TNT']].mean(0)[pi] - SREC['TNT3c'][idx['TNT3c']].mean(0)[pi]


def data_ratio(idx, th, oi):
    return FRAC[th]['TNT'][idx['TNT']].mean(0)[oi] / FRAC[th]['TNT3c'][idx['TNT3c']].mean(0)[oi]


if __name__ == '__main__':
    t0 = time.time()
    if CACHE and os.path.exists(CACHE):
        with np.load(CACHE, allow_pickle=False) as z:
            flat = {k: z[k] for k in z.files}
    else:
        flat = model_runs()
        if CACHE: np.savez_compressed(CACHE, **flat)
    print(f'model ready ({time.time() - t0:.0f} s)', flush=True)
    MBS = list(MB)
    rs = np.random.default_rng(7)
    ALL = {c: np.arange(len(SREC[c])) for c in COND}
    DRAWS, HALVES = boot_draws(rs, NB), split_halves(rs, 500)
    rows_fit, rows_null, rows_pair, rows_odor = [], [], [], []
    summary = {'data': {}, 'model': {}, 'fits': {}}
    keep = {}
    for th in (TH_MAIN, TH_ALT):
        for rn, R in RESP.items():
            od = list(R.columns); n = len(od); pi = pair_index(od); oi = np.array([O.index(o) for o in od])
            mm = lambda kind, nm: np.nanmean([flat[f'{th}|{rn}|{k}|{kind}|{nm}'] for k in MBS], axis=0)
            Sb = mm('S', 'blocked'); clb = mm('cl', 'blocked')
            dSm = {v: Sb - mm('S', v) for v in VERS}
            ratm = {v: clb / mm('cl', v) for v in VERS}
            y, dr = data_dS(ALL, pi), data_ratio(ALL, th, oi)
            boot_r = {v: [] for v in VERS}; boot_rho = {v: [] for v in VERS}; RB = []
            for d in DRAWS:
                yb, rb = data_dS(d, pi), data_ratio(d, th, oi); RB.append(rb)
                for v in VERS:
                    boot_r[v].append(np.corrcoef(yb, dSm[v])[0, 1]); boot_rho[v].append(spearmanr(rb, ratm[v])[0])
            boot_r = {v: np.array(x) for v, x in boot_r.items()}; boot_rho = {v: np.array(x) for v, x in boot_rho.items()}; RB = np.array(RB)
            fit = {}
            for v in VERS:
                r = np.corrcoef(y, dSm[v])[0, 1]; rho = spearmanr(dr, ratm[v])[0]; dU = boot_r[v] - boot_r['U']
                row = dict(threshold=th, responses=rn, version=v, label=VLABEL[v], r_pairs=r,
                           r_pairs_p_odor_permutation=mantel_p(y, dSm[v], n, rs),
                           r_pairs_boot_5=np.nanpercentile(boot_r[v], 5), r_pairs_boot_95=np.nanpercentile(boot_r[v], 95),
                           r_minus_U=r - fit.get('U', {}).get('r_pairs', r) if v != 'U' else 0.0,
                           r_minus_U_boot_5=np.nanpercentile(dU, 5), r_minus_U_boot_95=np.nanpercentile(dU, 95),
                           frac_draws_above_U=np.mean(dU > 0) if v != 'U' else np.nan,
                           rho_suppression=rho, rho_boot_5=np.nanpercentile(boot_rho[v], 5), rho_boot_95=np.nanpercentile(boot_rho[v], 95),
                           model_mean_dS=float(np.nanmean(dSm[v])), data_mean_dS=float(y.mean()))
                if th == TH_MAIN and v in NULLS:
                    Sn = np.nanmean([flat[f'{th}|{rn}|{k}|nullS|{v}'] for k in MBS], axis=0)          # sets x pairs
                    cn = np.nanmean([flat[f'{th}|{rn}|{k}|nullcl|{v}'] for k in MBS], axis=0)         # sets x odors
                    rn_ = np.array([np.corrcoef(y, Sb - s)[0, 1] for s in Sn]); rh_ = np.array([spearmanr(dr, clb / c)[0] for c in cn])
                    row.update(null_r_mean=rn_[1:].mean(), null_r_5=np.percentile(rn_[1:], 5), null_r_95=np.percentile(rn_[1:], 95),
                               null_p_r=(1 + np.sum(rn_[1:] >= rn_[0])) / (1 + NS),
                               null_rho_mean=rh_[1:].mean(), null_rho_5=np.percentile(rh_[1:], 5), null_rho_95=np.percentile(rh_[1:], 95),
                               null_p_rho=(1 + np.sum(rh_[1:] >= rh_[0])) / (1 + NS),
                               real_vs_solver_check=float(max(flat[f'{th}|{rn}|{k}|check|{v}'] for k in MBS)))
                    rows_null += [dict(responses=rn, test=v, shuffle=s, r_pairs=rn_[s], rho_suppression=rh_[s]) for s in range(len(rn_))]
                    keep[(rn, v)] = (rn_, rh_)
                fit[v] = row; rows_fit.append(row)
            # per-MB spread of r for U
            per_mb = {k: float(np.corrcoef(y, flat[f'{th}|{rn}|{k}|S|blocked'] - flat[f'{th}|{rn}|{k}|S|U'])[0, 1]) for k in MBS}
            # additive decomposition: data and uniform model
            fd, resd, r2d = additive(y, n); fm, resm, r2m = additive(dSm['U'], n)
            SC = SREC['TNT3c'].mean(0)[pi]; ST = SREC['TNT'].mean(0)[pi]
            rel = [np.corrcoef(data_dS(a, pi), data_dS(b, pi))[0, 1] for a, b in HALVES]
            relr = [spearmanr(data_ratio(a, th, oi), data_ratio(b, th, oi))[0] for a, b in HALVES]
            relres = [np.corrcoef(additive(data_dS(a, pi), n)[1], additive(data_dS(b, pi), n)[1])[0, 1] for a, b in HALVES]
            sb = lambda r: 2 * r / (1 + r)
            key = f'threshold {th}, {rn}'
            summary['fits'][key] = {
                'odors': n, 'r(data dS, model dS) by version': {v: round(fit[v]['r_pairs'], 3) for v in VERS},
                'uniform U: per mushroom body r': {k: round(x, 2) for k, x in per_mb.items()},
                'spearman(data suppression, model suppression) by version': {v: round(fit[v]['rho_suppression'], 3) for v in VERS},
                'data: share of dS variance that is per-odor (additive R2)': round(r2d, 3), 'model U: same': round(r2m, 3),
                'per-odor effects, data vs model U (spearman)': round(spearmanr(fd, fm)[0], 3),
                'pair-specific residuals, data vs model U (r)': round(np.corrcoef(resd, resm)[0, 1], 3),
                'data residual vs TNT3c similarity (r)': round(np.corrcoef(resd, SC)[0, 1], 3),
                'model U residual vs model intact similarity (r)': round(np.corrcoef(resm, Sb - dSm['U'])[0, 1], 3),
                'split-half (recordings) reliability of dS, full-data estimate': round(sb(np.mean(rel)), 3),
                'ceiling for r with a model (sqrt of that)': round(float(np.sqrt(sb(np.mean(rel)))), 3),
                'split-half reliability of the pair-specific residuals, full-data estimate': round(sb(np.mean(relres)), 3),
                'split-half reliability of per-odor suppression ranks, full-data estimate': round(sb(np.mean(relr)), 3),
                'mean dS: data / model U / model M': [round(float(y.mean()), 3), round(float(np.nanmean(dSm['U'])), 3), round(float(np.nanmean(dSm['M'])), 3)],
                'mean S: data TNT3c / TNT, model intact U / blocked': [round(float(SC.mean()), 3), round(float(ST.mean()), 3),
                                                                        round(float(np.nanmean(Sb - dSm['U'])), 3), round(float(np.nanmean(Sb)), 3)]}
            for t, (a, b) in enumerate(zip(np.array(od)[np.triu_indices(n, 1)[0]], np.array(od)[np.triu_indices(n, 1)[1]])):
                rows_pair.append(dict(threshold=th, responses=rn, odor_a=a, odor_b=b, S_TNT3c=SC[t], S_TNT=ST[t], dS_data=y[t],
                                      dS_data_per_odor_fit=y[t] - resd[t], dS_data_pair_specific=resd[t], S_model_blocked=Sb[t],
                                      **{f'dS_model_{v}': dSm[v][t] for v in VERS}))
            for t, o in enumerate(od):
                rows_odor.append(dict(threshold=th, responses=rn, odor=o, frac_responding_TNT3c=CLD[th]['TNT3c'][o],
                                      frac_responding_TNT=CLD[th]['TNT'][o], suppression_data=dr[t],
                                      suppression_boot_5=np.percentile(RB[:, t], 5), suppression_boot_95=np.percentile(RB[:, t], 95),
                                      odor_effect_dS_data=fd[t], odor_effect_dS_model_U=fm[t], model_frac_blocked=clb[t],
                                      **{f'suppression_model_{v}': ratm[v][t] for v in VERS}))
            if th == TH_MAIN:
                summary['fits'][key]['real wiring vs shuffles: r over pairs (real, shuffle mean, p) / rho suppression (real, shuffle mean, p)'] = {
                    v: [round(fit[v]['r_pairs'], 3), round(fit[v]['null_r_mean'], 3), round(fit[v]['null_p_r'], 3),
                        round(fit[v]['rho_suppression'], 3), round(fit[v]['null_rho_mean'], 3), round(fit[v]['null_p_rho'], 3)] for v in NULLS}
                keep[(rn, 'main')] = dict(od=od, y=y, dSm=dSm, Sb=Sb, dr=dr, ratm=ratm, fd=fd, fm=fm, fit=fit, SC=SC, ST=ST)
    # data similarity with responses thresholded like the model's output
    TR = {'raw': lambda x: x, 'clipped at 0': lambda x: np.maximum(x, 0), '[x - 1]+': lambda x: np.maximum(x - 1, 0),
          'x > 1 (0/1)': lambda x: (x > 1).astype(float)}
    SP = {nm: {c: np.nanmean([cos(f(v.values))[IU] for v in D[c].values()], axis=0) for c in COND} for nm, f in TR.items()}
    SL = {nm: {c: float(np.nanmean(d[c])) for c in COND} for nm, d in SP.items()}
    dS_tr = {nm: d['TNT'] - d['TNT3c'] for nm, d in SP.items()}
    summary['data'] = {
        'recordings (flies)': {c: f'{len(SREC[c])} ({len(np.unique(FLY[c]))})' for c in COND},
        'fraction of units > 1, mean over odors': {c: round(float(CLD[TH_MAIN][c].mean()), 4) for c in COND},
        'suppression (TNT / TNT3c fraction > 1) range over odors': [round(float((CLD[TH_MAIN]['TNT'] / CLD[TH_MAIN]['TNT3c']).min()), 2),
                                                                    round(float((CLD[TH_MAIN]['TNT'] / CLD[TH_MAIN]['TNT3c']).max()), 2)],
        'pairs more similar in TNT': f"{int(np.sum(SREC['TNT'].mean(0) > SREC['TNT3c'].mean(0)))} of {len(IU[0])}",
        'mean cosine similarity by response transform (TNT3c, TNT)': {nm: [round(d['TNT3c'], 3), round(d['TNT'], 3)] for nm, d in SL.items()},
        'change pattern by response transform vs raw (r over odor pairs)': {nm: round(float(np.corrcoef(x, dS_tr['raw'])[0, 1]), 3) for nm, x in dS_tr.items()},
        'pairs more similar in TNT by response transform': {nm: int(np.sum(x > 0)) for nm, x in dS_tr.items()}}
    summary['model'] = {'mushroom bodies (KCs)': SIDES, 'shuffles per mushroom body': NS, 'bootstrap draws (recordings resampled)': NB,
                        'pinning': 'per-odor input gain so blocked fraction = TNT fraction > threshold; APL gain so mean intact fraction = TNT3c mean'}
    pd.DataFrame(rows_fit).round(4).to_csv(OUT + 'TNT_vs_model_fits.csv', index=False)
    pd.DataFrame(rows_null).round(4).to_csv(OUT + 'TNT_vs_model_nulls.csv', index=False)
    pd.DataFrame(rows_pair).round(4).to_csv(OUT + 'TNT_vs_model_odor_pairs.csv', index=False)
    pd.DataFrame(rows_odor).round(4).to_csv(OUT + 'TNT_vs_model_per_odor.csv', index=False)
    json.dump(summary, open(OUT + 'TNT_vs_model_summary.json', 'w'), indent=1, default=float)

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(18, 13.6))
    gs = fig.add_gridspec(3, 4, left=0.05, right=0.985, top=0.885, bottom=0.085, hspace=0.5, wspace=0.34)
    ax = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(4)]
    K = keep[('ORN', 'main')]

    def labels(a, xs, ys, names):
        """Odor labels next to points, nudged down when they would sit on an earlier label."""
        (x0, x1), (y0, y1) = a.get_xlim(), a.get_ylim(); hpt = a.get_window_extent().height * 72 / fig.dpi; placed = []
        for o, xx, yy in zip(names, xs, ys):
            fx, fy, dy = (xx - x0) / (x1 - x0), (yy - y0) / (y1 - y0), 2
            while any(abs(fx - px) < 0.09 and abs(fy + dy / hpt - py) < 0.035 for px, py in placed): dy -= 8
            placed.append((fx, fy + dy / hpt))
            a.annotate(o, (xx, yy), fontsize=7, xytext=(3, dy), textcoords='offset points', color=INK)

    def tag(a, s):
        a.text(-0.12, 1.06, s, transform=a.transAxes, fontsize=13, fontweight='bold', va='bottom', color=INK)

    def heat(a, vals, odors, title, vmax, cmap):
        n = len(odors); M = np.full((n, n), np.nan); i, j = np.triu_indices(n, 1); M[i, j] = vals; M[j, i] = vals
        cm = plt.get_cmap(cmap).copy(); cm.set_bad(NAN_C)
        im = a.imshow(M, cmap=cm, vmin=0, vmax=vmax)
        a.set_xticks(range(n)); a.set_xticklabels(odors, rotation=90, fontsize=7); a.set_yticks(range(n)); a.set_yticklabels(odors, fontsize=7)
        a.set_title(title, loc='left'); a.tick_params(length=0)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
        return im

    smax = float(max(K['ST'].max(), K['SC'].max()))
    heat(ax[0], K['SC'], O, 'TNT3c: odor similarity (cosine)', smax, 'viridis')
    heat(ax[1], K['ST'], O, 'TNT: odor similarity (cosine)', smax, 'viridis')
    heat(ax[2], K['y'], O, f"Change, TNT - TNT3c (mean +{K['y'].mean():.2f})", float(K['y'].max()), 'magma')
    for a, s in zip(ax[:3], 'abc'): tag(a, s)
    # d: per-odor fraction responding
    a = ax[3]; tag(a, 'd'); od = O
    supp = (CLD[TH_MAIN]['TNT'] / CLD[TH_MAIN]['TNT3c']).sort_values()
    for yy, o in enumerate(supp.index):
        a.plot([CLD[TH_MAIN]['TNT3c'][o], CLD[TH_MAIN]['TNT'][o]], [yy, yy], color=GRID, lw=2, zorder=1)
        for c in COND: a.scatter(CLD[TH_MAIN][c][o], yy, color=CC[c], s=28, zorder=2, label=c if yy == 0 else None)
        a.text(0.262, yy, f'x{supp[o]:.2f}', va='center', fontsize=7, color=INK)
    a.set_yticks(range(len(supp))); a.set_yticklabels(supp.index, fontsize=7.5); a.set_xlim(0, 0.29)
    a.set_xlabel('fraction of units with response > 1'); a.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.42, 0.0), fontsize=7.5)
    a.set_title('Responding units per odor (TNT / TNT3c)', loc='left')
    # e: model U dS
    heat(ax[4], K['dSm']['U'], O, f"Model, uniform APL: change (mean +{np.nanmean(K['dSm']['U']):.2f})", float(np.nanmax(K['dSm']['U'])), 'magma'); tag(ax[4], 'e')
    # f: scatter data vs model U pairs
    a = ax[5]; tag(a, 'f')
    a.scatter(K['dSm']['U'], K['y'], s=14, color=RC['ORN'], alpha=0.75, lw=0)
    fu = K['fit']['U']
    a.set_title(f"Odor pairs: data vs uniform model, r = {fu['r_pairs']:.2f}", loc='left')
    a.set_xlabel('model change in similarity (blocked - intact)'); a.set_ylabel('data change in similarity (TNT - TNT3c)')
    pi16 = list(zip(np.array(O)[IU[0]], np.array(O)[IU[1]]))
    for t in np.argsort(-np.abs(K['y'] - np.polyval(np.polyfit(K['dSm']['U'], K['y'], 1), K['dSm']['U'])))[:6]:
        a.annotate(f'{pi16[t][0]}-{pi16[t][1]}', (K['dSm']['U'][t], K['y'][t]), fontsize=6.5, color=MUTED, xytext=(3, 2), textcoords='offset points')
    # g: per-odor effect on dS
    a = ax[6]; tag(a, 'g')
    a.axhline(0, color=GRID, lw=1, zorder=0); a.axvline(0, color=GRID, lw=1, zorder=0)
    a.scatter(K['fm'], K['fd'], s=20, color=RC['ORN'])
    labels(a, K['fm'], K['fd'], O)
    a.set_title(f"Per-odor part of the change: rho = {spearmanr(K['fd'], K['fm'])[0]:.2f}", loc='left')
    a.set_xlabel('model, uniform APL: per-odor effect on the change'); a.set_ylabel('data: per-odor effect on the change')
    # h: per-odor suppression
    a = ax[7]; tag(a, 'h')
    a.scatter(K['ratm']['U'], K['dr'], s=20, color=RC['ORN'])
    labels(a, K['ratm']['U'], K['dr'], O)
    a.set_title(f"Suppression per odor: rho = {fu['rho_suppression']:.2f}", loc='left')
    a.set_xlabel('model: fraction responding, blocked / intact'); a.set_ylabel('data: fraction > 1, TNT / TNT3c')
    # i: r per version
    a = ax[8]; tag(a, 'i')
    for s_, rn in enumerate(RESP):
        F_ = {r['version']: r for r in rows_fit if r['threshold'] == TH_MAIN and r['responses'] == rn}
        xs = np.arange(len(VERS)) + (s_ - 0.5) * 0.28
        for xx, v in zip(xs, VERS):
            a.plot([xx, xx], [F_[v]['r_pairs_boot_5'], F_[v]['r_pairs_boot_95']], color=RC[rn], lw=1.2, alpha=0.6)
        a.scatter(xs, [F_[v]['r_pairs'] for v in VERS], edgecolor=RC[rn], s=26, zorder=3, label=f'{rn} input', facecolor=RC[rn] if rn == 'ORN' else 'white')
        a.axhline(F_['U']['r_pairs'], color=RC[rn], lw=0.8, ls='--', zorder=0)
    ceil = summary['fits'][f'threshold {TH_MAIN}, ORN']['ceiling for r with a model (sqrt of that)']
    a.axhline(ceil, color=MUTED, lw=0.8, ls=':'); a.text(len(VERS) - 0.6, ceil, 'data reliability ceiling (ORN odors)', ha='right', va='bottom', fontsize=7, color=MUTED)
    a.set_xticks(range(len(VERS))); a.set_xticklabels([f'{v}: {VLABEL[v]}' for v in VERS], rotation=40, ha='right', fontsize=7)
    a.set_ylabel('r(data change, model change) over odor pairs'); a.legend(frameon=False, fontsize=7.5, loc='lower left')
    a.set_title('APL versions (bars: 90% bootstrap over recordings)', loc='left')
    # j, k: nulls
    for a, (col, lab, letter) in zip(ax[9:11], [(0, 'r over odor pairs (change in similarity)', 'j'), (1, 'rho over odors (suppression)', 'k')]):
        tag(a, letter)
        for s_, rn in enumerate(RESP):
            for t, v in enumerate(NULLS):
                vals = keep[(rn, v)][col]; xx = t + (s_ - 0.5) * 0.36
                vp = a.violinplot(vals[1:], positions=[xx], widths=0.32, showextrema=False)
                for b in vp['bodies']: b.set_facecolor(RC[rn]); b.set_alpha(0.25); b.set_edgecolor('none')
                a.scatter(xx, vals[0], marker='D', s=26, edgecolor=RC[rn], facecolor=RC[rn] if rn == 'ORN' else 'white', zorder=3)
        a.set_xticks(range(len(NULLS))); a.set_xticklabels([NLABEL[v] for v in NULLS], fontsize=6.8)
        a.set_ylabel(lab)
        a.set_title(f'Real wiring (diamond) vs {NS} shuffles (violin)', loc='left')
        a.set_xlabel('filled: ORN input, open: GH146 input', fontsize=7.5, color=MUTED)
    # l: similarity level by transform
    a = ax[11]; tag(a, 'l')
    labs = list(SL) + ['model (ORN)']
    v3c = [SL[nm]['TNT3c'] for nm in SL] + [float(np.nanmean(K['Sb'] - K['dSm']['U']))]
    vt = [SL[nm]['TNT'] for nm in SL] + [float(np.nanmean(K['Sb']))]
    xs = np.arange(len(labs))
    a.bar(xs - 0.2, v3c, 0.38, color=CC['TNT3c'], label='TNT3c / model intact (U)')
    a.bar(xs + 0.2, vt, 0.38, color=CC['TNT'], label='TNT / model blocked')
    a.set_xticks(xs); a.set_xticklabels(labs, fontsize=7.5, rotation=20); a.set_ylabel('mean cosine similarity over odor pairs')
    a.legend(frameon=False, fontsize=7.5); a.set_title('Similarity level, data by response transform', loc='left')
    for a in [ax[3]] + ax[5:]:
        a.spines[['top', 'right']].set_visible(False)
    MB_NAMES = ', '.join(k.replace('malecns', 'male CNS').replace('flywire', 'FlyWire').replace('banc', 'BANC') for k in MB)
    fig.suptitle('TNT vs TNT3c KC responses compared with the connectome APL model (official ROIs; TNT3c = control, TNT = APL blocked)',
                 x=0.05, y=0.975, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    fig.text(0.05, 0.928, 'Similarity = cosine of two odors\' responses over all units, mean over recordings (= 1 - the RDMs); change = TNT - TNT3c. '
             'Model pinned to the data: each odor\'s fraction of KCs responding with APL blocked = its TNT fraction > 1,\n'
             f'and one APL gain per version sets the mean intact fraction to TNT3c\'s. Model values are means over {len(MB)} mushroom bodies '
             f'({MB_NAMES}); e-h use ORN inputs.', fontsize=8.5, color=MUTED, va='bottom')
    fig.savefig(OUT + 'TNT_vs_model.png', dpi=150)
    print(json.dumps(summary['fits'], indent=1, default=float)[:6000])
    print(f'done ({time.time() - t0:.0f} s)')
