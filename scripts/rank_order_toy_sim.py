"""Toy simulations of the three rank-order cases in odor_weighted/TNT_vs_model_notes.md ("Rank order", "In principle";
Yang asked for them on 2026-10-21).

Completely hypothetical: no PN responses and no connectome. Each of N model KCs gets a random input for each of 16 odors.
Odors overlap through 4 shared random inputs (each odor takes its own mix of them) plus a private input. KC response =
max(input - threshold, 0). Similarity of two odors = cosine of their KC responses, as in the TNT comparison. APL blocked:
20% of KCs respond to an odor on average; APL intact: 10% (as in the connectome model).

  1   Inhibition only scales: each odor's responses are divided by 1 + a x (that odor's APL activity, its mean
      response). The inputs are those of 3a (odors differ in strength), so strong odors are divided more.
      Per-unit versions (Yang asked, 2026-10-22; in the JSON and the pairs table, not in the figure): KC k's responses
      to every odor are divided by its own pi_k instead, (i) random in the same range as the per-odor divisors, or
      (ii) 1 + c x (number of odors KC k responds to with APL blocked); or (iii) by 1 + a x u_k x A_o, u_k random
      (mean 1, CV 0.5), with a and A_o of case 1. In (ii), c is set so half the total response is kept, as in case 1.
  2   One threshold raise for every odor, and every odor's inputs have the same statistics (standard normal, jointly
      Gaussian). Similarity is then a rising function of the input correlation at both thresholds.
  3a  As 2, but each odor's inputs are multiplied by its own strength (0.65 to 1.5). Also run with feedback (each
      odor's threshold rises with its own APL activity).
  3b  As 2, but for odors 9-16 the shared input is sparse: max(s - 1.5, 0), large in a few KCs and zero elsewhere.
      Every odor's inputs are then mapped onto exactly the same set of values, so all odors stay alike in strength.

A swap: of two odor pairs, one is more similar with APL blocked and the other with APL intact. The 120 pairs give 7,140
such comparisons.
Outputs: odor_weighted/rank_order_toy_simulation.png, rank_order_toy_simulation.json, rank_order_toy_simulation_pairs.csv.
Run: python3 -I rank_order_toy_sim.py (under a minute, about 2 GB of memory)."""
import json
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, norm
from scipy.integrate import quad

OUT = '/mnt/project-files/apl_synapses/odor_weighted/'
N, M = 1_000_000, 16
FB, FI = 0.20, 0.10                      # fraction of KCs responding per odor on average: APL blocked, intact
KAPPA = 1.5                              # 3b: sparse shared input = max(s - KAPPA, 0)
IU = np.triu_indices(M, 1); PI, PJ = IU
NAMES = [f'odor {i + 1}' for i in range(M)]
rng = np.random.default_rng(5)
INK, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#898781'


def cos(Y):
    n = np.linalg.norm(Y, axis=0)
    return (Y.T @ Y) / np.outer(n, n)


def bisect(f, target, lo, hi, n=80):
    """x with f(x) = target, for f decreasing."""
    for _ in range(n):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if f(m) > target else (lo, m)
    return (lo + hi) / 2


class Sorted:
    """Per-odor sorted inputs, for the fraction responding and the mean response at any per-odor threshold."""
    def __init__(self, X):
        self.S = np.sort(X, axis=0)
        self.T = np.vstack([np.cumsum(self.S[::-1], axis=0)[::-1], np.zeros((1, X.shape[1]))])   # T[i] = sum of S[i:]

    def idx(self, t):
        return np.array([np.searchsorted(self.S[:, j], t[j], 'right') for j in range(self.S.shape[1])])

    def frac(self, t):
        return (N - self.idx(t)) / N

    def mean(self, t):
        i = self.idx(t)
        return (self.T[i, np.arange(len(t))] - t * (N - i)) / N


def stats(sb, si):
    """Spearman over pairs and swaps between pairs."""
    db, di = (sb[:, None] - sb[None])[np.triu_indices(len(sb), 1)], (si[:, None] - si[None])[np.triu_indices(len(sb), 1)]
    sw = np.sign(db) != np.sign(di)
    return {'spearman': float(spearmanr(sb, si)[0]), 'kendall_tau': float(kendalltau(sb, si)[0]),
            'swapped comparisons': int(sw.sum()), 'comparisons': int(len(sw)), 'share swapped': float(sw.mean()),
            'largest similarity gap between two swapped pairs, APL blocked': float(np.abs(db[sw]).max()) if sw.any() else 0.0,
            'largest similarity gap between two swapped pairs, APL intact': float(np.abs(di[sw]).max()) if sw.any() else 0.0}


def popcos(rho, t):
    """Cosine of max(X - t, 0) and max(Y - t, 0) for standard bivariate normal X, Y with correlation rho (infinitely many KCs)."""
    s = np.sqrt(1 - rho ** 2)
    f = lambda x: (x - t) * norm.pdf(x) * (s * norm.pdf((rho * x - t) / s) + (rho * x - t) * norm.cdf((rho * x - t) / s))
    return quad(f, t, np.inf)[0] / ((1 + t * t) * norm.sf(t) - t * norm.pdf(t))


# ---------------- inputs ----------------
F = rng.standard_normal((N, 4))                                   # shared inputs
L = rng.random((M, 4)) ** 2; L /= np.linalg.norm(L, axis=1, keepdims=True)
C = rng.uniform(0.2, 0.95, M)                                     # weight of the shared part
E = rng.standard_normal((N, M))                                   # private inputs
U = F @ L.T                                                       # each odor's shared input, standard normal
del F
Z = np.sqrt(C) * U + np.sqrt(1 - C) * E                           # 2: odors alike
Z = (Z - Z.mean(0)) / Z.std(0)
RHO = np.corrcoef(Z.T)[IU]
G = np.geomspace(0.65, 1.5, M); rng.shuffle(G)                    # 3a and 1: odor strengths
X = Z * G
SPARSE = np.arange(M) >= 8                                        # 3b: odors 9-16 get a sparse shared input
V = U.copy()
V[:, SPARSE] = np.maximum(U[:, SPARSE] - KAPPA, 0)
V = (V - V.mean(0)) / V.std(0)
raw = np.sqrt(C) * V + np.sqrt(1 - C) * E
del U, V, E
Q = norm.ppf((np.arange(N) + 0.5) / N)                            # the same set of input values for every odor
Z3 = np.empty_like(raw)
for o in range(M):
    Z3[np.argsort(raw[:, o]), o] = Q
del raw
RHO3 = np.corrcoef(Z3.T)[IU]

# ---------------- scenarios ----------------
R = {}
# 2: one threshold for all odors, odors alike
t2b = bisect(lambda t: (Z > t).mean(), FB, 0, 5); t2i = bisect(lambda t: (Z > t).mean(), FI, 0, 5)
Yb, Yi = np.maximum(Z - t2b, 0), np.maximum(Z - t2i, 0)
R['2'] = dict(sb=cos(Yb)[IU], si=cos(Yi)[IU], fb=(Yb > 0).mean(0), fi=(Yi > 0).mean(0))
del Yb, Yi

# 3a: one threshold for all odors, odors differ in strength
SX = Sorted(X)
t3b = bisect(lambda t: SX.frac(np.full(M, t)).mean(), FB, 0, 10)
t3i = bisect(lambda t: SX.frac(np.full(M, t)).mean(), FI, 0, 10)
Yb3 = np.maximum(X - t3b, 0)
Yi = np.maximum(X - t3i, 0)
R['3a'] = dict(sb=cos(Yb3)[IU], si=cos(Yi)[IU], fb=(Yb3 > 0).mean(0), fi=(Yi > 0).mean(0))
del Yi


def feedback_A(a):
    """APL activity per odor with uniform feedback: A = mean response at threshold t3b + a A, solved per odor."""
    lo, hi = np.zeros(M), SX.mean(np.full(M, t3b)) + 1e-12
    for _ in range(80):
        A = (lo + hi) / 2
        gap = A - SX.mean(t3b + a * A)
        hi = np.where(gap > 0, A, hi); lo = np.where(gap > 0, lo, A)
    return (lo + hi) / 2


a_fb = bisect(lambda a: SX.frac(t3b + a * feedback_A(a)).mean(), FI, 0, 1e4)
Yi = np.maximum(X - (t3b + a_fb * feedback_A(a_fb)), 0)
R['3a feedback'] = dict(sb=R['3a']['sb'], si=cos(Yi)[IU], fb=R['3a']['fb'], fi=(Yi > 0).mean(0))
del Yi, SX

# 1: the blocked responses of 3a divided by 1 + a A per odor, A = mean intact response (closed form)
mb = Yb3.mean(0)
Aof = lambda a: (np.sqrt(1 + 4 * a * mb) - 1) / (2 * a)
a1 = bisect(lambda a: np.mean(1 / (1 + a * Aof(a))), 0.5, 1e-6, 1e4)      # responses halved on average
S1 = 1 / (1 + a1 * Aof(a1))
R['1'] = dict(sb=R['3a']['sb'], si=cos(Yb3 * S1)[IU], fb=R['3a']['fb'], fi=R['3a']['fb'])

# 1 with a factor per KC instead of per odor. Own random generator, so the draws below and the figure stay the same.
rk = np.random.default_rng(11)
NK = (Yb3 > 0).sum(1)                                       # number of odors each KC responds to, APL blocked
kept = lambda D: float((Yb3 / D).sum() / Yb3.sum())         # share of the total response left with APL intact
cK = bisect(lambda c: kept((1 + c * NK)[:, None]), 0.5, 0, 100)
sig = np.sqrt(np.log(1 + 0.5 ** 2)); UK = rk.lognormal(-sig ** 2 / 2, sig, N)        # mean 1, CV 0.5
PIK = {'1, pi_k random': rk.uniform(1 / S1.max(), 1 / S1.min(), N)[:, None],
       '1, pi_k grows with the odors KC k responds to': (1 + cK * NK)[:, None],
       '1, 1 + a u_k A_o, u_k random': 1 + a1 * UK[:, None] * Aof(a1)[None]}
R1U = {k: dict(sb=R['3a']['sb'], si=cos(Yb3 / D)[IU], kept=kept(D), rng=[float(D[NK > 0].min()), float(D[NK > 0].max())])
       for k, D in PIK.items()}
del PIK

# 3b: one threshold for all odors; every odor has exactly the same input values
t3bb = (Q[int(N * (1 - FB)) - 1] + Q[int(N * (1 - FB))]) / 2
t3bi = (Q[int(N * (1 - FI)) - 1] + Q[int(N * (1 - FI))]) / 2
Yb, Yi = np.maximum(Z3 - t3bb, 0), np.maximum(Z3 - t3bi, 0)
R['3b'] = dict(sb=cos(Yb)[IU], si=cos(Yi)[IU], fb=(Yb > 0).mean(0), fi=(Yi > 0).mean(0))
del Yb, Yi

ST = {k: stats(v['sb'], v['si']) for k, v in R.items()}
for k in ST:
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in ST[k].items()})

# pair types
strong = G >= np.median(G)
TYPE3A = np.where(strong[PI] & strong[PJ], 'two strong odors', np.where(~strong[PI] & ~strong[PJ], 'two weak odors', 'one of each'))
TYPE3B = np.where(SPARSE[PI] & SPARSE[PJ], 'both odors 9-16 (sparse)', np.where(~SPARSE[PI] & ~SPARSE[PJ], 'both odors 1-8', 'one of each'))


def clearest(key, types, want):
    """Clearest swap in which a pair of type want[0] (P) is less similar than one of type want[1] (Q) with APL blocked
    and more similar with APL intact (largest smaller gap)."""
    sb, si = R[key]['sb'], R[key]['si']
    best, out = 0, None
    for i in np.where(types == want[0])[0]:
        for j in np.where(types == want[1])[0]:
            db, di = sb[j] - sb[i], si[i] - si[j]
            if db > 0 and di > 0 and min(db, di) > best:
                best, out = min(db, di), (int(i), int(j))
    return out


SW3A = clearest('3a', TYPE3A, ('two strong odors', 'two weak odors'))
SW3B = clearest('3b', TYPE3B, ('both odors 9-16 (sparse)', 'both odors 1-8'))
pairname = lambda k: f'{NAMES[PI[k]]}-{NAMES[PJ[k]]}'
short = lambda k: f'{PI[k] + 1} and {PJ[k] + 1}'

# 2: the infinitely-many-KC curve, and how far the simulated similarities sit from it
RR = np.linspace(0, 0.97, 80)
CB = np.array([popcos(r, t2b) for r in RR]); CI = np.array([popcos(r, t2i) for r in RR])
dev_b = R['2']['sb'] - np.interp(RHO, RR, CB); dev_i = R['2']['si'] - np.interp(RHO, RR, CI)

# ---------------- outputs ----------------
rows = []
for key, types, rho in [('1', TYPE3A, RHO), ('2', np.array(['odors alike'] * len(RHO)), RHO), ('3a', TYPE3A, RHO),
                        ('3a feedback', TYPE3A, RHO), ('3b', TYPE3B, RHO3)]:
    for k in range(len(RHO)):
        rows.append(dict(scenario=key, odor_a=NAMES[PI[k]], odor_b=NAMES[PJ[k]], pair_type=types[k],
                         input_correlation=rho[k], cos_blocked=R[key]['sb'][k], cos_intact=R[key]['si'][k]))
for key, v in R1U.items():
    for k in range(len(RHO)):
        rows.append(dict(scenario=key, odor_a=NAMES[PI[k]], odor_b=NAMES[PJ[k]], pair_type=TYPE3A[k],
                         input_correlation=RHO[k], cos_blocked=v['sb'][k], cos_intact=v['si'][k]))
pd.DataFrame(rows).round(5).to_csv(OUT + 'rank_order_toy_simulation_pairs.csv', index=False)

r3 = lambda v: round(float(v), 3)
summary = {
    'setup': {'model KCs': N, 'odors': M, 'pairs': len(RHO), 'comparisons between pairs': len(RHO) * (len(RHO) - 1) // 2,
              'fraction responding per odor, mean: APL blocked / intact': [FB, FI],
              'similarity': 'cosine of two odors\' KC responses', 'response': 'max(input - threshold, 0)',
              'input correlation range, 2 (odors alike)': [r3(RHO.min()), r3(RHO.max())],
              'odor strengths (3a and 1)': [r3(G.min()), r3(G.max())],
              'input correlation range, 3b': [r3(RHO3.min()), r3(RHO3.max())],
              '3b sparse shared input': f'max(s - {KAPPA}, 0) for odors 9-16'},
    'scenarios': {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in ST.items()},
    '1: divisor per odor (min, max)': [round(float(1 / S1.max()), 2), round(float(1 / S1.min()), 2)],
    '1: largest |cos intact - cos blocked|': float(np.abs(R['1']['si'] - R['1']['sb']).max()),
    '1 with a factor per KC (not in the figure)': {k: {
        **{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in stats(v['sb'], v['si']).items()},
        'change in cosine (intact - blocked): mean, min, max': [round(float(d), 4) for d in
                                                                 ((v['si'] - v['sb']).mean(), (v['si'] - v['sb']).min(), (v['si'] - v['sb']).max())],
        'divisor range over KCs that respond to any odor': [round(x, 2) for x in v['rng']],
        'share of the total response kept': round(v['kept'], 3)} for k, v in R1U.items()},
    '1 with a factor per KC: c in 1 + c x (number of odors)': round(float(cK), 3),
    '2: largest distance of a simulated similarity from the infinite-KC curve, blocked / intact': [round(float(np.abs(dev_b).max()), 4), round(float(np.abs(dev_i).max()), 4)],
    '3a: fraction responding blocked -> intact, one raise': {
        'strongest odor': [r3(R['3a']['fb'][np.argmax(G)]), r3(R['3a']['fi'][np.argmax(G)])],
        'weakest odor': [r3(R['3a']['fb'][np.argmin(G)]), r3(R['3a']['fi'][np.argmin(G)])]},
    '3a: fraction responding blocked -> intact, feedback': {
        'strongest odor': [r3(R['3a feedback']['fb'][np.argmax(G)]), r3(R['3a feedback']['fi'][np.argmax(G)])],
        'weakest odor': [r3(R['3a feedback']['fb'][np.argmin(G)]), r3(R['3a feedback']['fi'][np.argmin(G)])]},
    '3a: mean change in cosine (intact - blocked) by pair type': {t: r3((R['3a']['si'] - R['3a']['sb'])[TYPE3A == t].mean()) for t in np.unique(TYPE3A)},
    '3a: example swap P, Q: cos blocked, intact': {pairname(k): [r3(R['3a']['sb'][k]), r3(R['3a']['si'][k])] for k in SW3A},
    '3b: mean change in cosine (intact - blocked) by pair type': {t: r3((R['3b']['si'] - R['3b']['sb'])[TYPE3B == t].mean()) for t in np.unique(TYPE3B)},
    '3b: pairs more similar with APL intact': int((R['3b']['si'] > R['3b']['sb']).sum()),
    '3b: example swap P, Q: cos blocked, intact': {pairname(k): [r3(R['3b']['sb'][k]), r3(R['3b']['si'][k])] for k in SW3B},
}
json.dump(summary, open(OUT + 'rank_order_toy_simulation.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != 'scenarios'}, indent=1))

# ---------------- figure ----------------
# 3b, panel d: where the KCs that respond to both odors of P and of Q sit (same random KCs for both pairs)
SUBD = rng.choice(N, 12000, replace=False)
for p in SW3B:
    a, b = Z3[:, PI[p]], Z3[:, PJ[p]]
    ya, yb = np.maximum(a - t3bb, 0), np.maximum(b - t3bb, 0)
    both = (ya > 0) & (yb > 0)
    summary.setdefault('3b: example pairs, share of an odor\'s responders that also respond to the other odor, APL blocked / intact', {})[pairname(p)] = [
        r3(both.mean() / FB), r3(((a > t3bi) & (b > t3bi)).mean() / FI)]
    za, zb = np.maximum(a - t3bi, 0), np.maximum(b - t3bi, 0)
    summary.setdefault('3b: example pairs, overlap (sum over KCs of the product of the two responses) kept with APL intact', {})[pairname(p)] = r3((za @ zb) / (ya @ yb))
    summary.setdefault('3b: example pairs, correlation of the two odors\' inputs among KCs responding to both (APL blocked)', {})[pairname(p)] = r3(np.corrcoef(a[both], b[both])[0, 1])
OWN = float((np.maximum(Q - t3bi, 0) ** 2).sum() / (np.maximum(Q - t3bb, 0) ** 2).sum())
summary['3b: every odor keeps this share of its own sum of squared responses with APL intact'] = r3(OWN)
json.dump(summary, open(OUT + 'rank_order_toy_simulation.json', 'w'), indent=1)

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9.5})
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.12], width_ratios=[1, 1, 1, 1.3], hspace=0.55, wspace=0.3, top=0.795, bottom=0.15,
                      left=0.045, right=0.985)
AX0 = [fig.add_subplot(gs[0, k]) for k in range(3)]
gd = gs[0, 3].subgridspec(1, 2, wspace=0.1)
AXD = [fig.add_subplot(gd[0]), fig.add_subplot(gd[1])]
AX0.append(AXD[0])
AX1 = [fig.add_subplot(gs[1, k]) for k in range(4)]


def style(ax, grid=True):
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if grid: ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)


HEAD = ['1  Inhibition only scales', '2  One threshold raise, odors alike', '3a  Odors differ in strength', '3b  Overlap in strongly driven KCs']
SUB = ['Each odor\'s responses are divided by one number,\nlarger for odors that drive APL more.',
       'Every odor drives the KCs with the same statistics\n(Gaussian inputs with the same spread).',
       'One threshold raise as in 2, but each odor\'s inputs\nare scaled by its own strength (×0.65 to ×1.5).',
       'One threshold raise and odors alike in strength, but\nfor odors 9–16 the shared input is sparse: large in a\nfew KCs, zero in the rest.']
for k, ax in enumerate(AX0):
    x0 = ax.get_position().x0 - 0.012
    fig.text(x0, 0.892, HEAD[k], fontsize=11.5, fontweight='bold', color=INK, ha='left', va='bottom')
    fig.text(x0, 0.886, SUB[k], fontsize=9.3, color=MUTED, ha='left', va='top', linespacing=1.3)


def vline_labels(ax, xb, xi, y=0.985, flip=False):
    """Labels for the blocked (dashed) and intact (solid) lines, at the top of the panel."""
    tr = ax.get_xaxis_transform()
    ax.text(xb, y, 'blocked ', transform=tr, ha='left' if flip else 'right', va='top', fontsize=8, color=INK)
    ax.text(xi, y, ' intact', transform=tr, ha='right' if flip else 'left', va='top', fontsize=8, color=INK)


# a: scaling, per-KC responses of three odors
ax = AX0[0]; style(ax)
pick = [int(np.argmax(G)), int(np.argsort(G)[M // 2]), int(np.argmin(G))]
xmax = 0
for o, lab in zip(pick, ['strongest odor', 'middle odor', 'weakest odor']):
    ix = np.where(Yb3[:, o] > 0)[0]; ix = rng.choice(ix, 250, replace=False)
    ax.scatter(Yb3[ix, o], Yb3[ix, o] * S1[o], s=7, color=INK, alpha=0.35, lw=0)
    xm = np.quantile(Yb3[Yb3[:, o] > 0, o], 0.999); xmax = max(xmax, xm)
    ax.plot([0, xm], [0, xm * S1[o]], color=INK, lw=1.2)
    ax.annotate(f'{lab}: ÷{1 / S1[o]:.1f}', xy=(xm, xm * S1[o]), xytext=(4, 0), textcoords='offset points', fontsize=8.5,
                color=INK, va='center', ha='left')
ax.set_xlim(0, xmax * 1.75); ax.set_ylim(0, None)
ax.set_xlabel('KC response, APL blocked', fontsize=9, color=MUTED); ax.set_ylabel('KC response, APL intact', fontsize=9, color=MUTED)
ax.set_title('a  Each KC keeps its share of the odor\'s\nresponse, and no KC stops responding', fontsize=9.5, color=INK, loc='left')

# b: cosine vs input correlation at the two thresholds
ax = AX0[1]; style(ax)
ax.plot(RR, CB, color=INK, lw=1.4, ls=(0, (4, 2.5))); ax.plot(RR, CI, color=INK, lw=1.6)
ax.scatter(RHO, R['2']['sb'], s=16, facecolor='white', edgecolor=INK, lw=0.9, zorder=3)
ax.scatter(RHO, R['2']['si'], s=12, color=INK, lw=0, zorder=3)
ax.annotate('APL blocked\n(20% respond)', xy=(0.7, np.interp(0.7, RR, CB)), xytext=(-10, 12), textcoords='offset points', ha='right',
            fontsize=8.5, color=INK, linespacing=1.2)
ax.annotate('APL intact\n(10% respond)', xy=(0.7, np.interp(0.7, RR, CI)), xytext=(10, -10), textcoords='offset points', ha='left', va='top',
            fontsize=8.5, color=INK, linespacing=1.2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel('input correlation of the two odors (over KCs)', fontsize=9, color=MUTED)
ax.set_ylabel('similarity of KC responses (cosine)', fontsize=9, color=MUTED)
ax.set_title('b  Similarity depends only on input correlation:\none rising curve per threshold (dots: odor pairs)', fontsize=9.5, color=INK, loc='left')

# c: input distributions of the strongest and weakest odor, with both thresholds
ax = AX0[2]; style(ax, grid=False)
xs = np.linspace(-4.5, 5, 400)
lines, top = [], 0
for o, colr, lab, xl in [(int(np.argmax(G)), BLUE, 'strongest odor', -1.9), (int(np.argmin(G)), ORANGE, 'weakest odor', -0.62)]:
    yv = norm.pdf(xs / G[o]) / G[o]; top = max(top, yv.max())
    ax.plot(xs, yv, color=colr, lw=1.8)
    ax.annotate(lab, xy=(xl, norm.pdf(xl / G[o]) / G[o]), xytext=(-5, 2), textcoords='offset points', ha='right', fontsize=8.5, color=INK)
    fb_, fi_ = R['3a']['fb'][o], R['3a']['fi'][o]
    lines.append(f'{lab}: {fb_:.0%} → {fi_:.0%} of KCs respond (keeps {fi_ / fb_:.0%})')
ax.axvline(t3b, color=INK, lw=1.3, ls=(0, (4, 2.5)), ymax=0.74); ax.axvline(t3i, color=INK, lw=1.4, ymax=0.74)
vline_labels(ax, t3b, t3i, y=0.79)
ax.text(0.5, 0.995, '\n'.join(lines), transform=ax.transAxes, ha='center', va='top', fontsize=8.5, color=INK, linespacing=1.4)
ax.set_xlim(-4.5, 5); ax.set_ylim(0, top / 0.72); ax.set_yticks([]); ax.set_xticks([])
ax.set_xlabel('input to a KC (low → high)', fontsize=9, color=MUTED); ax.set_ylabel('share of KCs', fontsize=9, color=MUTED)
ax.set_title('c  The same raise silences a larger share\nof a weak odor\'s responders', fontsize=9.5, color=INK, loc='left')

# d: KCs responding to both odors of P and of Q with APL blocked (the same random KCs for both pairs)
KEPT = summary['3b: example pairs, overlap (sum over KCs of the product of the two responses) kept with APL intact']
lo_, hi_ = t3bb - 0.2, 4.2
for k, (ax, p, colr, lab) in enumerate(zip(AXD, SW3B, [BLUE, ORANGE], ['P', 'Q'])):
    style(ax, grid=False)
    a, b = Z3[SUBD, PI[p]], Z3[SUBD, PJ[p]]
    both = (a > t3bb) & (b > t3bb)
    ax.scatter(a[both], b[both], s=6, color=colr, alpha=0.8, lw=0, zorder=3)
    for t, ls in [(t3bb, (0, (4, 2.5))), (t3bi, '-')]:
        ax.plot([t, t], [t, hi_], color=INK, lw=1.1, ls=ls, zorder=4); ax.plot([t, hi_], [t, t], color=INK, lw=1.1, ls=ls, zorder=4)
    ax.text(hi_, t3bb - 0.04, 'blocked', ha='right', va='top', fontsize=7.5, color=INK)
    ax.text(hi_, t3bi + 0.04, 'intact', ha='right', va='bottom', fontsize=7.5, color=INK,
            bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.85), zorder=6)
    ax.text(0.97, 0.97, f'{lab}: odors {short(p)}\n{both.sum()} KCs; overlap\nkept: {KEPT[pairname(p)]:.0%}', transform=ax.transAxes, ha='right',
            va='top', fontsize=8.3, color=INK, linespacing=1.3, bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none', alpha=0.85), zorder=6)
    ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_); ax.set_aspect('equal')
    ax.set_xticks([1, 2, 3, 4]); ax.set_yticks([1, 2, 3, 4])
    ax.set_xlabel('input, one odor', fontsize=8.5, color=MUTED)
    if k == 0:
        ax.set_ylabel('input, the other odor', fontsize=8.5, color=MUTED)
    else:
        ax.set_yticklabels([])
AXD[0].set_title('d  KCs responding to both odors (APL blocked). P\'s are\noften driven hard by both, Q\'s by just one. '
                 f'Each odor keeps {OWN:.0%}\nof its own response (sum of squares) with APL intact', fontsize=9.5, color=INK, loc='left')

# e-h: blocked vs intact similarity of every pair
TITLES = ['e  Every pair keeps its similarity:\norder kept exactly', 'f  Every pair loses similarity along one\ncurve, so none overtakes another: order kept',
          'g  Pairs of weak odors lose more:\nsome pairs swap', 'h  Pairs that share a sparse input lose\nless: some pairs swap']
LIM = 0.85
for k, key in enumerate(['1', '2', '3a', '3b']):
    ax = AX1[k]; style(ax)
    ax.plot([0, LIM], [0, LIM], color=BASE, lw=1, zorder=1)
    if k:
        ax.plot(CB, CI, color=GRAY, lw=1.3, zorder=1)
    sb, si = R[key]['sb'], R[key]['si']
    if key in ('1', '2'):
        ax.scatter(sb, si, s=16, color=INK, lw=0, alpha=0.8, zorder=3)
    else:
        types = TYPE3A if key == '3a' else TYPE3B
        order = [('one of each', GRAY), ('two weak odors', ORANGE), ('two strong odors', BLUE)] if key == '3a' else \
                [('one of each', GRAY), ('both odors 1-8', ORANGE), ('both odors 9-16 (sparse)', BLUE)]
        for t, colr in order:
            m = types == t
            ax.scatter(sb[m], si[m], s=18, color=colr, lw=0.6, edgecolor='white', zorder=3, label=f'{t.replace("-", "–")} ({m.sum()})')
        sw_ = SW3A if key == '3a' else SW3B
        for p, lab in zip(sw_, ['P', 'Q']):
            ax.scatter([sb[p]], [si[p]], s=70, facecolor='none', edgecolor=INK, lw=1.3, zorder=4)
            ax.annotate(lab, xy=(sb[p], si[p]), xytext=(7, -2) if lab == 'Q' else (-7, 3), textcoords='offset points', fontsize=10,
                        fontweight='bold', color=INK, ha='left' if lab == 'Q' else 'right', va='top' if lab == 'Q' else 'bottom')
        ax.legend(loc='lower right', fontsize=8, frameon=False, handletextpad=0.2, borderaxespad=0.2, title='pairs of', title_fontsize=8,
                  alignment='left')
        # inset: P and Q, blocked -> intact
        ins = ax.inset_axes([0.1, 0.40, 0.25, 0.26])
        for p, lab, colr in zip(sw_, ['P', 'Q'], [BLUE, ORANGE]):
            ins.plot([0, 1], [sb[p], si[p]], color=colr, lw=1.8, marker='o', ms=4)
            ins.text(-0.12, sb[p], lab, ha='right', va='center', fontsize=8.5, fontweight='bold', color=INK)
        ins.set_xlim(-0.35, 1.15); ins.set_xticks([0, 1]); ins.set_xticklabels(['blocked', 'intact'], fontsize=7.5, color=MUTED)
        ins.tick_params(axis='y', labelsize=7, colors=MUTED, length=2); ins.tick_params(axis='x', length=0)
        for sp in ['top', 'right']: ins.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']: ins.spines[sp].set_color(BASE)
        ins.set_facecolor('none')
    s = ST[key]
    txt = f'Spearman {s["spearman"]:.3f}\nswaps: {s["swapped comparisons"]:,} of {s["comparisons"]:,} ({s["share swapped"]:.1%})'
    if key == '2':
        txt += (f'\nall between pairs < {s["largest similarity gap between two swapped pairs, APL blocked"]:.3f} apart, within\n'
                f'the scatter of simulated KCs around the curve')
    if key == '3a':
        txt += f'\nwith feedback instead of one raise:\nSpearman {ST["3a feedback"]["spearman"]:.3f}, swaps {ST["3a feedback"]["share swapped"]:.1%}'
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha='left', va='top', fontsize=8.8, color=INK, linespacing=1.35)
    if key == '2':
        ax.annotate('curve from b', xy=(np.interp(0.6, RR, CB), np.interp(0.6, RR, CI)), xytext=(10, -22), textcoords='offset points',
                    fontsize=8, color=MUTED, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7))
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect('equal')
    ax.set_xlabel('similarity, APL blocked (cosine)', fontsize=9, color=MUTED)
    ax.set_ylabel('similarity, APL intact (cosine)', fontsize=9, color=MUTED)
    ax.set_title(TITLES[k], fontsize=9.5, color=INK, loc='left')

fig.suptitle('Does uniform inhibition keep the rank order of odor-pair similarity?  Toy simulations without PN responses or connectome: '
             f'{N:,} model KCs, {M} odors, {len(RHO)} odor pairs', fontsize=12.5, color=INK, x=0.033, ha='left', y=0.985)
fig.text(0.033, 0.952, 'Each KC gets a random input for each odor; odors overlap through shared random inputs. KC response = input − threshold, '
         'if positive. APL blocked: 20% of KCs respond to an odor on average; intact: 10%.\nSimilarity = cosine of two odors\' KC responses. '
         'A swap: of two odor pairs, one is more similar with APL blocked and the other with APL intact (7,140 comparisons between the 120 pairs).',
         fontsize=9.5, color=MUTED, ha='left', va='top', linespacing=1.4)
fig.text(0.033, 0.075,
         'e–h: light line, the same similarity with and without APL; grey curve (f–h), where pairs fall when all odors are alike, as in b. '
         f'P and Q (g, h, insets) are the clearest swap of each kind: g, odors {short(SW3A[0])} vs {short(SW3A[1])}; h, odors {short(SW3B[0])} vs {short(SW3B[1])}.\n'
         'Why: in 1 the order cannot change, because cosine ignores how strongly each odor drives the KCs. In 2 every pair sits on the same rising curve at both '
         'thresholds. In 3a and 3b pairs leave that curve.\nIn 3a the raise removes a different share of each odor\'s responders (c). In 3b every odor\'s own response shrinks '
         'alike, so similarity follows the overlap (sum over KCs of the product of the two responses): a KC driven hard by both odors keeps most of its product,\n'
         'one driven hard by one odor and barely by the other loses it. Pairs sharing a sparse input have more KCs of the first kind (d).',
         fontsize=9.3, color=INK, ha='left', va='top', linespacing=1.5)
fig.savefig(OUT + 'rank_order_toy_simulation.png', dpi=150)
print('saved')
