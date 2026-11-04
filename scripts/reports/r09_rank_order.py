"""Report 09: rank order of odor-pair similarity with and without APL (tnt_rank_order.py, rank_order_toy_sim.py).
Rebuilt 2026-11-04 for Yang's revision: corrected calyx, new claws, new ORN table (APL_VARIANT=claws_v2m2), version MC, and the logic of
the noise correction step by step with a simulation. The toy simulations do not use the connectome and are read from odor_weighted/.
Old version in archive_2026-11-03/."""
import sys, json, os
os.environ['APL_VARIANT'] = 'claws_v2m2'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import report_lib as L
import tnt_compare as tc
L.style(); L.DATE = '2026-11-04'
F = L.fmt
OWV = tc.OUT                                  # with_outside_ROI_synapses/claws_v2m2/odor_weighted/
TJ = json.load(open(L.OW + 'rank_order_toy_simulation.json')); TP = pd.read_csv(L.OW + 'rank_order_toy_simulation_pairs.csv')
RO = pd.read_csv(OWV + 'TNT_rank_order.csv'); PR = pd.read_csv(OWV + 'TNT_vs_model_odor_pairs.csv'); PR = PR[PR.threshold == 1.0]
VERS = ['U', 'C', 'P0', 'P', 'L', 'M0', 'M', 'MC', 'F0', 'F']
ro = lambda resp, comp: float(RO[(RO.source == 'data') & (RO.responses == resp) & (RO.comparison == comp)].spearman.iloc[0])

R = L.Report('09', 'Does blocking APL change the rank order of odor-pair similarity, in theory, in the model and in the recordings?',
             inputs=['The recordings (Report 06)', 'The pinned model (Report 07; corrected calyx, new claws, new ORN table)'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/odor_weighted/TNT_rank_order.csv`',
                      '`odor_weighted/rank_order_toy_simulation.png`, `.json`, `_pairs.csv` (no connectome, unchanged)'],
             scripts=['`scripts/tnt_rank_order.py`', '`scripts/rank_order_toy_sim.py`', '`scripts/reports/r09_rank_order.py` (this report)'])
R.p('Reports 06 and 07 look at how much each pair\'s similarity changes. A different question is whether the **order** of the pairs changes: '
    'if pair A is more similar than pair B with APL intact, is it still more similar with APL blocked? A purely global inhibition might be '
    'expected to keep the order. This report shows when it does and when it does not, then compares model and data.')
R.box('What changed in this revision (2026-11-04)', ["New: the logic of the noise correction, step by step, with a simulation.", "Model numbers come from the run with the corrected calyx, the new claws and the new ORN table; version MC is added."], 'plain')
R.symbols([
    ('S^{\\mathrm{on}}_{ab},\\ S^{\\mathrm{off}}_{ab}', 'Cosine similarity of odors $a$, $b$ with inhibition on (APL intact) and off (APL blocked)', '−1 to 1'),
    ('\\rho', 'Spearman rank correlation over the odor pairs between $S^{\\mathrm{on}}$ and $S^{\\mathrm{off}}$', '1 = same order'),
    ('\\tau', 'Kendall rank correlation over the same pairs', '−1 to 1'),
    ('\\text{swap}', 'Two odor pairs whose order differs between on and off', ''),
    ('s_o', 'A divisor applied to every KC\'s response to odor $o$ (case 1)', '> 0'),
    ('\\theta', 'KC threshold', ''),
    ('\\rho_{\\mathrm{within}},\\ \\rho_{\\mathrm{between}}', 'Spearman between two halves of the recordings: same condition, or TNT vs TNT3c', '−1 to 1'),
])

# ---------------------------------------------------------------- 1 measures
R.section('How a change of order is measured')
R.p('With 16 odors there are 120 pairs and 7,140 comparisons between two pairs. Spearman $\\rho$ correlates the ranks of $S^{\\mathrm{on}}$ and '
    '$S^{\\mathrm{off}}$ over the pairs. Kendall $\\tau$ counts comparisons directly; without ties,')
R.eq(r'\tau = 1 - 2\,\frac{\#\,\text{swaps}}{\#\,\text{comparisons}}, \qquad \text{share swapped} = \frac{1 - \tau}{2}')
R.p('$\\rho$ = 1 means no swaps. $\\rho$ is less sensitive than $\\tau$ to swaps between pairs with nearly equal similarity.')

# ---------------------------------------------------------------- 2 theory
R.section('When does global inhibition keep the order?')
R.ul(['**Case 1: inhibition only scales.** If every KC\'s response to odor $o$ is divided by the same number $s_o$ (which may differ '
      'between odors), cosine similarity does not change at all, because cosine ignores the length of each response vector (Equation 2). '
      'The order is kept exactly.',
      '**Case 2: one threshold raise, odors alike.** If inhibition raises one threshold for every odor and KC, and every odor drives the '
      'KCs with the same statistics, then similarity depends only on how correlated the two odors\' inputs are, through one rising curve '
      'at each threshold. The order is kept (up to sampling noise).',
      '**Case 3: odors differ.** If odors differ in strength (3a), the same raise removes a larger share of a weak odor\'s responders, so '
      'pairs of weak odors lose more similarity than pairs of strong ones. If some odors share their overlap mostly in a few strongly '
      'driven KCs (3b), those pairs keep more of their overlap. Either way some pairs swap.'])
R.eq(r'\cos\!\left(\frac{\mathbf{y}_a}{s_a}, \frac{\mathbf{y}_b}{s_b}\right) = \frac{\mathbf{y}_a\cdot\mathbf{y}_b / (s_a s_b)}{\lVert\mathbf{y}_a\rVert\,\lVert\mathbf{y}_b\rVert / (s_a s_b)} = \cos(\mathbf{y}_a, \mathbf{y}_b)')

# ---------------------------------------------------------------- 3 toy
R.section('Toy simulations of the three cases')
R.p('No PN responses and no connectome: 1,000,000 model KCs each get a random input for each of 16 odors. Odors overlap through 4 shared '
    'random inputs (each odor takes its own mix) plus a private input. A KC\'s response is $\\max(\\text{input} - \\theta, 0)$. With APL '
    'blocked 20% of KCs respond to an odor on average, and with APL intact 10%. In case 1 the divisor of each odor is $1 + a A_o$, with '
    '$A_o$ the odor\'s mean response solved self-consistently, and one $a$ for all odors set so the intact/blocked ratio averages 0.5.')
SC = [('1', '1  Scaling only'), ('2', '2  One raise, odors alike'), ('3a', '3a  Odors differ in strength'), ('3b', '3b  Sparse shared overlap')]
PT = {'two weak odors': L.INTACT, 'two strong odors': L.BLOCKED, 'one of each': L.GRAY}


def fig_toy():
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.5))
    for a, (sc, t) in zip(axs, SC):
        P = TP[TP.scenario.astype(str) == sc]
        types = P.pair_type.unique()
        for ty in types:
            Q = P[P.pair_type == ty]
            col = L.INTACT if ('weak' in ty or '1-8' in ty) else L.BLOCKED if ('strong' in ty or '9-16' in ty) else L.GRAY
            a.scatter(Q.cos_blocked, Q.cos_intact, s=8, color=col, lw=0, alpha=0.8, label=ty[:40])
        lim = [min(P.cos_blocked.min(), P.cos_intact.min()), max(P.cos_blocked.max(), P.cos_intact.max())]
        a.plot(lim, lim, color=L.GRID, lw=0.8, zorder=0)
        j = TJ['scenarios'][sc]
        a.set_title(f'{t}\nρ = {j["spearman"]:.3f}, {L.pct(j["share swapped"], 1)} swapped', fontsize=8.5)
        a.set_xlabel('cosine, APL blocked'); a.set_ylabel('cosine, APL intact' if sc == '1' else '')
        if len(types) > 1: a.legend(fontsize=6.5, loc='upper left', markerscale=1.5)
    fig.tight_layout(); return fig


R.fig(fig_toy(), 'Each point is one of the 120 odor pairs of a toy simulation. Horizontal: cosine similarity with APL blocked (20% of KCs '
      'respond per odor). Vertical: with APL intact (10%). The grey line marks equal values. A monotone cloud means the order is kept. '
      'Colors (where odors differ) show whether the pair has two weak odors, two strong ones, or one of each; in 3b, whether it shares '
      'the sparse overlap. Titles give Spearman $\\rho$ and the share of the 7,140 comparisons that swap.', 'Rank order in the toy cases.')
R.ul([f'Scaling: every similarity unchanged to 10⁻¹⁴ ($\\rho$ = 1).',
      f'One raise, odors alike: $\\rho$ = {TJ["scenarios"]["2"]["spearman"]:.4f}; the {TJ["scenarios"]["2"]["swapped comparisons"]} swaps are between pairs less than 0.003 apart (sampling noise).',
      f'Odors differing in strength: $\\rho$ = {TJ["scenarios"]["3a"]["spearman"]:.3f} ({L.pct(TJ["scenarios"]["3a"]["share swapped"], 1)} swapped), and '
      f'{TJ["scenarios"]["3a feedback"]["spearman"]:.3f} ({L.pct(TJ["scenarios"]["3a feedback"]["share swapped"], 1)}) with feedback instead of one raise.',
      f'Sparse shared overlap: $\\rho$ = {TJ["scenarios"]["3b"]["spearman"]:.3f} ({L.pct(TJ["scenarios"]["3b"]["share swapped"], 1)} swapped).',
      'A divisor per KC instead of per odor keeps the order when it is random (ρ 0.9999) but not quite when it grows with the number of odors the KC responds to (ρ 0.995).'])

# ---------------------------------------------------------------- 4 model
R.section('The connectome model')
R.p('In the pinned model of Report 07, $\\rho$ is computed between the blocked and intact similarity over the odor pairs. '
    f'Table {R.n_tab + 1} uses the '
    'similarity averaged over the three mushroom bodies first; Report 11 averages $\\rho$ over the mushroom bodies instead, which gives '
    'slightly lower values.')
rows, RHO = [], {}
for v in VERS:
    r1 = [spearmanr(Q.S_model_blocked, Q.S_model_blocked - Q[f'dS_model_{v}'])[0] for rn in ['ORN', 'GH146'] for Q in [PR[PR.responses == rn]]]
    rows.append([v, F(r1[0], 3), F(r1[1], 3)]); RHO[v] = r1[0]
R.table(['version (Report 05 / 07)', '$\\rho$, ORN input (120 pairs)', '$\\rho$, GH146 input (78 pairs)'], rows,
        title='Rank order kept by every model version.', caption='Spearman over odor pairs between blocked and intact similarity of the pinned model, '
        'similarity averaged over hemibrain R and male CNS L and R.', num_cols=[1, 2])
mU = RO[(RO.source == 'model') & (RO.responses == 'ORN')]
VMIN = min(RHO, key=RHO.get)
R.p('Per mushroom body, uniform feedback gives ' + ', '.join(f'{r.spearman:.3f}' for r in mU[mU.comparison.str.contains('uniform')].itertuples()) +
    ' and a fixed threshold raise ' + ', '.join(f'{r.spearman:.3f}' for r in mU[mU.comparison.str.contains('fixed')].itertuples()) +
    f' (ORN input; hemibrain R, male CNS L, male CNS R). The lowest value is version {VMIN}\'s ({RHO[VMIN]:.3f}); every other version keeps '
    f'$\\rho$ at {min(v_ for k_, v_ in RHO.items() if k_ != VMIN):.3f} or above.')

# ---------------------------------------------------------------- 5 data
R.section('The recordings')
S3, ST = tc.SREC['TNT3c'], tc.SREC['TNT']
rs = np.random.default_rng(3); w3, wt, bt = [], [], []
for _ in range(1000):
    a, b = rs.permutation(len(S3)), rs.permutation(len(ST))
    A3, B3 = S3[a[:5]].mean(0), S3[a[5:10]].mean(0); AT, BT = ST[b[:4]].mean(0), ST[b[4:8]].mean(0)
    w3.append(spearmanr(A3, B3)[0]); wt.append(spearmanr(AT, BT)[0]); bt.append(np.mean([spearmanr(AT, B3)[0], spearmanr(BT, A3)[0]]))
w3, wt, bt = map(np.array, (w3, wt, bt))
full = spearmanr(ST.mean(0), S3.mean(0))[0]
R.p(f'Over the 120 pairs, TNT and TNT3c similarity give $\\rho$ = {full:.3f}, lower than any model version. Part of that is noise: even two '
    'sets of recordings of the same condition do not give $\\rho$ = 1. To separate the two, the recordings are split at random into halves '
    '(5 + 5 of the 11 TNT3c, 4 + 4 of the 8 TNT), 1,000 times. Two halves of the same condition give $\\rho_{\\mathrm{within}}$; a TNT half '
    'against a TNT3c half gives $\\rho_{\\mathrm{between}}$. If blocking APL kept the order, the two would be equal. Their ratio estimates the '
    'rank correlation without noise:')
R.eq(r'\rho_{\mathrm{true}} \approx \frac{\rho_{\mathrm{between}}}{\sqrt{\rho^{\mathrm{TNT}}_{\mathrm{within}}\;\rho^{\mathrm{TNT3c}}_{\mathrm{within}}}}',
     note='the usual correction for attenuation, applied to rank correlations of half-sample means')
R.sub('The logic of the correction')
R.p('Equation 3 is Spearman\'s correction for attenuation. Step by step:')
R.ul(['**1. Every measurement is pattern + noise.** The similarity of a pair measured from a set of recordings is its true value in '
      'that condition plus recording noise. A different set of recordings gives the same true value with different noise.',
      '**2. Noise lowers correlations.** If two measurements $X$ and $Y$ each carry independent noise, their correlation is the correlation '
      'of their true values times a shrink factor: $\\mathrm{corr}(X, Y) = \\mathrm{corr}(T_X, T_Y) \\cdot \\sqrt{\\mathrm{rel}_X\\,\\mathrm{rel}_Y}$, '
      'where rel = (variance of the true values) ÷ (variance of the measurement) is the share of a measurement that is not noise.',
      '**3. The within-condition correlation measures rel.** Two halves of the same condition share the true values and differ only in '
      'noise, so their correlation equals rel: $\\rho^{\\mathrm{TNT}}_{\\mathrm{within}} \\approx \\mathrm{rel}_{\\mathrm{TNT}}$, and the same for TNT3c.',
      '**4. Undo the shrink.** $\\rho_{\\mathrm{between}}$ (a TNT half against a TNT3c half) is the true TNT-vs-TNT3c correlation shrunk by '
      '$\\sqrt{\\mathrm{rel}_{\\mathrm{TNT}}\\,\\mathrm{rel}_{\\mathrm{TNT3c}}}$. Dividing by that factor gives Equation 3.',
      '**5. Why halves on both sides.** The between value is computed from half-size samples, exactly like the within values, so both '
      'carry the same amount of noise and no Spearman–Brown step is needed.'])


def fig_data():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    a = axs[0]; a.scatter(S3.mean(0), ST.mean(0), s=14, color=L.INK, alpha=0.75, lw=0)
    lim = [min(S3.mean(0).min(), 0), 1]; a.plot(lim, lim, color=L.GRID, lw=0.8)
    a.set_xlabel('similarity with APL intact, $S^{TNT3c}_{ab}$'); a.set_ylabel('similarity with APL blocked, $S^{TNT}_{ab}$')
    a.set_title(f'a  Data, all recordings: ρ = {full:.3f}')
    a = axs[1]; bins = np.linspace(0.6, 1.0, 41)
    a.hist(w3, bins=bins, histtype='step', lw=1.6, color=L.INTACT, label=f'within TNT3c (mean {w3.mean():.3f})')
    a.hist(wt, bins=bins, histtype='step', lw=1.6, color=L.BLOCKED, label=f'within TNT (mean {wt.mean():.3f})')
    a.hist(bt, bins=bins, histtype='stepfilled', alpha=0.35, color=L.GRAY, label=f'TNT vs TNT3c (mean {bt.mean():.3f})')
    a.set_xlabel('Spearman ρ between two half-sample means (120 pairs)'); a.set_ylabel('number of splits (of 1,000)')
    a.legend(fontsize=7.5, loc='upper left'); a.set_title('b  Noise vs real reordering')
    fig.tight_layout(); return fig


R.fig(fig_data(), '**a**, Each point is one odor pair: its mean similarity in TNT3c (horizontal) and TNT (vertical); the grey line marks equal '
      'values. All pairs lie above it, and the cloud is less tight than in the model. **b**, Distributions over 1,000 random splits of the '
      'recordings into halves: two halves of TNT3c (blue), two halves of TNT (orange), and a TNT half against a TNT3c half (grey). The '
      'between-condition values sit lower, so blocking APL changes the order more than noise does.', 'Rank order in the recordings.')
corr = bt.mean() / np.sqrt(w3.mean() * wt.mean())
R.ul([f'$\\rho_{{\\mathrm{{between}}}}$ = {bt.mean():.3f} vs $\\rho_{{\\mathrm{{within}}}}$ = {w3.mean():.3f} (TNT3c) and {wt.mean():.3f} (TNT); the between '
      f'value is lower than both within values in {L.pct(np.mean((bt < w3) & (bt < wt)), 1)} of splits.',
      f'Corrected for noise, $\\rho_{{\\mathrm{{true}}}}$ ≈ {corr:.2f}. The model versions give {min(RHO.values()):.2f}–{max(RHO.values()):.2f}, so '
      + ('**the recordings reorder odor pairs more than uniform or connectome-weighted feedback inhibition on the KCs predicts**.' if corr < min(RHO.values())
         else 'some model versions reorder as much as the recordings.'),
      f'Counting only responses above 1 (0/1) weakens the effect: {ro("responses > 1 (0/1)", "half vs half, TNT vs TNT3c"):.2f} between vs '
      f'{ro("responses > 1 (0/1)", "half vs half, within TNT3c (mean of 1000 splits)"):.2f} and {ro("responses > 1 (0/1)", "half vs half, within TNT"):.2f} within, '
      f'lower in {L.pct(ro("responses > 1 (0/1)", "fraction of splits with between < both withins"), 0)} of splits.'])
rs9 = np.random.default_rng(9)


def sim9(rt, rel, n=1000):
    sd = np.sqrt(1 / rel - 1); acc = []
    for _ in range(n):
        TC = rs9.standard_normal(120); TT = rt * TC + np.sqrt(1 - rt ** 2) * rs9.standard_normal(120)
        c1, c2 = TC + sd * rs9.standard_normal(120), TC + sd * rs9.standard_normal(120)
        t1, t2 = TT + sd * rs9.standard_normal(120), TT + sd * rs9.standard_normal(120)
        wC, wT, bw = spearmanr(c1, c2)[0], spearmanr(t1, t2)[0], spearmanr(t1, c1)[0]
        acc.append((wC, wT, bw, bw / np.sqrt(wC * wT), spearmanr(TT, TC)[0]))
    return np.mean(acc, 0)


wdat = (w3.mean() + wt.mean()) / 2
RELG = np.linspace(0.85, 0.98, 27); REL9 = RELG[np.argmin([abs(sim9(1.0, r_, 200)[0] - wdat) for r_ in RELG])]
RTG = np.linspace(0.80, 1.0, 41); RT9 = RTG[np.argmin([abs(sim9(r_, REL9, 200)[2] - bt.mean()) for r_ in RTG])]
tab9 = []
for rt in (1.0, RT9):
    a9 = sim9(rt, REL9)
    tab9.append([f'{rt:.3f}', f'{a9[4]:.3f}', f'{a9[0]:.3f}', f'{a9[1]:.3f}', f'{a9[2]:.3f}', f'**{a9[3]:.3f}**'])
tab9.append(['the data', '?', f'{w3.mean():.3f}', f'{wt.mean():.3f}', f'{bt.mean():.3f}', f'**{corr:.3f}**'])
R.table(['true correlation (Pearson)', 'true ρ (ranks)', '$\\rho^{\\mathrm{TNT3c}}_{\\mathrm{within}}$', '$\\rho^{\\mathrm{TNT}}_{\\mathrm{within}}$',
         '$\\rho_{\\mathrm{between}}$', 'corrected'], tab9,
        f'Mock data: 120 pairs, a true TNT3c pattern and a true TNT pattern that correlate as in column 1, and noise in each half sized '
        f'(rel = {REL9:.3f}) so that two halves of one condition correlate as in the recordings. Means of 1,000 simulations. In the first row '
        'blocking APL keeps the order perfectly, yet the between value is below 1 because of noise; the correction brings it back near 1. '
        'In the second row the true correlation is chosen so that the between value matches the data\'s, and the correction recovers it closely.',
        'The correction in a simulation.')
R.p(f'So a between value of {bt.mean():.2f} with within values of about {wdat:.2f} is what a true rank correlation of about {corr:.2f} looks '
    'like through this much noise. The correction is exact for Pearson correlations with independent noise and a close approximation for '
    'Spearman, as the table shows.')
R.box('What could reorder more', ['Inhibition that changes the input pattern differently for each odor, rather than one threshold per odor. '
      f'In the model, the version that reorders most is {VMIN} (ρ = {RHO[VMIN]:.3f}). Report 11 tests a local, non-spiking APL on KCs and boutons.',
      'Differences between the recorded flies and the connectome flies, or response nonlinearities not in the model.'], 'plain')
R.save()
