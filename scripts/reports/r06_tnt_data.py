"""Report 06: the TNT and TNT3c recordings, what is measured and what is derived from them (tnt_compare.py, data part).
Rebuilt 2026-11-04 for Yang's revision: responses are FC z-scores; what the per-odor effect means mechanistically (chance overlap)
and its statistics (from the Q&A page of 2026-11-03). The data part does not depend on the connectome; the model numbers quoted
come from the claws_v2m2 run. The old version is in archive_2026-11-03/."""
import sys, json, os
os.environ['APL_VARIANT'] = 'claws_v2m2'
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import report_lib as L
import tnt_compare as tc
L.style(); L.DATE = '2026-11-04'
F = L.fmt
rs6 = np.random.default_rng(11)
O, IU, D, SREC = tc.O, tc.IU, tc.D, tc.SREC
CC = {'TNT': L.BLOCKED, 'TNT3c': L.INTACT}
CN = {'TNT': 'TNT (APL blocked)', 'TNT3c': 'TNT3c (control, APL intact)'}
import apl_block_model as abm
CHK = json.load(open(L.OW + 'TNT_data_checks.json'))                 # data only: the same for every connectome version
RO = pd.read_csv(abm.OUT + 'TNT_rank_order.csv')                       # data rows identical to the official run
TVM = json.load(open(abm.OUT + 'TNT_vs_model_summary.json'))
ro = lambda resp, comp: float(RO[(RO.source == 'data') & (RO.responses == resp) & (RO.comparison == comp)].spearman.iloc[0])

R = L.Report('06', 'What do the TNT and TNT3c recordings contain, and which numbers are taken from them for the comparison with the model?',
             inputs=['`TNT_ovlp16_resp_perfly.pkl` and `TNT3c_ovlp16_resp_perfly.pkl` (Yang, 2026-10-15): per recording, a table of '
                     'units × 16 odors', '`TNT_ovlp16_cosine_mean_RDM.csv`, `TNT3c_ovlp16_cosine_mean_RDM.csv`: Yang\'s mean RDMs, used as a check'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/odor_weighted/TNT_vs_model_per_odor.csv`, `TNT_vs_model_odor_pairs.csv` (data columns), '
                      '`TNT_rank_order.csv` (data rows)', '`odor_weighted/TNT_data_checks.csv` / `.json` (data only)'],
             scripts=['`scripts/tnt_compare.py` (section "data" and the helpers)', '`scripts/tnt_data_checks.py`', '`scripts/tnt_rank_order.py`',
                      '`scripts/reports/r06_tnt_data.py` (this report)'])
R.p('Yang recorded KC responses to 16 odors in two genotypes. In **TNT**, APL\'s synaptic output is blocked (tetanus toxin light chain '
    'expressed in APL). **TNT3c** is the control, with APL intact (Yang, 2026-10-15). This report describes the files, the quantities taken '
    'from them, and how reliable those quantities are. The pickles are read with a restricted loader (`scripts/orn_load.py`) that only '
    'accepts pandas and numpy classes.')
R.box('What changed in this revision (2026-11-04)', ["The responses are FC z-scores (Yang, 2026-11-03); \"responding\" means above 1 z-score unit.", "New: what the per-odor effect means mechanistically, tested against chance overlap computed from the data alone.", "New: the statistics behind the per-odor effects (permutation null for the additive fit, bootstrap ranges, split-half reproducibility).", "Numbers that involve the model now come from the run with the corrected calyx and the new claws."], 'plain')
R.symbols([
    ('c', 'Condition: TNT (APL blocked) or TNT3c (control, APL intact)', ''),
    ('n', 'A recording. Each recording is one hemisphere and is treated as independent (Yang, 2026-10-15)', '8 TNT, 11 TNT3c'),
    ('i', 'A unit (one row of a recording\'s table): a segmented KC response region', '1,638–2,462 per recording'),
    ('o, a, b', 'Odors', '16, the odors of the ORN table'),
    ('r^{(n)}_{io}', 'Response of unit $i$ in recording $n$ to odor $o$, as provided', 'FC z-score (`fc_zscore`); −2.6 to 7.4'),
    ('\\phi^{(n)}_o', 'Fraction of recording $n$\'s units with $r^{(n)}_{io} > 1$ ("responding")', '0 to 1'),
    ('\\phi^c_o', 'Mean of $\\phi^{(n)}_o$ over the recordings of condition $c$', '0 to 1'),
    ('S^{(n)}_{ab}', 'Cosine similarity of odors $a$ and $b$ over the units of recording $n$', '−1 to 1'),
    ('S^c_{ab}', 'Mean of $S^{(n)}_{ab}$ over the recordings of condition $c$ (= 1 − Yang\'s RDM)', '−1 to 1'),
    ('\\Delta S_{ab}', 'Effect of blocking APL on the pair: $S^{\\mathrm{TNT}}_{ab} - S^{\\mathrm{TNT3c}}_{ab}$', 'positive = more similar when APL is blocked'),
    ('s_o', 'Suppression of odor $o$ by APL: $\\phi^{\\mathrm{TNT}}_o / \\phi^{\\mathrm{TNT3c}}_o$', '≥ 1 means more units respond when APL is blocked'),
    ('\\mu,\\ e_a', 'Additive fit of $\\Delta S$: mean change $\\mu$ and per-odor effect $e_a$ (written $f_a$ in the notes)', 'Section 5'),
    ('\\rho_s', 'Spearman rank correlation', '−1 to 1'),
])

# ---------------------------------------------------------------- 1 recordings
R.section('The recordings')
rows = []
for c in tc.COND:
    for name, df in D[c].items():
        rows.append(dict(condition=c, recording=name, fly=name.split('F')[0], side=name.split('F')[1][0] if 'F' in name else '?',
                         units=len(df), resp=float((df.values > 1).mean())))
REC = pd.DataFrame(rows)
R.p(f'There are {sum(REC.condition == "TNT")} TNT recordings from {REC[REC.condition == "TNT"].fly.nunique()} flies and '
    f'{sum(REC.condition == "TNT3c")} TNT3c recordings from {REC[REC.condition == "TNT3c"].fly.nunique()} flies. A recording name reads '
    '`date_fly` + `FL`/`FR` (left or right side) + `fun` and a number. Where one fly gives two recordings, they are its two hemispheres. The '
    'label 20240802_5 appears in both files but names two different flies (Yang). Every recording is a table with one row per unit and one '
    'column per odor; the 16 odors are the same, in the same order, as in the ORN table.')
R.table(['condition', 'recording', 'side', 'units', 'values > 1'],
        [[CN[r.condition], f'`{r.recording}`', r.side, f'{r.units:,}', L.pct(r.resp)] for r in REC.itertuples()],
        title='The recordings.', caption='"values > 1" is the share of all unit × odor responses above 1, the "responding" criterion used throughout.',
        num_cols=[3, 4])

# ---------------------------------------------------------------- 2 responses
R.section('Response values and the "responding" criterion')
R.p('The responses $r^{(n)}_{io}$ are **FC z-scores** (`fc_zscore`; Yang, 2026-11-03). Most values lie near 0, and a minority are '
    'large. A unit **responds** to an odor if its value is above 1 z-score unit. This is the threshold used in all comparisons (0.5 is '
    'checked as an alternative):')
R.eq(r'\phi^{(n)}_o = \frac{1}{N_n}\,\#\{\, i : r^{(n)}_{io} > 1 \,\}, \qquad \phi^c_o = \frac{1}{|c|}\sum_{n \in c} \phi^{(n)}_o',
     note='$N_n$ = units in recording $n$; $|c|$ = recordings of condition $c$; every recording counts equally')


def fig_hist():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = axs[0]; bins = np.linspace(-2.7, 7.5, 103)
    for c in tc.COND:
        v = np.concatenate([df.values.ravel() for df in D[c].values()])
        a.hist(v, bins=bins, histtype='step', lw=1.6, color=CC[c], density=True, label=f'{CN[c]}: {len(v):,} values, {L.pct((v > 1).mean())} > 1')
    a.axvline(1, color=L.INK, ls='--', lw=1); a.text(1.05, 1.5, 'threshold 1', fontsize=7.5)
    a.set_yscale('log'); a.set_xlabel('response $r^{(n)}_{io}$ (FC z-score)'); a.set_ylabel('density (log scale)')
    a.legend(fontsize=7.5); a.set_title('a  All responses, pooled over recordings')
    a = axs[1]
    for c in tc.COND:
        x = REC[REC.condition == c]
        a.scatter(x.units, x.resp * 100, color=CC[c], s=26, label=CN[c], edgecolor='white', lw=0.5)
    a.set_xlabel('units in the recording'); a.set_ylabel('responses > 1, % of unit × odor values'); a.legend(fontsize=7.5)
    a.set_title('b  Per recording'); a.set_ylim(0, None)
    fig.tight_layout(); return fig


R.fig(fig_hist(), '**a**, Distribution of every response value (unit × odor × recording) in each condition, as a density on a log scale. '
      'The dashed line is the responding threshold of 1. TNT has a heavier right tail. **b**, Each point is one recording: its number of '
      'units (horizontal) and the percentage of its unit × odor values above 1 (vertical).', 'Response values in the two conditions.')


def fig_frac():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    fr = {c: tc.FRAC[1.0][c] for c in tc.COND}; m = {c: fr[c].mean(0) for c in tc.COND}
    order = np.argsort(m['TNT']); x = np.arange(len(O))
    a = axs[0]
    for dx, c in [(-0.2, 'TNT3c'), (0.2, 'TNT')]:
        a.bar(x + dx, m[c][order] * 100, 0.38, color=CC[c], alpha=0.85, label=f'{CN[c]}, mean of recordings')
        for row in fr[c]: a.scatter(x + dx, row[order] * 100, s=5, color=L.INK, alpha=0.45, lw=0, zorder=3)
    a.scatter([], [], s=5, color=L.INK, alpha=0.6, label='single recordings')
    a.set_xticks(x); a.set_xticklabels(np.array(O)[order], rotation=90); a.set_ylabel('units responding (> 1), % ($\\phi_o$ × 100)')
    a.set_xlabel('odor (sorted by the TNT mean)'); a.legend(fontsize=7.2); a.set_title('a  Fraction of units responding per odor')
    a = axs[1]; s = m['TNT'] / m['TNT3c']; o2 = np.argsort(s)
    a.barh(np.arange(len(O)), s[o2], color=L.GRAY); a.set_yticks(np.arange(len(O))); a.set_yticklabels(np.array(O)[o2], fontsize=7.5)
    a.axvline(1, color=L.INK, lw=0.8); a.set_xlabel('suppression $s_o = \\phi^{TNT}_o / \\phi^{TNT3c}_o$ (×)')
    a.set_title('b  How much APL suppresses each odor')
    fig.tight_layout(); return fig, m, s


fg, M_, S_ = fig_frac()
R.fig(fg, '**a**, Percentage of units responding (> 1) to each odor. Bars are the mean over recordings (blue TNT3c, orange TNT); black dots '
      'are single recordings. **b**, Suppression per odor: the TNT mean divided by the TNT3c mean. A value of 2 means twice as many units '
      'respond when APL is blocked.', 'Fraction responding per odor, and suppression.')
R.eq(r's_o = \frac{\phi^{\mathrm{TNT}}_o}{\phi^{\mathrm{TNT3c}}_o}')
R.p(f'On average over odors, {L.pct(M_["TNT"].mean())} of units respond in TNT and {L.pct(M_["TNT3c"].mean())} in TNT3c. Suppression ranges '
    f'from ×{S_.min():.2f} ({O[int(np.argmin(S_))]}) to ×{S_.max():.2f} ({O[int(np.argmax(S_))]}). Report 07 uses these per-odor fractions to pin '
    'the model: with APL blocked, the model responds to each odor in the TNT fraction, and with APL intact its mean equals the TNT3c mean.')

# ---------------------------------------------------------------- 3 similarity
R.section('Odor-pair similarity')
R.p('For each recording, the similarity of two odors is the cosine of their response vectors over all of its units (raw values, '
    'not thresholded). The condition\'s similarity is the mean over its recordings:')
R.eq(r'S^{(n)}_{ab} = \frac{\sum_i r^{(n)}_{ia}\, r^{(n)}_{ib}}{\sqrt{\sum_i \big(r^{(n)}_{ia}\big)^2}\,\sqrt{\sum_i \big(r^{(n)}_{ib}\big)^2}}, \qquad '
     r'S^c_{ab} = \frac{1}{|c|}\sum_{n\in c} S^{(n)}_{ab}')
R.p('$1 - S^c_{ab}$ is exactly Yang\'s RDM for that condition (checked to 10⁻⁶ for every pair). The effect of blocking APL on a pair is')
R.eq(r'\Delta S_{ab} = S^{\mathrm{TNT}}_{ab} - S^{\mathrm{TNT3c}}_{ab}')
S3, ST = SREC['TNT3c'].mean(0), SREC['TNT'].mean(0); dS = ST - S3


def tomat(v):
    M = np.full((len(O), len(O)), np.nan); M[IU] = v; M[(IU[1], IU[0])] = v; return M


def fig_sim():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    vmax = float(max(S3.max(), ST.max()))
    for a, v, t, cm, lab, vm in [(axs[0], S3, 'a  $S^{TNT3c}_{ab}$ (APL intact)', L.SEQ, 'cosine similarity', (0, vmax)),
                                 (axs[1], ST, 'b  $S^{TNT}_{ab}$ (APL blocked)', L.SEQ, 'cosine similarity', (0, vmax)),
                                 (axs[2], dS, 'c  $\\Delta S_{ab}$ = TNT − TNT3c', L.DIV, '$\\Delta S_{ab}$ (difference in cosine)', (-0.4, 0.4))]:
        im = a.imshow(tomat(v), cmap=cm, vmin=vm[0], vmax=vm[1], interpolation='nearest')
        a.set_xticks(range(16)); a.set_xticklabels(O, rotation=90, fontsize=6.8); a.set_yticks(range(16)); a.set_yticklabels(O, fontsize=6.8)
        a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]; a.set_title(t, fontsize=9); L.cbar(fig, im, a, lab)
    fig.tight_layout(); return fig


R.fig(fig_sim(), '**a**, **b**, Mean cosine similarity of the units\' responses for every odor pair, in TNT3c and TNT, on one scale (white 0, '
      'dark blue the highest value; slightly negative values, such as pa–-aPine in TNT3c, also show as white). **c**, The change, on a '
      'diverging scale (orange = more similar in TNT). Every pair is orange: blocking APL makes all 120 pairs more similar. Diagonal cells '
      'are left blank.', 'Odor-pair similarity in the two conditions.')
R.p(f'All {len(dS)} pairs are more similar in TNT. Mean similarity rises from {S3.mean():.2f} to {ST.mean():.2f} (mean $\\Delta S$ = '
    f'{F(dS.mean(), 3, True)}, range {F(dS.min(), 2, True)} to {F(dS.max(), 2, True)}).')

# ---------------------------------------------------------------- 4 decomposition
R.section('Splitting the change into per-odor and pair-specific parts')
R.p('Some odors gain similarity with every partner. To separate that from what is specific to a pair, the change is fitted by least '
    'squares as a mean plus one effect per odor:')
R.eq(r'\Delta S_{ab} \approx \mu + e_a + e_b, \qquad \sum_a e_a = 0', note='$\\mu$ = mean change; $e_a$ = how much more the pairs of odor $a$ change than the average pair')
eff, res, r2 = tc.additive(dS, len(O))
R.p(f'$R^2$ of this fit is {r2:.2f}: {L.pct(r2, 0)} of the variation of $\\Delta S$ across pairs is per odor. The residuals, '
    '$\\Delta S_{ab} - (\\mu + e_a + e_b)$, are the pair-specific part.')


def fig_eff():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    a = axs[0]; o = np.argsort(eff)
    a.barh(np.arange(len(O)), eff[o], color=[L.BLOCKED if x > 0 else L.INTACT for x in eff[o]])
    a.set_yticks(np.arange(len(O))); a.set_yticklabels(np.array(O)[o], fontsize=7.5); a.axvline(0, color=L.INK, lw=0.8)
    a.set_xlabel('per-odor effect $e_a$ (change in cosine, relative to the mean change)'); a.set_title(f'a  Per-odor effects (μ = {dS.mean():+.3f})')
    a = axs[1]
    a.scatter(S3, dS, s=14, color=L.INK, alpha=0.7, lw=0)
    r_ = np.corrcoef(S3, dS)[0, 1]
    a.set_xlabel('similarity with APL intact, $S^{TNT3c}_{ab}$'); a.set_ylabel('$\\Delta S_{ab}$ (TNT − TNT3c)')
    a.set_title(f'b  Change against starting similarity (r = {r_:.2f})'); a.axhline(0, color=L.GRID, lw=0.8)
    fig.tight_layout(); return fig, r_


fg, rS = fig_eff()
R.fig(fg, '**a**, The per-odor effect $e_a$ of the additive fit, sorted. Orange: the odor\'s pairs rise more than average when APL is '
      'blocked. Blue: less. **b**, Each point is one odor pair: its similarity in TNT3c (horizontal) against its change (vertical).',
      'Which odors drive the change.')

# ---------------------------------------------------------------- per-odor effect: meaning, mechanism and statistics (Q&A 2026-11-03)
R.sub('What the per-odor effect means, mechanistically')
NO = len(O); FRr = tc.FRAC[1.0]
A6 = np.zeros((len(IU[0]), NO + 1)); A6[:, 0] = 1; A6[np.arange(len(IU[0])), IU[0] + 1] = 1; A6[np.arange(len(IU[0])), IU[1] + 1] = 1


def additive6(y):
    b = np.linalg.lstsq(A6, y, rcond=None)[0]; fit = A6 @ b
    return b[1:] - b[1:].mean(), 1 - np.var(y - fit) / np.var(y)


e6, r26 = additive6(dS)
assert np.allclose(e6, eff) and abs(r26 - r2) < 1e-9
null_r2 = np.array([additive6(rs6.permutation(dS))[1] for _ in range(5000)])
p_r2 = (1 + np.sum(null_r2 >= r26)) / (1 + len(null_r2))
boot = []
for _ in range(2000):
    t, c = rs6.choice(len(SREC['TNT']), len(SREC['TNT'])), rs6.choice(len(SREC['TNT3c']), len(SREC['TNT3c']))
    boot.append(additive6(SREC['TNT'][t].mean(0) - SREC['TNT3c'][c].mean(0))[0])
boot = np.array(boot); blo, bhi = np.percentile(boot, [2.5, 97.5], 0); sig = (blo > 0) | (bhi < 0)
rel = []
for _ in range(1000):
    t, c = rs6.permutation(len(SREC['TNT'])), rs6.permutation(len(SREC['TNT3c']))
    rel.append(np.corrcoef(additive6(SREC['TNT'][t[:4]].mean(0) - SREC['TNT3c'][c[:5]].mean(0))[0],
                           additive6(SREC['TNT'][t[4:]].mean(0) - SREC['TNT3c'][c[5:]].mean(0))[0])[0, 1])
rel_h = float(np.mean(rel)); rel_f = 2 * rel_h / (1 + rel_h)
chance = {c: np.mean([np.sqrt(np.outer(p_, p_))[IU] for p_ in FRr[c]], 0) for c in FRr}
e_ch, r2_ch = additive6(chance['TNT'] - chance['TNT3c'])
fT, fC = FRr['TNT'].mean(0), FRr['TNT3c'].mean(0)
rho_ch = spearmanr(e6, e_ch)[0]; rho_fT = spearmanr(e6, fT)[0]; rho_df = spearmanr(e6, fT - fC)[0]; rho_ratio = spearmanr(e6, fT / fC)[0]
dpp = pd.Series((fT - fC) * 100, index=O)
R.p('The fit $\\Delta S_{ab} \\approx \\mu + e_a + e_b$ says: the change of a pair is the average change, plus a bit that belongs to '
    'each of its two odors. $e_a$ is how much more (or less) the pairs that contain odor $a$ change than the average pair. For example '
    f'$e_{{\\mathrm{{eb}}}}$ = {F(e6[O.index("eb")], 3, True)}: pairs with eb gain about {F(e6[O.index("eb")], 2)} more cosine similarity '
    'than average when APL is blocked, whatever the partner. An odor with a large $e_a$ has its whole row of the similarity matrix move up.')
R.p('**Mechanism: chance overlap.** When an odor drives more units, it shares more responding units with every other odor just by chance: '
    'two unrelated 0/1 codes in which fractions $p_a$ and $p_b$ of equally excitable units respond have cosine $\\sqrt{p_a p_b}$ (Report 08). '
    'So an odor that gains many responders when APL is blocked should gain similarity with all partners, which is exactly a large $e_a$. '
    'This can be tested on the data alone: compute the chance overlap $\\sqrt{p_a p_b}$ from each recording\'s fractions responding, take '
    'its change TNT − TNT3c, fit the same additive model to it, and compare its per-odor effects with the real $e_a$ (Figure '
    f'{R.n_fig + 1}b).')


def fig_e():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1, 1.05]})
    a = axs[0]; o = np.argsort(e6); y = np.arange(NO)
    a.hlines(y, blo[o], bhi[o], color=L.GRAY, lw=1.6)
    a.scatter(e6[o], y, s=36, color=np.where(e6[o] > 0, L.BLOCKED, L.INTACT), zorder=3, edgecolor='white', linewidth=0.6)
    a.axvline(0, color=L.INK, lw=0.7)
    a.set_yticks(y); a.set_yticklabels([O[i] + (' *' if sig[i] else '') for i in o], fontsize=8)
    a.set_xlabel('per-odor effect $e_a$ (cosine, relative to the mean change)')
    a.set_title(f'a  $e_a$ with 95% bootstrap range ({int(sig.sum())} of {NO} exclude 0, *)', fontsize=9.5)
    b = axs[1]
    b.scatter(e_ch, e6, s=36, color=L.INTACT, edgecolor='white', linewidth=0.6, zorder=3)
    for i, od in enumerate(O): b.annotate(od, (e_ch[i], e6[i]), xytext=(3, 2), textcoords='offset points', fontsize=7, color=L.MUTED)
    k = np.polyfit(e_ch, e6, 1); xs = np.linspace(e_ch.min(), e_ch.max(), 10); b.plot(xs, np.polyval(k, xs), color=L.BLOCKED, lw=1.2)
    b.axhline(0, color=L.GRAY, lw=0.6, ls=':'); b.axvline(0, color=L.GRAY, lw=0.6, ls=':')
    b.set_xlabel('per-odor effect predicted by chance overlap alone'); b.set_ylabel('per-odor effect $e_a$ in the data')
    b.set_title(f'b  Chance overlap predicts $e_a$: Spearman ρ = {rho_ch:.2f}', fontsize=9.5)
    fig.tight_layout(); return fig


R.fig(fig_e(), f'**a**, The per-odor effects of Figure {R.n_fig}a, with their 95% bootstrap ranges (2,000 draws; recordings resampled with '
      'replacement within each condition; each draw refits the additive model). **b**, Each point is one odor. Horizontal: the per-odor '
      'effect of the additive fit to the change in chance overlap $\\sqrt{p_a p_b}$ (fractions responding > 1, per recording, averaged). '
      'Vertical: the per-odor effect of the real change in similarity. The orange line is the least-squares fit.',
      'Per-odor effects: how sure, and why.')
R.p(f'The per-odor effects follow chance overlap (Spearman ρ = {rho_ch:.2f}). They also follow how many units an odor drives in TNT '
    f'(ρ = {rho_fT:.2f}) and how many it gains (TNT − TNT3c, ρ = {rho_df:.2f}), but not the ratio TNT ÷ TNT3c (ρ = {rho_ratio:.2f}). '
    'So the per-odor part of the change is mostly the arithmetic of denser codes: odors that drive many units in TNT overlap more with '
    f'everything. -aPine and euc gain the fewest responding units when APL is blocked ({dpp["-aPine"]:.1f} and {dpp["euc"]:.1f} percentage '
    f'points, against {dpp.drop(["-aPine", "euc"]).min():.1f}–{dpp.max():.1f} for the other odors) and have the most negative effects. In '
    'mechanistic terms: if APL\'s feedback grows with the number of active KCs, as in the model of Report 05, it removes the most responders '
    'from odors that drive many KCs. Blocking it then adds the most responders to exactly those odors, and their codes overlap more with '
    f'every partner. The data agree with the first step: the units an odor gains (TNT − TNT3c) rise with how many it drives in TNT3c '
    f'(Spearman ρ = {spearmanr(fC, fT - fC)[0]:.2f} over odors).')
R.sub('Are the per-odor effects real? The statistics')
R.table(['test', 'how', 'result'], [
    ['Is the additive fit better than chance?', 'Permutation: the 120 values of $\\Delta S$ are shuffled across pairs (5,000 times), which '
     'removes any odor structure, and the additive model is refitted', f'$R^2$ = {r26:.2f} in the data; null mean {null_r2.mean():.2f}, 95th '
     f'percentile {np.percentile(null_r2, 95):.2f}; p = {p_r2:.4f}' + (' (none of 5,000 shuffles reach it)' if np.sum(null_r2 >= r26) == 0 else '')],
    ['Which $e_a$ differ from 0?', f'95% bootstrap range over recordings (Figure {R.n_fig}a)', f'{int(sig.sum())} of {NO} odors: '
     + ', '.join(O[i] for i in np.argsort(e6) if sig[i])],
    ['Are the $e_a$ reproducible?', 'Split-half: two random halves of the recordings (4 + 5 and 4 + 6), $e_a$ fitted in each, Pearson r '
     'over odors; 1,000 splits; Spearman–Brown to full data', f'r = {rel_h:.2f} between halves, {rel_f:.2f} for all the data']],
    title='Statistics for the per-odor effects.',
    caption=f'With {NO} odors, the additive model has {NO} free parameters for {len(dS)} pairs, so even random numbers give $R^2$ ≈ '
            f'{NO - 1}/{len(dS) - 1} ≈ {(NO - 1) / (len(dS) - 1):.2f}. That is why the null is needed: {r26:.2f} is far above it.')

# ---------------------------------------------------------------- 5 reliability
R.section('How reliable these numbers are')
R.p('The recordings are the unit of resampling everywhere: bootstrap draws resample recordings with replacement within each condition, '
    'and split-half tests divide each condition\'s recordings into two random halves. A split-half correlation describes half the data. '
    'It is turned into an estimate for all the data with the Spearman–Brown formula, $r_{\\mathrm{full}} = 2r_{\\mathrm{half}}/(1 + r_{\\mathrm{half}})$.')
rk = {k: ro('raw responses', k) for k in ['TNT vs TNT3c, all recordings', 'half vs half, within TNT3c (mean of 1000 splits)',
                                          'half vs half, within TNT', 'half vs half, TNT vs TNT3c', 'fraction of splits with between < both withins']}
R.ul([f'**Rank order of similarity.** Over the 120 pairs, $S^{{\\mathrm{{TNT}}}}$ and $S^{{\\mathrm{{TNT3c}}}}$ have $\\rho_s$ = {rk["TNT vs TNT3c, all recordings"]:.2f}. '
      f'With half the recordings, two halves of the same condition give {rk["half vs half, within TNT3c (mean of 1000 splits)"]:.2f} (TNT3c) and '
      f'{rk["half vs half, within TNT"]:.2f} (TNT), while TNT vs TNT3c gives {rk["half vs half, TNT vs TNT3c"]:.2f}. The between-condition value is '
      f'lower in {L.pct(rk["fraction of splits with between < both withins"], 1)} of splits. Dividing the between-condition value by the '
      f'within-condition one gives about {rk["half vs half, TNT vs TNT3c"] / np.mean([rk["half vs half, within TNT3c (mean of 1000 splits)"], rk["half vs half, within TNT"]]):.2f}: '
      'the rank correlation expected without measurement noise. Report 09 and the local model use this as the data\'s value.',
      f'**Change per pair.** The split-half reliability of $\\Delta S$ is {TVM["fits"]["threshold 1.0, ORN"]["split-half (recordings) reliability of dS, full-data estimate"]:.2f} '
      f'(Spearman–Brown, all recordings), which caps any model\'s Pearson r with it near '
      f'{TVM["fits"]["threshold 1.0, ORN"]["ceiling for r with a model (sqrt of that)"]:.2f} (Report 07 explains why).',
      f'**Duplicates.** No unit is repeated within or across recordings. Units whose nearest neighbour in their own recording is closer than '
      f'expected are rare ({L.pct(CHK["near-duplicate units (own-recording neighbour closer than 1st percentile of other recordings), mean over recordings"]["TNT"]["responsive units"], 1)} '
      f'of responsive TNT units, {L.pct(CHK["near-duplicate units (own-recording neighbour closer than 1st percentile of other recordings), mean over recordings"]["TNT3c"]["responsive units"], 1)} TNT3c).',
      f'**Two hemispheres of one fly.** Weighting each fly once instead of each recording changes the mean $\\Delta S$ from '
      f'{CHK["change in similarity, recordings weighted equally vs flies weighted equally"]["mean"][0]:.3f} to '
      f'{CHK["change in similarity, recordings weighted equally vs flies weighted equally"]["mean"][1]:.3f}, and the pattern over pairs correlates '
      f'{CHK["change in similarity, recordings weighted equally vs flies weighted equally"]["pattern r over odor pairs"]:.2f}.'])


def fig_rel():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
    a = axs[0]
    for c in tc.COND:
        for v in SREC[c]: a.plot(np.arange(len(IU[0])), np.sort(v), color=CC[c], lw=0.6, alpha=0.5)
        a.plot([], [], color=CC[c], label=f'{CN[c]}: single recordings')
    a.set_xlabel('odor pairs, sorted within each recording'); a.set_ylabel('cosine similarity $S^{(n)}_{ab}$')
    a.legend(fontsize=7.5); a.set_title('a  Similarity in every recording')
    a = axs[1]
    for c, off in [('TNT3c', -0.12), ('TNT', 0.12)]:
        Sm = SREC[c]; r = [spearmanr(Sm[i], np.delete(Sm, i, 0).mean(0))[0] for i in range(len(Sm))]
        a.scatter(np.full(len(r), {'TNT3c': 0, 'TNT': 1}[c]) + off, r, color=CC[c], s=24, edgecolor='white', lw=0.5)
    a.set_xticks([0, 1]); a.set_xticklabels([CN['TNT3c'], CN['TNT']], fontsize=8); a.set_xlim(-0.6, 1.6)
    a.set_ylabel('$\\rho_s$: one recording vs the mean of the rest\n(same condition, over 120 pairs)')
    a.set_title('b  How well one recording predicts the others')
    fig.tight_layout(); return fig


R.fig(fig_rel(), '**a**, Each line is one recording: its 120 pair similarities sorted from lowest to highest. TNT recordings (orange) sit '
      'higher throughout. **b**, Each point is one recording: the Spearman correlation of its pair similarities with the mean of the other '
      'recordings of the same condition. It measures how consistent the rank order of pairs is from hemisphere to hemisphere.',
      'Consistency across recordings.')
R.box('Caveats', ['The units are not matched across recordings or conditions, so every comparison is between condition means.',
                  'The responses are FC z-scores. Thresholding at 1 assumes that a z-score of 1 means the same in both genotypes.',
                  'Noise lowers cosine similarity more when responses are small. TNT3c has smaller responses, so part of the rise in TNT may be '
                  'noise. Thresholding at 1 before the cosine gives the same picture (every pair still rises; pattern r 0.94 with the raw change).'], 'caveat')
R.save()
