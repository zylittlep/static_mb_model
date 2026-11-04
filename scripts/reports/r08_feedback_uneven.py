"""Report 08: why APL feedback makes the effect of blocking APL uneven across odors (tnt_feedback_vs_fixed.py, chance_overlap_sim.py).
Rebuilt 2026-11-04 for Yang's revision: corrected calyx, new claws, new ORN table (APL_VARIANT=claws_v2m2); the four inhibition rules as
a table (only feedback is a model); how the Spearman values and -aPine's ratio are computed; gamma_o renamed "blocked / intact ratio";
dots drawn so none hides another; odor strength from TNT only. Old version in archive_2026-11-03/."""
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
FV = pd.read_csv(OWV + 'TNT_feedback_vs_fixed.csv'); FJ = json.load(open(OWV + 'TNT_feedback_vs_fixed.json'))
CO = pd.read_csv(OWV + 'chance_overlap_data_pairs.csv'); CS = pd.read_csv(OWV + 'chance_overlap_simulation.csv')
CJ = json.load(open(OWV + 'chance_overlap_simulation.json'))
RULES = ['feedback', 'fixed', 'same share', 'TNT3c per odor']
RC = {'feedback': L.INTACT, 'fixed': L.GRAY, 'same share': L.GREEN, 'TNT3c per odor': L.PURPLE}
J = FJ['ORN']
FO_ = FV[FV.responses == 'ORN']; T3M, TNM = float(FO_.frac_TNT3c.mean()), float(FO_.frac_TNT.mean())

R = L.Report('08', 'Why does blocking a uniform, feedback APL raise the similarity of some odor pairs more than others, and do the recordings '
             'show the same signature?',
             inputs=['The pinned model of Report 07 (uniform APL, hemibrain R and male CNS L and R; corrected calyx, new claws, new ORN table)',
                     'The recordings (Report 06)'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/odor_weighted/`: `TNT_feedback_vs_fixed.csv`, `TNT_feedback_vs_fixed.json`, '
                      '`chance_overlap_simulation.png`, `.csv`, `.json`, `chance_overlap_data_pairs.csv`, `TNT_feedback_signature_split_half.csv`'],
             scripts=['`scripts/tnt_feedback_vs_fixed.py`', '`scripts/chance_overlap_sim.py`', '`scripts/reports/r08_feedback_uneven.py` (this report)'])
R.p('Report 07 found that a uniform APL with feedback predicts which pairs gain similarity when APL is blocked about as well as any '
    'wiring version. This report explains why, in two steps. First, feedback makes the '
    'inhibition odor-dependent, so blocking it adds the most responders to strong odors. Second, an odor that gains responders '
    'becomes more similar to every other odor by chance overlap alone.')
R.box('What changed in this revision (2026-11-04)', ["The four inhibition rules as a table: only feedback is a model; the other three are reference rules.", "New: how the Spearman values are computed, with -aPine as a worked example.", "$\\gamma_o$ is now called the \"blocked ÷ intact ratio\", so it is not confused with the input gain $g_o$ of Report 07.", "In the ratio figure the data are drawn as open diamonds, so model points at the same place stay visible.", "Odor strength in the data is now measured in TNT only, with the earlier mean of TNT and TNT3c kept for comparison.", "Numbers come from the run with the corrected calyx, the new claws and the new ORN table."], 'plain')
R.symbols([
    ('\\phi_o', 'Fraction of KCs (model) or units (data) responding to odor $o$', '0 to 1'),
    ('\\gamma_o', '**Blocked ÷ intact ratio** of odor $o$: $\\phi^{\\mathrm{blocked}}_o / \\phi^{\\mathrm{intact}}_o$ (in the data, TNT ÷ TNT3c; called the '
     'suppression $s_o$ in Report 07). Not the input gain $g_o$ of Report 07', '×'),
    ('\\mathrm{CV}', 'Coefficient of variation across odors: standard deviation ÷ mean', 'dimensionless'),
    ('n_a,\\ n_{ab}', 'Number of units responding to odor $a$, and to both $a$ and $b$', 'counts'),
    ('N', 'Number of units (or KCs)', ''),
    ('p_a', '$n_a / N$, the fraction responding to $a$ (same as $\\phi_a$)', '0 to 1'),
    ('\\sqrt{p_a p_b}', 'Chance overlap: expected cosine of two unrelated 0/1 codes', '0 to 1'),
    ('\\pi_k', 'Probability that unit $k$ responds to a random odor (its excitability)', '0 to 1'),
    ('\\Delta S_{ab}', 'Change in similarity when APL is blocked', 'difference in cosine'),
])

# ---------------------------------------------------------------- 1 feedback = normalization
R.section('Step 1: feedback inhibits strong odors more')
R.p('With a uniform APL, every KC gets the same inhibition $a A_o$ for odor $o$ (Report 05). It is the same for every KC, but not for every '
    'odor: $A_o$ is the mean KC response, so an odor that drives many KCs recruits a lot of APL and is held down hard, and one that drives '
    'few recruits little. Feedback therefore evens out how many KCs the odors activate, like a normalization, and blocking APL undoes it. '
    'To show that feedback is what matters, three other rules for the inhibition are compared. All four start from the same pinned, '
    f'blocked responses $y^{{\\mathrm{{blocked}}}}_{{ko}} = [x_{{ko}} - 1]_+$ and give the same mean fraction responding over odors (TNT3c\'s, '
    f'{L.pct(T3M, 2)}). They differ only in how the inhibition is shared out between odors (Table {R.n_tab + 1}).')
R.table(['rule', 'intact response', 'how the inhibition is set', 'is it a model?'], [
    ['feedback (version U)', '$[x_{ko} - 1 - a\\,A_o]_+$, with $A_o = \\frac{1}{N}\\sum_k y_{ko}$', 'one gain $a$ for all odors; $A_o$ '
     'comes out of the KC responses themselves, so strong odors get more inhibition', 'yes: the model of Reports 05 and 07'],
    ['fixed', '$[x_{ko} - 1 - c]_+$', 'one number $c$ for all KCs and all odors (per mushroom body): the same as raising the threshold from '
     '1 to $1 + c$', 'no: a reference rule'],
    ['same share', '$[x_{ko} - 1 - c_o]_+$', 'one $c_o$ per odor, chosen so that every odor keeps the same share of its blocked '
     'responders: $\\phi^{\\mathrm{intact}}_o = \\phi^{\\mathrm{blocked}}_o \\times \\bar\\phi^{\\mathrm{TNT3c}} / \\bar\\phi^{\\mathrm{TNT}}$ '
     f'(about {T3M / TNM * 100:.0f}%)', 'no: a reference rule'],
    ['TNT3c per odor', '$[x_{ko} - 1 - c_o]_+$', 'one $c_o$ per odor, chosen so that $\\phi^{\\mathrm{intact}}_o = \\phi^{\\mathrm{TNT3c}}_o$, '
     'the data\'s own fraction', 'no: uses the answer from the data']],
    '$c$ and $c_o$ are found by bisection in each mushroom body. In all four, the inhibition is the same for every KC for a given odor.',
    'The four inhibition rules.')
R.p('The three extra rules are not new models of APL. They are controls that keep everything the same except how the inhibition '
    'depends on the odor: not at all (fixed), in proportion to the odor\'s own blocked response (same share), or exactly as in the data '
    '(TNT3c per odor). Comparing feedback with them shows whether the odor-dependence created by feedback is what matters for the fit.')


def fig_gain():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    P = FV[FV.responses == 'ORN']
    a = axs[0]
    a.scatter(P.frac_TNT * 100, P.gain_data, facecolor='none', edgecolor=L.BLOCKED, marker='D', s=70, lw=1.3, label='data (TNT ÷ TNT3c)', zorder=2)
    for rule, col, mk, sz in [('fixed', L.GRAY, 's', 30), ('same_share', L.GREEN, '^', 26), ('feedback', L.INTACT, 'o', 20)]:
        a.scatter(P.model_frac_blocked * 100, P[f'gain_model_{rule}'], facecolor='none' if rule != 'feedback' else col, edgecolor=col,
                  marker=mk, s=sz, lw=1.1, label=rule.replace('_', ' '), zorder=3 if rule == 'feedback' else 2.5)
    ap_ = P[P.odor == '-aPine'].iloc[0]
    a.annotate('-aPine', (ap_.frac_TNT * 100, ap_.gain_data), xytext=(6, -12), textcoords='offset points', fontsize=7, color=L.MUTED,
               arrowprops=dict(arrowstyle='-', color=L.MUTED, lw=0.6))
    a.set_xlabel('fraction responding with APL blocked, % (model = TNT by construction)'); a.set_ylabel('blocked ÷ intact ratio $\\gamma_o$')
    a.legend(fontsize=7.5); a.set_title('a  Blocked ÷ intact ratio per odor, ORN input (mean of 3 MBs)', fontsize=9.5)
    a = axs[1]; x = np.arange(len(P)); o = np.argsort(P.model_frac_blocked.values)
    a.plot(x, P.model_frac_blocked.values[o] * 100, color=L.BLOCKED, lw=1.8, marker='o', ms=3, label='blocked (= TNT)')
    a.plot(x, P.model_frac_intact_feedback.values[o] * 100, color=L.INTACT, lw=1.8, marker='o', ms=3, label=f'intact, feedback (CV {J["spread of fraction responding across odors (coefficient of variation)"]["model intact, feedback"]:.2f})')
    a.plot(x, P.model_frac_intact_fixed.values[o] * 100, color=L.GRAY, lw=1.8, marker='s', ms=3, label=f'intact, fixed (CV {J["spread of fraction responding across odors (coefficient of variation)"]["model intact, fixed"]:.2f})')
    a.plot(x, P.frac_TNT3c.values[o] * 100, color=L.INK, lw=1, ls='--', label=f'data TNT3c (CV {J["spread of fraction responding across odors (coefficient of variation)"]["data TNT3c"]:.2f})')
    a.set_xticks(x); a.set_xticklabels(P.odor.values[o], rotation=90, fontsize=7.5); a.set_ylabel('KCs / units responding, %')
    a.legend(fontsize=7); a.set_title('b  Feedback evens out the odors; a fixed raise spreads them')
    fig.tight_layout(); return fig


R.fig(fig_gain(), '**a**, Each point is one odor. Horizontal: the fraction responding with APL blocked (the model is pinned to TNT). Vertical: '
      'how many times more respond blocked than intact. Filled blue dots: feedback. Open grey squares: fixed raise. Open green triangles: '
      'same share. Large open orange diamonds: data, drawn open so that a model point at the same place stays visible inside. '
      '**b**, Per odor (sorted by the blocked fraction), the percentage responding blocked (orange), intact with feedback (blue) and intact '
      'with a fixed raise (grey), with the data\'s TNT3c values (dashed). CV is the spread across odors.', 'Feedback acts like a normalization across odors.')
SPJ = J["spearman over odors, gain vs fraction responding with APL blocked"]; RGJ = J["gain per odor (blocked / intact fraction responding), range"]
R.p(f'With feedback, the blocked ÷ intact ratio rises with odor strength (Spearman {SPJ["feedback"]:+.2f} over odors), from '
    f'×{RGJ["feedback"][0]:.2f} to ×{RGJ["feedback"][1]:.2f}. A fixed raise does the opposite (Spearman {SPJ["fixed"]:+.2f}): weak odors lose the '
    'largest share of their few responders, because a fixed raise removes a fixed slice of the drive distribution near threshold. '
    f'In the data the ratio is barely related to the TNT fraction (Spearman {SPJ["data"]:+.2f}).')
R.sub('How the Spearman values are computed: -aPine as an example')
ap = FO_[FO_.odor == '-aPine'].iloc[0]; rk = FO_.gain_model_feedback.rank(method='min').astype(int); rkx = FO_.model_frac_blocked.rank(method='min').astype(int)
iap = FO_.index[FO_.odor == '-aPine'][0]
R.p('For each of the 16 odors there are two numbers, each the mean over the three mushroom bodies: the fraction responding with APL blocked '
    '($\\phi^{\\mathrm{blocked}}_o$, equal to the TNT fraction by pinning) and the ratio $\\gamma_o = \\phi^{\\mathrm{blocked}}_o / '
    '\\phi^{\\mathrm{intact}}_o$. The Spearman correlation is the Pearson correlation of their ranks over the 16 odors.')
R.p(f'For -aPine, $\\phi^{{\\mathrm{{blocked}}}}$ = {ap.model_frac_blocked * 100:.2f}% and $\\phi^{{\\mathrm{{intact}}}}$ with feedback = '
    f'{ap.model_frac_intact_feedback * 100:.2f}%, so $\\gamma$ = {ap.model_frac_blocked * 100:.2f} / {ap.model_frac_intact_feedback * 100:.2f} '
    f'= **{ap.gain_model_feedback:.2f}** (rank {rk[iap]} of 16 by ratio, rank {rkx[iap]} of 16 by blocked fraction). The data\'s value is '
    f'{ap.frac_TNT * 100:.2f} / {ap.frac_TNT3c * 100:.2f} = {ap.gain_data:.2f}. This $\\gamma_o$ is not the input gain $g_o$ of Report 07, '
    f'which for -aPine is large because its ORN responses are weak.')

# ---------------------------------------------------------------- 2 chance overlap
R.section('Step 2: from more responders to more similarity')
R.p('Denser codes overlap more by chance. For 0/1 responses the cosine similarity is the overlap divided by the geometric mean of the code sizes:')
R.eq(r'\cos(a, b) = \frac{n_{ab}}{\sqrt{n_a\, n_b}}')
R.p('If odor $b$\'s responders are placed at random, independently of $a$\'s, the overlap $n_{ab}$ follows a hypergeometric distribution with mean '
    '$n_a n_b / N$. The expected cosine is therefore')
R.eq(r'\mathbb{E}[\cos(a, b)] = \frac{n_a n_b / N}{\sqrt{n_a n_b}} = \sqrt{p_a\, p_b}, \qquad \mathrm{SD} \approx \sqrt{\frac{(1-p_a)(1-p_b)}{N-1}}')
R.p('So an odor that gains responders becomes more similar to every other odor, and its whole row of the similarity matrix rises. This is the '
    'link between Step 1 (per-odor gains) and the per-odor part of $\\Delta S$ (Report 06, Section 4).')
R.box('A floor, not the exact chance level, for KCs', [
    'The formula assumes every unit is equally likely to respond. If unit $k$ responds with probability $\\pi_k$, the expected cosine is about '
    '$(p_a p_b + \\mathrm{cov}_k(\\pi_{a}, \\pi_{b}))/\\sqrt{p_a p_b}$, which is higher when the same units respond easily to both odors. '
    f'In the connectome model, two unrelated odors that each drive 10% of KCs have a cosine of {CJ["b connectome model, mean over 3 mushroom bodies"]["ORN-table glomeruli (as in the model)"]["cosine / sqrt(pa pb) at pa = pb = 0.1"]:.2f} × '
    f'$\\sqrt{{p_a p_b}}$ with the ORN-table glomeruli, and {CJ["b connectome model, mean over 3 mushroom bodies"]["every glomerulus responding"]["cosine / sqrt(pa pb) at pa = pb = 0.1"]:.2f} × with '
    'every glomerulus responding, because some KCs are easier to drive than others.'], 'plain')


def fig_chance():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1))
    a = axs[0]; s = CS[CS.part.str.startswith('a')]
    a.errorbar(s.sqrt_pa_pb, s.cosine_mean, yerr=[s.cosine_mean - s.cosine_5, s.cosine_95 - s.cosine_mean], fmt='o', ms=3, color=L.INK, ecolor=L.GRAY, elinewidth=0.8)
    lim = [0, s.sqrt_pa_pb.max() * 1.05]; a.plot(lim, lim, color=L.BLOCKED, lw=1, ls='--', label='$\\sqrt{p_a p_b}$')
    a.set_xlabel('$\\sqrt{p_a p_b}$ (chance overlap formula)'); a.set_ylabel('cosine of two random 0/1 codes\n(mean and 5–95%, N = 2,000 units)')
    a.legend(fontsize=7.5); a.set_title('a  Random codes follow the formula')
    a = axs[1]
    a.scatter(CO.d_chance, CO.d_s01, s=14, color=L.INK, alpha=0.75, lw=0)
    b = np.polyfit(CO.d_chance, CO.d_s01, 1); xs = np.linspace(CO.d_chance.min(), CO.d_chance.max(), 10); a.plot(xs, np.polyval(b, xs), color=L.BLOCKED, lw=1.2)
    r_ = np.corrcoef(CO.d_chance, CO.d_s01)[0, 1]
    a.set_xlabel('change in chance overlap $\\sqrt{p_a p_b}$, TNT − TNT3c'); a.set_ylabel('change in 0/1 cosine similarity, TNT − TNT3c')
    a.set_title(f'b  Data: per pair, r = {r_:.2f}, slope {b[0]:.1f}')
    fig.tight_layout(); return fig


R.fig(fig_chance(), '**a**, Simulation: for pairs of random, unrelated 0/1 codes over 2,000 units, the mean cosine (dots) and its 5–95% range '
      '(grey) against $\\sqrt{p_a p_b}$; the dashed line is equality. **b**, Data: each point is an odor pair. Horizontal: the change in '
      '$\\sqrt{p_a p_b}$ between TNT and TNT3c, computed per recording from the fractions responding (> 1) and averaged. Vertical: the change '
      'in cosine similarity of the 0/1 responses. The orange line is the least-squares fit.', 'Chance overlap and the similarity change.')
c = CJ['c data, mean over the 120 odor pairs']
R.p(f'In the data the real 0/1 similarity is about twice $\\sqrt{{p_a p_b}}$ in both conditions ({c["TNT3c"]["0/1 similarity"]:.3f} vs '
    f'{c["TNT3c"]["sqrt(pa pb)"]:.3f} in TNT3c, {c["TNT"]["0/1 similarity"]:.3f} vs {c["TNT"]["sqrt(pa pb)"]:.3f} in TNT). That excess can come '
    'from related odors or from units that respond to many odors. A shuffle that keeps each unit\'s number of responses and each odor\'s '
    f'fraction reproduces the mean ({c["TNT3c"]["shuffled keeping each unit\'s number of responses"]:.3f} in TNT3c), so with 16 odors the two cannot be told apart.')

# ---------------------------------------------------------------- 3 which rule fits
R.section('Step 3: which inhibition rule fits the data')


def fig_rules():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for a, rn in zip(axs, ['ORN', 'GH146']):
        j = FJ[rn]; y = np.arange(len(RULES))[::-1]
        for yi, ru in zip(y, RULES):
            lo, hi = j['bootstrap over recordings, r 5-95%'][ru]
            a.plot([lo, hi], [yi, yi], color=L.GRAY, lw=1.4)
            a.scatter(j['r over pairs with the data'][ru], yi, color=RC[ru], s=40, zorder=3)
            for k, mk in zip(j['r over pairs per mushroom body'][ru], ['o', 's', '^']):
                a.scatter(j['r over pairs per mushroom body'][ru][k], yi - 0.25, marker=mk, s=14, facecolor='white', edgecolor=RC[ru], lw=0.9)
        a.set_yticks(y); a.set_yticklabels(RULES); a.set_xlim(0, 0.7); a.set_xlabel('r over odor pairs, model vs data change in similarity')
        a.set_title(f'{"a" if rn == "ORN" else "b"}  {rn} input')
    hs = [plt.Line2D([], [], marker=m, ls='', mfc='white', mec=L.MUTED, label=k) for m, k in zip(['o', 's', '^'], ['hemibrain R', 'male CNS L', 'male CNS R'])]
    hs.append(plt.Line2D([], [], color=L.GRAY, lw=1.4, label='bootstrap 5–95%'))
    axs[1].legend(handles=hs, fontsize=7, loc='lower right')
    fig.tight_layout(); return fig


R.fig(fig_rules(), 'For each inhibition rule, the large dot is $r$ between the model\'s change in similarity (mean of the three mushroom bodies) '
      'and the data\'s, over odor pairs. The grey line is its bootstrap 5–95% range over recordings. The small open markers are the single '
      'mushroom bodies.', 'Fit of the four inhibition rules.')
_pm = J['r over pairs per mushroom body']; _nmb = sum(_pm['feedback'][k] > _pm['fixed'][k] for k in _pm['feedback'])
_mbtxt = 'every mushroom body' if _nmb == len(_pm['feedback']) else f'{_nmb} of {len(_pm["feedback"])} mushroom bodies'
R.ul([f'Feedback fits better than a fixed raise in {_mbtxt} and in {L.pct(J["bootstrap over recordings, fraction of draws with feedback r higher than"]["fixed"], 0)} of '
      f'bootstrap draws (r {J["r over pairs with the data"]["feedback"]:.2f} vs {J["r over pairs with the data"]["fixed"]:.2f} with ORN input).',
      f'Feedback does as well as setting each odor\'s intact fraction from the data (r {J["r over pairs with the data"]["TNT3c per odor"]:.2f}), '
      'although it is not given the per-odor TNT3c fractions.',
      f'The per-odor part of the change matches the data with Spearman {J["spearman of per-odor parts with the data"]["feedback"]:.2f} for feedback and '
      f'{J["spearman of per-odor parts with the data"]["fixed"]:.2f} for a fixed raise.'])

# ---------------------------------------------------------------- 4 data signature
R.section('Step 4: is the feedback signature in the recordings?')
d = FJ['data, 16 odors']
sp = d['spearman, odor strength (half of the recordings) vs gain TNT / TNT3c (other half), 2000 splits: mean, 5%, 95%, fraction > 0']
SIZE = {'fraction > 1': lambda X: (X > 1).mean(0), 'mean positive response': lambda X: np.clip(X, 0, None).mean(0)}
ST = {k: {c: np.array([f(v.values) for v in tc.D[c].values()]) for c in tc.COND} for k, f in SIZE.items()}
NT, N3 = len(ST['fraction > 1']['TNT']), len(ST['fraction > 1']['TNT3c'])
rs8 = np.random.default_rng(5); res8 = {}
for _ in range(2000):
    t, n = rs8.permutation(NT), rs8.permutation(N3)
    for k in SIZE:
        gain = ST[k]['TNT'][t[4:]].mean(0) / ST[k]['TNT3c'][n[5:]].mean(0)
        for sk, s_ in {'TNT only': ST[k]['TNT'][t[:4]].mean(0), 'mean of TNT and TNT3c (earlier version)': (ST[k]['TNT'][t[:4]].mean(0) + ST[k]['TNT3c'][n[:5]].mean(0)) / 2,
                       'TNT3c only': ST[k]['TNT3c'][n[:5]].mean(0)}.items():
            res8.setdefault((sk, k), []).append(spearmanr(s_, gain)[0])
SPL = pd.DataFrame([dict(strength=sk, measure=k, mean_spearman=np.mean(v), q5=np.quantile(v, .05), q95=np.quantile(v, .95), share_positive=np.mean(np.array(v) > 0))
                    for (sk, k), v in res8.items()])
SPL.round(4).to_csv(OWV + 'TNT_feedback_signature_split_half.csv', index=False)
sp_ = lambda sk, k: SPL[(SPL.strength == sk) & (SPL.measure == k)].iloc[0]
R.p('If APL acted as uniform feedback, odors that drive more units should gain more units when APL is blocked. Odor strength is measured '
    'in TNT, where APL is blocked: there it reflects the feed-forward drive from the PNs, and it is what the model\'s horizontal axis in '
    'Figure 1 uses. TNT3c is already evened out by feedback, so mixing it in would blur the measure. To avoid shared noise, strength and '
    f'ratio come from separate halves of the recordings: 4 TNT recordings give the strength, and the other {NT - 4} TNT and {N3 - 5} of the '
    f'{N3} TNT3c recordings give the ratio. This is repeated for 2,000 random splits.')


def fig_data():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fr = {c: tc.FRAC[1.0][c].mean(0) for c in tc.COND}; strength = fr['TNT']; gain = fr['TNT'] / fr['TNT3c']
    mp = {c: np.array([np.clip(v.values, 0, None).mean(0) for v in tc.D[c].values()]).mean(0) for c in tc.COND}      # as SIZE in tnt_feedback_vs_fixed.py
    a = axs[0]; a.scatter(strength * 100, gain, color=L.INK, s=22)
    for o, x, y in zip(tc.O, strength * 100, gain): a.annotate(o, (x, y), xytext=(3, 2), textcoords='offset points', fontsize=6.8, color=L.MUTED)
    a.set_xlabel('odor strength: fraction responding in TNT, %'); a.set_ylabel('blocked ÷ intact ratio of units responding (TNT ÷ TNT3c)')
    a.set_title(f'a  Units responding: split-half ρ = {sp_("TNT only", "fraction > 1").mean_spearman:+.2f}')
    a = axs[1]; s2 = mp['TNT']; g2 = mp['TNT'] / mp['TNT3c']
    a.scatter(s2, g2, color=L.INK, s=22)
    for o, x, y in zip(tc.O, s2, g2): a.annotate(o, (x, y), xytext=(3, 2), textcoords='offset points', fontsize=6.8, color=L.MUTED)
    a.set_xlabel('odor strength: mean positive response in TNT (FC z-score)'); a.set_ylabel('TNT ÷ TNT3c ratio of mean positive response')
    a.set_title(f'b  Response size: split-half ρ = {sp_("TNT only", "mean positive response").mean_spearman:+.2f}')
    fig.tight_layout(); return fig


R.fig(fig_data(), 'Each point is one odor, using all recordings (the split-half Spearman ρ in the titles uses separate halves for the two axes, '
      'mean of 2,000 splits). **a**, Odor strength as the fraction of units responding (> 1) in TNT against how many times more units respond in '
      'TNT than in TNT3c. **b**, Odor strength as the mean positive response in TNT (negative responses set to 0, then averaged over all units; '
      'FC z-score) against its ratio TNT ÷ TNT3c.', 'The feedback signature in the data.')
TAB_SIG = R.table(['odor strength measured from', 'measure (strength and ratio)', 'mean Spearman ρ (2,000 splits)', '5–95%', 'share of splits > 0'],
        [[sk, k, F(r_.mean_spearman, 2, True), f'{F(r_.q5, 2, True)} to {F(r_.q95, 2, True)}', f'{r_.share_positive * 100:.0f}%']
         for k in SIZE for sk in ['TNT only', 'mean of TNT and TNT3c (earlier version)', 'TNT3c only'] for r_ in [sp_(sk, k)]],
        f'Strength and ratio use the same measure in each row. The ratio always comes from the other half of the recordings. The model with '
        f'uniform feedback predicts ρ = {SPJ["feedback"]:+.2f} for the fraction responding.', 'The feedback signature with different measures of odor strength.')
f1, f2 = sp_('TNT only', 'fraction > 1'), sp_('TNT only', 'mean positive response')
CVd = d["spread across odors (coefficient of variation), TNT and TNT3c"]
R.ul([f'**Units responding**: the ratio is {"weakly related" if abs(f1.mean_spearman) < 0.4 else "related"} to odor strength (mean split-half '
      f'Spearman {F(f1.mean_spearman, 2, True)}, 5–95% {F(f1.q5, 2, True)} to {F(f1.q95, 2, True)}, positive in {f1.share_positive * 100:.0f}% of splits), far from the '
      f'{SPJ["feedback"]:+.2f} that uniform feedback predicts. The spread across odors is about the same in TNT and TNT3c (CV '
      f'{CVd["fraction > 1"][0]:.2f} and {CVd["fraction > 1"][1]:.2f}), whereas uniform feedback predicts less spread with APL intact.',
      f'**Response size**: stronger odors do gain more (Spearman {f2.mean_spearman:+.2f} for the mean positive response, positive in '
      f'{min(f2.share_positive * 100, 99.9):.{0 if f2.share_positive < 0.995 else 1}f}% of splits).',
      f'Measuring strength from the mean of TNT and TNT3c, as in the earlier version, gives the same conclusion (Table {TAB_SIG}).',
      'So the recordings show the feedback signature in response size but only weakly, if at all, in the number of responding units.'])
R.box('Caveats', ['The rules other than feedback are not mechanisms; they are reference points with the same mean inhibition.',
                  'Noise inflates TNT3c\'s apparent number of weak responders differently from TNT\'s, which can blur the gain-vs-strength relation.'], 'caveat')
R.save()
