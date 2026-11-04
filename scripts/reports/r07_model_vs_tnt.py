"""Report 07: the block model pinned to the TNT / TNT3c recordings, and the comparison (tnt_compare.py).
Rebuilt 2026-11-04 for Yang's revision: corrected calyx, new claws, new ORN table (APL_VARIANT=claws_v2m2), version MC; Equation 1
worked through with a mock example; why the ratio to the ceiling is squared; Figure 4 with a y = x line. Old version in archive_2026-11-03/."""
import sys, json, os
os.environ['APL_VARIANT'] = 'claws_v2m2'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import report_lib as L
import tnt_compare as tc
abm = tc.abm
L.style(); L.DATE = '2026-11-04'
F = L.fmt
O = tc.O
OWV = tc.OUT                                  # with_outside_ROI_synapses/claws_v2m2/odor_weighted/
FIT = pd.read_csv(OWV + 'TNT_vs_model_fits.csv'); PR = pd.read_csv(OWV + 'TNT_vs_model_odor_pairs.csv')
PO = pd.read_csv(OWV + 'TNT_vs_model_per_odor.csv'); NUL = pd.read_csv(OWV + 'TNT_vs_model_nulls.csv')
SUMM = json.load(open(OWV + 'TNT_vs_model_summary.json'))
VERS = ['U', 'C', 'P0', 'P', 'L', 'M0', 'M', 'MC', 'F0', 'F']
assert set(VERS) == set(tc.VERS), tc.VERS
VL = {'U': 'U  uniform', 'C': 'C  per-KC counts', 'P0': 'P0  boutons, alike', 'P': 'P  boutons per glom.', 'L': 'L  claws per glom.',
      'M0': 'M0  U + boutons', 'M': 'M  U + boutons per glom.', 'MC': 'MC  C + boutons per glom.', 'F0': 'F0  PN drive', 'F': 'F  PN drive per glom.'}
FO = SUMM['fits']['threshold 1.0, ORN']
REL = FO['split-half (recordings) reliability of dS, full-data estimate']; CEIL = FO['ceiling for r with a model (sqrt of that)']
fit = lambda rn, v, col: float(FIT[(FIT.threshold == 1.0) & (FIT.responses == rn) & (FIT.version == v)][col].iloc[0])

# ---- replicate the pinning for one mushroom body (hemibrain R, ORN, U)
k0 = 'hemibrain R'; mb = tc.MB[k0]; Rr = tc.RESP['ORN']
W = mb['W']; gl = [g for g in W.columns if g in Rr.index]; Wm, Rp = W[gl].values, np.clip(Rr.loc[gl].values, 0, None)
H = Wm @ Rp; clb = tc.CLD[1.0]['TNT'].reindex(Rr.columns).values; target = float(tc.CLD[1.0]['TNT3c'].reindex(Rr.columns).mean())
G = np.array([1.0 / np.quantile(H[:, j], 1 - clb[j]) for j in range(H.shape[1])]); X = Wm @ (Rp * G[None])
one, oneg = np.ones(len(X)), np.ones(len(gl))
aU = abm.calib(lambda a: abm.cl_of(abm.model_y(Wm, Rp * G[None], 1.0, a, 0, one, one, oneg)).mean(), target)
YU = abm.model_y(Wm, Rp * G[None], 1.0, aU, 0, one, one, oneg)
assert np.allclose((X > 1).mean(0), clb, atol=1.5 / len(X)) and abs(abm.cl_of(YU).mean() - target) < 1e-3

R = L.Report('07', 'When the block model is pinned to the recordings, how well does it predict which odor pairs become more similar when APL is '
             'blocked, and does APL\'s specific wiring improve that prediction?',
             inputs=['The model of Report 05 (hemibrain R, male CNS L and R; corrected calyx, new claws, new ORN table)',
                     'The recordings and the quantities of Report 06'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/odor_weighted/`: `TNT_vs_model_fits.csv`, `TNT_vs_model_nulls.csv`, '
                      '`TNT_vs_model_odor_pairs.csv`, `TNT_vs_model_per_odor.csv`, `TNT_vs_model_summary.json`, `TNT_vs_model.png`'],
             scripts=['`scripts/tnt_compare.py`', '`scripts/reports/r07_model_vs_tnt.py` (this report)'])
R.p('Report 05 ran the model with assumed coding levels (20% of KCs respond with APL blocked, 10% intact). Here the model is instead '
    '**pinned** to the recordings: the fraction of KCs responding to each odor with APL blocked is set to the TNT fraction for that odor, '
    'and the mean intact fraction to TNT3c\'s. The model\'s change in similarity is then compared with the data\'s, pair by pair. The '
    'pinning sets coding levels only. The pattern of similarity changes across pairs is not fitted, so it is a prediction.')
R.box('What changed in this revision (2026-11-04)', ["The model is rebuilt on the corrected calyx, the new claws and the new ORN table; version MC (per-KC counts plus bouton inhibition per glomerulus) is added.", "Equation 1 (the input gain $g_o$) is worked through with a mock example.", "New: why the ratio to the ceiling is squared, with a simulation.", "The suppression figure has a y = x line and equal axes."], 'plain')
R.symbols([
    ('x_{ko},\\ y_{ko},\\ A_o,\\ a,\\ u_k,\\ v_k,\\ p_g', 'As in Report 05', ''),
    ('\\theta', 'KC threshold, fixed at 1 here (the scale is carried by $g_o$)', '1'),
    ('g_o', 'Input gain of odor $o$: multiplies every glomerulus\'s response to $o$ so that the blocked model matches TNT', 'one per odor, mushroom body and input table'),
    ('\\phi^{\\mathrm{TNT}}_o,\\ \\phi^{\\mathrm{TNT3c}}_o', 'Fraction of units responding (> 1) in the data (Report 06)', '0 to 1'),
    ('Q_o(q)', 'The $q$-quantile of the drive $\\sum_g W_{kg} r_{go}$ over KCs', 'response units'),
    ('\\Delta S^{\\mathrm{data}}_{ab}', '$S^{\\mathrm{TNT}}_{ab} - S^{\\mathrm{TNT3c}}_{ab}$ (Report 06)', 'difference in cosine'),
    ('\\Delta S^{\\mathrm{model}}_{ab}', 'Model similarity blocked − intact, each averaged over the three mushroom bodies', 'difference in cosine'),
    ('r', 'Pearson correlation over odor pairs between $\\Delta S^{\\mathrm{data}}$ and $\\Delta S^{\\mathrm{model}}$', '−1 to 1'),
    ('s_o', 'Suppression of odor $o$: blocked ÷ intact fraction responding (data or model)', '×'),
    ('\\rho', 'Spearman correlation over odors between data and model suppression', '−1 to 1'),
    ('p', 'Probability under a null (odor labels permuted, or wiring shuffled) of a value at least as large', '0 to 1'),
])

# ---------------------------------------------------------------- 1 pinning
R.section('Step 1: pinning each odor to the TNT fraction')
R.p('With APL blocked the model is $y_{ko} = [x_{ko} - \\theta]_+$. Instead of one threshold for all odors, each odor\'s glomerular responses '
    'are multiplied by a gain $g_o$, with $\\theta = 1$. The gain is chosen so that exactly the TNT fraction of KCs exceeds the threshold:')
R.eq(r'x_{ko} = g_o \sum_g W_{kg}\, r_{go}, \qquad g_o = \frac{\theta}{Q_o\!\left(1 - \phi^{\mathrm{TNT}}_o\right)}',
     note='$Q_o(q)$ = the $q$-quantile over KCs of $\\sum_g W_{kg} r_{go}$; dividing by it puts the KC at that quantile exactly at threshold')
R.p('So an odor that drives 25% of units in TNT drives 25% of model KCs with APL blocked. Only the order of the KCs\' drives comes from the '
    'connectome and the ORN (or GH146) responses; the level comes from the data. Then, with APL intact, a single APL gain $a$ per mushroom '
    f'body and version is calibrated so that the mean fraction over odors equals TNT3c\'s ({L.pct(target, 2)} for the 16 odors). How much each '
    'odor is suppressed is left to the model.')
R.sub('Equation 1 with numbers: a mock example')
x7 = np.array([0.12, 0.30, 0.05, 0.42, 0.22, 0.18, 0.60, 0.35, 0.08, 0.50]); phi7 = 0.20
Q7 = float(np.quantile(x7, 1 - phi7)); g7 = 1 / Q7; srt = np.sort(x7)
R.p('The goal of pinning is that, with APL blocked, the model drives exactly the fraction of KCs that TNT drives for each odor. The '
    'threshold is fixed at θ = 1, and each odor\'s input is multiplied by a gain $g_o$ chosen so that the right number of KCs end up '
    'above 1. Here is the calculation for a mock mushroom body of 10 KCs and one odor that drives 20% of units in TNT:')
R.ul([f'**1. Unscaled drives** $\\sum_g W_{{kg}} r_{{go}}$ of the 10 KCs: {", ".join(f"{v:.2f}" for v in x7)}.',
      f'**2. Sort them:** {", ".join(f"{v:.2f}" for v in srt)}.',
      f'**3. The quantile.** 20% should respond, so the threshold must sit at the 80th percentile, $Q_o(1 - 0.20) = Q_o(0.80)$. With 10 '
      f'values its position is 0.80 × 9 = 7.2 places from the bottom (counting from 0): between the 8th value ({srt[7]:.2f}) and the 9th '
      f'({srt[8]:.2f}), 20% of the way: {srt[7]:.2f} + 0.2 × ({srt[8]:.2f} − {srt[7]:.2f}) = **{Q7:.3f}**.',
      f'**4. The gain.** $g_o = \\theta / Q_o(0.80) = 1 / {Q7:.3f}$ = **{g7:.3f}**.',
      f'**5. Scaled drives** $x_{{ko}} = g_o \\sum_g W_{{kg}} r_{{go}}$: {", ".join(f"{v * g7:.2f}" for v in x7)}. Exactly two (from '
      f'{srt[9]:.2f} and {srt[8]:.2f}) are above 1, so 2 of 10 = 20% respond, as in TNT. The KC at {srt[7]:.2f} ends just below, at {srt[7] * g7:.2f}.'])
assert (x7 * g7 > 1).sum() == 2
R.p('Dividing by the quantile only moves the threshold to the right place in each odor\'s distribution. The order of the KCs\' drives '
    '(which KC responds) still comes from the connectome and the ORN responses. An odor with weak ORN responses has small drives, a small '
    'quantile, and therefore a large gain (Figure 1b).')


def fig_pin():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = axs[0]; o = np.argsort(clb); x = np.arange(len(O))
    a.bar(x - 0.2, clb[o] * 100, 0.38, color=L.BLOCKED, alpha=0.35, label='data TNT, $\\phi^{TNT}_o$')
    a.scatter(x - 0.2, (X > 1).mean(0)[o] * 100, color=L.BLOCKED, s=18, zorder=3, label='model blocked (pinned)')
    d3 = tc.CLD[1.0]['TNT3c'].reindex(Rr.columns).values
    a.bar(x + 0.2, d3[o] * 100, 0.38, color=L.INTACT, alpha=0.35, label='data TNT3c, $\\phi^{TNT3c}_o$')
    a.scatter(x + 0.2, abm.cl_of(YU)[o] * 100, color=L.INTACT, s=18, zorder=3, label='model intact (U; only the mean is pinned)')
    a.set_xticks(x); a.set_xticklabels(np.array(O)[o], rotation=90); a.set_ylabel('KCs / units responding, %')
    a.legend(fontsize=7); a.set_title('a  Data (bars) and pinned model (dots), hemibrain R, ORN')
    a = axs[1]
    a.scatter(np.arange(len(O)), G[o], color=L.INK, s=18)
    a.set_xticks(x); a.set_xticklabels(np.array(O)[o], rotation=90); a.set_ylabel('input gain $g_o$ (multiplies $r_{go}$)')
    a.set_title('b  The per-odor gains'); a.set_yscale('log')
    fig.tight_layout(); return fig


R.fig(fig_pin(), '**a**, For each odor (sorted by the TNT fraction), the percentage of units responding in the data (bars: orange TNT, blue '
      'TNT3c) and of model KCs responding (dots), hemibrain R with ORN input and uniform APL. Orange dots sit on the bars by construction. '
      'Blue dots are free per odor; only their mean is set to the TNT3c mean. **b**, The gain $g_o$ that pins each odor (log scale). Odors '
      'with weak ORN responses but many TNT responders need large gains.', 'Pinning the model to the recordings.')

# ---------------------------------------------------------------- 2 versions
R.section('Step 2: the APL versions compared')
R.p('All versions of Report 05 are run with the same pinning, plus two in which APL\'s activity is fed forward from the PNs instead of '
    'fed back from the KCs:')
R.eq(r'\text{F0: } y_{ko} = \left[x_{ko} - 1 - a\,A^{\mathrm{ff}}_o\right]_+, \quad A^{\mathrm{ff}}_o = \frac{\sum_g w_g\, g_o r_{go}}{\sum_g w_g}; \qquad '
     r'\text{F: the same with } w_g \text{ replaced by } w_g\,\tilde p^{\,\mathrm{in}}_g',
     note='$w_g = \\sum_k W_{kg}$ (total claw input of glomerulus $g$); $\\tilde p^{\\mathrm{in}}_g$ = PN→APL per PN→KC synapse, scaled to a claw-input-weighted mean of 1 (Report 04)')
R.p('In F0, APL\'s activity is the odor\'s mean KC input, so it does not depend on how many KCs respond. In F, glomeruli with more PN→APL '
    'synapses contribute more to it.')

# ---------------------------------------------------------------- 3 measures
R.section('Step 3: how model and data are compared')
R.ul(['**Odor-pair pattern**, $r$: the Pearson correlation over odor pairs between $\\Delta S^{\\mathrm{data}}_{ab}$ and $\\Delta S^{\\mathrm{model}}_{ab}$. '
      'The model value is averaged over the three mushroom bodies before the correlation. Every pair counts (120 with ORN input, 78 with '
      'GH146).',
      '**Odor-label permutation** $p$: the odor labels of the model matrix are permuted 5,000 times. This keeps the matrix structure and '
      'tests whether the match depends on which odor is which.',
      '**Bootstrap range**: recordings are resampled with replacement within each condition (2,000 draws), and the 5th–95th percentile of $r$ is reported.',
      '**Wiring nulls**: each wiring version is re-solved with its wiring shuffled 200 times per mushroom body. C swaps $(u_k, v_k)$ pairs '
      'among KCs of the same class. P and M shuffle $p_g$ across glomeruli. F shuffles $\\tilde p^{\\mathrm{in}}_g$. $p$ is the share of shuffles '
      'with $r$ at least as high as the real wiring.',
      f'**Ceiling**: because the data are noisy, even a perfect model would not reach $r = 1$. The split-half reliability of $\\Delta S^{{\\mathrm{{data}}}}$ '
      f'(Spearman–Brown corrected) is {REL:.2f}, which caps $r$ near $\\sqrt{{{REL:.2f}}}$ = {CEIL:.2f} (explained in Section 4).',
      '**Suppression**: Spearman $\\rho$ over odors between data and model $s_o$.'])

# ---------------------------------------------------------------- 4 results
R.section('Results')
P1 = PR[(PR.threshold == 1.0) & (PR.responses == 'ORN')]


def fig_scatter():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for a, rn in zip(axs, ['ORN', 'GH146']):
        P = PR[(PR.threshold == 1.0) & (PR.responses == rn)]
        a.scatter(P.dS_model_U, P.dS_data, s=16, color=L.INK, alpha=0.75, lw=0)
        b = np.polyfit(P.dS_model_U, P.dS_data, 1); xs = np.linspace(P.dS_model_U.min(), P.dS_model_U.max(), 10)
        a.plot(xs, np.polyval(b, xs), color=L.BLOCKED, lw=1.2)
        top = P.assign(s=P.dS_data + P.dS_model_U * 2).sort_values('s')
        for _, r in pd.concat([top.head(2), top.tail(3)]).iterrows():
            a.annotate(f'{r.odor_a}–{r.odor_b}', (r.dS_model_U, r.dS_data), xytext=(3, 2), textcoords='offset points', fontsize=6.5, color=L.MUTED)
        a.set_xlabel('model $\\Delta S$ (blocked − intact), uniform APL, mean of 3 MBs'); a.set_ylabel('data $\\Delta S$ (TNT − TNT3c)')
        a.set_title(f'{"a" if rn == "ORN" else "b"}  {rn} input: r = {fit(rn, "U", "r_pairs"):.2f} ({len(P)} pairs)')
    fig.tight_layout(); return fig


R.fig(fig_scatter(), 'Each point is one odor pair. Horizontal: the model\'s change in similarity when APL is blocked (uniform APL, pinned, '
      'mean of hemibrain R and male CNS L and R). Vertical: the data\'s change, TNT − TNT3c. The orange line is the least-squares fit. Note '
      'the different axis ranges: the model\'s change is smaller than the data\'s.', 'Model against data, pair by pair.')
rU = fit('ORN', 'U', 'r_pairs')
R.p(f'With uniform APL, $r$ = {fit("ORN", "U", "r_pairs"):.2f} with ORN input (permutation $p$ = {fit("ORN", "U", "r_pairs_p_odor_permutation"):.3f}; '
    f'bootstrap {fit("ORN", "U", "r_pairs_boot_5"):.2f}–{fit("ORN", "U", "r_pairs_boot_95"):.2f}) and {fit("GH146", "U", "r_pairs"):.2f} with GH146 input. '
    f'Against the ceiling of {CEIL:.2f}, that is about $({rU:.2f}/{CEIL:.2f})^2 \\approx$ {(rU / CEIL) ** 2 * 100:.0f}% of the reproducible pattern '
    '(why squared: below). Per mushroom body $r$ is '
    + ', '.join(f'{v:.2f} ({k.replace("malecns", "male CNS")})' for k, v in FO['uniform U: per mushroom body r'].items()) + '.')
R.sub('Why the ratio to the ceiling is squared')
R.p('A correlation r is not a share. Its square, $r^2$, is: the share of the data\'s variance that the model accounts for. The '
    'reasoning goes in three steps.')
R.ul([f'**The data are part pattern, part noise.** The split-half reliability of $\\Delta S$ is {REL:.3f}: {REL * 100:.0f}% of its variance '
      f'over pairs is reproducible pattern, and {100 - REL * 100:.0f}% is recording noise that changes from one set of recordings to the next.',
      f'**A perfect model can explain only the pattern.** Its $r^2$ would be {REL:.3f}, so its r would be $\\sqrt{{{REL:.3f}}}$ = {CEIL:.3f}. '
      'That is the ceiling.',
      f'**The uniform model.** r = {rU:.2f}, so $r^2$ = {rU ** 2:.3f}: it accounts for {rU ** 2 * 100:.0f}% of all the variance, which is '
      f'{rU ** 2 / REL * 100:.0f}% of the {REL * 100:.0f}% that is pattern. That is $r^2 / \\text{{reliability}} = (r / \\text{{ceiling}})^2$.'])
rs7 = np.random.default_rng(7); sim = []
for _ in range(400):
    sig_ = rs7.standard_normal(120); data = np.sqrt(REL) * sig_ + np.sqrt(1 - REL) * rs7.standard_normal(120)
    model = (rU / CEIL) * sig_ + np.sqrt(1 - (rU / CEIL) ** 2) * rs7.standard_normal(120)
    sim.append((np.corrcoef(model, data)[0, 1], np.corrcoef(model, sig_)[0, 1]))
sim = np.array(sim).mean(0)
R.p(f'A quick simulation confirms it: 120 mock pairs whose "data" are {REL * 100:.0f}% pattern and {100 - REL * 100:.0f}% noise, and a mock '
    f'model that shares {(rU / CEIL) ** 2 * 100:.0f}% of its variance with the pattern. The model correlates with the noisy data at r = '
    f'{sim[0]:.2f} on average and with the noise-free pattern at r = {sim[1]:.2f} (${rU / CEIL:.2f}^2$ = {(rU / CEIL) ** 2:.2f}). The unsquared '
    f'ratio {rU:.2f}/{CEIL:.2f} = {rU / CEIL:.2f} is that second correlation; squaring turns it into a share.')


def fig_versions():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, rn in zip(axs, ['ORN', 'GH146']):
        y = np.arange(len(VERS))[::-1]
        for yi, v in zip(y, VERS):
            r = fit(rn, v, 'r_pairs'); lo, hi = fit(rn, v, 'r_pairs_boot_5'), fit(rn, v, 'r_pairs_boot_95')
            a.plot([lo, hi], [yi, yi], color=L.GRAY, lw=1.4); a.scatter(r, yi, color=L.INK, s=28, zorder=3)
            nl = NUL[(NUL.responses == rn) & (NUL.test == v) & (NUL.shuffle > 0)]
            if len(nl):
                q5, q95 = nl.r_pairs.quantile([.05, .95])
                a.plot([q5, q95], [yi - 0.28, yi - 0.28], color=L.PURPLE, lw=3, alpha=0.6, solid_capstyle='butt')
                a.text(1.0, yi, f'shuffle p = {fit(rn, v, "null_p_r"):.2f}', fontsize=6.8, color=L.PURPLE, va='center', ha='right')
        cl_ = SUMM['fits'][f'threshold 1.0, {rn}']['ceiling for r with a model (sqrt of that)']
        a.axvline(fit(rn, 'U', 'r_pairs'), color=L.INK, ls=':', lw=0.8); a.axvline(cl_, color=L.BLOCKED, ls='--', lw=1)
        a.text(cl_, len(VERS) - 0.4, f' ceiling ≈ {cl_:.2f}', fontsize=7, color=L.BLOCKED)
        a.set_yticks(y); a.set_yticklabels([VL[v] for v in VERS], fontsize=7.8); a.set_xlim(0, 1.0)
        a.set_xlabel('r over odor pairs, model vs data change'); a.set_title(f'{"a" if rn == "ORN" else "b"}  {rn} input')
    h_ = [plt.Line2D([], [], color=L.GRAY, lw=1.4, label='bootstrap 5–95% (recordings resampled)'),
          plt.Line2D([], [], color=L.PURPLE, lw=3, alpha=0.6, label='wiring shuffled: 5–95%')]
    fig.legend(handles=h_, fontsize=7.5, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); return fig


R.fig(fig_versions(), 'For each APL version, black dots are $r$ between the model\'s and the data\'s change in similarity over odor pairs. Grey '
      'lines are the bootstrap 5–95% range over recordings. Purple bars are the 5–95% range of $r$ when that version\'s wiring is shuffled '
      '(200 shuffles per mushroom body), and $p$ is the share of shuffles that do at least as well as the real wiring. The dotted line marks '
      'uniform APL; the dashed orange line marks the ceiling set by the data\'s noise.', 'Every version against the data.')
WV = [v for v in VERS if not np.isnan(fit('ORN', v, 'null_p_r'))]
WIRE = ['C', 'P0', 'P', 'M0', 'M', 'MC']
dlo, dhi = min(fit('ORN', v, 'r_minus_U') for v in WIRE), max(fit('ORN', v, 'r_minus_U') for v in WIRE)
dlo_g, dhi_g = min(fit('GH146', v, 'r_minus_U') for v in WIRE), max(fit('GH146', v, 'r_minus_U') for v in WIRE)
PASSES = [(rn, v) for rn in ['ORN', 'GH146'] for v in WV if fit(rn, v, 'null_p_r') < 0.05]
PASS_LIST = ' and '.join(f'{v} with {rn} input ($p$ = {fit(rn, v, "null_p_r"):.3f}, r = {fit(rn, v, "r_pairs"):.2f})' for rn, v in PASSES)
PASS_TXT = PASS_LIST + (' passes' if len(PASSES) == 1 else ' pass')
NUMW = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}
R.table(['version', '$r$ ORN', 'vs U', 'shuffle $p$', '$r$ GH146', 'vs U', 'shuffle $p$', 'suppression $\\rho$ ORN', 'mean model $\\Delta S$ (ORN)'],
        [[VL[v], F(fit('ORN', v, 'r_pairs'), 2), F(fit('ORN', v, 'r_minus_U'), 2, True), '' if np.isnan(fit('ORN', v, 'null_p_r')) else F(fit('ORN', v, 'null_p_r'), 2),
          F(fit('GH146', v, 'r_pairs'), 2), F(fit('GH146', v, 'r_minus_U'), 2, True), '' if np.isnan(fit('GH146', v, 'null_p_r')) else F(fit('GH146', v, 'null_p_r'), 2),
          F(fit('ORN', v, 'rho_suppression'), 2), F(fit('ORN', v, 'model_mean_dS'), 3)] for v in VERS],
        title='Fit of every version (threshold 1).', caption=f'Data mean $\\Delta S$ = {fit("ORN", "U", "data_mean_dS"):.3f} (16 odors). Shuffle $p$ only for the '
        f'{len(WV)} wiring versions with a null ({", ".join(WV)}).',
        num_cols=list(range(1, 9)))
R.ul([f'**Feedback {"matters" if max(fit("ORN", "F0", "r_pairs"), fit("ORN", "F", "r_pairs")) < rU else "and feed-forward do about as well"}.** Uniform '
      f'feedback gives r = {rU:.2f}; inhibition fed forward from the PNs (F0, F) gives r = {fit("ORN", "F0", "r_pairs"):.2f} and '
      f'{fit("ORN", "F", "r_pairs"):.2f}. Report 08 explains why.',
      f'**Specific wiring.** The wiring versions ({", ".join(WIRE)}) change $r$ from U\'s by {F(dlo, 2, True)} to {F(dhi, 2, True)} with ORN input, and '
      f'{F(dlo_g, 2, True)} to {F(dhi_g, 2, True)} with GH146. ' + (f'Against their own shuffles, {PASS_TXT} at $p$ < 0.05; every other test does not. '
      if PASSES else 'None beats its own shuffles at $p$ < 0.05. ') + f'L (inhibition on claws per glomerulus) gives r = {fit("ORN", "L", "r_pairs"):.2f} '
      f'(ORN) and {fit("GH146", "L", "r_pairs"):.2f} (GH146).',
      f'**Suppression per odor is not predicted** (ρ ≤ {max(fit("ORN", v, "rho_suppression") for v in VERS):.2f}).',
      f'**Size.** The model\'s mean change ({fit("ORN", "U", "model_mean_dS"):+.3f}) is {fit("ORN", "U", "model_mean_dS") / fit("ORN", "U", "data_mean_dS"):.2f} '
      f'times the data\'s ({fit("ORN", "U", "data_mean_dS"):+.3f}).'])


PO1 = PO[(PO.threshold == 1.0) & (PO.responses == 'ORN')]; N_ABOVE = int((PO1.suppression_data > PO1.suppression_model_U).sum())


def fig_supp():
    fig, a = plt.subplots(figsize=(6, 5.6))
    P = PO[(PO.threshold == 1.0) & (PO.responses == 'ORN')]
    a.errorbar(P.suppression_model_U, P.suppression_data, yerr=[P.suppression_data - P.suppression_boot_5, P.suppression_boot_95 - P.suppression_data],
               fmt='o', color=L.INK, ms=4, ecolor=L.GRAY, elinewidth=1)
    for _, r in P.iterrows(): a.annotate(r.odor, (r.suppression_model_U, r.suppression_data), xytext=(4, 2), textcoords='offset points', fontsize=7, color=L.MUTED)
    lo_ = min(1.0, P.suppression_model_U.min(), P.suppression_boot_5.min()) * 0.95; hi_ = max(P.suppression_model_U.max(), P.suppression_boot_95.max()) * 1.05
    a.plot([lo_, hi_], [lo_, hi_], color=L.BLOCKED, ls='--', lw=1, label='y = x (model = data)')
    a.axhline(1, color=L.GRAY, lw=0.6, ls=':'); a.axvline(1, color=L.GRAY, lw=0.6, ls=':')
    a.set_xlim(lo_, hi_); a.set_ylim(lo_, hi_); a.set_aspect('equal'); a.legend(fontsize=7, loc='upper left')
    a.set_xlabel('model suppression $s_o$, uniform APL (blocked ÷ intact fraction)'); a.set_ylabel('data suppression $s_o$ (TNT ÷ TNT3c fraction)')
    a.set_title(f'Suppression per odor, ORN input (ρ = {fit("ORN", "U", "rho_suppression"):.2f})')
    fig.tight_layout(); return fig


R.fig(fig_supp(), 'Each point is one odor. Horizontal: how many times more model KCs respond with APL blocked than intact (uniform APL, '
      'mean of three mushroom bodies). Vertical: the same ratio in the data, with its bootstrap 5–95% range over recordings (grey). '
      'The dashed orange line is y = x, where model and data would agree; the dotted lines mark 1 (no suppression). Both axes use the same '
      f'scale. {N_ABOVE} of {len(PO1)} odors lie above the line: the data suppress them more than the model does.', 'Which odors APL suppresses most.')
PMB = list(FO['uniform U: per mushroom body r'].values())
R.box('Caveats', ['Inputs are ORN or GH146 averages from other flies, not the recorded flies\' PNs. Imaged units are not identified KCs.',
                  f'Three mushroom bodies from two connectomes. Their uniform-model fits range from {min(PMB):.2f} to {max(PMB):.2f} (ORN input), so '
                  'wiring tests that average over them have little power.',
                  f'Many tests: of {2 * len(WV)} real-vs-shuffle tests on odor pairs, {NUMW.get(len(PASSES), len(PASSES))} '
                  f'{"passes" if len(PASSES) == 1 else "pass"} at $p$ < 0.05' + (f': {PASS_LIST}' if PASSES else '') + f'. With {2 * len(WV)} tests, '
                  f'about {0.05 * 2 * len(WV):.1f} would pass by chance alone.'], 'caveat')
R.save()
