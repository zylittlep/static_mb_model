"""Report 07: the block model pinned to the TNT / TNT3c recordings, and the comparison (tnt_compare.py)."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import report_lib as L
import tnt_compare as tc
abm = tc.abm
L.style()
F = L.fmt
O = tc.O
FIT = pd.read_csv(L.OW + 'TNT_vs_model_fits.csv'); PR = pd.read_csv(L.OW + 'TNT_vs_model_odor_pairs.csv')
PO = pd.read_csv(L.OW + 'TNT_vs_model_per_odor.csv'); NUL = pd.read_csv(L.OW + 'TNT_vs_model_nulls.csv')
SUMM = json.load(open(L.OW + 'TNT_vs_model_summary.json'))
VERS = ['U', 'C', 'P0', 'P', 'L', 'M0', 'M', 'F0', 'F']
VL = {'U': 'U  uniform', 'C': 'C  per-KC counts', 'P0': 'P0  boutons, alike', 'P': 'P  boutons per glom.', 'L': 'L  claws per glom.',
      'M0': 'M0  U + boutons', 'M': 'M  U + boutons per glom.', 'F0': 'F0  PN drive', 'F': 'F  PN drive per glom.'}
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
             inputs=['The model of Report 05 (hemibrain R, male CNS L and R)', 'The recordings and the quantities of Report 06'],
             outputs=['`odor_weighted/TNT_vs_model_fits.csv`, `TNT_vs_model_nulls.csv`, `TNT_vs_model_odor_pairs.csv`, '
                      '`TNT_vs_model_per_odor.csv`, `TNT_vs_model_summary.json`, `TNT_vs_model.png`'],
             scripts=['`scripts/tnt_compare.py`', '`scripts/reports/r07_model_vs_tnt.py` (this report)'])
R.p('Report 05 ran the model with assumed coding levels (20% of KCs respond with APL blocked, 10% intact). Here the model is instead '
    '**pinned** to the recordings: the fraction of KCs responding to each odor with APL blocked is set to the TNT fraction for that odor, '
    'and the mean intact fraction to TNT3c\'s. The model\'s change in similarity is then compared with the data\'s, pair by pair. The '
    'pinning sets coding levels only. The pattern of similarity changes across pairs is not fitted, so it is a prediction.')
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
      'The model value is averaged over the three mushroom bodies before the correlation. Pairs with an odor that drives fewer than 0.5% of '
      'model KCs are excluded only in Report 05; here every pair counts (120 with ORN input, 78 with GH146).',
      '**Odor-label permutation** $p$: the odor labels of the model matrix are permuted 5,000 times. This keeps the matrix structure and '
      'tests whether the match depends on which odor is which.',
      '**Bootstrap range**: recordings are resampled with replacement within each condition (2,000 draws), and the 5th–95th percentile of $r$ is reported.',
      '**Wiring nulls**: each wiring version is re-solved with its wiring shuffled 200 times per mushroom body. C swaps $(u_k, v_k)$ pairs '
      'among KCs of the same class. P and M shuffle $p_g$ across glomeruli. F shuffles $\\tilde p^{\\mathrm{in}}_g$. $p$ is the share of shuffles '
      'with $r$ at least as high as the real wiring.',
      '**Ceiling**: because the data are noisy, even a perfect model would not reach $r = 1$. The split-half reliability of $\\Delta S^{\\mathrm{data}}$ '
      '(Spearman–Brown corrected) caps $r$ near 0.84.',
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
R.p(f'With uniform APL, $r$ = {fit("ORN", "U", "r_pairs"):.2f} with ORN input (permutation $p$ = {fit("ORN", "U", "r_pairs_p_odor_permutation"):.3f}; '
    f'bootstrap {fit("ORN", "U", "r_pairs_boot_5"):.2f}–{fit("ORN", "U", "r_pairs_boot_95"):.2f}) and {fit("GH146", "U", "r_pairs"):.2f} with GH146 input. '
    f'Against the ceiling of 0.84, that is about $(0.46/0.84)^2 \\approx$ 30% of the reproducible pattern. Per mushroom body $r$ is '
    + ', '.join(f'{v:.2f} ({k.replace("malecns", "male CNS")})' for k, v in SUMM['fits']['threshold 1.0, ORN']['uniform U: per mushroom body r'].items()) + '.')


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
        a.axvline(fit(rn, 'U', 'r_pairs'), color=L.INK, ls=':', lw=0.8); a.axvline(0.84, color=L.BLOCKED, ls='--', lw=1)
        a.text(0.84, len(VERS) - 0.4, ' ceiling ≈ 0.84', fontsize=7, color=L.BLOCKED)
        a.set_yticks(y); a.set_yticklabels([VL[v] for v in VERS], fontsize=7.8); a.set_xlim(0, 1.0)
        a.set_xlabel('r over odor pairs, model vs data change'); a.set_title(f'{"a" if rn == "ORN" else "b"}  {rn} input')
    a.plot([], [], color=L.GRAY, lw=1.4, label='bootstrap 5–95% (recordings resampled)'); a.plot([], [], color=L.PURPLE, lw=3, alpha=0.6, label='wiring shuffled: 5–95%')
    a.legend(fontsize=7, loc='lower right')
    fig.tight_layout(); return fig


R.fig(fig_versions(), 'For each APL version, black dots are $r$ between the model\'s and the data\'s change in similarity over odor pairs. Grey '
      'lines are the bootstrap 5–95% range over recordings. Purple bars are the 5–95% range of $r$ when that version\'s wiring is shuffled '
      '(200 shuffles per mushroom body), and $p$ is the share of shuffles that do at least as well as the real wiring. The dotted line marks '
      'uniform APL; the dashed orange line marks the ceiling set by the data\'s noise.', 'Every version against the data.')
R.table(['version', '$r$ ORN', 'vs U', 'shuffle $p$', '$r$ GH146', 'vs U', 'shuffle $p$', 'suppression $\\rho$ ORN', 'mean model $\\Delta S$ (ORN)'],
        [[VL[v], F(fit('ORN', v, 'r_pairs'), 2), F(fit('ORN', v, 'r_minus_U'), 2, True), '' if np.isnan(fit('ORN', v, 'null_p_r')) else F(fit('ORN', v, 'null_p_r'), 2),
          F(fit('GH146', v, 'r_pairs'), 2), F(fit('GH146', v, 'r_minus_U'), 2, True), '' if np.isnan(fit('GH146', v, 'null_p_r')) else F(fit('GH146', v, 'null_p_r'), 2),
          F(fit('ORN', v, 'rho_suppression'), 2), F(fit('ORN', v, 'model_mean_dS'), 3)] for v in VERS],
        title='Fit of every version (threshold 1).', caption=f'Data mean $\\Delta S$ = {fit("ORN", "U", "data_mean_dS"):.3f} (16 odors). Shuffle $p$ only for the four wiring versions with a null.',
        num_cols=list(range(1, 9)))
R.ul(['**Feedback matters.** Uniform feedback predicts the pattern; inhibition of fixed size or fed forward from the PNs (F0, F) does worse '
      f'(r = {fit("ORN", "F0", "r_pairs"):.2f} and {fit("ORN", "F", "r_pairs"):.2f}). Report 08 explains why.',
      '**Specific wiring adds little.** C, P and M change $r$ by a few hundredths, and none beats its own shuffles at $p$ < 0.05 with ORN '
      'input. L (inhibition on claws per glomerulus) fits worse.',
      f'**Suppression per odor is not predicted** (ρ ≤ {max(fit("ORN", v, "rho_suppression") for v in VERS):.2f}).',
      f'**Size.** The model\'s mean change ({fit("ORN", "U", "model_mean_dS"):+.3f}) is about half the data\'s ({fit("ORN", "U", "data_mean_dS"):+.3f}).'])


def fig_supp():
    fig, a = plt.subplots(figsize=(6, 4.2))
    P = PO[(PO.threshold == 1.0) & (PO.responses == 'ORN')]
    a.errorbar(P.suppression_model_U, P.suppression_data, yerr=[P.suppression_data - P.suppression_boot_5, P.suppression_boot_95 - P.suppression_data],
               fmt='o', color=L.INK, ms=4, ecolor=L.GRAY, elinewidth=1)
    for _, r in P.iterrows(): a.annotate(r.odor, (r.suppression_model_U, r.suppression_data), xytext=(4, 2), textcoords='offset points', fontsize=7, color=L.MUTED)
    a.set_xlabel('model suppression $s_o$, uniform APL (blocked ÷ intact fraction)'); a.set_ylabel('data suppression $s_o$ (TNT ÷ TNT3c fraction)')
    a.set_title(f'Suppression per odor, ORN input (ρ = {fit("ORN", "U", "rho_suppression"):.2f})')
    fig.tight_layout(); return fig


R.fig(fig_supp(), 'Each point is one odor. Horizontal: how many times more model KCs respond with APL blocked than intact (uniform APL, '
      'mean of three mushroom bodies). Vertical: the same ratio in the data, with its bootstrap 5–95% range over recordings (grey). '
      'The model does not rank the odors the way the data do.', 'Which odors APL suppresses most.')
R.box('Caveats', ['Inputs are ORN or GH146 averages from other flies, not the recorded flies\' PNs. Imaged units are not identified KCs.',
                  'Three mushroom bodies from two connectomes. Their uniform-model fits range from 0.27 to 0.53, so wiring tests that average '
                  'over them have little power.',
                  'Many tests: of 8 real-vs-shuffle tests on odor pairs, one passes at $p$ < 0.05 (F with ORN input), and that version still fits '
                  'worse than uniform feedback.'], 'caveat')
R.save()
