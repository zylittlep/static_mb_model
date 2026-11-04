"""Answers to Yang's questions of 2026-11-03 on Reports 05-11 (explanations, mock examples, extra statistics).
Writes reports/QA_2026-11-03_reports_05-11.html/.pdf. Run from anywhere: python3 -I qa_2026_11_03.py
Model numbers come from the saved outputs of Reports 05-11 (official ROIs, old claws); the uniform model of Report 05 is
re-solved here only to draw the pairs that include ms."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import report_lib as L
from orn_load import load
import apl_block_model as abm

L.REPORTS['QA'] = ('QA_2026-11-03_reports_05-11', 'Answers to the questions on Reports 05–11 (2026-11-03)')
L.DATE = '2026-11-03'
L.style()
rs = np.random.default_rng(11)
SUM5 = json.load(open(L.OW + 'APL_block_model_summary.json'))
FB = json.load(open(L.OW + 'TNT_feedback_vs_fixed.json'))
FBC = pd.read_csv(L.OW + 'TNT_feedback_vs_fixed.csv')
TVM = json.load(open(L.OW + 'TNT_vs_model_summary.json'))
F = L.fmt

R = L.Report('QA', 'What do the numbers and equations in Reports 05–11 mean, and do the claims hold up? Each question from 2026-11-03 is '
             'answered here, with small worked examples. Requests that need the model to be rerun are listed at the end of each section; '
             'they go into the rebuilt reports.',
             inputs=['Saved outputs of Reports 05–11 (`odor_weighted/`), official-ROI version and old claws',
                     'TNT and TNT3c recordings (`resp_perfly.pkl`, FC z-scores)'],
             outputs=['reports/QA_2026-11-03_reports_05-11.html, .pdf'],
             scripts=['scripts/reports/qa_2026_11_03.py (this page)'])
R.p('All numbers below come from the current reports (official ROIs, old 2 µm claws), so they match what you read. The rebuilt '
    'reports will use the corrected calyx and the claws you choose, and some values will shift.')

# =====================================================================================================================
# Report 05
# =====================================================================================================================
R.section('Report 05: the block model')
R.sub('The sentence about APL, rewritten')
R.p('**Old:** "APL is one cell whose activity, for each odor, is the average response of the KCs. It inhibits every KC by subtracting a '
    'fixed multiple of that activity."')
R.p('**New:** "APL is modelled as a single unit with one activity level per odor. Every KC excites APL, so this activity is the mean '
    'response of all KCs to that odor. APL then inhibits every KC equally: it subtracts its activity times a gain $a$ from each KC\'s '
    'input. The gain is the same for every odor, but the inhibition is not, because APL\'s activity is higher for odors that drive more KCs."')

# ---------------------------------------------------------------- p_g, q_g
R.sub('Why the bouton and claw weights of P and L are counted per synapse')
R.p('In versions P and L, inhibition does not subtract from the KC. It **divides** the input that glomerulus $g$ sends to every KC by '
    '$1 + b\\,\\tilde p_g\\,A_o$. So $\\tilde p_g$ has to answer one question: _how strongly is each unit of $g$\'s input inhibited?_ '
    'Raw APL synapse counts answer a different question, because big glomeruli have more of everything. DC1 has 6,125 PN→KC synapses '
    'and VA1v 1,965. If both had the same number of APL→PN synapses per bouton, DC1 would still have about 3 times as many in total. '
    'The size of a glomerulus\'s input already enters the model through $W_{kg}$. Using raw APL counts would count size a second time and '
    'make big glomeruli look strongly inhibited simply because they are big. Dividing by the excitatory synapses removes that: $p_g$ is '
    'the number of APL synapses that each PN→KC synapse of $g$ "sees".')
R.p('(In the sentence you copied, the symbol $g$ is typeset as math, which is why it dropped out of the copied text as "glomerulus \'s".)')
G = pd.read_csv(L.OW + 'APL_wiring_by_glomerulus.csv')
h = G[G.dataset == 'hemibrain'].set_index('glomerulus')
gl_ex = ['DC1', 'DM6', 'DM2', 'DA1', 'DL5', 'VM2', 'VA1v', 'DA2']
rows = [[g, int(h.loc[g, 'n_PNs']), f"{int(h.loc[g, 'PN_to_KC']):,}", int(h.loc[g, 'APL_to_PN']), F(h.loc[g, 'APL_to_PN_per_PN_to_KC'], 3),
         f"{int(h.loc[g, 'PN_synapses_on_claws']):,}", int(h.loc[g, 'APL_to_KC_on_claws']), F(h.loc[g, 'APL_to_KC_per_claw_PN_synapse'], 3)]
        for g in gl_ex]
R.table(['glomerulus', 'PNs', 'PN→KC synapses', 'APL→PN synapses (boutons)', '$p_g$ = APL→PN ÷ PN→KC', 'PN synapses on its claws',
         'APL→KC synapses on its claws', '$q_g$ = APL→KC ÷ claw PN synapses'], rows,
        'Eight hemibrain glomeruli. The raw counts (columns 4 and 7) mostly follow the glomerulus\'s size (columns 3 and 6). The ratios '
        '(columns 5 and 8) are what P and L use, after scaling to a claw-input-weighted mean of 1.', 'Raw counts and the per-synapse '
        'weights, hemibrain.', num_cols=[1, 2, 3, 4, 5, 6, 7])
R.table(['', 'glomerulus A (big)', 'glomerulus B (small)', 'what it means'],
        [['PN→KC synapses', '1,000', '200', 'A sends 5 times more input to KCs'],
         ['APL→PN synapses on the boutons', '100', '40', 'A receives 2.5 times more APL synapses in total'],
         ['$p_g$ (per PN→KC synapse)', '0.10', '0.20', 'but each unit of B\'s output is inhibited twice as strongly'],
         ['a KC with half its input from each', '$\\frac{0.5\\,r_A}{1 + 0.10\\,b A_o}$', '$\\frac{0.5\\,r_B}{1 + 0.20\\,b A_o}$',
          'B\'s half of the input is divided more']],
        'A mock example with two glomeruli. Raw counts would rank A as the more inhibited glomerulus; per synapse, B is.',
        'Why per synapse, in a mock example.')
R.p('$q_g$ (version L) follows the same logic for APL→KC synapses that sit on the claws of glomerulus $g$: they are divided by the PN '
    'synapses on those claws, so a glomerulus with many large claws is not counted as more inhibited only because it has more claw '
    'membrane.')

# ---------------------------------------------------------------- W normalization
R.sub('Why the input weights are divided by all of the KC\'s claw PN synapses')
cl = pd.read_csv(L.B + 'hemibrain_KC_claws_derived.csv.gz', dtype={'kc_id': str})
kw = pd.read_csv(L.B + 'KC_weights_APL_PN_per_KC.csv', dtype={'kc_id': str})
tot = cl.groupby('kc_id').n_PN_synapses.sum().to_frame('n').join(kw[kw.dataset == 'hemibrain'].set_index('kc_id').kc_class)
byc = tot.groupby('kc_class').n.agg(['mean', 'count'])
R.p('Dividing by the KC\'s total makes its drive $x_{ko}$ a **weighted average** of its glomeruli\'s responses, so every KC\'s drive is '
    'on the same scale whatever its size. Without it, the drive would be $\\sum_g n_{kg}\\,r_{go}$, which grows with the number of '
    f'synapses. KCs differ a lot in that number: hemibrain KCs have a median of {int(tot.n.median())} claw PN synapses (5–95% '
    f'{int(tot.n.quantile(.05))}–{int(tot.n.quantile(.95))}), and γ KCs have about twice as many as the others '
    f'({byc.loc["γ", "mean"]:.0f} on average, against {byc.loc["αβ", "mean"]:.0f} for αβ and {byc.loc["α′β′", "mean"]:.0f} for α′β′). '
    'With raw counts and one threshold, γ KCs would take most of the responses. The averaging assumes that a KC with more input synapses '
    'also needs more input to reach threshold, which is roughly what a larger dendrite (lower input resistance) does.')
R.p('The denominator also includes claws whose main PN is not in the response table or is multiglomerular. Those claws count as input '
    'of 0, so a KC whose claws come mostly from unmeasured glomeruli gets a smaller drive, instead of having its few measured claws scaled up.')
R.p('This is a modelling choice, and the data do not settle it. The rebuilt Report 05 will add two alternatives so the effect can be '
    'seen: raw synapse counts ($W_{kg} = n_{kg}$, one threshold), and equal weight per claw ($W_{kg}$ = share of the KC\'s claws from $g$).')

# ---------------------------------------------------------------- results numbers
R.sub('What each number in "Results of the prediction run" is')
so, sg = SUM5['model']['ORN 20% -> 10%'], SUM5['model']['GH146 20% -> 10%']
R.p('"r" in Report 05 is always the **Pearson** correlation. Spearman (rank) correlations are written $\\rho$ or $\\rho_s$ in all '
    'reports (for example the rank order in Reports 06 and 09). The rebuilt reports will spell out "Pearson r" or "Spearman ρ" each time.')
R.p('The r = 0.99 is **not** a correlation between KC responses and ORN responses. "With ORN input" only says which response table '
    'drove the model (as opposed to GH146). The correlation is over odors, between how many KCs an odor drives with APL blocked and how '
    'many more KCs respond when APL is blocked.')
R.table(['statement', 'what is computed', 'over what', 'value (ORN / GH146)'], [
    ['KCs gained by blocking track odor strength, r = 0.99', 'Pearson r between $\\phi^{\\text{blocked}}_o - \\phi^{\\text{intact,U}}_o$ (KCs gained) '
     'and $\\phi^{\\text{blocked}}_o$ (odor strength); each fraction averaged over the 3 MBs first', '16 odors (ORN) / 13 (GH146)',
     f'{so["per-odor gain U vs blocked fraction: r"]:.3f} / {sg["per-odor gain U vs blocked fraction: r"]:.3f}'],
    ['and so do all versions, r ≥ 0.999', 'Pearson r between the KCs gained under U and under each other version', 'odors',
     f'{min(so["per-odor gain (blocked - intact): r between versions (mean over MBs)"].values()):.3f} / '
     f'{min(sg["per-odor gain (blocked - intact): r between versions (mean over MBs)"].values()):.3f} (lowest version)'],
    ['mean $\\Delta S$', 'mean of $\\Delta S_{ab} = S^{\\text{blocked}}_{ab} - S^{\\text{intact,U}}_{ab}$ (cosine), each pair averaged over the 3 MBs',
     f'{so["pairs kept"]} pairs (ORN, ms left out) / {sg["pairs kept"]}',
     f'{F(so["dS U: mean, sd, min, max over pairs (mean over MBs)"][0], 3, True)} / {F(sg["dS U: mean, sd, min, max over pairs (mean over MBs)"][0], 3, True)}'],
    ['same pattern in all three MBs', 'Pearson r between two MBs\' $\\Delta S$ over pairs, for each of the 3 MB pairs; the median of the 3',
     'pairs', f'{so["dS U: r between MBs (median)"]:.2f} / {sg["dS U: r between MBs (median)"]:.2f}'],
    ['C changes $\\Delta S$ by 0.029', 'mean over pairs and MBs of $|\\Delta S^{C}_{ab} - \\Delta S^{U}_{ab}|$', 'pairs × 3 MBs',
     f'{F(so["dS C - dS U"]["mean |diff| over pairs and MBs"], 3)} / {F(sg["dS C - dS U"]["mean |diff| over pairs and MBs"], 3)}'],
    ['shuffled counts do much the same', 'Pearson r between the maps $\\Delta S^{C} - \\Delta S^{U}$ and $\\Delta S^{Cs} - \\Delta S^{U}$ (Cs = counts shuffled within class)',
     'pairs (mean over MBs)', f'{so["dS C-U vs Cs-U maps: r (mean over MBs)"]:.2f} / {sg["dS C-U vs Cs-U maps: r (mean over MBs)"]:.2f}'],
    ['what is left differs between MBs', 'as row 4, for $\\Delta S^{C} - \\Delta S^{Cs}$', 'pairs',
     f'{so["dS C - dS Cs"]["r between MBs (median)"]:.2f} / {sg["dS C - dS Cs"]["r between MBs (median)"]:.2f}'],
    ['P vs P0: consistent pattern', 'as rows 4 and 5, for $\\Delta S^{P} - \\Delta S^{P0}$', 'pairs',
     f'r {so["dS P - dS P0"]["r between MBs (median)"]:.2f}, |diff| {F(so["dS P - dS P0"]["mean |diff| over pairs and MBs"], 3)} / '
     f'r {sg["dS P - dS P0"]["r between MBs (median)"]:.2f}, |diff| {F(sg["dS P - dS P0"]["mean |diff| over pairs and MBs"], 3)}'],
    ['M vs M0', 'mean $|\\Delta S^{M} - \\Delta S^{M0}|$', 'pairs × 3 MBs',
     f'{F(so["dS M - dS M0"]["mean |diff| over pairs and MBs"], 3)} / {F(sg["dS M - dS M0"]["mean |diff| over pairs and MBs"], 3)}'],
    ['L vs P0', 'mean $|\\Delta S^{L} - \\Delta S^{P0}|$', 'pairs × 3 MBs',
     f'{F(so["dS L - dS P0"]["mean |diff| over pairs and MBs"], 3)} / {F(sg["dS L - dS P0"]["mean |diff| over pairs and MBs"], 3)}']],
    '$\\phi_o$ is the fraction of KCs responding to odor $o$. All values are for the main setting (20% of KCs respond with APL blocked, 10% '
    'intact).', 'Every number in the Results section of Report 05.')

# ---------------------------------------------------------------- strong pairs plot (with ms)
R.sub('Does similarity rise most for pairs of strong odors? (ms included)')
RESP = abm.responses(); MB, _ = abm.mushroom_bodies()
prs = []
for rn, Rt in RESP.items():
    for k, mb in MB.items():
        W = mb['W']; gl = [g for g in W.columns if g in Rt.index]
        Wm, Rp = W[gl].values, np.clip(Rt.loc[gl].values, 0, None); H = Wm @ Rp
        lo, hi = 0.0, H.max()
        for _ in range(60):
            m = (lo + hi) / 2
            if (H > m).mean(0).mean() > 0.2: lo = m
            else: hi = m
        th = (lo + hi) / 2; one, oneg = np.ones(H.shape[0]), np.ones(len(gl))
        a = abm.calib(lambda a: abm.cl_of(abm.model_y(Wm, Rp, th, a, 0, one, one, oneg)).mean(), 0.1)
        Yb, Yi = np.maximum(H - th, 0), abm.model_y(Wm, Rp, th, a, 0, one, one, oneg)
        with np.errstate(invalid='ignore'):
            Cb, Ci = abm.cosm(Yb), abm.cosm(Yi)
        fb, fi = abm.cl_of(Yb), abm.cl_of(Yi); Od = list(Rt.columns)
        for i in range(len(Od)):
            for j in range(i + 1, len(Od)):
                prs.append(dict(responses=rn, mb=k, a=Od[i], b=Od[j], dS=Cb[i, j] - Ci[i, j], strength=np.sqrt(fb[i] * fb[j]),
                                nb_a=int((Yb[:, i] > 0).sum()), nb_b=int((Yb[:, j] > 0).sum())))
PRS = pd.DataFrame(prs)
ms_n = {}
for k in MB:
    q = PRS[(PRS.responses == 'ORN') & (PRS.mb == k) & (PRS.a == 'ms')]
    q2 = PRS[(PRS.responses == 'ORN') & (PRS.mb == k) & (PRS.b == 'ms')]
    ms_n[k] = int(q.nb_a.iloc[0]) if len(q) else int(q2.nb_b.iloc[0])
PM = PRS.groupby(['responses', 'a', 'b'])[['dS', 'strength']].mean().reset_index()     # nan-skipping: ms pairs from male CNS only
PM['ms'] = (PM.a == 'ms') | (PM.b == 'ms')
stat = {}


def fig_strong():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, rn in zip(axs, ['ORN', 'GH146']):
        g = PM[(PM.responses == rn) & PM.dS.notna()]
        rho_all = spearmanr(g.strength, g.dS)[0]; rho_no = spearmanr(g.strength[~g.ms], g.dS[~g.ms])[0]
        stat[rn] = (rho_all, rho_no, len(g), int(g.ms.sum()))
        a.scatter(g.strength[~g.ms] * 100, g.dS[~g.ms], s=26, color=L.INTACT, edgecolor='white', linewidth=0.6, label='pairs without ms', zorder=3)
        a.scatter(g.strength[g.ms] * 100, g.dS[g.ms], s=34, facecolor='white', edgecolor=L.BLOCKED, linewidth=1.4, label='pairs with ms', zorder=4)
        a.axhline(0, color=L.GRAY, lw=0.7, ls=':')
        a.set_xlabel('pair strength $\\sqrt{\\phi_a\\,\\phi_b}$ with APL blocked, % of KCs')
        a.set_ylabel('$\\Delta S_{ab}$ = blocked − intact (cosine)')
        a.set_title(f'{"ab"[rn == "GH146"]}  {rn} input: Spearman ρ = {rho_all:.2f} (with ms), {rho_no:.2f} (without)', fontsize=9.5)
        a.legend(fontsize=7.5, loc='upper left')
    fig.tight_layout(); return fig


R.fig(fig_strong(), 'Each point is one odor pair (uniform APL, mean of the three mushroom bodies, 20% → 10%). Horizontal: the geometric '
      'mean of the two odors\' fractions of KCs responding with APL blocked. Vertical: how much more similar the pair is with APL blocked. '
      'Open orange circles are the pairs with ms. With ORN input, ms drives '
      f'{ms_n["hemibrain R"]} KCs in hemibrain R even with APL blocked, so its similarity is undefined there; its pairs here are the mean '
      f'of male CNS L and R, where it drives {ms_n["malecns L"]} and {ms_n["malecns R"]} KCs. With GH146 input ms drives 3–4% of KCs.',
      'Similarity change against pair strength.')
R.p(f'The claim holds: pairs of strongly driven odors gain the most similarity (Spearman ρ = {stat["ORN"][0]:.2f} with ORN input, '
    f'{stat["GH146"][0]:.2f} with GH146). The pairs with ms sit at ΔS ≈ 0 with ORN input, because ms barely drives any KC: its ORN '
    'responses are weak, and with one shared threshold almost no KC reaches it. This is a property of the Report 05 setup only. From '
    'Report 07 on, each odor is pinned to its TNT fraction (ms: 11.6% of units), so ms is fully part of every comparison there.')
R.p('In the rebuilt Report 05, ms stays in every figure and table. Where its similarity is undefined (no KC responds), the cell is '
    'marked as such instead of being dropped.')

R.sub('What goes into the rebuilt Report 05')
R.ul(['**PN-level transformation** (item 1): ORN → PN through the divisive normalization of Olsen et al. (2010), as used in Kennedy\'s model '
      '(2019, "Learning with naturalistic odor representations in a dynamic model of the Drosophila olfactory system"). Its output is '
      'compared with the GH146 results. If you meant a different Kennedy paper, tell me.',
      '**Negative $r_{go}$** (item 4): a version that keeps inhibitory glomerular responses instead of setting them to 0.',
      '**Inhibition on both PNs and KCs** (item 5): APL subtracts from KCs and divides bouton input per glomerulus at the same time, with the '
      'per-glomerulus bouton weights (like M, with C-style per-KC weights on the KC side).',
      '**Figures 8 and 9 for every version** (item 8), each caption stating the mean change in similarity with and without inhibition, and a '
      'comparison of every version with U.',
      '**Marked pairs** (item 9): in the $\\Delta S$ heat maps, the pairs that change most and least are outlined.',
      '**Data at the end** (item 10): the response-similarity heat maps of the recordings (Report 06, Figure 3), for direct comparison.',
      '**The two alternative $W$ normalizations** above, and the sentence rewrite.'])

# =====================================================================================================================
# Report 06
# =====================================================================================================================
R.section('Report 06: the recordings')
R.p('**Units.** The responses are FC z-scores (`fc_zscore`). The rebuilt report says so, and "responding" means a response above 1 '
    'z-score unit.')
UP = '/mnt/project-files/uploads/hearth/'
D = {'TNT': load(UP + '2e7fb552-f4ec-49d7-b06f-14ee778f6433'), 'TNT3c': load(UP + '76debf61-257e-4375-b4d8-a02b48c4f62a')}
O = list(next(iter(D['TNT'].values())).columns); NO = len(O); IU = np.triu_indices(NO, 1)


def cosx(X):
    n = np.linalg.norm(X, axis=0); return (X.T @ X) / np.outer(n, n)


SR = {c: np.array([cosx(v.values)[IU] for v in d.values()]) for c, d in D.items()}
FR = {c: np.array([(v.values > 1).mean(0) for v in d.values()]) for c, d in D.items()}
A6 = np.zeros((len(IU[0]), NO + 1)); A6[:, 0] = 1; A6[np.arange(len(IU[0])), IU[0] + 1] = 1; A6[np.arange(len(IU[0])), IU[1] + 1] = 1


def additive(y):
    b = np.linalg.lstsq(A6, y, rcond=None)[0]; fit = A6 @ b
    return b[1:] - b[1:].mean(), 1 - np.var(y - fit) / np.var(y)


dS6 = SR['TNT'].mean(0) - SR['TNT3c'].mean(0)
e6, r26 = additive(dS6)
null_r2 = np.array([additive(rs.permutation(dS6))[1] for _ in range(5000)])
p_r2 = (1 + np.sum(null_r2 >= r26)) / (1 + len(null_r2))
boot = []
for _ in range(2000):
    t, c = rs.choice(len(SR['TNT']), len(SR['TNT'])), rs.choice(len(SR['TNT3c']), len(SR['TNT3c']))
    boot.append(additive(SR['TNT'][t].mean(0) - SR['TNT3c'][c].mean(0))[0])
boot = np.array(boot); blo, bhi = np.percentile(boot, [2.5, 97.5], 0)
sig = (blo > 0) | (bhi < 0)
rel = []
for _ in range(1000):
    t, c = rs.permutation(len(SR['TNT'])), rs.permutation(len(SR['TNT3c']))
    rel.append(np.corrcoef(additive(SR['TNT'][t[:4]].mean(0) - SR['TNT3c'][c[:5]].mean(0))[0],
                           additive(SR['TNT'][t[4:]].mean(0) - SR['TNT3c'][c[5:]].mean(0))[0])[0, 1])
rel_h = float(np.mean(rel)); rel_f = 2 * rel_h / (1 + rel_h)
chance = {c: np.mean([np.sqrt(np.outer(p, p))[IU] for p in FR[c]], 0) for c in FR}
e_ch, r2_ch = additive(chance['TNT'] - chance['TNT3c'])
fT, fC = FR['TNT'].mean(0), FR['TNT3c'].mean(0)
rho_ch = spearmanr(e6, e_ch)[0]; rho_fT = spearmanr(e6, fT)[0]; rho_fC = spearmanr(e6, fC)[0]; rho_df = spearmanr(e6, fT - fC)[0]
rho_ratio = spearmanr(e6, fT / fC)[0]
dpp = pd.Series((fT - fC) * 100, index=O)

R.sub('What the per-odor effect is, and what it means mechanistically')
R.p('The fit $\\Delta S_{ab} \\approx \\mu + e_a + e_b$ says: the change of a pair is the average change, plus a bit that belongs to '
    'each of its two odors. $e_a$ is how much more (or less) the pairs that contain odor $a$ change than the average pair. For example '
    f'$e_{{\\text{{eb}}}}$ = {F(e6[O.index("eb")], 3, True)}: pairs with eb gain about {F(e6[O.index("eb")], 2)} more cosine similarity '
    'than average when APL is blocked, whatever the partner. An odor with a large $e_a$ has its whole row of the similarity matrix move up.')
R.p('**Mechanism: chance overlap.** When an odor drives more units, it shares more responding units with every other odor just by chance '
    '(Report 08, Step 2: two unrelated 0/1 codes have cosine $\\sqrt{p_a p_b}$). So an odor that gains many responders when APL is blocked '
    'should gain similarity with all partners, which is exactly a large $e_a$. This can be tested on the data alone: compute the chance '
    'overlap $\\sqrt{p_a p_b}$ from each recording\'s fractions responding, take its change TNT − TNT3c, fit the same additive model to it, '
    'and compare its per-odor effects with the real $e_a$.')


def fig_e():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1, 1.05]})
    a = axs[0]; o = np.argsort(e6); y = np.arange(NO)
    a.hlines(y, blo[o], bhi[o], color=L.GRAY, lw=1.6)
    a.scatter(e6[o], y, s=36, color=np.where(e6[o] > 0, L.BLOCKED, L.INTACT), zorder=3, edgecolor='white', linewidth=0.6)
    a.axvline(0, color=L.INK if hasattr(L, 'INK') else '#333', lw=0.7)
    a.set_yticks(y); a.set_yticklabels([O[i] + (' *' if sig[i] else '') for i in o], fontsize=8)
    a.set_xlabel('per-odor effect $e_a$ (cosine, relative to the mean change)')
    a.set_title(f'a  $e_a$ with 95% bootstrap range ({int(sig.sum())} of 16 exclude 0, *)', fontsize=9.5)
    b = axs[1]
    b.scatter(e_ch, e6, s=36, color=L.INTACT, edgecolor='white', linewidth=0.6, zorder=3)
    for i, od in enumerate(O): b.annotate(od, (e_ch[i], e6[i]), xytext=(3, 2), textcoords='offset points', fontsize=7, color='#555')
    k = np.polyfit(e_ch, e6, 1); xs = np.linspace(e_ch.min(), e_ch.max(), 10); b.plot(xs, np.polyval(k, xs), color=L.BLOCKED, lw=1.2)
    b.axhline(0, color=L.GRAY, lw=0.6, ls=':'); b.axvline(0, color=L.GRAY, lw=0.6, ls=':')
    b.set_xlabel('per-odor effect predicted by chance overlap alone')
    b.set_ylabel('per-odor effect $e_a$ in the data')
    b.set_title(f'b  Chance overlap predicts $e_a$: Spearman ρ = {rho_ch:.2f}', fontsize=9.5)
    fig.tight_layout(); return fig


R.fig(fig_e(), '**a**, The per-odor effects of Report 06, Figure 4a, with their 95% bootstrap ranges (2,000 draws; recordings resampled '
      'with replacement within each condition; each draw refits the additive model). **b**, Each point is one odor. Horizontal: the '
      'per-odor effect of the additive fit to the change in chance overlap $\\sqrt{p_a p_b}$ (fractions responding > 1, per recording, '
      'averaged). Vertical: the per-odor effect of the real change in similarity. The orange line is the least-squares fit.',
      'Per-odor effects: how sure, and why.')
R.p(f'The per-odor effects follow chance overlap (Spearman ρ = {rho_ch:.2f}). They also follow how many units an odor drives in TNT '
    f'(ρ = {rho_fT:.2f}) and how many it gains (TNT − TNT3c, ρ = {rho_df:.2f}), but not the ratio TNT ÷ TNT3c (ρ = {rho_ratio:.2f}). '
    'So the per-odor part of the change is mostly the arithmetic of denser codes: odors that drive many units in TNT, such as eb, 2h and IaA, '
    f'overlap more with everything. -aPine and euc gain the fewest responding units when APL is blocked ({dpp["-aPine"]:.1f} and '
    f'{dpp["euc"]:.1f} percentage points, against {dpp.drop(["-aPine", "euc"]).min():.1f}–{dpp.max():.1f} for the other odors) and have '
    'the most negative effects.')
R.sub('Are the per-odor effects real? The statistics')
R.table(['test', 'how', 'result'], [
    ['Is the additive fit better than chance?', 'Permutation: the 120 values of $\\Delta S$ are shuffled across pairs (5,000 times), which '
     'removes any odor structure, and the additive model is refitted', f'$R^2$ = {r26:.2f} in the data; null mean {null_r2.mean():.2f}, 95th '
     f'percentile {np.percentile(null_r2, 95):.2f}; p = {p_r2:.4f} (none of 5,000 shuffles reach it)'],
    ['Which $e_a$ differ from 0?', '95% bootstrap range over recordings (Figure 2a)', f'{int(sig.sum())} of 16 odors: '
     + ', '.join(O[i] for i in np.argsort(e6) if sig[i])],
    ['Are the $e_a$ reproducible?', 'Split-half: two random halves of the recordings (4 + 5 and 4 + 6), $e_a$ fitted in each, Pearson r '
     'over odors; 1,000 splits; Spearman–Brown to full data', f'r = {rel_h:.2f} between halves, {rel_f:.2f} for all the data']],
    'With 16 odors, the additive model has 16 free parameters for 120 pairs, so even random numbers give $R^2$ ≈ 15/119 ≈ 0.13. That is '
    'why the null is needed: 0.65 is far above it.', 'Statistics for the per-odor effects.')

# =====================================================================================================================
# Report 07
# =====================================================================================================================
R.section('Report 07: pinning and the comparison')
R.sub('Equation 1 with numbers: the per-odor input gain')
R.p('The goal of pinning is that, with APL blocked, the model drives exactly the fraction of KCs that TNT drives for each odor. The '
    'threshold is fixed at θ = 1, and each odor\'s input is multiplied by a gain $g_o$ chosen so that the right number of KCs end up '
    'above 1. Here is the calculation for a mock mushroom body of 10 KCs and one odor that drives 20% of units in TNT:')
x7 = np.array([0.12, 0.30, 0.05, 0.42, 0.22, 0.18, 0.60, 0.35, 0.08, 0.50]); phi7 = 0.20
Q7 = float(np.quantile(x7, 1 - phi7)); g7 = 1 / Q7
srt = np.sort(x7)
R.ul([f'**1. Unscaled drives** $\\sum_g W_{{kg}} r_{{go}}$ of the 10 KCs: {", ".join(f"{v:.2f}" for v in x7)}.',
      f'**2. Sort them:** {", ".join(f"{v:.2f}" for v in srt)}.',
      f'**3. The quantile.** 20% should respond, so the threshold must sit at the 80th percentile, $Q_o(1 - 0.20) = Q_o(0.80)$. With 10 '
      f'values its position is 0.80 × 9 = 7.2 places from the bottom (counting from 0): between the 8th value (0.42) and the 9th (0.50), '
      f'20% of the way: 0.42 + 0.2 × (0.50 − 0.42) = **{Q7:.3f}**.',
      f'**4. The gain.** $g_o = \\theta / Q_o(0.80) = 1 / {Q7:.3f}$ = **{g7:.3f}**.',
      f'**5. Scaled drives** $x_{{ko}} = g_o \\sum_g W_{{kg}} r_{{go}}$: {", ".join(f"{v * g7:.2f}" for v in x7)}. Exactly two (from '
      f'0.60 and 0.50) are above 1, so 2 of 10 = 20% respond, as in TNT. The KC at 0.42 ends just below, at {0.42 * g7:.2f}.'])
R.p('Dividing by the quantile only moves the threshold to the right place in each odor\'s distribution. The order of the KCs\' drives '
    '(which KC responds) still comes from the connectome and the ORN responses. An odor with weak ORN responses has small drives, a small '
    'quantile, and therefore a large gain.')

R.sub('Why the ratio is squared: (0.46/0.84)² ≈ 30%')
fit_o = TVM['fits']['threshold 1.0, ORN']
relS = fit_o['split-half (recordings) reliability of dS, full-data estimate']; ceil = fit_o['ceiling for r with a model (sqrt of that)']
rU = fit_o['r(data dS, model dS) by version']['U']
R.p('A correlation r is not a share. Its square, $r^2$, is: the share of the data\'s variance that the model accounts for. The '
    'reasoning goes in three steps.')
R.ul([f'**The data are part pattern, part noise.** The split-half reliability of $\\Delta S$ is {relS:.3f}: {relS * 100:.0f}% of its variance '
      f'over pairs is reproducible pattern, and {100 - relS * 100:.0f}% is recording noise that changes from one set of recordings to the next.',
      f'**A perfect model can explain only the pattern.** Its $r^2$ would be {relS:.3f}, so its r would be $\\sqrt{{{relS:.3f}}}$ = {ceil:.3f}. '
      'That is the ceiling of 0.84.',
      f'**The uniform model.** r = {rU:.2f}, so $r^2$ = {rU ** 2:.3f}: it accounts for {rU ** 2 * 100:.0f}% of all the variance, which is '
      f'{rU ** 2 / relS * 100:.0f}% of the {relS * 100:.0f}% that is pattern. That is $r^2 / \\text{{reliability}} = (r / \\text{{ceiling}})^2$.'])
sim = []
for _ in range(400):
    sig_ = rs.standard_normal(120); noise = rs.standard_normal(120)
    data = np.sqrt(relS) * sig_ + np.sqrt(1 - relS) * noise
    rho_ms = rU / ceil
    model = rho_ms * sig_ + np.sqrt(1 - rho_ms ** 2) * rs.standard_normal(120)
    sim.append((np.corrcoef(model, data)[0, 1], np.corrcoef(model, sig_)[0, 1]))
sim = np.array(sim).mean(0)
R.p(f'A quick simulation confirms it: 120 mock pairs whose "data" are {relS * 100:.0f}% pattern and {100 - relS * 100:.0f}% noise, and a mock '
    f'model that shares 30% of its variance with the pattern. The model correlates with the noisy data at r = {sim[0]:.2f} on average '
    f'and with the noise-free pattern at r = {sim[1]:.2f} ($0.55^2$ = 0.30). The unsquared ratio 0.46/0.84 = 0.55 is that second '
    'correlation; squaring turns it into a share.')
R.p('**Figure 4** will get a $y = x$ line in the rebuild (both axes are blocked ÷ intact ratios, so points on the line mean the model '
    'gets that odor\'s suppression right).')

# =====================================================================================================================
# Report 08
# =====================================================================================================================
R.section('Report 08: the inhibition rules')
R.sub('The four rules as equations')
R.p('All four rules start from the same pinned, blocked responses $y^{\\text{blocked}}_{ko} = [x_{ko} - 1]_+$ and all give the same mean '
    'fraction responding over odors (TNT3c\'s, 6.83%). They differ only in how the inhibition is shared out between odors. Your reading '
    'of the fixed rule is right.')
R.table(['rule', 'intact response', 'how the inhibition is set', 'is it a model?'], [
    ['feedback (version U)', '$[x_{ko} - 1 - a\\,A_o]_+$, with $A_o = \\frac{1}{N}\\sum_k y_{ko}$', 'one gain $a$ for all odors; $A_o$ '
     'comes out of the KC responses themselves, so strong odors get more inhibition', 'yes: the model of Reports 05 and 07'],
    ['fixed', '$[x_{ko} - 1 - c]_+$', 'one number $c$ for all KCs and all odors (per mushroom body): the same as raising the threshold from '
     '1 to $1 + c$', 'no: a reference rule'],
    ['same share', '$[x_{ko} - 1 - c_o]_+$', 'one $c_o$ per odor, chosen so that every odor keeps the same share of its blocked '
     'responders: $\\phi^{\\text{intact}}_o = \\phi^{\\text{blocked}}_o \\times \\bar\\phi^{\\text{TNT3c}} / \\bar\\phi^{\\text{TNT}}$ (about 45%)',
     'no: a reference rule'],
    ['TNT3c per odor', '$[x_{ko} - 1 - c_o]_+$', 'one $c_o$ per odor, chosen so that $\\phi^{\\text{intact}}_o = \\phi^{\\text{TNT3c}}_o$, '
     'the data\'s own fraction', 'no: uses the answer from the data']],
    '$c$ and $c_o$ are found by bisection in each mushroom body. In all four, the inhibition is the same for every KC for a given odor.',
    'The inhibition rules of Report 08.')
R.p('The three extra rules are not new models of APL. They are controls that keep everything the same except how the inhibition '
    'depends on the odor: not at all (fixed), in proportion to the odor\'s own blocked response (same share), or exactly as in the data '
    '(TNT3c per odor). Comparing feedback with them shows that the odor-dependence created by feedback is what matters for the fit.')

R.sub('How the Spearman values were computed, and -aPine\'s 1.71')
ap = FBC[(FBC.responses == 'ORN') & (FBC.odor == '-aPine')].iloc[0]
R.p('For each of the 16 odors, two numbers, each the mean over the three mushroom bodies: the fraction responding with APL blocked '
    '($\\phi^{\\text{blocked}}_o$, which equals the TNT fraction by pinning) and the gain $\\gamma_o = \\phi^{\\text{blocked}}_o / '
    '\\phi^{\\text{intact}}_o$. The Spearman correlation over the 16 odors between them is +0.96 for feedback and −0.62 for the fixed rule.')
R.p(f'For -aPine, $\\phi^{{\\text{{blocked}}}}$ = {ap.model_frac_blocked * 100:.2f}% and $\\phi^{{\\text{{intact}}}}$ with feedback = '
    f'{ap.model_frac_intact_feedback * 100:.2f}%, so $\\gamma$ = {ap.model_frac_blocked * 100:.2f} / {ap.model_frac_intact_feedback * 100:.2f} '
    f'= **{ap.gain_model_feedback:.2f}**, the smallest of the 16 (Report 08, Figure 1b, leftmost blue and orange points). In Figure 1a its '
    f'blue dot is hidden under the orange diamond of the data, whose value is almost the same ({ap.gain_data:.2f}). Two things to fix in '
    'the rebuild: the dots will be drawn so none hides another, and $\\gamma_o$ will be called the "blocked ÷ intact ratio", because "gain" '
    'is also used for the input gain $g_o$ of Report 07. For -aPine that one is large, since its ORN responses are weak.')

R.sub('Why odor strength was the mean of TNT and TNT3c, and the TNT-only result')
SIZE = {'fraction > 1': lambda X: (X > 1).mean(0), 'mean positive response': lambda X: np.clip(X, 0, None).mean(0)}
ST = {k: {c: np.array([f(v.values) for v in d.values()]) for c, d in D.items()} for k, f in SIZE.items()}
rs8 = np.random.default_rng(5); res8 = {}
for _ in range(2000):
    t, n = rs8.permutation(len(ST['fraction > 1']['TNT'])), rs8.permutation(len(ST['fraction > 1']['TNT3c']))
    for k in SIZE:
        gain = ST[k]['TNT'][t[4:]].mean(0) / ST[k]['TNT3c'][n[5:]].mean(0)
        for sk, s in {'mean of TNT and TNT3c (Report 08)': (ST[k]['TNT'][t[:4]].mean(0) + ST[k]['TNT3c'][n[:5]].mean(0)) / 2,
                      'TNT only': ST[k]['TNT'][t[:4]].mean(0), 'TNT3c only': ST[k]['TNT3c'][n[:5]].mean(0)}.items():
            res8.setdefault((sk, k), []).append(spearmanr(s, gain)[0])
R.p('I used the mean of the two conditions as a neutral measure of how strongly an odor drives the units, one that does not lean '
    'towards either condition. You are right that TNT alone is the better choice: TNT has no APL, so it measures the feed-forward drive '
    'directly, and it is also what the model\'s horizontal axis uses. TNT3c is already flattened by feedback, so mixing it in blurs the '
    'measure. The strength and the gain still come from separate halves of the recordings (4 TNT recordings for strength; the other 4 TNT '
    'and 6 of the TNT3c recordings for the gain), so they share no noise.')
R.table(['odor strength measured as', 'gain = ratio of', 'mean Spearman ρ (2,000 splits)', '5–95%', 'share of splits > 0'],
        [[sk, k, F(np.mean(v), 2, True), f'{np.quantile(v, .05):+.2f} to {np.quantile(v, .95):+.2f}', f'{np.mean(np.array(v) > 0) * 100:.0f}%']
         for k in SIZE for sk in ['mean of TNT and TNT3c (Report 08)', 'TNT only', 'TNT3c only'] for v in [res8[(sk, k)]]],
        'Strength and gain use the same measure in each row (here, unlike Report 08, where strength was always the fraction > 1). The model '
        'with uniform feedback predicts ρ = +0.96 for the fraction.', 'The feedback signature with TNT-only odor strength.')
v_t = np.array(res8[('TNT only', 'fraction > 1')]); v_r = np.array(res8[('TNT only', 'mean positive response')])
R.p(f'With TNT alone the signature is a bit stronger but the conclusion does not change: for the number of responding units it is weak '
    f'(ρ = {v_t.mean():+.2f}, positive in {np.mean(v_t > 0) * 100:.0f}% of splits), far from the +0.96 that uniform feedback predicts; for '
    f'response size it is clear (ρ = {v_r.mean():+.2f}, positive in {np.mean(v_r > 0) * 100:.0f}% of splits). The rebuilt Report 08 will use '
    'TNT-only strength.')

# =====================================================================================================================
# Report 09
# =====================================================================================================================
R.section('Report 09: the noise correction')
R.p('Equation 3 is Spearman\'s correction for attenuation. The logic, step by step:')
R.ul(['**1. Every measurement is pattern + noise.** The similarity of a pair measured from a set of recordings is its true value in '
      'that condition plus recording noise. A different set of recordings gives the same true value with different noise.',
      '**2. Noise lowers correlations.** If two measurements $X$ and $Y$ each carry independent noise, their correlation is the correlation '
      'of their true values times a shrink factor: $\\text{corr}(X, Y) = \\text{corr}(T_X, T_Y) \\cdot \\sqrt{\\text{rel}_X\\,\\text{rel}_Y}$, '
      'where $\\text{rel}$ = (variance of the true values) ÷ (variance of the measurement) is the share of a measurement that is not noise.',
      '**3. The within-condition correlation measures rel.** Two halves of the same condition share the true values and differ only in '
      'noise, so their correlation equals rel: $\\rho^{\\text{TNT}}_{\\text{within}} \\approx \\text{rel}_{\\text{TNT}}$, and the same for TNT3c.',
      '**4. Undo the shrink.** $\\rho_{\\text{between}}$ (a TNT half against a TNT3c half) is the true TNT-vs-TNT3c correlation shrunk by '
      '$\\sqrt{\\text{rel}_{\\text{TNT}}\\,\\text{rel}_{\\text{TNT3c}}}$. Dividing by that factor gives Equation 3.',
      '**5. Why halves on both sides.** The between value is computed from half-size samples, exactly like the within values, so both '
      'carry the same amount of noise and no Spearman–Brown step is needed.'])
tab9 = []
for rt in (1.0, 0.92):
    acc = []
    for _ in range(2000):
        TC = rs.standard_normal(120); TT = rt * TC + np.sqrt(1 - rt ** 2) * rs.standard_normal(120)
        s = np.sqrt(1 / 0.917 - 1)
        c1, c2 = TC + s * rs.standard_normal(120), TC + s * rs.standard_normal(120)
        t1, t2 = TT + s * rs.standard_normal(120), TT + s * rs.standard_normal(120)
        wC, wT, bw = spearmanr(c1, c2)[0], spearmanr(t1, t2)[0], spearmanr(t1, c1)[0]
        acc.append((wC, wT, bw, bw / np.sqrt(wC * wT), spearmanr(TT, TC)[0]))
    acc = np.mean(acc, 0)
    tab9.append([f'{rt:.2f}', f'{acc[4]:.3f}', f'{acc[0]:.3f}', f'{acc[1]:.3f}', f'{acc[2]:.3f}', f'**{acc[3]:.3f}**'])
tab9.append(['data', '?', '0.904', '0.906', '0.831', '**0.92**'])
R.table(['true correlation (Pearson)', 'true ρ (ranks)', '$\\rho^{\\text{TNT3c}}_{\\text{within}}$', '$\\rho^{\\text{TNT}}_{\\text{within}}$',
         '$\\rho_{\\text{between}}$', 'corrected'], tab9,
        'Mock data: 120 pairs, a true TNT3c pattern and a true TNT pattern that correlate as in column 1, and noise in each half sized so '
        'that two halves of one condition correlate at about 0.90, as in the recordings. Means of 2,000 simulations. In the first row '
        'blocking APL keeps the order perfectly, yet the between value is only about 0.90 because of noise; the correction brings it back '
        'near 1. In the second row the between value drops to about 0.83, as in the data, and the correction recovers the true value.',
        'The correction in a simulation.')
R.p('So a between value of 0.83 with within values of 0.90 is what a true rank correlation of about 0.92 looks like through this much '
    'noise. The correction is exact for Pearson correlations with independent noise and a close approximation for Spearman, as the table '
    'shows.')

# =====================================================================================================================
# Report 11
# =====================================================================================================================
R.section('Report 11: the local APL model')
R.sub('θ in Equation 1')
R.p('Yes, θ = 1. Because each odor\'s input is pinned (Report 07), the threshold can be any positive number: the gains $g_o$ scale the '
    'inputs to it. Setting θ = 1 just fixes the units, so $x_{ko}$ is measured in "threshold units" (1 = exactly at threshold).')
R.sub('Equation 3, every variable, with a mock tree')
R.p('**What it computes.** For the local model we need $\\Phi_{vj}$: the total signal that KC $j$\'s inputs to APL produce at a point $v$ '
    'of APL, each input decayed by its distance along APL, $\\Phi_{vj} = \\sum_{t \\in \\text{in}(j)} e^{-d(v,t)/\\lambda}$. Summing it over KC '
    '$k$\'s output sites gives $M_{kj}$. Equation 3 computes $\\Phi$ for every node without ever listing distances, using one fact: the '
    'decay along a path is the product of the decays of its edges, $e^{-(\\ell_1 + \\ell_2)/\\lambda} = e^{-\\ell_1/\\lambda}\\,e^{-\\ell_2/\\lambda}$.')
R.symbols([('v,\\ w', 'Nodes of the (reduced) APL skeleton tree. One node is picked as the root; every other node has one parent and may have children', 'nodes'),
           ('\\text{parent}(v),\\ \\text{children}(v)', 'The neighbour of $v$ one step towards the root, and the neighbours one step away from it', ''),
           ('\\ell_v', 'Length of the edge from $v$ up to its parent', 'µm'),
           ('\\lambda', 'Length constant: the distance over which a passive signal falls to 37%', 'µm'),
           ('\\kappa_v = e^{-\\ell_v/\\lambda}', 'The share of a signal that survives crossing the edge from $v$ to its parent (either way)', '0 to 1'),
           ('n_{vj}', 'Number of KC $j$\'s KC→APL synapses at node $v$: where its signal enters APL', 'count'),
           ('\\text{up}_{vj}', 'Signal at $v$ from KC $j$\'s inputs **in $v$\'s own subtree** ($v$ and everything below it), each decayed by its distance to $v$', '≥ 0'),
           ('\\Phi_{vj}', 'Signal at $v$ from **all** of KC $j$\'s inputs, anywhere on the tree', '≥ 0'),
           ('M_{kj} = c\\sum_{s \\in \\text{out}(k)} \\Phi_{sj}', 'Coupling: KC $j$\'s signal summed over the nodes where APL synapses onto KC $k$', '')],
          'Variables of Equation 3')
# mock tree
par = {0: None, 1: 0, 2: 1, 3: 1, 4: 0}; ell = {1: 4.0, 2: 2.0, 3: 3.0, 4: 6.0}; nin = {0: 0, 1: 0, 2: 1, 3: 0, 4: 2}; lam = 5.0
kap = {v: np.exp(-ell[v] / lam) for v in ell}
ch = {v: [w for w, p in par.items() if p == v] for v in par}
up = {}
for v in [2, 3, 4, 1, 0]:
    up[v] = nin[v] + sum(kap[w] * up[w] for w in ch[v])
Phi = {0: up[0]}
for v in [1, 4, 2, 3]:
    Phi[v] = up[v] * (1 - kap[v] ** 2) + kap[v] * Phi[par[v]]


def dist(u, v):
    def path(x):
        p = [x]
        while par[p[-1]] is not None: p.append(par[p[-1]])
        return p
    pu, pv = path(u), path(v); common = next(x for x in pu if x in pv)
    return sum(ell[x] for x in pu[:pu.index(common)]) + sum(ell[x] for x in pv[:pv.index(common)])


brute = {v: sum(nin[t] * np.exp(-dist(v, t) / lam) for t in par) for v in par}
NAME = {0: 'R (root)', 1: 'A', 2: 'B', 3: 'C', 4: 'D'}


def fig_tree():
    fig, a = plt.subplots(figsize=(6.2, 3.4))
    pos = {0: (0, 0), 1: (-2, -1.3), 4: (2.2, -1.3), 2: (-3.1, -2.6), 3: (-0.9, -2.6)}
    for v, p in par.items():
        if p is None: continue
        (x0, y0), (x1, y1) = pos[v], pos[p]
        a.plot([x0, x1], [y0, y1], color=L.GRAY, lw=2, zorder=1)
        a.text((x0 + x1) / 2 + (0.3 if x0 > x1 else -0.3), (y0 + y1) / 2 + 0.1, f'{ell[v]:g} µm\nκ = {kap[v]:.3f}', fontsize=7.5,
               ha='left' if x0 > x1 else 'right', va='center', color='#444')
    for v, (x, y) in pos.items():
        a.scatter([x], [y], s=380, color='white', edgecolor='#333', zorder=2)
        a.text(x, y, NAME[v].split()[0], ha='center', va='center', fontsize=9, zorder=3)
        if nin[v]:
            a.text(x, y - 0.42, f'{nin[v]} input{"s" if nin[v] > 1 else ""} from KC j', ha='center', va='top', fontsize=7.5, color=L.BLOCKED)
    a.text(pos[3][0], pos[3][1] - 0.42, 'output onto KC k', ha='center', va='top', fontsize=7.5, color=L.INTACT)
    a.text(pos[1][0] - 0.3, pos[1][1] + 0.05, 'output onto KC k', ha='right', va='bottom', fontsize=7.5, color=L.INTACT)
    a.set_xlim(-4.6, 3.6); a.set_ylim(-3.3, 0.5); a.axis('off')
    a.set_title(f'Mock APL tree, λ = {lam:g} µm', fontsize=9.5)
    fig.tight_layout(); return fig


R.fig(fig_tree(), 'Five nodes. KC $j$ excites APL with one synapse at B and two at D (orange). APL inhibits KC $k$ at A and C (blue). Edge '
      'lengths and their decay factors κ are written on the edges.', 'A mock tree for Equation 3.', width=70)
R.p('**Pass 1, leaves to root** ($\\text{up}_{vj} = n_{vj} + \\sum_{w} \\kappa_w\\,\\text{up}_{wj}$): each node collects what enters '
    'below it, decayed across each edge on the way up.')
R.p('**Pass 2, root to leaves** ($\\Phi_{vj} = \\text{up}_{vj}(1 - \\kappa_v^2) + \\kappa_v\\,\\Phi_{\\text{parent}(v),j}$). The '
    'parent knows the total signal, $\\Phi_{\\text{parent}}$. Bringing it down to $v$ costs one more crossing of the edge, so it is '
    'multiplied by $\\kappa_v$. But part of $\\Phi_{\\text{parent}}$ came from $v$\'s own subtree: that part reached the parent as '
    '$\\kappa_v\\,\\text{up}_v$, and bringing it back would multiply it by $\\kappa_v$ again, giving $\\kappa_v^2\\,\\text{up}_v$ although its '
    'true value at $v$ is $\\text{up}_v$. So: take $\\kappa_v \\Phi_{\\text{parent}}$, remove the wrong $\\kappa_v^2\\,\\text{up}_v$ and add the '
    'right $\\text{up}_v$. That is $\\Phi_v = \\kappa_v \\Phi_{\\text{parent}} - \\kappa_v^2\\,\\text{up}_v + \\text{up}_v$, which is Equation 3.')
rows = []
for v in [0, 1, 2, 3, 4]:
    if v == 0:
        pform = f'= up = {Phi[0]:.3f}'
    else:
        pform = f'{up[v]:.3f} × (1 − {kap[v]:.3f}²) + {kap[v]:.3f} × {Phi[par[v]]:.3f} = {Phi[v]:.3f}'
    uform = f'{nin[v]}' + ''.join(f' + {kap[w]:.3f} × {up[w]:.3f}' for w in ch[v]) + (f' = {up[v]:.3f}' if ch[v] else '')
    rows.append([NAME[v], NAME[par[v]].split()[0] if par[v] is not None else '–', f'{ell[v]:g}' if v in ell else '–',
                 f'{kap[v]:.3f}' if v in kap else '–', nin[v], uform, pform, f'{brute[v]:.3f}'])
R.table(['node $v$', 'parent', '$\\ell_v$ (µm)', '$\\kappa_v$', '$n_{vj}$', 'pass 1: $\\text{up}_{vj}$', 'pass 2: $\\Phi_{vj}$',
         'check: $\\sum_t e^{-d(v,t)/\\lambda}$ directly'], rows,
        'Pass 1 is filled from the bottom row up (B, C, D, then A, then R); pass 2 from R down. The last column sums over the three input '
        'synapses directly, using the distances along the tree; it equals pass 2 at every node.', 'Equation 3 on the mock tree.',
        num_cols=[2, 3, 4, 7])
R.p(f'For example at A: the input at B is 2 µm away (decay {np.exp(-2 / lam):.3f}), and the two inputs at D are 4 + 6 = 10 µm away '
    f'({np.exp(-10 / lam):.3f} each), so $\\Phi_{{A}}$ = {np.exp(-2 / lam):.3f} + 2 × {np.exp(-10 / lam):.3f} = {brute[1]:.3f}. KC $k$ '
    f'receives APL output at A and C, so $M_{{kj}} = c\\,(\\Phi_A + \\Phi_C) = c \\times$ {Phi[1] + Phi[3]:.3f}. On the real skeletons the '
    'same two passes run for every KC $j$ at once.')
R.sub('The C matrix')
R.p('Yes, it is the same per-KC weighting as version C of Reports 05 and 07, written as a matrix: $M_{kj} = u_k\\,v_j / N$, with $u_k$ '
    '= KC $k$\'s APL→KC synapses and $v_j$ = KC $j$\'s KC→APL synapses in the calyx, each divided by its mean. Putting it into Equation 1 '
    'gives $a\\sum_j M_{kj}\\,y_{jo} = a\\,u_k \\cdot \\frac{1}{N}\\sum_j v_j\\,y_{jo} = a\\,u_k\\,A_o$, which is exactly version C: one APL '
    'activity $A_o$ (the $v$-weighted mean KC response), applied to KC $k$ in proportion to $u_k$. M is "C" only to compare it with the '
    'local matrices, which have the same form but let the coupling depend on where on APL the synapses are.')

R.save()
