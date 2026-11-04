"""Report 05: the APL block model, built step by step (the prediction run of scripts/apl_block_model.py)."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import report_lib as L
import apl_block_model as abm
L.style()
F = L.fmt

RESP = abm.responses(); MB, SIDES = abm.mushroom_bodies(); MBS = list(MB)
SUM = json.load(open(L.OW + 'APL_block_model_summary.json'))
OD = pd.read_csv(L.OW + 'APL_block_model_per_odor.csv'); PR = pd.read_csv(L.OW + 'APL_block_model_odor_pairs.csv')
MAIN = '20% -> 10%'
MBNAME = {'hemibrain R': 'hemibrain R', 'malecns L': 'male CNS L', 'malecns R': 'male CNS R'}
ODORS = list(RESP['ORN'].columns)


# ---------------------------------------------------------------- replicate the model (one MB, one input) to show the steps
def setup(mb, R):
    W = mb['W']; gl = [g for g in W.columns if g in R.index]
    Wm, Rp = W[gl].values, np.clip(R.loc[gl].values, 0, None)
    return W, gl, Wm, Rp, Wm @ Rp


def theta_of(H, cb):
    lo, hi = 0.0, H.max(); trace = []
    for _ in range(60):
        m = (lo + hi) / 2; c = (H > m).mean(0).mean(); trace.append((m, c))
        if c > cb: lo = m
        else: hi = m
    return (lo + hi) / 2, trace


def norm_w(x, Wm, gl):
    x = x.reindex(gl)
    return (x / (np.nansum(Wm.sum(0) * x.values) / Wm[:, ~x.isna().values].sum())).fillna(1.0).values


def gains(mb, R, cb=0.2, ci=0.1):
    W, gl, Wm, Rp, H = setup(mb, R); th, _ = theta_of(H, cb)
    one, oneg = np.ones(len(H)), np.ones(len(gl)); u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean()
    p, q = norm_w(mb['p'], Wm, gl), norm_w(mb['q'], Wm, gl)
    out = {'theta': th}
    for nm, (uu, vv) in {'U': (one, one), 'C': (u, v)}.items():
        out[nm] = abm.calib(lambda a: abm.cl_of(abm.model_y(Wm, Rp, th, a, 0, uu, vv, oneg)).mean(), ci)
    for nm, pp in {'P0': oneg, 'P': p, 'L': q}.items():
        out[nm] = abm.calib(lambda b: abm.cl_of(abm.model_y(Wm, Rp, th, 0, b, one, one, pp)).mean(), ci)
    f = abm.F_SHARE[mb['ds']]
    for nm, pp in {'M0': oneg, 'M': p}.items():
        out[nm] = abm.calib(lambda s: abm.cl_of(abm.model_y(Wm, Rp, th, s * (1 - f) * out['U'], s * f * out['P0'], one, one, pp)).mean(), ci)
    return out


GAIN = {(rn, k): gains(MB[k], RESP[rn]) for rn in RESP for k in MBS}
# check the replication against the saved per-odor table (U, hemibrain R, ORN)
k0 = 'hemibrain R'; W0, gl0, Wm0, Rp0, H0 = setup(MB[k0], RESP['ORN']); th0 = GAIN[('ORN', k0)]['theta']
one0, oneg0 = np.ones(len(H0)), np.ones(len(gl0))
Y0U = abm.model_y(Wm0, Rp0, th0, GAIN[('ORN', k0)]['U'], 0, one0, one0, oneg0)
saved = OD[(OD.coding_levels == MAIN) & (OD.responses == 'ORN') & (OD.mushroom_body == k0)].set_index('odor')
assert np.allclose(abm.cl_of(Y0U), saved.frac_KCs_responding_intact_U.reindex(ODORS).values, atol=1e-4)
assert np.allclose(abm.cl_of(np.maximum(H0 - th0, 0)), saved.frac_KCs_responding_blocked.reindex(ODORS).values, atol=1e-4)

R = L.Report('05', 'How is the APL block model built, step by step, and what does it predict about blocking APL, before any '
             'comparison with the TNT / TNT3c recordings?',
             inputs=['Derived KC claws: `hemibrain_KC_claws_derived.csv.gz`, `malecns_KC_claws_derived.csv.gz` (Report 02)',
                     'Per-KC calyx APL synapse counts: `KC_weights_APL_PN_per_KC.csv` (Report 01)',
                     'Per-glomerulus APL wiring: `odor_weighted/APL_wiring_by_glomerulus.csv` (Report 04)',
                     "Yang's ORN table `ORN_ovlp16_resp_flyave_merge_dropna.csv` (36 glomeruli × 16 odors) and the GH146 means "
                     '`odor_weighted/GH146_mean_by_glomerulus.csv` (Report 03)'],
             outputs=['`odor_weighted/APL_block_model_per_odor.csv`, `APL_block_model_odor_pairs.csv`, `APL_block_model_summary.json`, '
                      '`APL_block_model.png`'],
             scripts=['`scripts/apl_block_model.py` (the model)', '`scripts/reports/r05_block_model.py` (this report; it re-runs the steps '
                      'for illustration and checks them against the saved tables)'])

R.p('This report explains the KC model that all later steps use. Report 07 takes the same model and pins it to the recordings. '
    'Here it runs in its **prediction** form, set up before the recordings arrived: 20% of KCs respond to an odor (averaged over '
    'odors) with APL blocked, and 10% with APL intact. Every symbol is defined in Table 1, and again where it first appears.')
R.box('The model in one paragraph', [
    'Each KC adds up its claws\' glomerular inputs (a weighted average of the glomeruli\'s odor responses, with weights from the '
    'connectome). It responds by how much that sum exceeds a threshold. APL is one cell whose activity, for each odor, is the average '
    'response of the KCs. It inhibits every KC by subtracting a fixed multiple of that activity. Because APL\'s activity depends on '
    'the KC responses and the KC responses depend on APL\'s activity, both are solved together for each odor ("self-consistently"). '
    '"Blocked" (TNT) means no APL at all. Later versions replace the uniform APL with connectome measurements of where APL\'s '
    'synapses are.'], 'plain')

R.symbols([
    ('k', 'Index of a Kenyon cell (KC) in one mushroom body', '$k = 1 \\dots N$'),
    ('N', 'Number of KCs in the mushroom body (KCs with at least one claw)', f'hemibrain R {SIDES["hemibrain R"]:,}; male CNS L {SIDES["malecns L"]:,}; male CNS R {SIDES["malecns R"]:,}'),
    ('g', 'Index of an antennal-lobe glomerulus (uniglomerular PN types only)', 'names such as DM2, VA1v'),
    ('o', 'Index of an odor', '16 odors (ORN table) or 13 (GH146)'),
    ('c', 'Index of a claw (a cluster of a KC\'s PN synapses, Report 02)', '5.6–8 claws per KC'),
    ('n_c', 'Number of PN→KC synapses on claw $c$', 'synapse count'),
    ('g(c)', 'Glomerulus of claw $c$: the glomerulus of the PN that makes most of the claw\'s PN synapses', ''),
    ('W_{kg}', 'Input weight: share of KC $k$\'s claw PN synapses that sit on claws of glomerulus $g$', '0 to 1; a row sums to the share from uniglomerular PNs (0.96–0.97 on average)'),
    ('r_{go}', 'Response of glomerulus $g$ to odor $o$; negative values set to 0', 'ORN table: fly-averaged response as provided (ΔF/F-like, units not stated), 0 to 2.06; GH146: mean peak ΔF/F'),
    ('x_{ko}', 'Drive of KC $k$ by odor $o$: weighted average of its glomeruli\'s responses', 'same units as $r_{go}$'),
    ('\\theta', 'KC threshold, one number per mushroom body and input table', 'set so 20% of KCs respond with APL blocked (Section 5)'),
    ('y_{ko}', 'Response of KC $k$ to odor $o$ (how far the drive exceeds threshold plus inhibition; 0 if below)', 'same units as $r_{go}$; ≥ 0'),
    ('A_o', 'APL activity for odor $o$: the (weighted) mean KC response', 'same units as $y$; one number per odor'),
    ('a', 'Subtractive APL gain (how strongly APL activity inhibits a KC), one number per mushroom body and version', 'dimensionless; set by calibration'),
    ('b', 'Divisive APL gain on PN boutons (versions P0, P, L)', 'dimensionless; set by calibration'),
    ('u_k', 'Per-KC inhibition weight: KC $k$\'s APL→KC synapses in the calyx ÷ the mean over KCs', 'mean 1; $u_k = 1$ in the uniform version'),
    ('v_k', 'Per-KC drive weight: KC $k$\'s KC→APL synapses in the calyx ÷ the mean over KCs', 'mean 1; $v_k = 1$ in the uniform version'),
    ('p_g', 'Bouton weight of glomerulus $g$: APL→PN synapses per PN→KC synapse of its PNs, scaled to a claw-input-weighted mean of 1', 'version P (Report 04)'),
    ('q_g', 'Claw weight of glomerulus $g$: APL→KC synapses on its claws per PN→KC synapse on them, scaled the same way', 'version L (Report 04)'),
    ('f', 'Boutons\' share of APL\'s calyx output synapses: APL→PN ÷ (APL→PN + APL→KC)', f'hemibrain {SUM["bouton_share_of_APL_calyx_output_synapses"]["hemibrain"]}, male CNS {SUM["bouton_share_of_APL_calyx_output_synapses"]["malecns"]}'),
    ('s', 'Common scale of the KC and bouton gains in versions M0 and M', 'set by calibration'),
    ('F_o(A)', 'The APL activity the KCs would produce for odor $o$ if APL\'s activity were $A$', 'used to solve for $A_o$'),
    ('\\phi_o', 'Fraction of KCs responding to odor $o$ ($y_{ko} > 0$), also called the coding level', '0 to 1'),
    ('S_{ab}', 'Cosine similarity of the KC population responses to odors $a$ and $b$', '0 to 1 here (responses are ≥ 0)'),
    ('\\Delta S_{ab}', 'Effect of blocking APL on the similarity of odors $a$ and $b$: $S_{ab}$ blocked − $S_{ab}$ intact (written Δcos in the figure files)', 'positive = more similar when blocked'),
    ('w_g', 'Total claw input from glomerulus $g$ in the mushroom body, $\\sum_k W_{kg}$', 'used to scale glomerulus weights'),
])

# ---------------------------------------------------------------- 1 overview / schematic
R.section('What the model is')
R.p('The model is a rate model of the calyx. A rate model describes each cell by one number per odor (its response), not by spikes '
    'or a time course. There is one model per mushroom body: hemibrain R, male CNS L and male CNS R. FlyWire and BANC were dropped on '
    '2026-10-21 at Yang\'s request. Figure 1 shows the circuit and where each symbol acts.')


def schematic():
    fig, ax = plt.subplots(figsize=(10, 5.6)); ax.set_xlim(0, 12.4); ax.set_ylim(0.3, 7.4); ax.axis('off')
    G = [('glomerulus 1', 5.9), ('glomerulus 2', 4.5), ('glomerulus 3', 3.1), ('glomerulus 4', 1.7)]
    for nm, yy in G:
        ax.add_patch(FancyBboxPatch((0.25, yy - 0.42), 2.3, 0.84, boxstyle='round,pad=0.02,rounding_size=0.15', fc='#f6f5f1', ec=L.MUTED, lw=1))
        ax.text(1.4, yy + 0.12, nm, ha='center', va='center', fontsize=9, color=L.INK)
        ax.text(1.4, yy - 0.2, r'PN bouton(s), response $r_{go}$', ha='center', va='center', fontsize=7.5, color=L.MUTED)
    KC = [(6.2, 5.3), (6.2, 3.8), (6.2, 2.3)]
    con = {0: [(0, 3.0), (1, 1.5)], 1: [(1, 1.0), (2, 2.5), (3, 1.0)], 2: [(0, 1.2), (3, 2.8)]}
    for ki, (kx, ky) in enumerate(KC):
        for gi, w in con[ki]:
            ax.plot([2.55, kx - 0.45], [G[gi][1], ky], color=L.GRAY, lw=w, alpha=0.75, solid_capstyle='round', zorder=1)
        ax.add_patch(Circle((kx, ky), 0.45, fc='white', ec=L.INK, lw=1.3, zorder=3))
        ax.text(kx, ky, f'KC {ki + 1}', ha='center', va='center', fontsize=8.5, zorder=4)
    ax.text(3.9, 6.55, r'claws: weight $W_{kg}$ (line width)', fontsize=8.5, color=L.INK, ha='center')
    ax.text(6.2, 6.35, r'drive $x_{ko}=\sum_g W_{kg}\,r_{go}$', fontsize=8.5, ha='center', color=L.INK)
    ax.text(6.2, 1.35, r'response $y_{ko}=[\,x_{ko}-\theta-a\,u_k A_o\,]_+$', fontsize=8.5, ha='center', color=L.INK)
    ax.add_patch(FancyBboxPatch((9.6, 1.5), 1.6, 4.6, boxstyle='round,pad=0.02,rounding_size=0.3', fc='#e8f1fc', ec=L.INTACT, lw=1.3))
    ax.text(10.4, 4.15, 'APL', ha='center', fontsize=12, fontweight='bold', color=L.INTACT)
    ax.text(10.4, 3.6, r'activity $A_o$', ha='center', fontsize=9, color=L.INK)
    ax.text(10.4, 3.1, r'$=\frac{1}{N}\sum_k v_k\,y_{ko}$', ha='center', fontsize=9, color=L.INK)
    for kx, ky in KC:
        ax.annotate('', xy=(9.58, ky + 0.2), xytext=(kx + 0.47, ky + 0.2), arrowprops=dict(arrowstyle='-|>', color=L.INK, lw=1.1))
        ax.annotate('', xy=(kx + 0.47, ky - 0.2), xytext=(9.58, ky - 0.2), arrowprops=dict(arrowstyle='-[, widthB=0.35, lengthB=0.15', color=L.INTACT, lw=1.4))
    ax.text(8.05, 5.62, r'KC→APL: drive $v_k\,y_{ko}$', fontsize=8, ha='center', color=L.INK)
    ax.text(8.05, 4.83, r'APL→KC: $-\,a\,u_k\,A_o$', fontsize=8, ha='center', color=L.INTACT)
    ax.plot([10.4, 10.4, 1.4, 1.4], [6.12, 7.05, 7.05, 6.35], color=L.INTACT, lw=1.4, ls=(0, (4, 2)))
    ax.annotate('', xy=(1.4, 6.34), xytext=(1.4, 6.6), arrowprops=dict(arrowstyle='-[, widthB=0.35, lengthB=0.15', color=L.INTACT, lw=1.4))
    ax.text(5.9, 7.15, r'APL→PN boutons (versions P0, P, L, M0, M): input of glomerulus $g$ divided by $1+b\,p_g A_o$', fontsize=8.2, ha='center', color=L.INTACT)
    ax.text(0.25, 0.45, 'Blocked (TNT): no APL, so $a = b = 0$ and $y_{ko}=[x_{ko}-\\theta]_+$.   Arrow = excitation;  bar = inhibition.', fontsize=8.2, color=L.MUTED)
    return fig


R.fig(schematic(), 'Glomeruli (left) send PN input to KCs (middle) through claws. Line width shows the input weight $W_{kg}$, the share '
      'of a KC\'s claw PN synapses that come from each glomerulus (three example KCs). A KC\'s drive $x_{ko}$ is the $W$-weighted sum of '
      'its glomeruli\'s responses $r_{go}$. KCs excite APL (black arrows), and APL\'s activity $A_o$ is their mean response, each KC weighted '
      'by $v_k$. APL inhibits each KC by subtracting $a\\,u_k\\,A_o$ (blue bars). In the bouton versions it instead divides the input of '
      'glomerulus $g$ by $1 + b\\,p_g\\,A_o$ (dashed blue path). In the uniform version every $u_k$, $v_k$ and $p_g$ is 1.',
      'The circuit in the model and where each symbol acts.')

# ---------------------------------------------------------------- 2 mushroom bodies
R.section('Step 1: one model per mushroom body')
R.p(f'The hemibrain contains one calyx, the right one ({SIDES["hemibrain R"]:,} KCs with claws). The male CNS contains both. Its KCs are '
    'split into left and right in two steps. First, each KC\'s position is the mean of its claw centroids. Second, two-means clustering '
    '(k-means with $k = 2$, seed 0) on those positions gives two groups. Each group is named after the APL that its KCs\' on-claw APL '
    f'synapses mostly contact (APL_L or APL_R). This gives {SIDES["malecns L"]:,} KCs on the left and {SIDES["malecns R"]:,} on the right '
    '(Figure 2). Each mushroom body gets its own $W$, $u$, $v$ and threshold, and its own calibrated gains. Results are averaged over the '
    'three only at the end.')


def fig_sides():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1, 2.4]))
    for a, ds in zip(axs, ['hemibrain', 'malecns']):
        cl = abm.CLAWS[ds]; xyz = cl.groupby('kc_id')[['centroid_x_nm', 'centroid_y_nm', 'centroid_z_nm']].mean() / 1e3
        for k, col in zip([m for m in MBS if MB[m]['ds'] == ds], [L.BLOCKED, L.INTACT]):
            ids = MB[k]['W'].index; P = xyz.reindex(ids)
            X, Y = (-P.centroid_x_nm, -P.centroid_z_nm) if ds == 'hemibrain' else (-P.centroid_x_nm, -P.centroid_y_nm)
            a.scatter(X, Y, s=2, color=col, lw=0, label=f'{MBNAME[k]}: {len(ids):,} KCs', rasterized=True)
        a.set_aspect('equal', adjustable='datalim'); a.legend(markerscale=5, loc='lower center', fontsize=7.5)
        a.set_xlabel('fly\'s right → (µm; −x of the dataset)'); a.set_ylabel('dorsal ↑ (µm; ' + ('−z' if ds == 'hemibrain' else '−y') + ' of the dataset)')
        a.set_title(L.DS_LABEL[ds])
    fig.tight_layout(); return fig


R.fig(fig_sides(), 'Each dot is one KC, placed at the mean of its claw centroids and seen from behind the fly (posterior view: the fly\'s '
      'right on the right, dorsal up). Axes are the dataset\'s own coordinates in µm, sign-flipped for this view: hemibrain (−x, −z), '
      'male CNS (−x, −y). Color shows the mushroom body each KC is assigned to. In the male CNS the two groups come from two-means '
      'clustering and are named by the APL their on-claw synapses contact.', 'KCs of each mushroom body.')

# ---------------------------------------------------------------- 3 W
R.section('Step 2: input weights from the claws')
R.p('A **claw** is a cluster of one KC\'s postsynaptic sites from antennal-lobe PNs in the calyx. Sites within 2 µm of each other '
    '(single linkage) belong to the same claw. Report 02 describes how claws are found. Each claw is assigned to one glomerulus, '
    'the glomerulus of its **main PN** (the PN that makes most of the claw\'s PN synapses), and all its PN synapses count for that '
    'glomerulus. Only uniglomerular PN types count. Claws whose main PN is multiglomerular, bi-glomerular or untyped stay in the '
    'denominator but give no input. The input weight is')
R.eq(r'W_{kg} \;=\; \frac{\displaystyle\sum_{c \,\in\, \mathrm{claws}(k):\ g(c)=g} n_c}{\displaystyle\sum_{c \,\in\, \mathrm{claws}(k)} n_c}',
     note='$n_c$ = PN synapses on claw $c$; the sum on top runs over KC $k$\'s claws of glomerulus $g$, the sum below over all its claws')
# worked example KC
cl = abm.CLAWS['hemibrain'].copy(); cl['glom'] = [abm.uni(t, 'hemibrain') for t in cl.main_PN_type]
cand = cl[cl.kc_id.isin(W0.index)].groupby('kc_id').agg(n=('claw_id', 'size'), ng=('glom', lambda s: s[s != ''].nunique()),
                                                       uni=('glom', lambda s: (s != '').all()), cls=('kc_type', 'first'))
cand = cand[(cand.n == 6) & (cand.ng >= 4) & cand.uni & cand.cls.str.startswith('KCab')]
ex = None
for kid in cand.index:
    c = cl[cl.kc_id == kid]
    if c.glom.isin(RESP['ORN'].index).sum() >= 4 and (~c.glom.isin(RESP['ORN'].index)).sum() >= 1: ex = kid; break
ex = ex or cand.index[0]
c = cl[cl.kc_id == ex].sort_values('n_PN_synapses', ascending=False)
tot = c.n_PN_synapses.sum(); ki = list(W0.index).index(ex)
pa = RESP['ORN']['pa'].clip(lower=0)
rows = []
for i, (_, r) in enumerate(c.iterrows()):
    w = r.n_PN_synapses / tot; rr = pa.get(r.glom, np.nan)
    rows.append([f'claw {i + 1}', r.main_PN_type, r.glom or '—', r.n_PN_synapses, F(w, 3), F(rr, 3) if not np.isnan(rr) else 'not in table (0)',
                 F(w * (0 if np.isnan(rr) else rr), 3)])
xk = float(sum(r.n_PN_synapses / tot * (0 if np.isnan(pa.get(r.glom, np.nan)) else pa.get(r.glom)) for _, r in c.iterrows()))
assert abs(xk - H0[ki, ODORS.index('pa')]) < 1e-9
rows.append(['**sum**', '', '', tot, F(sum(r.n_PN_synapses for _, r in c.iterrows()) / tot, 3), '', f'**{F(xk, 3)}** = $x_{{k,\\mathrm{{pa}}}}$'])
R.p(f'**Worked example.** Table 2 follows one hemibrain αβ KC (body ID {ex}, type {c.kc_type.iloc[0]}) through Equations 1 and 2 for '
    f'the odor pentyl acetate (pa). Its {len(c)} claws carry {tot} PN synapses in total.')
R.table(['claw', 'main PN type', 'glomerulus $g(c)$', 'PN synapses $n_c$', 'share = contribution to $W_{kg}$', '$r_{g,\\mathrm{pa}}$ (ORN, ≥ 0)',
         '$W \\times r$'], rows, title='One KC\'s claws, its input weights and its drive by pentyl acetate.',
        caption='Each row is a claw. Its share of the KC\'s claw PN synapses is its contribution to $W_{kg}$; claws of the same glomerulus add up. '
                'The last column is the claw\'s contribution to the drive. Glomeruli missing from the ORN table count as 0.', num_cols=[3, 4, 5, 6])
R.p('The **drive** of KC $k$ by odor $o$ is the $W$-weighted sum of its glomeruli\'s responses:')
R.eq(r'x_{ko} \;=\; \sum_g W_{kg}\; r_{go}')
R.p('Because the weights of one KC add up to at most 1, $x_{ko}$ is a weighted average of the responses of the KC\'s glomeruli. '
    'Claws of glomeruli outside the response table count as 0. The drive is therefore smaller for KCs whose claws come mostly from '
    'unmeasured glomeruli.')


def fig_W():
    W = MB[k0]['W']; cls = pd.Series(MB[k0]['cls'], index=W.index)
    fig = plt.figure(figsize=(10.5, 4.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.1], wspace=0.28)
    a = fig.add_subplot(gs[0]); ng = (W > 0).sum(1)
    for c_, col in zip(['αβ', 'α′β′', 'γ'], [L.INTACT, L.BLOCKED, L.GREEN]):
        x = ng[cls == c_]; a.hist(x, bins=np.arange(0.5, 13.5), histtype='step', lw=1.6, color=col, label=f'{c_} KCs (n = {len(x)}, median {int(x.median())})')
    a.set_xlabel('number of glomeruli with $W_{kg} > 0$ (per KC)'); a.set_ylabel('number of KCs'); a.legend(fontsize=7, loc='upper right', frameon=True, framealpha=0.9)
    a.set_ylim(0, a.get_ylim()[1] * 1.35)
    a.set_title('a  Glomeruli per KC, hemibrain R')
    a2 = fig.add_subplot(gs[1])
    order_g = W.sum(0).sort_values(ascending=False).index
    ordk = np.lexsort((W[order_g].values.argmax(1), cls.map({'αβ': 0, 'α′β′': 1, 'γ': 2}).fillna(3).values))
    im = a2.imshow(W[order_g].values[ordk], aspect='auto', cmap=L.SEQ, vmin=0, vmax=0.6, interpolation='nearest')
    a2.set_xticks(range(len(order_g))); a2.set_xticklabels(order_g, rotation=90, fontsize=5.2)
    a2.set_ylabel(f'KCs ({len(W):,}), grouped by class (αβ, α′β′, γ),\nthen by their largest glomerulus')
    a2.set_xlabel('glomerulus (most total claw input on the left)')
    L.cbar(fig, im, a2, '$W_{kg}$: share of the KC\'s claw PN synapses\nfrom glomerulus $g$ (0–1; capped at 0.6 for color)')
    a2.set_title('b  The input-weight matrix $W$, hemibrain R')
    return fig


R.fig(fig_W(), '**a**, For each KC, the number of glomeruli that give it any input ($W_{kg} > 0$), as a histogram per KC class. **b**, The '
      'whole matrix $W$ for hemibrain R. Each row is a KC and each column a glomerulus; color is $W_{kg}$ (white = 0). Values above 0.6 '
      'share the darkest color. Every KC samples a few glomeruli, and different KCs sample different combinations.',
      'Input weights $W_{kg}$ across all KCs.')

# ---------------------------------------------------------------- 4 odor inputs
R.section('Step 3: odor input to the glomeruli')
cov = {rn: {k: float(MB[k]['W'][[g for g in MB[k]['W'].columns if g in RESP[rn].index]].values.sum(1).mean()) for k in MBS} for rn in RESP}
R.p('Two tables give $r_{go}$, the response of glomerulus $g$ to odor $o$. The model takes no PN-level transformation: these numbers '
    'go straight into Equation 2. Negative responses (inhibition below baseline) are set to 0.')
R.ul([f'**ORN table** (Yang\'s `ORN_ovlp16_resp_flyave_merge_dropna.csv`): {RESP["ORN"].shape[0]} glomeruli × {RESP["ORN"].shape[1]} odors, '
      'fly-averaged olfactory receptor neuron (ORN) responses. The file does not state units; the values behave like ΔF/F (range '
      f'{F(RESP["ORN"].values.min())} to {F(RESP["ORN"].values.max())}). Glomeruli in this table carry '
      f'{L.pct(min(cov["ORN"].values()))}–{L.pct(max(cov["ORN"].values()))} of the KCs\' claw PN synapses (mean over KCs, per mushroom body).',
      f'**GH146 table**: mean peak ΔF/F of GH146 PNs (7 flies), {RESP["GH146"].shape[0]} glomeruli × {RESP["GH146"].shape[1]} odors. '
      '-aPine, euc and B-myr are missing, and paa comes from the diagnostics panel at 10⁻⁵ because it is the only PAA presentation. '
      f'These glomeruli carry {L.pct(min(cov["GH146"].values()))}–{L.pct(max(cov["GH146"].values()))} of the claw PN synapses.'])


def fig_inputs():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 6.2), gridspec_kw=dict(width_ratios=[16, 13]))
    for a, rn, lab in zip(axs, ['ORN', 'GH146'], ['ORN response (fly average, units as provided)', 'GH146 PN peak ΔF/F (mean of 7 flies)']):
        X = RESP[rn].clip(lower=0)
        im = a.imshow(X.values, aspect='auto', cmap=L.SEQ, vmin=0, vmax=float(X.values.max()), interpolation='nearest')
        a.set_yticks(range(len(X))); a.set_yticklabels(X.index, fontsize=6.5); a.set_xticks(range(X.shape[1])); a.set_xticklabels(X.columns, rotation=90, fontsize=7.5)
        a.set_xlabel('odor $o$'); a.set_ylabel('glomerulus $g$'); a.set_title(f'{"a" if rn == "ORN" else "b"}  $r_{{go}}$ from the {rn} table (negatives shown as 0)')
        L.cbar(fig, im, a, lab)
    fig.tight_layout(); return fig


R.fig(fig_inputs(), 'Each cell is one glomerulus\'s response to one odor after negative values are set to 0. Color runs from 0 (white) to '
      'the table\'s maximum (dark blue), with a separate scale for each panel. **a**, ORN table, 36 glomeruli × 16 odors. **b**, GH146 '
      'table, 29 glomeruli × 13 odors. Glomeruli that are missing here give 0 input to the claws that belong to them.',
      'The glomerular responses $r_{go}$ that drive the model.')

# ---------------------------------------------------------------- 5 drive and threshold
R.section('Step 4: KC drive and the threshold with APL blocked')
R.p('With APL blocked there is no inhibition, so a KC responds by how far its drive exceeds the threshold $\\theta$:')
R.eq(r'y^{\mathrm{blocked}}_{ko} \;=\; \left[\, x_{ko} - \theta \,\right]_+ \qquad \text{where } [z]_+ = \max(z, 0)')
R.p('$\\theta$ is one number per mushroom body and input table. It is found by bisection so that, averaged over odors, 20% of KCs '
    'respond ($y_{ko} > 0$). The bisection keeps an interval $[\\theta_{lo}, \\theta_{hi}]$, which starts at 0 and the largest drive. At each '
    'step it tests the midpoint: if more than 20% respond there, the threshold must be higher, so the midpoint becomes $\\theta_{lo}$; '
    'otherwise it becomes $\\theta_{hi}$. 60 halvings make the interval negligibly small. Individual odors then respond at different levels '
    '(Figure 5). Strong odors drive many KCs past the shared threshold, and weak ones drive few.')
R.eq(r'\frac{1}{O}\sum_{o=1}^{O} \phi_o^{\mathrm{blocked}} = 0.20, \qquad \phi_o = \frac{1}{N}\#\{k : y_{ko} > 0\}',
     note='$O$ = number of odors; $\\phi_o$ = fraction of KCs responding to odor $o$')


def fig_drive():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = axs[0]; fr = (H0 > th0).mean(0); order = np.argsort(fr)
    picks = [ODORS[order[2]], ODORS[order[8]], ODORS[order[-1]]]
    bins = np.linspace(0, np.quantile(H0, 0.999), 70)
    for o, col in zip(picks, [L.GRAY, L.GREEN, L.BLOCKED]):
        j = ODORS.index(o); a.hist(H0[:, j], bins=bins, histtype='step', lw=1.6, color=col, label=f'{o}: {L.pct(fr[j])} of KCs above θ')
    a.axvline(th0, color=L.INK, lw=1, ls='--'); a.text(th0, a.get_ylim()[1] * 0.95, f'  θ = {F(th0, 3)}', fontsize=8, va='top')
    a.set_xlabel('KC drive $x_{ko}$ (ORN response units: weighted mean over the KC\'s claws)'); a.set_ylabel(f'number of KCs (of {len(H0):,})')
    a.legend(fontsize=7.5); a.set_title('a  Drive for three odors, hemibrain R, ORN input')
    a = axs[1]; _, tr = theta_of(H0, 0.2)
    ths = np.linspace(0, np.quantile(H0, 0.995), 300)
    for j in range(H0.shape[1]): a.plot(ths, [(H0[:, j] > t).mean() for t in ths], color=L.GRID, lw=0.8)
    a.plot(ths, [(H0 > t).mean(0).mean() for t in ths], color=L.BLOCKED, lw=2, label='mean over the 16 odors')
    a.plot([], [], color=L.GRID, lw=0.8, label='single odors')
    a.axhline(0.2, color=L.INK, lw=0.8, ls=':'); a.axvline(th0, color=L.INK, lw=1, ls='--')
    a.scatter([m for m, _ in tr[:8]], [c_ for _, c_ in tr[:8]], s=14, color=L.INK, zorder=4, label='first 8 bisection midpoints')
    a.set_xlabel('threshold θ (ORN response units)'); a.set_ylabel('fraction of KCs with $x_{ko} > θ$'); a.legend(fontsize=7.5)
    a.set_title('b  Choosing θ: mean fraction = 0.20')
    fig.tight_layout(); return fig


R.fig(fig_drive(), f'**a**, Histograms of the drive $x_{{ko}}$ over the {len(H0):,} hemibrain KCs for a weak, a middle and the strongest odor '
      '(ORN input). The dashed line is the threshold θ. KCs to its right respond when APL is blocked. **b**, The fraction of KCs above '
      'a threshold, as the threshold rises: thin grey lines are single odors, and the orange line is their mean. θ is where the orange line '
      'crosses 0.20 (dotted). Black dots are the first bisection midpoints as they close in on it.',
      'The KC drive and how the threshold θ is set.')
tab = [[MBNAME[k], F(GAIN[('ORN', k)]['theta'], 3), F(GAIN[('GH146', k)]['theta'], 3)] for k in MBS]
R.table(['mushroom body', 'θ with ORN input', 'θ with GH146 input'], tab, title='Thresholds.',
        caption='In each case θ makes 20% of KCs respond with APL blocked, averaged over odors. The units are those of the response table.', num_cols=[1, 2])

# ---------------------------------------------------------------- 6 APL feedback, fixed point
R.section('Step 5: APL feedback and what "solved self-consistently" means')
R.p('With APL intact, each KC is inhibited in proportion to APL\'s activity for that odor:')
R.eq(r'y_{ko} \;=\; \left[\, x_{ko} - \theta - a\, u_k\, A_o \,\right]_+')
R.p('and APL\'s activity is the mean response of the KCs, each weighted by how strongly it drives APL:')
R.eq(r'A_o \;=\; \frac{1}{N}\sum_{k=1}^{N} v_k\, y_{ko}')
R.p('In the uniform version (U) every $u_k = v_k = 1$. Every KC then gets the same inhibition $a A_o$ for a given odor, and APL\'s '
    'activity is simply the mean KC response. The gain $a$ is one number per mushroom body and version. It is the same for every odor '
    'and every KC. The inhibition still differs between odors, because $A_o$ does.')
R.box('Why "self-consistently"', [
    '$A_o$ appears on both sides. The KC responses (Equation 5) depend on APL\'s activity, and APL\'s activity (Equation 6) depends on '
    'the KC responses. The model needs the one value of $A_o$ that reproduces itself: the value that, when used to compute the KC '
    'responses, gives back the same $A_o$ as their mean. Mathematically this is a fixed point. Physically, it is the steady state in which '
    'APL and the KCs have settled.'], 'plain')
R.p('To find it, define for each odor the function')
R.eq(r'F_o(A) \;=\; \frac{1}{N}\sum_{k} v_k \left[\, x_{ko} - \theta - a\, u_k\, A \,\right]_+',
     note='the APL activity the KCs would produce if APL\'s activity were $A$')
R.p('and solve $A = F_o(A)$. $F_o$ starts at $F_o(0)$, the mean (weighted) response with no inhibition, and can only fall as $A$ grows, '
    'because more inhibition never raises a KC\'s response. The line $A$ rises from 0. The two cross exactly once, which makes the '
    'solution unique (Figure 6). The code finds it by bisection on $[0, F_o(0)]$: if $A > F_o(A)$ at the midpoint, the crossing lies '
    'below it; otherwise above. 60 halvings, done for all odors at once (function `fixed_point` in `apl_block_model.py`).')
aU0 = GAIN[('ORN', k0)]['U']


def Fo(A, j, a=aU0):
    return np.maximum(H0[:, j] - th0 - a * A, 0).mean()


def fig_fp():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    a = axs[0]; fr = (H0 > th0).mean(0); order = np.argsort(fr)
    picks = [ODORS[order[2]], ODORS[order[8]], ODORS[order[-1]]]
    Amax = max(Fo(0, ODORS.index(o)) for o in picks) * 1.05; As = np.linspace(0, Amax, 300)
    a.plot(As, As, color=L.INK, lw=1, ls='--', label='$A$ (the line $F = A$)')
    for o, col in zip(picks, [L.GRAY, L.GREEN, L.BLOCKED]):
        j = ODORS.index(o); Fv = [Fo(x, j) for x in As]; a.plot(As, Fv, color=col, lw=1.8, label=f'$F_o(A)$, {o}')
        Ao = Y0U[:, j].mean(); a.scatter([Ao], [Ao], s=40, color=col, edgecolor='white', zorder=4)
        a.annotate(f'$A_o$ = {Ao:.4f}', (Ao, Ao), xytext=(6, -12), textcoords='offset points', fontsize=7.5, color=col)
    a.set_xlabel('assumed APL activity $A$ (mean KC response units)'); a.set_ylabel('APL activity the KCs produce, $F_o(A)$')
    a.set_title(f'a  Fixed point per odor (hemibrain R, ORN, U, a = {aU0:.3g})'); a.legend(fontsize=7.5)
    a = axs[1]; j = ODORS.index(picks[-1]); lo, hi = 0.0, Fo(0, j); rows_ = []
    for it in range(10):
        m = (lo + hi) / 2; fm = Fo(m, j); rows_.append((it + 1, lo, hi, m, fm))
        if m > fm: hi = m
        else: lo = m
    a.plot(As, [Fo(x, j) - x for x in As], color=L.BLOCKED, lw=1.8, label=f'$F_o(A) - A$, {picks[-1]}')
    a.axhline(0, color=L.INK, lw=0.8)
    for it, lo_, hi_, m, fm in rows_[:6]:
        a.plot([lo_, hi_], [-0.004 * it - 0.002] * 2, color=L.INTACT, lw=2.4, solid_capstyle='butt')
        a.text(hi_, -0.004 * it - 0.002, f'  step {it}', fontsize=6.5, va='center', color=L.MUTED)
    a.set_xlabel('assumed APL activity $A$'); a.set_ylabel('$F_o(A) - A$ (zero at the solution)')
    a.set_title('b  Bisection: the search interval halves each step'); a.legend(fontsize=7.5, loc='upper right')
    fig.tight_layout(); return fig, rows_, picks[-1]


fg, bis, ostrong = fig_fp()
R.fig(fg, '**a**, For three odors (a weak one, a middle one and the strongest; hemibrain R, ORN input, uniform APL at its calibrated gain), '
      'the APL activity that the KC responses would produce, $F_o(A)$ (colored), against the assumed activity $A$. The dashed diagonal is '
      '$F = A$, and the dots mark where each curve crosses it: the self-consistent APL activity $A_o$. Stronger odors cross further out, so '
      'they recruit more APL. **b**, The bisection for the strongest odor. Blue bars are the search interval $[A_{lo}, A_{hi}]$ at each step '
      '(drawn below zero for visibility). Each step halves the interval around the zero of $F_o(A) - A$.',
      'Solving for APL\'s activity: the fixed point $A_o = F_o(A_o)$.')
R.table(['step', '$A_{lo}$', '$A_{hi}$', 'midpoint $m$', '$F_o(m)$', 'decision'],
        [[it, F(lo, 5), F(hi, 5), F(m, 5), F(fm, 5), 'm > F(m): solution below m, so $A_{hi} = m$' if m > fm else 'm ≤ F(m): solution above m, so $A_{lo} = m$']
         for it, lo, hi, m, fm in bis[:8]],
        title=f'The first bisection steps for {ostrong} (hemibrain R, ORN input, uniform APL).',
        caption=f'The interval starts at $[0, F_o(0)]$. After 60 steps it has shrunk by a factor of 2⁶⁰. The final $A_o$ is {Y0U[:, ODORS.index(ostrong)].mean():.6f}.',
        num_cols=[1, 2, 3, 4])

# ---------------------------------------------------------------- 7 calibration
R.section('Step 6: calibrating the APL gain')
R.p('The gain $a$ sets how strong APL is. It is chosen so that, with APL intact, 10% of KCs respond on average over odors, half the '
    'blocked level. More gain means more inhibition and fewer KCs responding, so the mean fraction falls steadily as $a$ grows. The code '
    '(function `calib`) starts at $a = 1$ and doubles it until the fraction drops to 10% or below. It then bisects 50 times between the '
    'last two values. Every evaluation solves the fixed point of Step 5 for all odors.')


def fig_cal():
    fig, a = plt.subplots(figsize=(6.6, 3.9))
    ones, oneg = np.ones(len(H0)), np.ones(len(gl0))
    As = np.geomspace(0.05, 200, 60)
    C = np.array([abm.cl_of(abm.model_y(Wm0, Rp0, th0, x, 0, ones, ones, oneg)) for x in As])
    for j in range(C.shape[1]): a.plot(As, C[:, j], color=L.GRID, lw=0.8)
    a.plot(As, C.mean(1), color=L.INTACT, lw=2, label='mean over the 16 odors')
    a.plot([], [], color=L.GRID, lw=0.8, label='single odors')
    a.axhline(0.1, color=L.INK, ls=':', lw=0.8); a.axhline(0.2, color=L.BLOCKED, ls=':', lw=0.8)
    a.axvline(aU0, color=L.INK, ls='--', lw=1); a.text(aU0, 0.33, f'  a = {aU0:.3g}', fontsize=8)
    a.text(0.055, 0.205, 'blocked level 0.20 (a → 0)', fontsize=7.5, color=L.BLOCKED, va='bottom')
    a.set_xscale('log'); a.set_xlabel('APL gain $a$ (log scale)'); a.set_ylabel('fraction of KCs responding, $\\phi_o$')
    a.legend(fontsize=7.5, loc='upper right'); a.set_title('Calibrating $a$ (hemibrain R, ORN input, uniform APL)')
    fig.tight_layout(); return fig


R.fig(fig_cal(), 'The fraction of KCs responding with APL intact as the gain $a$ grows (log axis). Thin grey lines are single odors, and '
      'the blue line is their mean. The calibrated gain (dashed) is where the mean reaches 0.10 (dotted black). At $a \\to 0$ the model is '
      'the blocked model, with a mean of 0.20 (dotted orange).', 'Calibrating the APL gain.')
VERS = ['U', 'C', 'P0', 'P', 'L', 'M0', 'M']
R.table(['mushroom body', 'input'] + [f'{v}' for v in VERS],
        [[MBNAME[k], rn] + [f'{GAIN[(rn, k)][v]:.3g}' for v in VERS] for rn in RESP for k in MBS],
        title='Calibrated gains (20% → 10%).',
        caption='U and C give $a$ (subtractive); P0, P and L give $b$ (divisive, on PN boutons or claws); M0 and M give the common scale $s$ '
                '(Section 8). The gains are not comparable across columns, because each version uses a different form and normalization.',
        num_cols=list(range(2, 2 + len(VERS))))

# ---------------------------------------------------------------- 8 readouts
R.section('Step 7: what is read out')
R.p('Three quantities are read out for every mushroom body, version and input table:')
R.ul(['**Fraction responding per odor**, $\\phi_o$, blocked and intact.',
      '**Odor-pair similarity** $S_{ab}$: the cosine similarity of the two odors\' KC population response vectors $\\mathbf{y}_a = (y_{1a}, \\dots, y_{Na})$ '
      'and $\\mathbf{y}_b$ (Equation 8). It is the same measure as Yang\'s RDMs (which hold $1 - S$). Because responses are ≥ 0, $S$ runs from 0 '
      '(no KC responds to both) to 1 (the same pattern up to a scale factor). It is undefined when an odor drives no KC.',
      '**The effect of blocking APL** on a pair, $\\Delta S_{ab} = S^{\\mathrm{blocked}}_{ab} - S^{\\mathrm{intact}}_{ab}$ (Equation 9). It is computed per '
      'mushroom body and then averaged over the three.'])
R.eq(r'S_{ab} \;=\; \cos(\mathbf{y}_a, \mathbf{y}_b) \;=\; \frac{\sum_k y_{ka}\, y_{kb}}{\sqrt{\sum_k y_{ka}^2}\;\sqrt{\sum_k y_{kb}^2}}')
R.eq(r'\Delta S_{ab} \;=\; S^{\mathrm{blocked}}_{ab} - S^{\mathrm{intact}}_{ab}')
drop = SUM['odors_left_out_of_pair_results']
R.p(f'Odors that drive fewer than 0.5% of KCs with APL intact (in any version and mushroom body) are left out of the pair results, because '
    f'their similarity rests on a handful of KCs. In the main setting this drops {", ".join(drop["ORN " + MAIN]) or "none"} with ORN input and '
    f'{", ".join(drop["GH146 " + MAIN]) or "none"} with GH146 input.')


def od_mean(rn, col):
    o = OD[(OD.coding_levels == MAIN) & (OD.responses == rn)]
    return o.groupby('odor', sort=False)[col].mean()


def fig_frac():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    fb, fi = od_mean('ORN', 'frac_KCs_responding_blocked'), od_mean('ORN', 'frac_KCs_responding_intact_U')
    order = fb.sort_values().index; x = np.arange(len(order))
    a = axs[0]
    a.bar(x - 0.2, fb[order] * 100, 0.38, color=L.BLOCKED, label='APL blocked'); a.bar(x + 0.2, fi[order] * 100, 0.38, color=L.INTACT, label='APL intact (uniform, U)')
    a.set_xticks(x); a.set_xticklabels(order, rotation=90); a.set_ylabel('KCs responding, % (mean of 3 mushroom bodies)')
    a.set_xlabel('odor (sorted by the blocked fraction)'); a.legend(fontsize=7.5); a.set_title('a  Fraction responding per odor, ORN input')
    a = axs[1]; gain = (fb - fi) * 100
    a.scatter(fb * 100, gain, s=22, color=L.INK)
    for o in order: a.annotate(o, (fb[o] * 100, gain[o]), xytext=(3, 2), textcoords='offset points', fontsize=6.5, color=L.MUTED)
    r_ = SUM['model'][f'ORN {MAIN}']['per-odor gain U vs blocked fraction: r']
    a.set_xlabel('KCs responding with APL blocked, %'); a.set_ylabel('KCs gained by blocking APL\n(blocked − intact), percentage points')
    a.set_title(f'b  Strong odors gain most (r = {r_:.2f})')
    fig.tight_layout(); return fig


R.fig(fig_frac(), '**a**, Percentage of KCs responding to each odor with APL blocked (orange) and intact (blue, uniform APL), averaged over '
      'the three mushroom bodies (ORN input, 20% → 10%). **b**, One point per odor. The horizontal axis is the blocked percentage (odor '
      'strength). The vertical axis is how many percentage points blocking APL adds. Strong odors recruit more APL, so removing APL adds the '
      'most KCs for them.', 'Per-odor fractions responding, blocked and intact.')


def matS(rn, ver, col='cos'):
    P = PR[(PR.coding_levels == MAIN) & (PR.responses == rn)]
    M = pd.DataFrame(np.nan, index=ODORS, columns=ODORS)
    for (o1, o2), x in P.groupby(['odor_a', 'odor_b'], sort=False)[f'{col}_{ver}'].mean().items():
        M.loc[o1, o2] = M.loc[o2, o1] = x
    return M


def fig_sim():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    Sb, Si = matS('ORN', 'blocked'), matS('ORN', 'intact_U'); D = Sb - Si
    for o in drop['ORN ' + MAIN]: D.loc[o] = np.nan; D[o] = np.nan
    vmax = float(np.nanmax([Sb.values, Si.values]))
    for a, M, t, cm, lab, vm in [(axs[0], Sb, 'a  $S_{ab}$, APL blocked', L.SEQ, 'cosine similarity $S_{ab}$ (0–1)', (0, vmax)),
                                 (axs[1], Si, 'b  $S_{ab}$, APL intact (uniform)', L.SEQ, 'cosine similarity $S_{ab}$ (0–1)', (0, vmax)),
                                 (axs[2], D, 'c  $\\Delta S_{ab}$ = blocked − intact', L.DIV, '$\\Delta S_{ab}$ (difference in cosine)', (-0.2, 0.2))]:
        Mv = M.values.copy(); np.fill_diagonal(Mv, np.nan)
        im = a.imshow(Mv, cmap=cm, vmin=vm[0], vmax=vm[1], interpolation='nearest')
        a.set_xticks(range(16)); a.set_xticklabels(ODORS, rotation=90, fontsize=6.8); a.set_yticks(range(16)); a.set_yticklabels(ODORS, fontsize=6.8)
        a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]; a.set_title(t, fontsize=9)
        L.cbar(fig, im, a, lab)
    fig.tight_layout(); return fig


dU = SUM['model'][f'ORN {MAIN}']['dS U: mean, sd, min, max over pairs (mean over MBs)']
R.fig(fig_sim(), f'Odor-by-odor matrices, mean of the three mushroom bodies, ORN input. **a**, **b**, Cosine similarity of the KC '
      f'population responses with APL blocked and intact (uniform APL), on one shared scale (white 0, dark blue the highest value). '
      f'**c**, Their difference, $\\Delta S_{{ab}}$, on a diverging scale: orange = more similar when APL is blocked, blue = less similar, '
      f'grey-white = no change. Grey cells: diagonal, or an odor left out because fewer than 0.5% of KCs respond ({", ".join(drop["ORN " + MAIN])}). '
      f'Mean $\\Delta S$ = {F(dU[0], 3, True)}, range {F(dU[2], 2, True)} to {F(dU[3], 2, True)}.',
      'Odor-pair similarity with APL blocked and intact, and the change.')

# ---------------------------------------------------------------- 9 versions
R.section('Step 8: versions that use APL\'s connectome wiring')
R.p('Each version replaces part of the uniform APL with a connectome measurement. All of them keep the same blocked responses and the same '
    'θ, and all are calibrated to the same 10%. The versions therefore differ only in **where** the inhibition acts and **how it is '
    'weighted**. Table 6 summarizes them.')
R.table(['version', 'KC response $y_{ko}$', 'APL activity $A_o$', 'weights from the connectome'],
        [['U, uniform', '$[x_{ko}-\\theta-a A_o]_+$', '$\\frac1N\\sum_k y_{ko}$', 'none ($u = v = 1$)'],
         ['C, per-KC counts', '$[x_{ko}-\\theta-a u_k A_o]_+$', '$\\frac1N\\sum_k v_k y_{ko}$', '$u_k$ = APL→KC, $v_k$ = KC→APL synapses of KC $k$ in the calyx, each ÷ its mean'],
         ['Cs, shuffled control', 'as C', 'as C', '$(u_k, v_k)$ pairs shuffled among KCs of the same class (αβ, α′β′, γ)'],
         ['Cc, class means', 'as C', 'as C', '$u_k, v_k$ replaced by the mean of the KC\'s class'],
         ['P0, boutons, all alike', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+b A_o}-\\theta]_+$', '$\\frac1N\\sum_k y_{ko}$', 'none'],
         ['P, boutons per glomerulus', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+b\\,p_g A_o}-\\theta]_+$', 'as P0', '$p_g$ = APL→PN per PN→KC synapse of glomerulus $g$'],
         ['L, claws per glomerulus', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+b\\,q_g A_o}-\\theta]_+$', 'as P0', '$q_g$ = on-claw APL→KC per claw PN synapse of $g$'],
         ['M0, U + boutons at real share', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+s f b_{P0} A_o}-\\theta - s(1-f) a_U A_o]_+$', 'as U', '$f$ = boutons\' share of APL\'s calyx output'],
         ['M, U + boutons per glomerulus', 'as M0 with $p_g$ in the bouton term', 'as U', '$f$ and $p_g$']],
        title='The APL versions of the prediction run.',
        caption='$a_U$ and $b_{P0}$ are the calibrated gains of U and P0. In M0 and M only the common scale $s$ is calibrated. Report 07 adds two '
                'feed-forward versions (F0, F), where APL activity comes from the PN input instead of the KCs.')
R.sub('C: APL synapses per KC')
R.p('$u_k$ is KC $k$\'s number of APL→KC synapses in the calyx, divided by the mean over the mushroom body\'s KCs. $v_k$ is the same for '
    'KC→APL synapses (calyx only since 2026-10-21, at Yang\'s request). A KC with $u_k = 2$ receives twice the average inhibition for the '
    'same APL activity, and a KC with $v_k = 2$ counts twice as much toward APL\'s activity. APL is still one unit with one activity per '
    'odor. The two controls ask whether it matters **which** KC has which counts (Cs keeps each KC class\'s distribution but shuffles it) '
    'or only the class averages (Cc).')


def fig_uv():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    mb = MB[k0]; u, v = mb['u'] / mb['u'].mean(), mb['v'] / mb['v'].mean(); cls = mb['cls']
    a = axs[0]
    for c_, col in zip(['αβ', 'α′β′', 'γ'], [L.INTACT, L.BLOCKED, L.GREEN]):
        m = cls == c_; a.scatter(v[m], u[m], s=5, color=col, alpha=0.6, lw=0, label=f'{c_} (n = {m.sum()})')
    a.set_xlabel('$v_k$: KC→APL synapses in the calyx ÷ mean'); a.set_ylabel('$u_k$: APL→KC synapses in the calyx ÷ mean')
    a.legend(markerscale=3, fontsize=7.5); a.set_title('a  Per-KC weights, hemibrain R')
    a = axs[1]
    for k, col in zip(MBS, [L.BLOCKED, L.GREEN, L.INTACT]):
        uu = MB[k]['u'] / MB[k]['u'].mean(); a.hist(uu, bins=np.linspace(0, 4, 41), histtype='step', lw=1.5, color=col, label=MBNAME[k])
    a.set_xlabel('$u_k$ (APL→KC synapses in the calyx ÷ mean)'); a.set_ylabel('number of KCs'); a.legend(fontsize=7.5)
    a.set_title('b  Spread of $u_k$ in each mushroom body')
    fig.tight_layout(); return fig


R.fig(fig_uv(), '**a**, Each point is one hemibrain KC. Horizontal: KC→APL synapses in the calyx divided by the mushroom body mean ($v_k$). '
      'Vertical: APL→KC synapses divided by their mean ($u_k$). Colors are KC classes. **b**, Histograms of $u_k$ in each mushroom body. '
      'Uniform APL corresponds to every KC sitting at (1, 1).', 'The per-KC weights used by version C.')
R.sub('P, L, M: APL synapses per glomerulus')
R.p('In P and L, inhibition **divides** a glomerulus\'s input instead of subtracting from the KC, because APL→PN synapses act presynaptically '
    'on the PN boutons and scale down what the boutons pass to all their KCs. $p_g$ (P) is the number of APL→PN synapses on glomerulus $g$\'s '
    'boutons per PN→KC synapse of those boutons. $q_g$ (L) is the number of APL→KC synapses on $g$\'s claws per PN synapse on those claws. '
    'Report 04 shows both across glomeruli. Within each mushroom body, the weights are scaled so that their claw-input-weighted mean is 1:')
R.eq(r"\tilde p_g \;=\; \frac{p_g}{\left(\sum_{g'} w_{g'}\, p_{g'}\right) \big/ \sum_{g'} w_{g'}}, \qquad w_g = \sum_k W_{kg}",
     note='sums over glomeruli $g\'$ that have a value; glomeruli without a value get $\\tilde p_g = 1$')
R.p('With this scaling, P and P0 put the same total inhibition on the average claw input, and only its distribution across glomeruli '
    'differs. In M0 and M only a share $f$ of the inhibition goes to boutons. $f$ = APL→PN ÷ (APL→PN + APL→KC) synapses in the calyx, '
    f'{SUM["bouton_share_of_APL_calyx_output_synapses"]["hemibrain"]} in hemibrain and {SUM["bouton_share_of_APL_calyx_output_synapses"]["malecns"]} '
    'in male CNS. The rest is uniform on the KCs. The KC gain is $(1-f)\\,a_U$, the bouton gain $f\\,b_{P0}$, and both are scaled by one '
    'common factor $s$ to reach 10%.')


def fig_pq():
    W, gl, Wm, Rp, H = setup(MB[k0], RESP['ORN'])
    p = pd.Series(norm_w(MB[k0]['p'], Wm, gl), index=gl); q = pd.Series(norm_w(MB[k0]['q'], Wm, gl), index=gl)
    order = p.sort_values().index
    fig, a = plt.subplots(figsize=(10.5, 3.6)); x = np.arange(len(order))
    a.bar(x - 0.2, p[order], 0.38, color=L.GREEN, label='$\\tilde p_g$: APL→PN per PN→KC synapse (version P)')
    a.bar(x + 0.2, q[order], 0.38, color=L.PURPLE, label='$\\tilde q_g$: on-claw APL→KC per claw PN synapse (version L)')
    a.axhline(1, color=L.INK, lw=0.8, ls=':')
    a.set_xticks(x); a.set_xticklabels(order, rotation=90, fontsize=7); a.set_ylabel('weight ÷ claw-input-weighted mean')
    a.set_xlabel('glomerulus (ORN-table glomeruli, sorted by $\\tilde p_g$)'); a.legend(fontsize=7.5)
    a.set_title('Glomerulus weights used by P and L, hemibrain R')
    fig.tight_layout(); return fig


R.fig(fig_pq(), 'For each glomerulus of the ORN table (hemibrain R), the bouton weight $\\tilde p_g$ (green) and claw weight $\\tilde q_g$ '
      '(purple), each divided by its claw-input-weighted mean (Equation 10), so the dotted line at 1 is the average. A glomerulus at 2 gets '
      'twice the average inhibition per unit of input in that version.', 'Per-glomerulus weights after scaling.')

# ---------------------------------------------------------------- 10 results
R.section('Results of the prediction run')
s_o, s_g = SUM['model'][f'ORN {MAIN}'], SUM['model'][f'GH146 {MAIN}']
R.ul([f'**Uniform APL.** The KCs gained by blocking APL track odor strength (r = {s_o["per-odor gain U vs blocked fraction: r"]:.2f} with ORN input), '
      f'and so do all versions (r ≥ {min(s_o["per-odor gain (blocked - intact): r between versions (mean over MBs)"].values()):.3f} between versions). '
      f'Similarity rises most for pairs of strong odors: mean $\\Delta S$ {F(s_o["dS U: mean, sd, min, max over pairs (mean over MBs)"][0], 3, True)} '
      f'with ORN and {F(s_g["dS U: mean, sd, min, max over pairs (mean over MBs)"][0], 3, True)} with GH146. The pattern is the same in all three '
      f'mushroom bodies (median r between them {s_o["dS U: r between MBs (median)"]:.2f} and {s_g["dS U: r between MBs (median)"]:.2f}).',
      f'**Per-KC counts (C).** They change $\\Delta S$ by {F(s_o["dS C - dS U"]["mean |diff| over pairs and MBs"], 3)} on average (ORN), but the '
      f'same counts shuffled within class do much the same (maps r {s_o["dS C-U vs Cs-U maps: r (mean over MBs)"]:.2f}). What is left differs '
      f'between mushroom bodies (median r {s_o["dS C - dS Cs"]["r between MBs (median)"]:.2f}).',
      f'**Boutons per glomerulus (P vs P0).** A consistent pattern (median r {s_o["dS P - dS P0"]["r between MBs (median)"]:.2f} between mushroom bodies; '
      f'mean |change| {F(s_o["dS P - dS P0"]["mean |diff| over pairs and MBs"], 3)}): pa–eb and pa–ep rise more, eb and ep move away from 1-6ol and 1-8ol.',
      f'**At the boutons\' real share (M vs M0)** the same pattern shrinks to a mean |change| of {F(s_o["dS M - dS M0"]["mean |diff| over pairs and MBs"], 3)}.',
      f'**Claws (L vs P0)**: mean |change| {F(s_o["dS L - dS P0"]["mean |diff| over pairs and MBs"], 3)}.'])


def fig_diff():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    pairs = [('intact_U', 'intact_C', 'a  C − U: per-KC counts'), ('intact_P0', 'intact_P', 'b  P − P0: boutons per glomerulus'),
             ('intact_M0', 'intact_M', 'c  M − M0: boutons at real share')]
    for a, (ref, ver, t) in zip(axs, pairs):
        D = matS('ORN', ref) - matS('ORN', ver)          # (S_b - S_ver) - (S_b - S_ref) = S_ref - S_ver
        for o in drop['ORN ' + MAIN]: D.loc[o] = np.nan; D[o] = np.nan
        Mv = D.values.copy(); np.fill_diagonal(Mv, np.nan)
        im = a.imshow(Mv, cmap=L.DIV, vmin=-0.12, vmax=0.12, interpolation='nearest')
        a.set_xticks(range(16)); a.set_xticklabels(ODORS, rotation=90, fontsize=6.8); a.set_yticks(range(16)); a.set_yticklabels(ODORS, fontsize=6.8)
        a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]; a.set_title(t, fontsize=9)
        L.cbar(fig, im, a, 'difference in $\\Delta S_{ab}$ (version − reference)')
    fig.tight_layout(); return fig


R.fig(fig_diff(), 'How much each wiring version changes the effect of blocking APL, pair by pair (mean of three mushroom bodies, ORN input). '
      'Each cell is $\\Delta S_{ab}$ of the version minus $\\Delta S_{ab}$ of its reference. The blocked responses are identical, so this equals '
      'the reference\'s intact similarity minus the version\'s. Orange: blocking APL raises the pair\'s similarity more in this version. '
      'Blue: less. The scale is clipped at ±0.12. Grey: diagonal or an odor left out.',
      'Wiring versions compared with their uniform references.')
R.box('What this step does not include', [
    'One threshold for all KCs, with no differences in excitability between classes, and no spikes or time course.',
    'APL is one unit with one activity per odor in every version. P and L weight its output per glomerulus but do not make its activity local. '
    'A local, non-spiking APL is modelled separately (`scripts/apl_local_model.py`).',
    'The inputs are ORN or GH146 averages from other flies, not the recorded flies\' PNs.',
    'The coding levels (20% blocked, 10% intact) are assumptions here. Report 07 replaces them with the recordings.'], 'caveat')
R.save()
