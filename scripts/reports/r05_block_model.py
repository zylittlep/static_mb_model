"""Report 05: the APL block model, built step by step (the prediction run of scripts/apl_block_model.py).
Rebuilt 2026-11-04 for Yang's revision (2026-11-03): corrected calyx, new claws and new ORN table (APL_VARIANT=claws_v2m2), PN-level
transform, negative responses, version MC, two other input weightings, figures for every version, marked pairs, the recordings'
heat maps at the end, ms kept everywhere. The old version is in archive_2026-11-03/."""
import sys, json, os
os.environ['APL_VARIANT'] = 'claws_v2m2'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from scipy.stats import spearmanr, pearsonr
import report_lib as L
import apl_block_model as abm
L.style(); L.DATE = '2026-11-04'
F = L.fmt
OWV = abm.OUT                                   # with_outside_ROI_synapses/claws_v2m2/odor_weighted/

RESP = abm.responses(); RV = abm.responses_revision(); MB, SIDES = abm.mushroom_bodies(); MBS = list(MB)
SUM = json.load(open(OWV + 'APL_block_model_summary.json'))
OD = pd.read_csv(OWV + 'APL_block_model_per_odor.csv'); PR = pd.read_csv(OWV + 'APL_block_model_odor_pairs.csv')
OW2 = pd.read_csv(OWV + 'APL_block_model_other_weights_odor_pairs.csv'); OW2o = pd.read_csv(OWV + 'APL_block_model_other_weights_per_odor.csv')
VALL = abm.VARS                                 # U C Cs Cc P0 P L M0 M MC
MAIN = '20% -> 10%'
MBNAME = {'hemibrain R': 'hemibrain R', 'malecns L': 'male CNS L', 'malecns R': 'male CNS R'}
ODORS = list(RESP['ORN'].columns)


# ---------------------------------------------------------------- replicate the model (one MB, one input) to show the steps
def setup(mb, R, clip=True):
    W = mb['W']; gl = [g for g in W.columns if g in R.index]
    Wm, Rp = W[gl].values, (np.clip(R.loc[gl].values, 0, None) if clip else R.loc[gl].values.astype(float))
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


def gains(mb, R, cb=0.2, ci=0.1, clip=True):
    W, gl, Wm, Rp, H = setup(mb, R, clip); th, _ = theta_of(H, cb)
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
    out['MC'] = abm.calib(lambda s: abm.cl_of(abm.model_y(Wm, Rp, th, s * (1 - f) * out['C'], s * f * out['P'], u, v, p)).mean(), ci)
    return out


_cache = os.environ.get('R05_GAIN_CACHE')          # optional pickle for faster rebuilds while editing
if _cache and os.path.exists(_cache):
    import pickle; GAIN = pickle.load(open(_cache, 'rb'))
else:
    GAIN = {(rn, k): gains(MB[k], RV[rn], clip=abm.CLIP(rn)) for rn in RV for k in MBS}
    if _cache: import pickle; pickle.dump(GAIN, open(_cache, 'wb'))
# check the replication against the saved per-odor table (U, hemibrain R, ORN)
k0 = 'hemibrain R'; W0, gl0, Wm0, Rp0, H0 = setup(MB[k0], RESP['ORN']); th0 = GAIN[('ORN', k0)]['theta']
one0, oneg0 = np.ones(len(H0)), np.ones(len(gl0))
Y0U = abm.model_y(Wm0, Rp0, th0, GAIN[('ORN', k0)]['U'], 0, one0, one0, oneg0)
saved = OD[(OD.coding_levels == MAIN) & (OD.responses == 'ORN') & (OD.mushroom_body == k0)].set_index('odor')
assert np.allclose(abm.cl_of(Y0U), saved.frac_KCs_responding_intact_U.reindex(ODORS).values, atol=1e-4)
assert np.allclose(abm.cl_of(np.maximum(H0 - th0, 0)), saved.frac_KCs_responding_blocked.reindex(ODORS).values, atol=1e-4)
# ... and MC (the version added in this revision), male CNS L, GH146
_k, _rn = 'malecns L', 'GH146'; _W, _gl, _Wm, _Rp, _H = setup(MB[_k], RV[_rn]); _mb = MB[_k]; _f = abm.F_SHARE['malecns']
_u, _v = _mb['u'] / _mb['u'].mean(), _mb['v'] / _mb['v'].mean(); _p = norm_w(_mb['p'], _Wm, _gl); _G = GAIN[(_rn, _k)]
_Y = abm.model_y(_Wm, _Rp, _G['theta'], _G['MC'] * (1 - _f) * _G['C'], _G['MC'] * _f * _G['P'], _u, _v, _p)
_sv = OD[(OD.coding_levels == MAIN) & (OD.responses == _rn) & (OD.mushroom_body == _k)].set_index('odor')
assert np.allclose(abm.cl_of(_Y), _sv.frac_KCs_responding_intact_MC.reindex(RV[_rn].columns).values, atol=1e-4)

NCLAW = {ds: float(abm.CLAWS[ds].groupby('kc_id').size().median()) for ds in abm.DS}
_ws = [float(MB[k]['W'].values.sum(1).mean()) for k in MBS]; WSUM_LO, WSUM_HI = min(_ws), max(_ws)

R = L.Report('05', 'How is the APL block model built, step by step, and what does it predict about blocking APL, before any '
             'comparison with the TNT / TNT3c recordings?',
             inputs=['All in `with_outside_ROI_synapses/claws_v2m2/` (corrected calyx, new claws; Reports 02 and 04):',
                     'KC claws `hemibrain_KC_claws_derived.csv.gz`, `malecns_KC_claws_derived.csv.gz`; per-KC calyx APL synapse counts '
                     '`KC_weights_APL_PN_per_KC.csv`; per-glomerulus APL wiring `odor_weighted/APL_wiring_by_glomerulus.csv`',
                     'ORN responses: `orn_inputs_2026-11-03/ORN_TNT16_flymean.csv` (new table from Yang\'s `responses_wide.csv`; Report 03)',
                     'GH146 means `odor_weighted/GH146_mean_by_glomerulus.csv` (Report 03)',
                     'For Section 13 only: the TNT and TNT3c recordings (Report 06)'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/odor_weighted/`: `APL_block_model_per_odor.csv`, `APL_block_model_odor_pairs.csv`, '
                      '`APL_block_model_other_weights_*.csv`, `APL_block_model_summary.json`, `APL_block_model.png`'],
             scripts=['`scripts/apl_block_model.py` (the model)', '`scripts/reports/r05_block_model.py` (this report; it re-runs the steps '
                      'for illustration and checks them against the saved tables)'])

R.p('This report explains the KC model that all later steps use. Report 07 takes the same model and pins it to the recordings. '
    'Here it runs in its **prediction** form, set up before the recordings arrived: 20% of KCs respond to an odor (averaged over '
    'odors) with APL blocked, and 10% with APL intact. Every symbol is defined in Table 1, and again where it first appears.')
R.box('What changed in this revision (2026-11-04)', [
    'Built on the corrected calyx (synapses outside the official ROIs added) and the new claws (8 µm along the KC skeleton, one claw per '
    'PN with at least 2 PSDs; Report 02), and the new ORN table from `responses_wide.csv` (Report 03).',
    'New inputs: ORN responses passed through a PN-level transformation (Olsen et al. 2010, as used by Kennedy 2019), compared with GH146 '
    '(Section 4); ORN and GH146 with their negative responses kept (Section 4). Their results are in Section 11.',
    'New version MC: APL inhibits both the KCs (per KC, as in C) and the PN boutons (per glomerulus, as in P), at the boutons\' real share '
    'of APL\'s synapses (Section 9).',
    'Why the input weights are divided by the KC\'s total (Section 3), with two alternatives run in Section 12; why $p_g$ and $q_g$ are '
    'counted per synapse (Section 9, TAB_PQ).',
    'ms is kept everywhere; similarities that cannot be computed (an odor that drives no KC) are marked, not dropped.',
    'Every number in the Results is spelled out (TAB_NUM; "r" is always Pearson, "ρ" Spearman), the similarity maps are shown for every '
    'version with the pairs that change most and least outlined, and the recordings\' similarity maps are added at the end (Section 13).'], 'plain')
R.box('The model in one paragraph', [
    'Each KC adds up its claws\' glomerular inputs (a weighted average of the glomeruli\'s odor responses, with weights from the '
    'connectome). It responds by how much that sum exceeds a threshold. APL is modelled as a single unit with one activity level per '
    'odor. Every KC excites APL, so this activity is the mean response of all KCs to that odor. APL then inhibits every KC equally: it '
    'subtracts its activity times a gain $a$ from each KC\'s input. The gain is the same for every odor, but the inhibition is not, '
    'because APL\'s activity is higher for odors that drive more KCs. Because APL\'s activity depends on the KC responses and the KC '
    'responses depend on APL\'s activity, both are solved together for each odor ("self-consistently"). "Blocked" (TNT) means no APL '
    'at all. Later versions replace the uniform APL with connectome measurements of where APL\'s synapses are.'], 'plain')

R.symbols([
    ('k', 'Index of a Kenyon cell (KC) in one mushroom body', '$k = 1 \\dots N$'),
    ('N', 'Number of KCs in the mushroom body (KCs with at least one claw)', f'hemibrain R {SIDES["hemibrain R"]:,}; male CNS L {SIDES["malecns L"]:,}; male CNS R {SIDES["malecns R"]:,}'),
    ('g', 'Index of an antennal-lobe glomerulus (uniglomerular PN types only)', 'names such as DM2, VA1v'),
    ('o', 'Index of an odor', f'{RESP["ORN"].shape[1]} odors (ORN table) or {RESP["GH146"].shape[1]} (GH146)'),
    ('c', 'Index of a claw (one PN\'s synapses on one KC within one 8 µm cluster, Report 02)', f'median {NCLAW["hemibrain"]:.0f} claws per KC (hemibrain), {NCLAW["malecns"]:.0f} (male CNS)'),
    ('n_c', 'Number of PN→KC synapses on claw $c$', 'synapse count'),
    ('g(c)', 'Glomerulus of claw $c$: the glomerulus of its PN (each claw has one PN)', ''),
    ('W_{kg}', 'Input weight: share of KC $k$\'s claw PN synapses that sit on claws of glomerulus $g$', f'0 to 1; a row sums to the share from uniglomerular PNs ({WSUM_LO:.2f}–{WSUM_HI:.2f} on average)'),
    ('r_{go}', 'Response of glomerulus $g$ to odor $o$; negative values set to 0 (except in the "neg" inputs of Section 4)', f'ORN table: fly-averaged response as provided (units not stated), {F(RESP["ORN"].values.min())} to {F(RESP["ORN"].values.max())}; GH146: mean peak ΔF/F'),
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
    ('s', 'Common scale of the KC and bouton gains in versions M0, M and MC', 'set by calibration'),
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
    ax.text(5.9, 7.15, r'APL→PN boutons (versions P0, P, L, M0, M, MC): input of glomerulus $g$ divided by $1+b\,p_g A_o$', fontsize=8.2, ha='center', color=L.INTACT)
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
R.p('A **claw** is a group of one KC\'s postsynaptic sites from one antennal-lobe PN in the calyx (Report 02). The KC\'s calyx synapse '
    'sites are first clustered along its skeleton: sites within 8 µm of each other, measured along the KC\'s branches, belong to the same '
    'cluster (single linkage). Within a cluster, every PN that makes at least 2 synapses onto the KC gets its own claw. So each claw has '
    'exactly one PN and one glomerulus, $g(c)$, and PNs that make a single synapse in a cluster are not counted. Only uniglomerular PN '
    'types give input. Claws of multiglomerular, bi-glomerular or untyped PNs stay in the denominator but give no input. The input '
    'weight is')
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
R.sub('Why $W_{kg}$ is divided by all of the KC\'s claw PN synapses')
_tot = {}
for _ds in abm.DS:
    _c = abm.CLAWS[_ds].groupby('kc_id').n_PN_synapses.sum()
    _cls = pd.concat([pd.Series(MB[k]['cls'], index=MB[k]['W'].index) for k in MBS if MB[k]['ds'] == _ds])
    _tot[_ds] = pd.DataFrame({'n': _c.reindex(_cls.index), 'cls': _cls})
_t = _tot['hemibrain']; _byc = _t.groupby('cls').n.mean(); _nc = {ds: abm.CLAWS[ds].groupby('kc_id').size() for ds in abm.DS}
R.p('Dividing by the KC\'s total makes its drive a **weighted average** of its glomeruli\'s responses, so every KC\'s drive is on the '
    'same scale whatever its size. Without it, the drive would be $\\sum_g n_{kg}\\,r_{go}$ ($n_{kg}$ = the KC\'s claw PN synapses from '
    'glomerulus $g$), which grows with the number of synapses. KCs differ a lot in that number: hemibrain KCs have a median of '
    f'{_t.n.median():.0f} claw PN synapses (5–95%: {_t.n.quantile(.05):.0f}–{_t.n.quantile(.95):.0f}), and γ KCs have about '
    f'{_byc["γ"] / _byc[["αβ", "α′β′"]].mean():.1f} times as many as the others ({_byc["γ"]:.0f} on average, against {_byc["αβ"]:.0f} for '
    f'αβ and {_byc["α′β′"]:.0f} for α′β′; male CNS: {_tot["malecns"].groupby("cls").n.mean()["γ"]:.0f}, '
    f'{_tot["malecns"].groupby("cls").n.mean()["αβ"]:.0f} and {_tot["malecns"].groupby("cls").n.mean()["α′β′"]:.0f}). With raw counts and one '
    'threshold, γ KCs would take most of the responses (Section 12 shows this). The averaging assumes that a KC with more input synapses '
    'also needs more input to reach threshold, which is roughly what a larger dendrite (lower input resistance) does.')
R.p('The denominator also includes claws of PNs that are not in the response table or are multiglomerular. Those claws count as input '
    '0, so a KC whose claws come mostly from unmeasured glomeruli gets a smaller drive, instead of having its few measured claws scaled up.')
R.p('This is a modelling choice, and the data do not settle it. Section 12 runs two alternatives: raw synapse counts '
    '($W_{kg} = n_{kg}$, one threshold), and equal weight per claw ($W_{kg}$ = share of the KC\'s claws that come from $g$).')


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
R.p('Two tables give $r_{go}$, the response of glomerulus $g$ to odor $o$. In the main runs these numbers go straight into Equation 2, and '
    'negative responses (inhibition below baseline) are set to 0. Two variations follow below: a PN-level transformation of the ORN '
    'responses, and keeping the negative values.')
R.ul([f'**ORN table** (`orn_inputs_2026-11-03/ORN_TNT16_flymean.csv`, built from Yang\'s `responses_wide.csv`; Report 03): '
      f'{RESP["ORN"].shape[0]} glomeruli × {RESP["ORN"].shape[1]} odors, fly-averaged olfactory receptor neuron (ORN) responses. '
      '12 odors come from the megamat flies and 4 (paa, -aPine, euc, B-myr) from the validation2 flies. DL4 and VM4 were imaged only in '
      'the validation2 flies, so they have values for 4 of the 16 odors and are left out (their claws give input 0). The file does not '
      f'state units (range {F(RESP["ORN"].values.min())} to {F(RESP["ORN"].values.max())}). The values are about 0.7–0.8 times those of '
      'the old table, with the same patterns (Report 03); the shared threshold absorbs a common scale. Glomeruli in this table carry '
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
      f'the table\'s maximum (dark blue), with a separate scale for each panel. **a**, ORN table, {RESP["ORN"].shape[0]} glomeruli × '
      f'{RESP["ORN"].shape[1]} odors. **b**, GH146 table, {RESP["GH146"].shape[0]} glomeruli × {RESP["GH146"].shape[1]} odors. Glomeruli '
      'that are missing here give 0 input to the claws that belong to them.',
      'The glomerular responses $r_{go}$ that drive the model.')

# ---------------------------------------------------------------- PN-level transformation
R.sub('A PN-level transformation of the ORN responses')
OF = SUM['olsen_fit']
R.p('KCs receive input from PNs, not ORNs. Olsen, Bhandawat and Wilson (2010) showed that a PN\'s response is a saturating function of its '
    'own ORNs\' input, divided down by the summed ORN input to all glomeruli (lateral inhibition in the antennal lobe). Kennedy (2019, '
    '"Learning with naturalistic odor representations in a dynamic model of the Drosophila olfactory system") used the same function to '
    'turn ORN responses into PN input to KCs. Without its maximum rate $R_{\\max}$, which only scales all values and is absorbed by the '
    'threshold, it is')
R.eq(r'r^{\mathrm{PN}}_{go} \;=\; \frac{r_{go}^{1.5}}{r_{go}^{1.5} + \sigma^{1.5} + (m\, s_o)^{1.5}}, \qquad s_o = \frac{1}{G}\sum_{g} r_{go}',
     note='$r_{go}$ = ORN response (negative values set to 0); $s_o$ = mean ORN response over the $G$ measured glomeruli for odor $o$; '
          '$\\sigma$ = half-saturation constant; $m$ = strength of the lateral inhibition')
_pos = np.clip(RV['ORN'].values, 0, None); _pos = _pos[_pos > 0]
_dr = OF['r with GH146 (PN)'] - OF['r with GH146 (ORN, no transform)']
R.p(f'Olsen\'s $\\sigma$ and $m$ were fitted to spike rates and do not carry over to these calcium responses. So both were fitted here: '
    f'$\\sigma$ from the 2nd percentile of the positive ORN responses up to 10 times the largest one, and $m$ from 0 to 16, keeping the pair '
    f'whose PN responses best match GH146 (Pearson r over the {OF["glomeruli"]} glomeruli × {OF["odors"]} odors that both tables share). '
    f'The best fit is $\\sigma$ = {OF["sigma"]:.2f} and $m$ = {OF["m"]:g}.'
    + (f' With $m$ = 0 there is no lateral inhibition, and with $\\sigma$ '
       + (f'larger than every positive ORN response (the largest is {_pos.max():.2f})' if (_pos < OF['sigma']).all() else f'above {L.pct((_pos < OF["sigma"]).mean())} of the responses')
       + ' the function is close to a power law, $r^{1.5}$, with only the largest responses starting to saturate.' if OF['m'] == 0 else '')
    + f' Its match with GH146 is **r = {OF["r with GH146 (PN)"]:.3f}**, against **r = {OF["r with GH146 (ORN, no transform)"]:.3f}** for the '
    'untransformed ORN responses' + (': practically the same. ' if abs(_dr) < 0.01 else (': better. ' if _dr > 0 else ': worse. '))
    + 'Strong saturation (small $\\sigma$) matches GH146 clearly worse, and strong lateral inhibition (large $m$) slightly worse (Figure 5c). '
    + ('So on these data a PN-level transformation of this form barely brings the ORN input closer to the measured PN responses. ' if _dr < 0.02 else '')
    + 'The model '
    'is run with the best-fitting PN responses as the input "ORN-PN" (Section 11).')


def fig_olsen():
    orn, gh = RV['ORN'], RV['GH146']; gl = [g for g in orn.index if g in gh.index]; od = list(gh.columns)
    G = np.clip(gh.loc[gl, od].values, 0, None).ravel(); X0 = np.clip(orn.loc[gl, od].values, 0, None).ravel()
    PN = RV['ORN-PN'].loc[gl, od].values.ravel()
    sgs = OF['sigma grid']; ms_ = OF['m grid']
    Rg = np.array([[np.corrcoef(abm.olsen(orn, sg, m).loc[gl, od].values.ravel(), G)[0, 1] for m in ms_] for sg in sgs])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for a, X, t, lab in [(axs[0], X0, f'a  ORN, no transform: r = {np.corrcoef(X0, G)[0, 1]:.3f}', 'ORN response $r_{go}$ (≥ 0)'),
                         (axs[1], PN, f'b  ORN → PN (Equation 3): r = {np.corrcoef(PN, G)[0, 1]:.3f}', 'transformed response $r^{PN}_{go}$')]:
        a.scatter(X, G, s=9, color=L.INTACT, alpha=0.6, lw=0)
        a.set_xlabel(lab); a.set_ylabel('GH146 PN peak ΔF/F (≥ 0)'); a.set_title(t, fontsize=9)
    a = axs[2]; im = a.imshow(Rg, aspect='auto', cmap=L.SEQ, origin='lower', interpolation='nearest')
    a.set_xticks(range(len(ms_))); a.set_xticklabels([f'{m:g}' for m in ms_], fontsize=7); a.set_yticks(range(len(sgs)))
    a.set_yticklabels([f'{x:.2f}' for x in sgs], fontsize=7); a.set_xlabel('lateral inhibition $m$'); a.set_ylabel('half-saturation $\\sigma$')
    i, j = np.unravel_index(np.nanargmax(Rg), Rg.shape); a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=L.INK, lw=1.6))
    a.set_title('c  Fit: Pearson r with GH146 for each $\\sigma$, $m$', fontsize=9); L.cbar(fig, im, a, 'Pearson r with GH146')
    fig.tight_layout(); return fig


R.fig(fig_olsen(), f'Each point in **a** and **b** is one glomerulus–odor value ({OF["glomeruli"]} glomeruli × {OF["odors"]} odors shared by the '
      'ORN and GH146 tables). **a**, GH146 against the ORN responses as they are. **b**, GH146 against the ORN responses after the '
      'PN transformation at the best $\\sigma$ and $m$. **c**, The Pearson r with GH146 over the grid of $\\sigma$ (rows; the 2nd to 90th '
      'percentiles of the positive ORN responses, then 1 to 10 times the largest response) and $m$ (columns); the black square is the best pair.',
      'ORN → PN transformation compared with the GH146 responses.')

# ---------------------------------------------------------------- negative responses
R.sub('Negative responses')
neg = {rn: RV[rn] for rn in ['ORN', 'GH146']}
_ng = {rn: (X < 0).values for rn, X in neg.items()}
_wn = {rn: neg[rn].where(neg[rn] < 0).stack().sort_values().head(3) for rn in neg}
R.p(f'In the main runs, responses below baseline are set to 0. They are few and small: {_ng["ORN"].sum()} of {_ng["ORN"].size} ORN values '
    f'(lowest {F(neg["ORN"].values.min())}, against a maximum of {F(neg["ORN"].values.max())}) and {_ng["GH146"].sum()} of {_ng["GH146"].size} '
    f'GH146 values (lowest {F(neg["GH146"].values.min())}, maximum {F(neg["GH146"].values.max())}). The most negative are '
    + ', '.join(f'{g}–{o} ({F(v)})' for (g, o), v in _wn['ORN'].items()) + ' (ORN) and '
    + ', '.join(f'{g}–{o} ({F(v)})' for (g, o), v in _wn['GH146'].items()) + ' (GH146).')
R.p('The inputs "ORN neg" and "GH146 neg" keep them. A negative response then lowers the drive of every KC with claws from that '
    'glomerulus (Equation 2 is unchanged), and θ and the gains are recalibrated as before. One side effect concerns the versions that '
    'divide bouton input (P0, P, L, M0, M, MC): dividing a negative input by $1 + b\\,p_g A_o$ makes it less negative, so in those versions '
    'APL also weakens the inhibition that a glomerulus passes on. The results are in Section 11.')

# ---------------------------------------------------------------- 5 drive and threshold
R.section('Step 4: KC drive and the threshold with APL blocked')
R.p('With APL blocked there is no inhibition, so a KC responds by how far its drive exceeds the threshold $\\theta$:')
R.eq(r'y^{\mathrm{blocked}}_{ko} \;=\; \left[\, x_{ko} - \theta \,\right]_+ \qquad \text{where } [z]_+ = \max(z, 0)')
R.p('$\\theta$ is one number per mushroom body and input table. It is found by bisection so that, averaged over odors, 20% of KCs '
    'respond ($y_{ko} > 0$). The bisection keeps an interval $[\\theta_{lo}, \\theta_{hi}]$, which starts at 0 and the largest drive. At each '
    'step it tests the midpoint: if more than 20% respond there, the threshold must be higher, so the midpoint becomes $\\theta_{lo}$; '
    'otherwise it becomes $\\theta_{hi}$. 60 halvings make the interval negligibly small. Individual odors then respond at different levels '
    f'(Figure {R.n_fig + 1}). Strong odors drive many KCs past the shared threshold, and weak ones drive few.')
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
EQ_Y = R.eq(r'y_{ko} \;=\; \left[\, x_{ko} - \theta - a\, u_k\, A_o \,\right]_+')
R.p('and APL\'s activity is the mean response of the KCs, each weighted by how strongly it drives APL:')
EQ_A = R.eq(r'A_o \;=\; \frac{1}{N}\sum_{k=1}^{N} v_k\, y_{ko}')
R.p('In the uniform version (U) every $u_k = v_k = 1$. Every KC then gets the same inhibition $a A_o$ for a given odor, and APL\'s '
    'activity is simply the mean KC response. The gain $a$ is one number per mushroom body and version. It is the same for every odor '
    'and every KC. The inhibition still differs between odors, because $A_o$ does.')
R.box('Why "self-consistently"', [
    f'$A_o$ appears on both sides. The KC responses (Equation {EQ_Y}) depend on APL\'s activity, and APL\'s activity (Equation {EQ_A}) depends on '
    'the KC responses. The model needs the one value of $A_o$ that reproduces itself: the value that, when used to compute the KC '
    'responses, gives back the same $A_o$ as their mean. Mathematically this is a fixed point. Physically, it is the steady state in which '
    'APL and the KCs have settled.'], 'plain')
R.p('To find it, define for each odor the function')
R.eq(r'F_o(A) \;=\; \frac{1}{N}\sum_{k} v_k \left[\, x_{ko} - \theta - a\, u_k\, A \,\right]_+',
     note='the APL activity the KCs would produce if APL\'s activity were $A$')
R.p('and solve $A = F_o(A)$. $F_o$ starts at $F_o(0)$, the mean (weighted) response with no inhibition, and can only fall as $A$ grows, '
    'because more inhibition never raises a KC\'s response. The line $A$ rises from 0. The two cross exactly once, which makes the '
    f'solution unique (Figure {R.n_fig + 1}). The code finds it by bisection on $[0, F_o(0)]$: if $A > F_o(A)$ at the midpoint, the crossing lies '
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
VERS = ['U', 'C', 'P0', 'P', 'L', 'M0', 'M', 'MC']
R.table(['input', 'mushroom body', 'θ'] + [f'{v}' for v in VERS],
        [[rn, MBNAME[k], F(GAIN[(rn, k)]['theta'], 3)] + [f'{GAIN[(rn, k)][v]:.3g}' for v in VERS] for rn in RV for k in MBS],
        title='Thresholds and calibrated gains (20% → 10%), for every input.',
        caption='U and C give $a$ (subtractive); P0, P and L give $b$ (divisive, on PN boutons or claws); M0, M and MC give the common scale $s$ '
                '(Section 9). The gains are not comparable across columns, because each version uses a different form and normalization. '
                'Cs and Cc are calibrated in the same way as C and are not shown. The inputs are described in Section 4.',
        num_cols=list(range(2, 3 + len(VERS))))

# ---------------------------------------------------------------- 8 readouts
NAN_C = '#d6d6d4'
DIVc = L.DIV.copy(); DIVc.set_bad(NAN_C); SEQc = L.SEQ.copy(); SEQc.set_bad(NAN_C)
NKC = {(None, r.responses, r.mushroom_body, r.odor): r for r in OD[OD.coding_levels == MAIN].itertuples()}
NKC.update({(r.weights, r.responses, r.mushroom_body, r.odor): r for r in OW2o[OW2o.coding_levels == MAIN].itertuples()})
MIN_KC = 10          # cells resting on fewer responding KCs than this (in some mushroom body) are hatched


def nkc(rn, k, o, ver, w=None):
    r = NKC[(w, rn, k, o)]
    return round(getattr(r, 'frac_KCs_responding_blocked' if ver == 'blocked' else f'frac_KCs_responding_intact_{ver}') * SIDES[k])


def ptab(rn, src=None, w=None):
    P = (PR if src is None else src); P = P[(P.coding_levels == MAIN) & (P.responses == rn)]
    return P if w is None else P[P.weights == w]


def dS_mb(rn, ver, src=None, w=None):
    """Delta S per pair (rows, (odor_a, odor_b)) and mushroom body (columns); NaN where undefined."""
    X = ptab(rn, src, w).set_index(['odor_a', 'odor_b', 'mushroom_body'])
    return (X['cos_blocked'] - X[f'cos_intact_{ver}']).unstack('mushroom_body').reindex(columns=MBS)


def S_mb(rn, col, src=None, w=None):
    X = ptab(rn, src, w).set_index(['odor_a', 'odor_b', 'mushroom_body'])
    return X[col].unstack('mushroom_body').reindex(columns=MBS)


def tomat(ser, odors):
    M = pd.DataFrame(np.nan, index=odors, columns=odors)
    for (a, b), x in ser.items(): M.loc[a, b] = M.loc[b, a] = x
    return M


def weak(rn, vers, odors, w=None):
    """pairs to hatch: undefined in some mushroom body, or one odor drives < MIN_KC KCs there (blocked or intact)."""
    H = pd.DataFrame(False, index=odors, columns=odors)
    for o in odors:
        if any(nkc(rn, k, o, v, w) < MIN_KC for k in MBS for v in ['blocked'] + vers): H.loc[o] = True; H[o] = True
    return H


def weak_txt(rn, vers=('U',), w=None):
    """which odors are hatched, with how many KCs they drive (blocked) in each mushroom body"""
    out = []
    for o in RV[rn].columns:
        if any(nkc(rn, k, o, v, w) < MIN_KC for k in MBS for v in ['blocked', *vers]):
            out.append(f'{o} ({", ".join(str(nkc(rn, k, o, "blocked", w)) for k in MBS)} KCs with APL blocked in ' + ', '.join(MBNAME[k] for k in MBS) + ')')
    return 'Hatched: ' + '; '.join(out) + '.' if out else 'No cell is hatched.'


def extremes(ser, hatch, n=5):
    s = ser[[not hatch.loc[a, b] for a, b in ser.index]].dropna().sort_values()
    return list(s.index[-n:][::-1]), list(s.index[:n])


def heat(fig, a, M, cmap, vm, title, lab, hatch=None, top=(), bot=(), fs=6.6, cb=True):
    od = list(M.index); Mv = M.values.astype(float).copy(); np.fill_diagonal(Mv, np.nan)
    im = a.imshow(np.ma.masked_invalid(Mv), cmap=cmap, vmin=vm[0], vmax=vm[1], interpolation='nearest')
    a.set_xticks(range(len(od))); a.set_xticklabels(od, rotation=90, fontsize=fs); a.set_yticks(range(len(od))); a.set_yticklabels(od, fontsize=fs)
    a.tick_params(length=0); [s_.set_visible(False) for s_ in a.spines.values()]; a.set_title(title, fontsize=8.6)
    if hatch is not None:
        for i in range(len(od)):
            for j in range(len(od)):
                if i != j and hatch.iloc[i, j]: a.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='////', ec=L.MUTED, lw=0))
    for prs, ls in [(top, '-'), (bot, (0, (2, 1.2)))]:
        for p1, p2 in prs:
            for i, j in [(od.index(p1), od.index(p2)), (od.index(p2), od.index(p1))]:
                a.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False, ec=L.INK, lw=1.3, ls=ls))
    if cb: L.cbar(fig, im, a, lab)
    return im


def pairs_txt(prs, ser):
    return ', '.join(f'{a}–{b} ({F(ser[(a, b)], 3, True)})' for a, b in prs)


R.section('Step 7: what is read out')
R.p('Three quantities are read out for every mushroom body, version and input table:')
R.ul(['**Fraction responding per odor**, $\\phi_o$, blocked and intact.',
      '**Odor-pair similarity** $S_{ab}$: the cosine similarity of the two odors\' KC population response vectors $\\mathbf{y}_a = (y_{1a}, \\dots, y_{Na})$ '
      f'and $\\mathbf{{y}}_b$ (Equation {R.n_eq + 1}). It is the same measure as Yang\'s RDMs (which hold $1 - S$). Because responses are ≥ 0, $S$ runs from 0 '
      '(no KC responds to both) to 1 (the same pattern up to a scale factor). It is undefined (0/0) when an odor drives no KC.',
      '**The effect of blocking APL** on a pair, $\\Delta S_{ab} = S^{\\mathrm{blocked}}_{ab} - S^{\\mathrm{intact}}_{ab}$ '
      f'(Equation {R.n_eq + 2}). It is computed per mushroom body and then averaged over the three.'])
EQ_S = R.eq(r'S_{ab} \;=\; \cos(\mathbf{y}_a, \mathbf{y}_b) \;=\; \frac{\sum_k y_{ka}\, y_{kb}}{\sqrt{\sum_k y_{ka}^2}\;\sqrt{\sum_k y_{kb}^2}}')
EQ_D = R.eq(r'\Delta S_{ab} \;=\; S^{\mathrm{blocked}}_{ab} - S^{\mathrm{intact}}_{ab}')
_ms = {k: (nkc('ORN', k, 'ms', 'blocked'), nkc('ORN', k, 'ms', 'U')) for k in MBS}
_msg = {k: nkc('GH146', k, 'ms', 'blocked') for k in MBS}
R.p('**Every odor is kept, including ms.** The earlier version of this report left out odors that drove fewer than 0.5% of KCs. Now a '
    'similarity is only missing where it cannot be computed. This happens for one odor, ms with ORN input: with APL blocked it drives '
    + ', '.join(f'{_ms[k][0]} KCs in {MBNAME[k]}' for k in MBS) + ' (intact: ' + ', '.join(f'{_ms[k][1]}' for k in MBS) + '). Its ORN responses '
    f'are weak (largest {F(RESP["ORN"]["ms"].max())}, against {F(RESP["ORN"].max().median())} for the median odor), and with one threshold '
    'for all odors almost no KC reaches it. With GH146 input ms drives ' + ', '.join(f'{_msg[k]}' for k in MBS) + ' KCs, so its pairs are '
    'defined there. Averages over the three mushroom bodies use those where the value is defined. In the heat maps, cells that rest on '
    f'fewer than three mushroom bodies, or on an odor that drives fewer than {MIN_KC} KCs in some mushroom body, are **hatched**; grey '
    'means undefined in all three.')
R.p('**Correlations.** Every "r" in this report is a **Pearson** correlation; rank (Spearman) correlations are written ρ.')


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
    a.set_xlabel('KCs responding with APL blocked, %'); a.set_ylabel('KCs gained by blocking APL\n(blocked − intact), percentage points')
    a.set_title(f'b  Strong odors gain most (Pearson r = {np.corrcoef(fb, gain)[0, 1]:.2f})')
    fig.tight_layout(); return fig


R.fig(fig_frac(), '**a**, Percentage of KCs responding to each odor with APL blocked (orange) and intact (blue, uniform APL), averaged over '
      'the three mushroom bodies (ORN input, 20% → 10%). ms is at 0. **b**, One point per odor. The horizontal axis is the blocked percentage (odor '
      'strength). The vertical axis is how many percentage points blocking APL adds. Strong odors recruit more APL, so removing APL adds the '
      'most KCs for them.', 'Per-odor fractions responding, blocked and intact.')


def sim_set(rn, ver, src=None, w=None):
    """mean-over-MB matrices for one input and version: S blocked, S intact, dS, hatch, top/bottom pairs."""
    od = list(RV[rn].columns)
    d = dS_mb(rn, ver, src, w).mean(1)
    Sb = S_mb(rn, 'cos_blocked', src, w).mean(1); Si = S_mb(rn, f'cos_intact_{ver}', src, w).mean(1)
    nm = dS_mb(rn, ver, src, w).notna().sum(1)
    H = weak(rn, [ver], od, w)
    for (a_, b_), n_ in nm.items():
        if n_ < 3: H.loc[a_, b_] = H.loc[b_, a_] = True
    top, bot = extremes(d, H)
    return dict(od=od, d=d, Sb=tomat(Sb, od), Si=tomat(Si, od), D=tomat(d, od), H=H, top=top, bot=bot, nm=nm)


def dstats(d):
    return d.mean(), d.std(), d.min(), d.max(), int(d.notna().sum())


def fig_sim(rn, ver):
    z = sim_set(rn, ver); fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
    vmax = float(np.nanmax([z['Sb'].values, z['Si'].values]))
    heat(fig, axs[0], z['Sb'], SEQc, (0, vmax), 'a  $S_{ab}$, APL blocked', 'cosine similarity $S_{ab}$', z['H'])
    heat(fig, axs[1], z['Si'], SEQc, (0, vmax), f'b  $S_{{ab}}$, APL intact ({ver})', 'cosine similarity $S_{ab}$', z['H'])
    heat(fig, axs[2], z['D'], DIVc, (-0.2, 0.2), 'c  $\\Delta S_{ab}$ = blocked − intact', '$\\Delta S_{ab}$ (difference in cosine)', z['H'], z['top'], z['bot'])
    fig.tight_layout(); return fig, z


fg, zU = fig_sim('ORN', 'U'); mU = dstats(zU['d'])
R.fig(fg, f'Odor-by-odor matrices, mean of the three mushroom bodies, ORN input, uniform APL (U). **a**, **b**, Cosine similarity of the KC '
      f'population responses with APL blocked and intact, on one shared scale (white 0, dark blue the highest value). **c**, Their difference, '
      f'$\\Delta S_{{ab}}$, on a diverging scale: orange = more similar when APL is blocked, blue = less similar. **The average change in '
      f'similarity with minus without inhibition is {F(mU[0], 3, True)}** (sd {F(mU[1], 3)}, range {F(mU[2], 2, True)} to {F(mU[3], 2, True)}, '
      f'{mU[4]} pairs). Solid outlines: the 5 pairs that become most similar when APL is blocked ({pairs_txt(zU["top"], zU["d"])}); dashed '
      f'outlines: the 5 that change least or become less similar ({pairs_txt(zU["bot"], zU["d"])}). ' + weak_txt('ORN')
      + ' Grey: diagonal or undefined (Section 8).',
      'Odor-pair similarity with APL blocked and intact (uniform APL), and the change.')
FIG_SIMU = R.n_fig

# ---------------------------------------------------------------- 9 versions
R.section('Step 8: versions that use APL\'s connectome wiring')
R.p('Each version replaces part of the uniform APL with a connectome measurement. All of them keep the same blocked responses and the same '
    'θ, and all are calibrated to the same 10%. The versions therefore differ only in **where** the inhibition acts and **how it is '
    f'weighted**. Table {R.n_tab + 1} summarizes them.')
R.table(['version', 'KC response $y_{ko}$', 'APL activity $A_o$', 'weights from the connectome'],
        [['U, uniform', '$[x_{ko}-\\theta-a A_o]_+$', '$\\frac1N\\sum_k y_{ko}$', 'none ($u = v = 1$)'],
         ['C, per-KC counts', '$[x_{ko}-\\theta-a u_k A_o]_+$', '$\\frac1N\\sum_k v_k y_{ko}$', '$u_k$ = APL→KC, $v_k$ = KC→APL synapses of KC $k$ in the calyx, each ÷ its mean'],
         ['Cs, shuffled control', 'as C', 'as C', '$(u_k, v_k)$ pairs shuffled among KCs of the same class (αβ, α′β′, γ)'],
         ['Cc, class means', 'as C', 'as C', '$u_k, v_k$ replaced by the mean of the KC\'s class'],
         ['P0, boutons, all alike', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+b A_o}-\\theta]_+$', '$\\frac1N\\sum_k y_{ko}$', 'none'],
         ['P, boutons per glomerulus', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+b\\,p_g A_o}-\\theta]_+$', 'as P0', '$p_g$ = APL→PN per PN→KC synapse of glomerulus $g$'],
         ['L, claws per glomerulus', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+b\\,q_g A_o}-\\theta]_+$', 'as P0', '$q_g$ = on-claw APL→KC per claw PN synapse of $g$'],
         ['M0, U + boutons at real share', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+s f b_{P0} A_o}-\\theta - s(1-f) a_U A_o]_+$', 'as U', '$f$ = boutons\' share of APL\'s calyx output'],
         ['M, U + boutons per glomerulus', 'as M0 with $p_g$ in the bouton term', 'as U', '$f$ and $p_g$'],
         ['**MC, C + boutons per glomerulus (new)**', '$[\\sum_g W_{kg}\\frac{r_{go}}{1+s f b_{P}\\,p_g A_o}-\\theta - s(1-f) a_C\\,u_k A_o]_+$',
          'as C', '$u_k$, $v_k$, $p_g$ and $f$']],
        title='The APL versions of the prediction run.',
        caption='$a_U$, $a_C$, $b_{P0}$ and $b_P$ are the calibrated gains of U, C, P0 and P. In M0, M and MC only the common scale $s$ is '
                'calibrated. Report 07 adds two feed-forward versions (F0, F), where APL activity comes from the PN input instead of the KCs.')
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

R.sub('P and L: APL synapses per glomerulus, and why they are counted per synapse')
R.p('In P and L, inhibition **divides** a glomerulus\'s input instead of subtracting from the KC, because APL→PN synapses act presynaptically '
    'on the PN boutons and scale down what the boutons pass to all their KCs. The input that glomerulus $g$ sends to every KC is divided by '
    '$1 + b\\,\\tilde p_g\\,A_o$. So $\\tilde p_g$ has to answer one question: _how strongly is each unit of $g$\'s input inhibited?_')
GWr = pd.read_csv(OWV + 'APL_wiring_by_glomerulus.csv'); hgw = GWr[GWr.dataset == 'hemibrain'].set_index('glomerulus')
gl_ex = [g for g in ['DC1', 'DM6', 'DM2', 'DA1', 'DL5', 'VM2', 'VA1v', 'DA2'] if g in hgw.index]
R.p('Raw APL synapse counts answer a different question, because big glomeruli have more of everything. In the hemibrain, DC1 has '
    f'{int(hgw.loc["DC1", "PN_to_KC"]):,} PN→KC synapses and VA1v {int(hgw.loc["VA1v", "PN_to_KC"]):,}. If both had the same number of APL→PN '
    f'synapses per PN→KC synapse, DC1 would still have about {hgw.loc["DC1", "PN_to_KC"] / hgw.loc["VA1v", "PN_to_KC"]:.0f} times as many in total. '
    'The size of a glomerulus\'s input already enters the model through $W_{kg}$. Using raw APL counts would count size a second time and '
    'make big glomeruli look strongly inhibited simply because they are big. Dividing by the excitatory synapses removes that:')
R.ul(['$p_g$ (version P) is the number of APL→PN synapses on glomerulus $g$\'s boutons per PN→KC synapse of those boutons: the number of '
      'APL synapses that each PN→KC synapse of $g$ "sees".',
      '$q_g$ (version L) is the same for the APL→KC synapses that sit on the claws of glomerulus $g$, divided by the PN synapses on those '
      'claws. A glomerulus with many large claws is then not counted as more inhibited only because it has more claw membrane.'])
TAB_PQ = R.n_tab + 1
R.table(['glomerulus', 'PNs', 'PN→KC synapses', 'APL→PN synapses (boutons)', '$p_g$ = APL→PN ÷ PN→KC', 'PN synapses on its claws',
         'APL→KC synapses on its claws', '$q_g$ = APL→KC ÷ claw PN synapses'],
        [[g, int(hgw.loc[g, 'n_PNs']), f"{int(hgw.loc[g, 'PN_to_KC']):,}", int(hgw.loc[g, 'APL_to_PN']), F(hgw.loc[g, 'APL_to_PN_per_PN_to_KC'], 3),
          f"{int(hgw.loc[g, 'PN_synapses_on_claws']):,}", int(hgw.loc[g, 'APL_to_KC_on_claws']), F(hgw.loc[g, 'APL_to_KC_per_claw_PN_synapse'], 3)]
         for g in gl_ex],
        title='Raw counts and the per-synapse weights, hemibrain (corrected calyx, new claws).',
        caption='The raw counts (columns 4 and 7) mostly follow the glomerulus\'s size (columns 3 and 6). The ratios (columns 5 and 8) are '
                'what P and L use, after scaling to a claw-input-weighted mean of 1 (Equation ' + str(R.n_eq + 1) + ').',
        num_cols=[1, 2, 3, 4, 5, 6, 7])
R.table(['', 'glomerulus A (big)', 'glomerulus B (small)', 'what it means'],
        [['PN→KC synapses', '1,000', '200', 'A sends 5 times more input to KCs'],
         ['APL→PN synapses on the boutons', '100', '40', 'A receives 2.5 times more APL synapses in total'],
         ['$p_g$ (per PN→KC synapse)', '0.10', '0.20', 'but each unit of B\'s output is inhibited twice as strongly'],
         ['a KC with half its input from each', '$\\frac{0.5\\,r_A}{1 + 0.10\\,b A_o}$', '$\\frac{0.5\\,r_B}{1 + 0.20\\,b A_o}$',
          'B\'s half of the input is divided more']],
        title='Why per synapse, in a mock example.',
        caption='Two made-up glomeruli. Raw counts would rank A as the more inhibited glomerulus; per synapse, B is.')
R.p('Within each mushroom body, the weights are scaled so that their claw-input-weighted mean is 1:')
EQ_P = R.eq(r"\tilde p_g \;=\; \frac{p_g}{\left(\sum_{g'} w_{g'}\, p_{g'}\right) \big/ \sum_{g'} w_{g'}}, \qquad w_g = \sum_k W_{kg}",
            note='sums over glomeruli $g\'$ that have a value; glomeruli without a value get $\\tilde p_g = 1$')
R.p('With this scaling, P and P0 put the same total inhibition on the average claw input, and only its distribution across glomeruli '
    'differs. Report 04 shows $p_g$ and $q_g$ across glomeruli and connectomes.')


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
      f'(purple), each divided by its claw-input-weighted mean (Equation {EQ_P}), so the dotted line at 1 is the average. A glomerulus at 2 gets '
      'twice the average inhibition per unit of input in that version.', 'Per-glomerulus weights after scaling.')
R.sub('M0, M and MC: inhibition on both KCs and boutons')
FS = SUM['bouton_share_of_APL_calyx_output_synapses']
R.p('APL makes synapses onto both KCs and PN boutons in the calyx. In M0, M and MC, only a share $f$ of the inhibition goes to boutons, the '
    f'boutons\' share of APL\'s calyx output synapses: $f$ = APL→PN ÷ (APL→PN + APL→KC), {FS["hemibrain"]} in hemibrain and {FS["malecns"]} '
    'in male CNS. The rest acts on the KCs.')
R.ul(['**M0**: uniform on the KCs (as U) and on all boutons alike (as P0). KC gain $(1-f)\\,a_U$, bouton gain $f\\,b_{P0}$.',
      '**M**: as M0, but the bouton term uses the per-glomerulus weights $\\tilde p_g$ (as P).',
      '**MC** (new, item 5 of the revision): inhibition on both, each with its connectome weights. The KC term is that of C (per-KC $u_k$, '
      'and APL\'s activity weighted by $v_k$), with gain $(1-f)\\,a_C$; the bouton term is that of P, with gain $f\\,b_P$ and $\\tilde p_g$.'])
R.p('In all three, both gains are scaled by one common factor $s$, the only number calibrated, so that 10% of KCs respond.')

# ---------------------------------------------------------------- 10 results
VDESC = {'U': 'uniform APL', 'C': 'per-KC APL synapse counts', 'Cs': 'per-KC counts shuffled within KC class',
         'Cc': 'KC-class means of the counts', 'P0': 'all inhibition on PN boutons, every glomerulus alike',
         'P': 'all inhibition on PN boutons, per glomerulus ($\\tilde p_g$)', 'L': 'all inhibition on claws, per glomerulus ($\\tilde q_g$)',
         'M0': 'U plus boutons at their real share', 'M': 'U plus boutons per glomerulus at their real share',
         'MC': 'C plus boutons per glomerulus at their real share'}
TRI = np.triu_indices(len(MBS), 1)


def mbr(X):
    return float(np.median(X.corr().values[TRI]))


def mad(X):
    return float(np.nanmean(np.abs(X.values)))


def F3(x):
    return '< 0.001' if abs(x) < 0.0005 else F(x, 3)


def gain_odor(rn, src=None, w=None):
    o = OD if src is None else src
    o = o[(o.coding_levels == MAIN) & (o.responses == rn)]
    if w is not None: o = o[o.weights == w]
    m = o.groupby('odor', sort=False).mean(numeric_only=True)
    return m, {v: m.frac_KCs_responding_blocked - m[f'frac_KCs_responding_intact_{v}'] for v in VALL}


def strength(rn):
    o = OD[(OD.coding_levels == MAIN) & (OD.responses == rn)].set_index(['mushroom_body', 'odor']).frac_KCs_responding_blocked
    P = ptab(rn)
    s_ = np.sqrt(o.reindex(list(zip(P.mushroom_body, P.odor_a))).values * o.reindex(list(zip(P.mushroom_body, P.odor_b))).values)
    return pd.Series(s_, index=pd.MultiIndex.from_arrays([P.odor_a, P.odor_b, P.mushroom_body])).unstack('mushroom_body').mean(1)


NUM = {}
for rn in ['ORN', 'GH146']:
    m, g = gain_odor(rn); d = {v: dS_mb(rn, v) for v in VALL}
    vr = {v: np.corrcoef(g['U'], g[v])[0, 1] for v in VALL if v != 'U'}; vmin = min(vr, key=vr.get)
    st = strength(rn); dU = d['U'].mean(1); ok = dU.notna()
    NUM[rn] = dict(
        gain_r=np.corrcoef(g['U'], m.frac_KCs_responding_blocked)[0, 1], n_odor=len(m), ver_r=(vr[vmin], vmin), dU=dstats(dU),
        dU_mb=mbr(d['U']), C_U=mad(d['C'] - d['U']), CU_CsU=np.corrcoef((d['C'] - d['U']).mean(1)[ok], (d['Cs'] - d['U']).mean(1)[ok])[0, 1],
        C_Cs=mbr(d['C'] - d['Cs']), P_P0=(mbr(d['P'] - d['P0']), mad(d['P'] - d['P0'])), L_P0=(mbr(d['L'] - d['P0']), mad(d['L'] - d['P0'])),
        M_M0=(mbr(d['M'] - d['M0']), mad(d['M'] - d['M0'])), MC_M=(mbr(d['MC'] - d['M']), mad(d['MC'] - d['M'])),
        MC_C=(mbr(d['MC'] - d['C']), mad(d['MC'] - d['C'])),
        rho=spearmanr(st[ok], dU[ok])[0], d=d, st=st)
R.section('Results of the prediction run')
R.p('All results are for the main setting (20% of KCs respond with APL blocked, 10% intact), averaged over the three mushroom bodies. '
    f'Table {R.n_tab + 1} says exactly what each number is: what is correlated with what, and over which items. "With ORN input" only '
    'says which response table drove the model.')
TAB_NUM = R.n_tab + 1
_r = lambda k, f_=lambda x: f'{x:.3f}': f'{f_(NUM["ORN"][k])} / {f_(NUM["GH146"][k])}'
_two = lambda k: (f'r {NUM["ORN"][k][0]:.2f}, mean |diff| {F(NUM["ORN"][k][1], 3)} / r {NUM["GH146"][k][0]:.2f}, mean |diff| {F(NUM["GH146"][k][1], 3)}')
R.table(['statement', 'what is computed', 'over what', 'value (ORN / GH146)'], [
    ['KCs gained by blocking APL track odor strength', 'Pearson r between $\\phi^{\\mathrm{blocked}}_o - \\phi^{\\mathrm{intact,U}}_o$ (KCs gained) '
     'and $\\phi^{\\mathrm{blocked}}_o$ (odor strength); each fraction averaged over the 3 MBs first',
     f'{NUM["ORN"]["n_odor"]} odors (ORN) / {NUM["GH146"]["n_odor"]} (GH146)', _r('gain_r')],
    ['and so do all versions', 'Pearson r between the KCs gained under U and under each other version; the lowest version is given', 'odors',
     f'{NUM["ORN"]["ver_r"][0]:.3f} ({NUM["ORN"]["ver_r"][1]}) / {NUM["GH146"]["ver_r"][0]:.3f} ({NUM["GH146"]["ver_r"][1]})'],
    ['mean $\\Delta S$', f'mean of $\\Delta S_{{ab}}$ (Equation {EQ_D}; cosine), each pair averaged over the MBs where it is defined', 'odor pairs',
     f'{F(NUM["ORN"]["dU"][0], 3, True)} ({NUM["ORN"]["dU"][4]} pairs) / {F(NUM["GH146"]["dU"][0], 3, True)} ({NUM["GH146"]["dU"][4]} pairs)'],
    ['similarity rises most for pairs of strong odors', 'Spearman ρ between pair strength $\\sqrt{\\phi^{\\mathrm{blocked}}_a\\,\\phi^{\\mathrm{blocked}}_b}$ '
     f'and $\\Delta S_{{ab}}$ (U); Figure {R.n_fig + 1}', 'odor pairs', _r('rho', lambda x: f'{x:.2f}')],
    ['same pattern in all three MBs', 'Pearson r between two MBs\' $\\Delta S$ (U) over pairs, for each of the 3 MB pairs; the median of the 3',
     'odor pairs', _r('dU_mb', lambda x: f'{x:.2f}')],
    ['per-KC counts (C) change $\\Delta S$', 'mean over pairs and MBs of $|\\Delta S^{C}_{ab} - \\Delta S^{U}_{ab}|$', 'pairs × 3 MBs', _r('C_U')],
    ['shuffled counts do much the same', 'Pearson r between the maps $\\Delta S^{C} - \\Delta S^{U}$ and $\\Delta S^{Cs} - \\Delta S^{U}$ (MB means)',
     'odor pairs', _r('CU_CsU', lambda x: f'{x:.2f}')],
    ['what is left differs between MBs', 'as the "same pattern" row, for $\\Delta S^{C} - \\Delta S^{Cs}$', 'odor pairs', _r('C_Cs', lambda x: f'{x:.2f}')],
    ['P against P0', 'median Pearson r between MBs of $\\Delta S^{P} - \\Delta S^{P0}$, and mean $|\\Delta S^{P} - \\Delta S^{P0}|$', 'pairs (× 3 MBs)', _two('P_P0')],
    ['L against P0', 'the same for $\\Delta S^{L} - \\Delta S^{P0}$', 'pairs (× 3 MBs)', _two('L_P0')],
    ['M against M0', 'the same for $\\Delta S^{M} - \\Delta S^{M0}$', 'pairs (× 3 MBs)', _two('M_M0')],
    ['MC against M and C', 'the same for $\\Delta S^{MC} - \\Delta S^{M}$; then for $\\Delta S^{MC} - \\Delta S^{C}$', 'pairs (× 3 MBs)',
     _two('MC_M') + '; ' + _two('MC_C')]],
    title='Every number in the Results, and what it is.',
    caption='$\\phi_o$ is the fraction of KCs responding to odor $o$. "MB" = mushroom body. Pairs with ms count where they are defined '
            '(Section 8). r = Pearson correlation, ρ = Spearman (rank) correlation.')

R.sub('Similarity rises most for pairs of strong odors')


def fig_strong():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, rn in zip(axs, ['ORN', 'GH146']):
        st, dU = NUM[rn]['st'], NUM[rn]['d']['U'].mean(1)
        ms_ = np.array([('ms' in ab) for ab in st.index]); ok = dU.notna().values
        rho_no = spearmanr(st[ok & ~ms_], dU[ok & ~ms_])[0]
        a.scatter(st[~ms_ & ok] * 100, dU[~ms_ & ok], s=26, color=L.INTACT, edgecolor='white', linewidth=0.6, label='pairs without ms', zorder=3)
        a.scatter(st[ms_ & ok] * 100, dU[ms_ & ok], s=34, facecolor='white', edgecolor=L.BLOCKED, linewidth=1.4, label='pairs with ms', zorder=4)
        a.axhline(0, color=L.GRAY, lw=0.7, ls=':')
        a.set_xlabel('pair strength $\\sqrt{\\phi_a\\,\\phi_b}$ with APL blocked, % of KCs')
        a.set_ylabel('$\\Delta S_{ab}$ = blocked − intact (cosine), U')
        a.set_title(f'{"ab"[rn == "GH146"]}  {rn} input: Spearman ρ = {NUM[rn]["rho"]:.2f} (all), {rho_no:.2f} (without ms)', fontsize=9.5)
        a.legend(fontsize=7.5, loc='upper left')
    fig.tight_layout(); return fig


R.fig(fig_strong(), 'Each point is one odor pair (uniform APL, mean of the three mushroom bodies, 20% → 10%). Horizontal: the geometric '
      'mean of the two odors\' fractions of KCs responding with APL blocked. Vertical: how much more similar the pair is with APL blocked. '
      f'Open orange circles are the pairs with ms. With ORN input ms drives no KC in two mushroom bodies, so its pairs come from male CNS R '
      f'only, where it drives {_ms["malecns R"][0]} KC{"" if _ms["malecns R"][0] == 1 else "s"}; they sit near pair strength 0.',
      'Similarity change against pair strength.')
R.p(f'Pairs of strongly driven odors gain the most similarity (Spearman ρ = {NUM["ORN"]["rho"]:.2f} with ORN input, {NUM["GH146"]["rho"]:.2f} '
    'with GH146). Strong odors drive many KCs, recruit more APL, and so lose more of their responses to APL; when APL is blocked, '
    'their overlapping KCs come back and their patterns converge.')

R.sub('Every version: similarity with and without inhibition, and its difference from U')
VER_TAB = []
for v in VALL:
    row = [f'{v}: {VDESC[v]}']
    for rn in ['ORN', 'GH146']:
        dv, dU_ = NUM[rn]['d'][v], NUM[rn]['d']['U']; s_ = dstats(dv.mean(1)); ok = dv.mean(1).notna() & dU_.mean(1).notna()
        row += [f'{F(s_[0], 3, True)} ({F(s_[2], 2, True)} to {F(s_[3], 2, True)})',
                '—' if v == 'U' else f'{np.corrcoef(dv.mean(1)[ok], dU_.mean(1)[ok])[0, 1]:.2f}',
                '—' if v == 'U' else F3(mad(dv - dU_))]
    VER_TAB.append(row)
TAB_VER = R.n_tab + 1
R.table(['version', 'ORN: mean $\\Delta S$ (range)', 'ORN: r with U', 'ORN: mean $|\\Delta S - \\Delta S^U|$',
         'GH146: mean $\\Delta S$ (range)', 'GH146: r with U', 'GH146: mean $|\\Delta S - \\Delta S^U|$'], VER_TAB,
        title='The average change in similarity (blocked − intact) in every version, and how each version differs from U.',
        caption='Mean $\\Delta S$: mean over odor pairs of the MB-averaged $\\Delta S_{ab}$, with its smallest and largest pair. r with U: Pearson '
                'r between the version\'s and U\'s MB-averaged $\\Delta S$ over pairs. Mean $|\\Delta S - \\Delta S^U|$: over pairs and MBs. '
                'The blocked responses are the same in every version, so $\\Delta S - \\Delta S^U$ = $S^{\\mathrm{intact},U} - S^{\\mathrm{intact}}$.',
        num_cols=[1, 2, 3, 4, 5, 6])
_dd = pd.concat([(NUM['ORN']['d'][v] - NUM['ORN']['d']['U']).mean(1) for v in VALL if v != 'U'])
VMD = float(np.ceil(np.nanpercentile(np.abs(_dd), 99) / 0.05) * 0.05)
R.p(f'Figures {R.n_fig + 1}–{R.n_fig + len(VALL) - 1} show, for each version with ORN input, the intact similarity, the change $\\Delta S$ '
    f'(same layout and scale as Figure {FIG_SIMU}c, with the 5 most and 5 least changed pairs outlined), and the difference from U '
    f'(diverging scale clipped at ±{VMD:g}). Figure {R.n_fig + len(VALL)} shows all versions with GH146 input.')


def fig_ver(rn, v):
    z = sim_set(rn, v); zu = sim_set(rn, 'U'); fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
    heat(fig, axs[0], z['Si'], SEQc, (0, float(np.nanmax(z['Sb'].values))), f'a  $S_{{ab}}$, APL intact ({v})', 'cosine similarity $S_{ab}$', z['H'])
    heat(fig, axs[1], z['D'], DIVc, (-0.2, 0.2), f'b  $\\Delta S_{{ab}}$ = blocked − intact ({v})', '$\\Delta S_{ab}$ (difference in cosine)', z['H'], z['top'], z['bot'])
    Dd = z['D'] - zu['D']
    heat(fig, axs[2], Dd, DIVc, (-VMD, VMD), f'c  $\\Delta S^{{{v}}} - \\Delta S^{{U}}$', 'difference in $\\Delta S_{ab}$ (version − U)', z['H'] | zu['H'])
    fig.tight_layout(); return fig, z


for v in VALL:
    if v == 'U': continue
    fg, z = fig_ver('ORN', v); s_ = dstats(z['d']); dv, dU_ = NUM['ORN']['d'][v], NUM['ORN']['d']['U']
    ok = dv.mean(1).notna() & dU_.mean(1).notna()
    R.fig(fg, f'Version {v} ({VDESC[v]}), ORN input, mean of the three mushroom bodies. **a**, Cosine similarity with APL intact. '
          f'**b**, $\\Delta S_{{ab}}$ = blocked − intact. **The average change in similarity with minus without inhibition is '
          f'{F(s_[0], 3, True)}** (U: {F(mU[0], 3, True)}; range {F(s_[2], 2, True)} to {F(s_[3], 2, True)}). Solid outlines: the 5 pairs that '
          f'change most ({pairs_txt(z["top"], z["d"])}); dashed: the 5 that change least ({pairs_txt(z["bot"], z["d"])}). **c**, $\\Delta S$ of '
          f'this version minus that of U: orange = blocking APL raises the pair\'s similarity more than under U. Compared with U: Pearson r = '
          f'{np.corrcoef(dv.mean(1)[ok], dU_.mean(1)[ok])[0, 1]:.2f} over pairs, mean |difference| {F3(mad(dv - dU_))}. Hatched: ms (Section 8).',
          f'Version {v}: similarity with APL intact, the change, and the difference from U.')


def fig_grid(rn, diff=False):
    vs = [v for v in VALL if not (diff and v == 'U')]
    nc = 5 if not diff else 3; nr = int(np.ceil(len(vs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc + 1.2, 3.25 * nr + 0.3)); axs = np.atleast_2d(axs)
    zu = sim_set(rn, 'U'); out = {}
    for a, v in zip(axs.ravel(), vs):
        z = sim_set(rn, v); out[v] = z
        M = z['D'] - zu['D'] if diff else z['D']
        im = heat(fig, a, M, DIVc, (-VMD, VMD) if diff else (-0.2, 0.2),
                  f'{v}: mean {F((z["d"] - zu["d"]).mean() if diff else z["d"].mean(), 3, True)}', '', z['H'] | (zu['H'] if diff else False),
                  () if diff else z['top'], () if diff else z['bot'], fs=5.4, cb=False)
    for a in axs.ravel()[len(vs):]: a.axis('off')
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.25, 0.012, 0.5]); cb = fig.colorbar(im, cax=cax)
    cb.set_label('difference in $\\Delta S_{ab}$ (version − U)' if diff else '$\\Delta S_{ab}$ (blocked − intact)', fontsize=8); cb.ax.tick_params(labelsize=7)
    return fig, out


fg, zg = fig_grid('GH146')
R.fig(fg, f'$\\Delta S_{{ab}}$ = blocked − intact for every version with GH146 input ({RV["GH146"].shape[1]} odors), mean of the three mushroom '
      'bodies. The panel title gives the average change in similarity with minus without inhibition. Outlines as in Figure '
      f'{FIG_SIMU}c (the 5 pairs that change most, solid; least, dashed). ' + weak_txt('GH146', VALL),
      'Every version with GH146 input: the change in similarity.')
fg, _ = fig_grid('GH146', diff=True)
R.fig(fg, 'Each version\'s $\\Delta S_{ab}$ minus that of U, GH146 input, mean of the three mushroom bodies. The panel title gives the mean '
      f'difference over pairs. Orange: blocking APL raises the pair\'s similarity more than under U. Scale clipped at ±{VMD:g}.',
      'Every version with GH146 input: difference from U.')

R.sub('Wiring versions against their own references')


def fig_diff():
    cmp_ = [('C', 'U', 'C − U: per-KC counts'), ('C', 'Cs', 'C − Cs: counts vs shuffled'), ('P', 'P0', 'P − P0: boutons per glomerulus'),
            ('L', 'P0', 'L − P0: claws per glomerulus'), ('M', 'M0', 'M − M0: boutons at real share'), ('MC', 'M', 'MC − M: per-KC on top of M')]
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 8.4))
    for a, (v, ref, t) in zip(axs.ravel(), cmp_):
        z, zr = sim_set('ORN', v), sim_set('ORN', ref); X = NUM['ORN']['d'][v] - NUM['ORN']['d'][ref]
        heat(fig, a, z['D'] - zr['D'], DIVc, (-VMD, VMD), f'{"abcdef"[cmp_.index((v, ref, t))]}  {t}\n(MB agreement r = {mbr(X):.2f}; mean |diff| {F(mad(X), 3)})',
             'difference in $\\Delta S_{ab}$', z['H'] | zr['H'])
    fig.tight_layout(); return fig


R.fig(fig_diff(), 'How much each wiring version changes the effect of blocking APL, pair by pair (mean of three mushroom bodies, ORN input). '
      'Each cell is $\\Delta S_{ab}$ of the version minus $\\Delta S_{ab}$ of its reference. The blocked responses are identical, so this equals '
      'the reference\'s intact similarity minus the version\'s. Orange: blocking APL raises the pair\'s similarity more in this version. '
      f'Blue: less. The scale is clipped at ±{VMD:g}. "MB agreement" is the median Pearson r between the three mushroom bodies\' maps: a '
      'pattern that is the same in all three has r near 1. ' + weak_txt('ORN', VALL), 'Wiring versions compared with their references.')
# ---------------------------------------------------------------- numbers for the summary (Sections 11 and 12 draw them)
SH = list(RV['GH146'].columns)
def shared(rn, v, ref='GH146', vref=None):
    a_ = NUM_ALL[rn][v].mean(1); b_ = NUM_ALL[ref][vref or v].mean(1)
    a_ = a_[[(x in SH and y in SH) for x, y in a_.index]]; b_ = b_.reindex(a_.index)
    ok = a_.notna() & b_.notna(); return np.corrcoef(a_[ok], b_[ok])[0, 1], int(ok.sum())


NUM_ALL = {rn: {v: dS_mb(rn, v) for v in VALL} for rn in RV}
_o = {rn: dstats(NUM_ALL[rn]['U'].mean(1))[0] for rn in RV}
OTHER_TXT = (_o,)
def class_share():
    out = []
    for k in MBS:
        mb = MB[k]; cls = mb['cls']
        for wn in ['W', 'W_raw', 'W_claw']:
            W_ = mb[wn]; gl = [g for g in W_.columns if g in RESP['ORN'].index]
            H = W_[gl].values @ np.clip(RESP['ORN'].loc[gl].values, 0, None); th, _ = theta_of(H, 0.2); Rsp = H > th
            for c_ in ['αβ', 'α′β′', 'γ']:
                out.append(dict(mb=k, w=wn, cls=c_, frac=Rsp[cls == c_].mean(), share_kc=(cls == c_).mean(), share_resp=Rsp[cls == c_].sum() / Rsp.sum()))
    return pd.DataFrame(out)


CS = class_share()


OW2d = {(w, rn): {v: dS_mb(rn, v, OW2, w) for v in VALL} for w in ['W_raw', 'W_claw'] for rn in ['ORN', 'GH146']}
def rnan(a_, b_):
    b_ = b_.reindex(a_.index); ok = a_.notna() & b_.notna(); return float(np.corrcoef(a_[ok], b_[ok])[0, 1])


WT = {(w, rn): (OW2d[(w, rn)]['U'].mean(1).mean(), rnan(OW2d[(w, rn)]['U'].mean(1), NUM[rn]['d']['U'].mean(1)))
      for w in ['W_raw', 'W_claw'] for rn in ['ORN', 'GH146']}
_gsh = lambda w: CS[(CS.w == w) & (CS.cls == 'γ')]

# ---------------------------------------------------------------- summary bullets (inserted at the end of Section 10)
no, ng = NUM['ORN'], NUM['GH146']
pp = (no['d']['P'] - no['d']['P0']).mean(1).sort_values()
R.sub('Summary of the prediction run')
R.ul([f'**Uniform APL.** The KCs gained by blocking APL track odor strength (Pearson r = {no["gain_r"]:.2f} over odors with ORN input, '
      f'{ng["gain_r"]:.2f} with GH146), and the per-odor gain is nearly the same in all versions (r ≥ {min(no["ver_r"][0], ng["ver_r"][0]):.3f}). '
      f'Similarity rises on average by {F(no["dU"][0], 3, True)} with ORN and {F(ng["dU"][0], 3, True)} with GH146, most for pairs of strong odors '
      f'(Spearman ρ = {no["rho"]:.2f} and {ng["rho"]:.2f}). The pattern is the same in all three mushroom bodies (median r '
      f'{no["dU_mb"]:.2f} and {ng["dU_mb"]:.2f}).',
      f'**Per-KC counts (C).** They change $\\Delta S$ by {F(no["C_U"], 3)} on average (ORN; GH146 {F(ng["C_U"], 3)}). The same counts shuffled '
      f'within KC class do much the same (maps r {no["CU_CsU"]:.2f}). What is left differs between mushroom bodies (median r {no["C_Cs"]:.2f}).',
      f'**Boutons per glomerulus (P against P0).** A pattern that is consistent across mushroom bodies (median r {no["P_P0"][0]:.2f} with '
      f'ORN, {ng["P_P0"][0]:.2f} with GH146; mean |change| {F(no["P_P0"][1], 3)}). The pairs that gain most under P are '
      + ', '.join(f'{a}–{b}' for a, b in pp.index[-4:][::-1]) + '; those that gain least are ' + ', '.join(f'{a}–{b}' for a, b in pp.index[:3]) + '.',
      f'**At the boutons\' real share (M against M0)** the same pattern shrinks to a mean |change| of {F(no["M_M0"][1], 3)}. **Claws (L '
      f'against P0)**: mean |change| {F(no["L_P0"][1], 3)} (median r between mushroom bodies {no["L_P0"][0]:.2f}).',
      f'**Both on KCs and boutons with connectome weights (MC).** MC differs from M by {F(no["MC_M"][1], 3)} on average and from C by '
      f'{F(no["MC_C"][1], 3)} (ORN). Its mean $\\Delta S$ is in Table {TAB_VER}.',
      f'**Other inputs (Section 11).** The PN transformation gives a mean $\\Delta S$ of {F(OTHER_TXT[0]["ORN-PN"], 3, True)} (ORN: '
      f'{F(OTHER_TXT[0]["ORN"], 3, True)}). Keeping negative responses gives {F(OTHER_TXT[0]["ORN neg"], 3, True)} (ORN neg) and '
      f'{F(OTHER_TXT[0]["GH146 neg"], 3, True)} (GH146 neg; GH146: {F(OTHER_TXT[0]["GH146"], 3, True)}).',
      f'**Other weightings (Section 12).** With raw synapse counts, {L.pct(CS[(CS.w == "W_raw") & (CS.cls == "γ")].share_resp.mean())} of '
      f'the responses come from γ KCs (main weighting: {L.pct(CS[(CS.w == "W") & (CS.cls == "γ")].share_resp.mean())}). Mean $\\Delta S$ '
      f'(U, ORN) is {F(WT[("W_raw", "ORN")][0], 3, True)} with raw counts and {F(WT[("W_claw", "ORN")][0], 3, True)} with equal weight per '
      f'claw; their maps match the main one with Pearson r = {WT[("W_raw", "ORN")][1]:.2f} and {WT[("W_claw", "ORN")][1]:.2f}.'])


# ---------------------------------------------------------------- 11 other inputs
R.section('Other inputs: PN-level transformation and negative responses')
R.p('The same model was run with three more inputs (Section 4): **ORN-PN**, the ORN responses after the Olsen transformation; **ORN neg** '
    'and **GH146 neg**, with negative responses kept. Each gets its own threshold and gains. Table ' + str(R.n_tab + 1) + ' compares them with '
    'the main inputs.')
SH = list(RV['GH146'].columns)


base = {'ORN': 'ORN', 'ORN-PN': 'ORN', 'ORN neg': 'ORN', 'GH146': 'GH146', 'GH146 neg': 'GH146'}
rows = []
for rn in ['ORN', 'ORN-PN', 'ORN neg', 'GH146', 'GH146 neg']:
    m, g = gain_odor(rn); s_ = dstats(NUM_ALL[rn]['U'].mean(1))
    rb = '—' if base[rn] == rn else f'{np.corrcoef(*[NUM_ALL[x]["U"].mean(1).reindex(NUM_ALL[rn]["U"].index).values for x in (rn, base[rn])])[0, 1]:.2f}' \
        if NUM_ALL[rn]['U'].mean(1).notna().all() else '—'
    if base[rn] != rn:
        a_, b_ = NUM_ALL[rn]['U'].mean(1), NUM_ALL[base[rn]]['U'].mean(1).reindex(NUM_ALL[rn]['U'].index); ok = a_.notna() & b_.notna()
        rb = f'{np.corrcoef(a_[ok], b_[ok])[0, 1]:.2f}'
    rg = '—' if rn == 'GH146' else '{:.2f} ({} pairs)'.format(*shared(rn, 'U'))
    rows.append([rn, len(m), f'{F(s_[0], 3, True)} ({F(s_[2], 2, True)} to {F(s_[3], 2, True)})', f'{np.corrcoef(g["U"], m.frac_KCs_responding_blocked)[0, 1]:.2f}',
                 rb, rg])
R.table(['input', 'odors', 'mean $\\Delta S$, U (range)', 'KCs gained vs odor strength, r', 'r with the main input\'s $\\Delta S$ (U)',
         'r with GH146\'s $\\Delta S$ (U), shared odors'], rows,
        title='The uniform model with each input.',
        caption='"KCs gained vs odor strength": as in Table ' + str(TAB_NUM) + ', row 1. "r with the main input": Pearson r over pairs between the '
                'MB-averaged $\\Delta S$ maps of this input and of ORN (for ORN-PN and ORN neg) or GH146 (for GH146 neg). "r with GH146": the same '
                f'against GH146, over the pairs of the {len(SH)} odors that both have.', num_cols=[1, 2, 3, 4, 5])
rv_rows = []
for v in VALL:
    rv_rows.append([v] + [F(NUM_ALL[rn][v].mean(1).mean(), 3, True) for rn in RV] +
                   ['{:.2f}'.format(shared('ORN', v)[0]), '{:.2f}'.format(shared('ORN-PN', v)[0])])
R.table(['version'] + [f'mean $\\Delta S$, {rn}' for rn in RV] + ['ORN vs GH146: r', 'ORN-PN vs GH146: r'], rv_rows,
        title='Mean change in similarity for every version and input, and the match with GH146.',
        caption=f'The last two columns: Pearson r between the version\'s MB-averaged $\\Delta S$ maps with ORN (or ORN-PN) and with GH146 input, over '
                f'the pairs of the {len(SH)} shared odors. If the PN transformation made the ORN input more PN-like, the last column would be higher '
                'than the one before it.', num_cols=list(range(1, len(RV) + 3)))


def fig_inputs_dS():
    fig, axs = plt.subplots(1, 5, figsize=(16.5, 3.9))
    for a, rn in zip(axs, ['ORN', 'ORN-PN', 'ORN neg', 'GH146', 'GH146 neg']):
        z = sim_set(rn, 'U'); im = heat(fig, a, z['D'], DIVc, (-0.2, 0.2), f'{rn}: mean {F(z["d"].mean(), 3, True)}', '', z['H'], z['top'], z['bot'], fs=5.6, cb=False)
    fig.tight_layout(rect=(0, 0, 0.95, 1)); cax = fig.add_axes([0.958, 0.25, 0.008, 0.5]); cb = fig.colorbar(im, cax=cax)
    cb.set_label('$\\Delta S_{ab}$ (blocked − intact)', fontsize=8); cb.ax.tick_params(labelsize=7)
    return fig


R.fig(fig_inputs_dS(), '$\\Delta S_{ab}$ = blocked − intact for the uniform model with each input, mean of the three mushroom bodies. Outlines: '
      f'the 5 pairs that change most (solid) and least (dashed). Odors that drive fewer than {MIN_KC} KCs in some mushroom body are hatched: '
      + '; '.join(f'{rn}: ' + weak_txt(rn)[len('Hatched: '):].rstrip('.') for rn in ['ORN', 'ORN-PN', 'ORN neg', 'GH146', 'GH146 neg']
                  if weak_txt(rn).startswith('Hatched')) + '.', 'The change in similarity with each input (uniform APL).')
_neg = {rn: float(np.nanmean(np.abs(S_mb(rn, 'cos_KC_input').values - S_mb(rn + ' neg', 'cos_KC_input').values))) for rn in ['ORN', 'GH146']}
_negb = {rn: float(np.nanmean(np.abs(S_mb(rn, 'cos_blocked').values - S_mb(rn + ' neg', 'cos_blocked').values))) for rn in ['ORN', 'GH146']}
_rg = {rn: shared(rn, 'U')[0] for rn in ['ORN', 'ORN-PN']}; _rgP = {rn: shared(rn, 'P')[0] for rn in ['ORN', 'ORN-PN']}
R.ul([f'**PN transformation (ORN-PN).** Blocking APL raises similarity a little {"more" if _o["ORN-PN"] > _o["ORN"] else "less"} ({F(_o["ORN-PN"], 3, True)} against '
      f'{F(_o["ORN"], 3, True)} with ORN), and the map of which pairs change is nearly the same as with ORN (r = {rows[1][4]}). Over the shared '
      f'odors its $\\Delta S$ map matches GH146\'s with r = {_rg["ORN-PN"]:.2f}, against {_rg["ORN"]:.2f} for ORN (U; version P, which matches '
      f'GH146 best, gives {_rgP["ORN-PN"]:.2f} against {_rgP["ORN"]:.2f})'
      + (': the transformation moves the model only slightly closer to GH146.' if 0 < _rg['ORN-PN'] - _rg['ORN'] < 0.05 else
         ': the transformation does not move the model closer to GH146.' if _rg['ORN-PN'] - _rg['ORN'] <= 0 else '.'),
      f'**ORN and GH146 give different maps.** Even without the transformation, the ORN and GH146 $\\Delta S$ maps agree only with r = '
      f'{_rg["ORN"]:.2f} over their {shared("ORN", "U")[1]} shared pairs, although both inputs give the same per-odor pattern of KCs gained '
      f'(Table {R.n_tab - 1}, column 4). Which pairs change most depends on the input table.',
      f'**Negative responses** change almost nothing. They alter the similarity of the KC drives only slightly (mean |change| of the '
      f'input cosine {F(_neg["ORN"], 3)} with ORN, {F(_neg["GH146"], 3)} with GH146) and the KC responses even less (blocked cosine '
      f'{F(_negb["ORN"], 4)} and {F(_negb["GH146"], 4)}), as the threshold removes small drives anyway. The $\\Delta S$ maps match those of the '
      f'clipped inputs with r ≈ 1.00 in every version (Table {R.n_tab}).'])

# ---------------------------------------------------------------- 12 other weightings
R.section('Other input weightings')
R.p(f'Section 3 explains why $W_{{kg}}$ is divided by all of the KC\'s claw PN synapses. Two alternatives were run with the ORN and GH146 inputs:')
R.ul(['**$W^{\\mathrm{raw}}$**: raw synapse counts, $W_{kg} = n_{kg}$, so a KC\'s drive grows with its number of claw PN synapses. One '
      'threshold for all KCs, as before.',
      '**$W^{\\mathrm{claw}}$**: equal weight per claw, $W_{kg}$ = share of the KC\'s claws that come from glomerulus $g$, so a small claw counts '
      'as much as a large one.'])


def fig_weights():
    fig = plt.figure(figsize=(13, 4.3)); gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.42)
    a = fig.add_subplot(gs[0]); cols = {'W': L.INK, 'W_raw': L.BLOCKED, 'W_claw': L.GREEN}
    lab = {'W': '$W$ (share of synapses, main)', 'W_raw': '$W^{raw}$ (synapse counts)', 'W_claw': '$W^{claw}$ (share of claws)'}
    for i, c_ in enumerate(['αβ', 'α′β′', 'γ']):
        for j, wn in enumerate(['W', 'W_raw', 'W_claw']):
            q = CS[(CS.cls == c_) & (CS.w == wn)]; x = i + (j - 1) * 0.26
            a.bar(x, q.frac.mean() * 100, 0.24, color=cols[wn], label=lab[wn] if i == 0 else None)
            a.scatter(np.full(len(q), x), q.frac * 100, s=9, color='white', edgecolor=L.MUTED, lw=0.6, zorder=3)
    a.set_xticks(range(3)); a.set_xticklabels(['αβ KCs', 'α′β′ KCs', 'γ KCs']); a.set_ylabel('KCs responding with APL blocked, %\n(mean over odors)')
    a.legend(fontsize=7, loc='upper left'); a.set_title('a  Who responds, ORN input (bars: mean of 3 MBs; dots: each MB)', fontsize=8.6)
    for ax_i, wn in zip([1, 2], ['W_raw', 'W_claw']):
        a = fig.add_subplot(gs[ax_i]); z = sim_set('ORN', 'U', OW2, wn)
        heat(fig, a, z['D'], DIVc, (-0.2, 0.2), f'{"bc"[ax_i - 1]}  $\\Delta S_{{ab}}$, U, {lab[wn].split(" (")[0]}: mean {F(z["d"].mean(), 3, True)}',
             '$\\Delta S_{ab}$', z['H'], z['top'], z['bot'], fs=6)
    return fig


_cs = CS.groupby(['w', 'cls']).frac.mean()
R.fig(fig_weights(), '**a**, Percentage of KCs of each class that respond with APL blocked (one threshold, 20% of all KCs on average over odors), '
      'with the three weightings (ORN input). With raw synapse counts the γ KCs, which have the most claw PN synapses, take most of the '
      'responses. **b**, **c**, $\\Delta S_{ab}$ of the uniform model with the two other weightings (ORN input, mean of three mushroom bodies), '
      'with the 5 most (solid) and least (dashed) changed pairs outlined; same scale as the main figure. Raw counts: ' + weak_txt('ORN', w='W_raw')
      + ' Equal weight per claw: ' + weak_txt('ORN', w='W_claw'),
      'Other input weightings: who responds, and the change in similarity.')
wrows = []
for v in VALL:
    row = [v]
    for rn in ['ORN', 'GH146']:
        for w in ['W_raw', 'W_claw']:
            a_, b_ = OW2d[(w, rn)][v].mean(1), NUM[rn]['d'][v].mean(1).reindex(OW2d[(w, rn)][v].index); ok = a_.notna() & b_.notna()
            row += [f'{F(a_.mean(), 3, True)} (r {np.corrcoef(a_[ok], b_[ok])[0, 1]:.2f})']
    wrows.append(row)
R.table(['version', 'ORN, $W^{raw}$', 'ORN, $W^{claw}$', 'GH146, $W^{raw}$', 'GH146, $W^{claw}$'], wrows,
        title='Mean $\\Delta S$ for every version with the other weightings, and the match with the main weighting.',
        caption='Each cell: mean $\\Delta S$ over pairs (MB-averaged), and in brackets the Pearson r over pairs with the same version\'s '
                '$\\Delta S$ map under the main weighting $W$.', num_cols=[1, 2, 3, 4])
R.ul([f'**Equal weight per claw** gives nearly the same model: γ KCs give {L.pct(_gsh("W_claw").share_resp.mean())} of the responses '
      f'(main weighting {L.pct(_gsh("W").share_resp.mean())}; γ KCs are {L.pct(_gsh("W").share_kc.mean())} of all KCs), and the $\\Delta S$ maps '
      f'match the main ones with r ≥ {min(float(x.split("r ")[1].rstrip(")")) for row in wrows for x in row[2::2]):.2f}.',
      f'**Raw synapse counts** shift the responses to the γ KCs ({L.pct(_gsh("W_raw").share_resp.mean())} of the responses), because they have '
      f'the most claw PN synapses (Section 3). With a different set of KCs responding, the maps match the main ones less well (U: r = '
      f'{WT[("W_raw", "ORN")][1]:.2f} with ORN, {WT[("W_raw", "GH146")][1]:.2f} with GH146), and least for the versions with per-KC counts '
      '(C, Cs, Cc, MC).'])

# ---------------------------------------------------------------- 13 the recordings
R.section('The recordings, for comparison')
import tnt_compare as tc
assert sorted(tc.O) == sorted(ODORS), (tc.O, ODORS)
S3d, STd = tc.SREC['TNT3c'].mean(0), tc.SREC['TNT'].mean(0)
_ser = lambda v: pd.Series(v, index=pd.MultiIndex.from_arrays([[tc.O[i] for i in tc.IU[0]], [tc.O[j] for j in tc.IU[1]]]))
dSd = _ser(STd - S3d)
_top, _bot = extremes(dSd, pd.DataFrame(False, index=ODORS, columns=ODORS))
R.p('For direct comparison with the model maps above, Figure ' + str(R.n_fig + 1) + ' repeats the similarity maps of the recordings (Report 06, '
    f'Figure 3) with the odors in the same order. TNT3c ({len(tc.SREC["TNT3c"])} recordings) has APL intact; TNT ({len(tc.SREC["TNT"])} '
    'recordings) has APL blocked. Each recording\'s similarity is the cosine over its units of the raw responses (FC z-scores), and each '
    'condition\'s map is the mean over its recordings; $1 - S$ is Yang\'s RDM.')


def fig_data():
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
    A, B_ = tomat(_ser(S3d), ODORS), tomat(_ser(STd), ODORS); vmax = float(max(S3d.max(), STd.max()))
    heat(fig, axs[0], A, SEQc, (0, vmax), 'a  $S^{TNT3c}_{ab}$ (APL intact)', 'cosine similarity')
    heat(fig, axs[1], B_, SEQc, (0, vmax), 'b  $S^{TNT}_{ab}$ (APL blocked)', 'cosine similarity')
    heat(fig, axs[2], tomat(dSd, ODORS), DIVc, (-0.4, 0.4), 'c  $\\Delta S_{ab}$ = TNT − TNT3c', '$\\Delta S_{ab}$ (difference in cosine)', None, _top, _bot)
    fig.tight_layout(); return fig


R.fig(fig_data(), '**a**, **b**, Mean cosine similarity of the recorded units\' responses for every odor pair, in TNT3c and TNT, on one scale '
      '(white 0, dark blue the highest value; slightly negative values also show as white). **c**, The change, on a diverging scale '
      f'(orange = more similar in TNT), clipped at ±0.4, twice the model maps\' range. **The average change in similarity is {F(dSd.mean(), 3, True)}** '
      f'(range {F(dSd.min(), 2, True)} to {F(dSd.max(), 2, True)}; all {len(dSd)} pairs are more similar in TNT). Solid outlines: the 5 pairs '
      f'that change most ({pairs_txt(_top, dSd)}); dashed: the 5 that change least ({pairs_txt(_bot, dSd)}).',
      'The recordings: odor-pair similarity in TNT3c and TNT, and the change.')
R.p(f'In the recordings every pair becomes more similar, by {F(dSd.mean(), 2, True)} on average, against {F(mU[0], 3, True)} for the uniform '
    'model with ORN input. The model here is set to 20% → 10% of KCs responding for every odor on average, and it runs on another '
    'fly\'s glomerular responses, so its size of effect is not expected to match. Report 07 pins the model to each odor\'s measured '
    'fraction of responding units and compares the two pair by pair.')

R.box('What this step does not include', [
    'One threshold for all KCs, with no differences in excitability between classes, and no spikes or time course.',
    'APL is one unit with one activity per odor in every version. P, L, M and MC weight its output per glomerulus or per KC but do not make '
    'its activity local. A local, non-spiking APL is modelled separately (Report 11).',
    'The inputs are ORN or GH146 averages from other flies, not the recorded flies\' PNs. The PN transformation\'s two constants were fitted '
    'to GH146; the best fit matches GH146 about as well as the untransformed ORN responses.',
    'With negative responses kept, the divisive versions also shrink negative inputs (Section 4).',
    'The coding levels (20% blocked, 10% intact) are assumptions here. Report 07 replaces them with the recordings.'], 'caveat')
R.parts = [x.replace('TAB_PQ', f'Tables {TAB_PQ} and {TAB_PQ + 1}').replace('TAB_NUM', f'Table {TAB_NUM}') for x in R.parts]
R.save()
