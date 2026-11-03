"""Report 12: APL with its signals confined to its calyx sub-arbors (apl_compartment_model.py), step by step, with results in both
versions of the tables and the local model of Report 11 as reference."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import report_lib as L
L.style(); L.DATE = '2026-10-24'
F = L.fmt
OWS = {'official': L.B + 'odor_weighted/', 'outside': L.B + 'with_outside_ROI_synapses/odor_weighted/'}
TAGS = {'official': '', 'outside': '_outside_roi'}
DD = L.B + 'reports/data/'
MBS = ['hemibrain R', 'malecns L', 'malecns R']; MBN = {'hemibrain R': 'hemibrain R', 'malecns L': 'male CNS L', 'malecns R': 'male CNS R'}
MBC = {'hemibrain R': L.DS_COLOR['hemibrain'], 'malecns L': L.DS_COLOR['malecns'], 'malecns R': '#7ccfa9'}
MK = {'hemibrain R': 'o', 'malecns L': 's', 'malecns R': '^'}
RNS = ['ORN', 'GH146']
CORR, CEIL = 0.92, 0.84
COMP = '#a8820e'
PAL = ['#2a78d6', '#eb6834', '#1f9e6e', '#8a5cc2', '#c8a21c', '#d35d9b', '#4bb3c7', '#7a5c3e']    # sub-arbor colors of Report 10
RK = 'model: Spearman blocked vs intact (mean over MBs) by version'
RC = 'r over odor pairs, model change (MB-averaged) vs data change'
RL = 'r over odor pairs, model change vs data change'
DS = 'mean change in similarity: data, model by version'
LOC = ['200', '100', '50', '25', '10', '5', '2']
CL = ['200', '100', '50', '25', '10']


def load(ver):
    o, t = OWS[ver], TAGS[ver]
    return dict(SC=json.load(open(o + 'APL_compartment_model_summary.json')), SL=json.load(open(o + 'APL_local_model_summary.json')),
                S2=json.load(open(o + 'APL_local_boutons_summary.json')),
                PC={rn: pd.read_csv(o + f'APL_compartment_model_per_MB_{rn}.csv').set_index(['version', 'mushroom_body']) for rn in RNS},
                PL={rn: pd.read_csv(o + f'APL_local_model_per_MB_{rn}.csv').set_index(['version', 'mushroom_body']) for rn in RNS},
                NU=pd.read_csv(o + 'APL_compartment_model_nulls.csv'), CT=pd.read_csv(o + 'APL_compartment_model_compartments.csv'),
                SP=pd.read_csv(DD + f'r12_kc_spread{t}.csv'), MP=pd.read_csv(DD + f'r12_M_properties{t}.csv'),
                POS=pd.read_csv(DD + f'r12_positions{t}.csv'), NM=json.load(open(DD + f'r12_names{t}.json')))


V = {v: load(v) for v in OWS}
d0, d1 = V['official'], V['outside']


def lam_lab(key):
    t = key.split('compartments ')[-1].replace('local ', '')
    return {'inf': 'λ = ∞', 'inf, per length': 'λ = ∞, ÷ length'}.get(t, f'λ = {t}')


def versions():
    v = [('uniform (U)', 'ref', 'c', 'U'), ('per-KC counts (C)', 'ref', 'c', 'C')]
    v += [(f'λ = {l} µm', 'loc', 'l', f'local {l} µm') for l in LOC]
    v += [(lam_lab(k), 'cmp', 'c', k) for k in ['compartments inf', 'compartments inf, per length'] + [f'compartments {l} µm' for l in CL]]
    v += [(lam_lab(f'compartments {l}'), 'bo', 'c', f'boutons, compartments {l}') for l in ['inf', '50 µm']]
    v += [(lam_lab(f'compartments {l}'), 'kb', 'c', f'KC + boutons, compartments {l}') for l in ['inf', '50 µm']]
    return v


def rk(d, rn, src, key): return (d['SC'] if src == 'c' else d['SL'])[rn][RK][key]
def rr(d, rn, src, key): return d['SC'][rn][RC][key] if src == 'c' else d['SL'][rn][RL][key]
def ds(d, rn, src, key): return (d['SC'] if src == 'c' else d['SL'])[rn][DS][1][key]
def per(d, rn, src, key, mb, col): return (d['PC'] if src == 'c' else d['PL'])[rn].loc[(key, mb), col]
def nkey(d, rn): return next(k for k in d['SC'][rn] if k.startswith('null'))


GCOL = {'ref': L.GRAY, 'loc': L.PURPLE, 'cmp': COMP, 'bo': L.GREEN, 'kb': L.INK}
GNAME = {'ref': 'global APL', 'loc': 'local APL, no walls (Report 11)', 'cmp': 'APL confined to sub-arbors, inhibition on KCs',
         'bo': 'sub-arbors, inhibition on PN boutons only', 'kb': 'sub-arbors, KCs + boutons at the real bouton share'}
GSHORT = {'ref': 'global', 'loc': 'no walls', 'cmp': 'sub-arbors', 'bo': 'boutons\nonly', 'kb': 'KCs +\nboutons'}
DATA_RANK = {rn: d0['SC'][rn]['data: Spearman TNT vs TNT3c similarity over odor pairs'] for rn in RNS}
DATA_DS = {rn: d0['SC'][rn][DS][0] for rn in RNS}

R = L.Report('12', 'If signals inside APL cannot cross between its calyx sub-arbors (the compartments of Report 10), does the model '
             'reshape the KC representation, especially the rank order of odor-pair similarity with inhibition on versus off, more like '
             'TNT vs TNT3c than a uniform APL, an APL weighted by per-KC synapse counts, or the local APL without walls of Report 11?',
             inputs=['The pinned KC model of Report 07', 'APL skeletons and calyx sub-arbors (Report 10)',
                     'Calyx KC ↔ APL and APL → PN synapses with their positions (Reports 01, 02)', 'The local model of Report 11, as reference'],
             outputs=['`odor_weighted/APL_compartment_model_summary.json`, `_fits.csv`, `_per_MB_{ORN,GH146}.csv`, `_odor_pairs.csv`, `_nulls.csv`, '
                      '`_compartments.csv`', '`odor_weighted/APL_compartment_model.png`, `APL_compartment_model_notes.md`',
                      'the same in `with_outside_ROI_synapses/odor_weighted/`'],
             scripts=['`scripts/apl_compartment_model.py` (the model)', '`scripts/apl_compartment_model_figure.py`',
                      '`scripts/apl_compartment_model_notes.py`', '`scripts/reports/r12_prep.py` (compartment spread, matrix properties, positions)',
                      '`scripts/reports/r12_compartments.py` (this report)'])

R.p('Report 10 found that APL\'s calyx arbor is not one continuous tree in the calyx: it comes in several sub-arbors that join only farther '
    'down, through the peduncle or the lobes, so two synapses in different sub-arbors are 200–360 µm apart along APL even when they are close '
    'in space. Report 11 let APL\'s signal fade smoothly with distance along the skeleton. This report tries the other extreme suggested by '
    'that anatomy: each sub-arbor is a closed compartment, and a KC\'s activity can only inhibit KCs whose APL synapses sit in the same '
    'sub-arbor. Hemibrain R, male CNS L and R (Yang, 2026-10-21), both versions of the tables.')

R.symbols([
    ('k,\\ j', 'Kenyon cells (KCs); $k$ receives inhibition, $j$ excites APL', '$N$ = 1,793 (hemibrain R), 1,876 (male CNS L), 1,900 (male CNS R)'),
    ('o', 'An odor', '16 (ORN input) or 13 (GH146 input)'),
    ('x_{ko},\\ y_{ko}', 'Excitatory input and response of KC $k$ to odor $o$ (Report 07, pinned)', 'threshold units; ≥ 0'),
    ('\\theta', 'KC threshold', '1'),
    ('a', 'APL gain onto KCs, one per mushroom body and version, set so that the mean intact fraction responding equals TNT3c\'s', '> 0'),
    ('M_{kj}', 'Coupling: how strongly activity of KC $j$ inhibits KC $k$ through APL', 'mean over $k, j$ set to that of C'),
    ('c', 'A compartment: one connected calyx sub-arbor of APL (named S1, S2, … in Report 10; small pieces are compartments too)', ''),
    ('I_{cj}', 'KC → APL synapses from KC $j$ in compartment $c$ (APL\'s inputs)', 'count'),
    ('O_{ck}', 'APL → KC synapses onto KC $k$ in compartment $c$ (APL\'s outputs)', 'count'),
    ('L_c', 'Cable length of compartment $c$: summed skeleton length inside it', 'µm'),
    ('d(s, t)', 'Geodesic distance between sites $s$ and $t$ along APL\'s skeleton', 'µm'),
    ('\\lambda', 'Length constant of passive decay inside a compartment', '10–200 µm, or ∞ (no decay)'),
    ('\\kappa_v', 'Decay factor of the tree edge from node $v$ to its parent: $e^{-\\ell_v/\\lambda}$ inside a compartment, 0 on an edge that leaves one', '0 to 1'),
    ('\\Phi_{vj}', 'Signal from KC $j$ at skeleton node $v$', '≥ 0'),
    ('B_{kb},\\ Q_{bo},\\ \\beta', 'Bouton route of Report 11: share of KC $k$\'s claw input from bouton $b$; APL signal at bouton $b$; its divisive gain', ''),
    ('f', 'Boutons\' share of APL\'s calyx output synapses', '0.097 hemibrain, 0.112 male CNS'),
    ('S_{ab}', 'Cosine similarity of the KC population responses to odors $a$ and $b$', '0 to 1'),
    ('\\rho_{\\mathrm{rank}}', 'Spearman correlation over odor pairs between blocked and intact $S_{ab}$, mean over the three mushroom bodies', '−1 to 1; lower = more reordering'),
    ('r', 'Pearson correlation over odor pairs between the model\'s $\\Delta S_{ab}$ (blocked − intact, averaged over mushroom bodies) and the data\'s (TNT − TNT3c)', '−1 to 1'),
    ('\\pi', 'A random permutation of KCs within their class (αβ, α′β′, γ), for the null', ''),
])

# ---------------------------------------------------------------- 1 compartments
R.section('The compartments: APL\'s calyx sub-arbors')
R.p('The compartments are exactly the sub-arbors of Report 10. Skeleton nodes within 5 µm of a calyx APL synapse (from the table with '
    'synapses outside the official ROIs added, which covers the whole calyx) are calyx nodes. Keeping only those nodes and the skeleton edges '
    'between two of them splits APL\'s calyx arbor into connected pieces, because the pieces are joined only through nodes outside the calyx. '
    'Each piece is one compartment. Pieces holding at least 2% of APL\'s synapses on microglomeruli keep Report 10\'s names S1, S2, …; the '
    'rest are small pieces, and each small piece is a compartment of its own. Every calyx KC ↔ APL synapse is placed at its nearest skeleton '
    'node and so belongs to exactly one compartment (none fell outside a piece).')


def fig_map(d):
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.9))
    for ax, k in zip(axs, MBS):
        p = d['POS'][d['POS'].mushroom_body == k]
        nms = sorted([x for x in p.compartment.unique() if x.startswith('S')], key=lambda x: int(x[1:]))
        sp = p[p.compartment == 'small piece']
        ax.scatter(sp.x_um, sp.y_um, s=1.2, color=L.GRID, lw=0, rasterized=True)
        for i, n in enumerate(nms):
            q = p[p.compartment == n]
            ax.scatter(q.x_um, q.y_um, s=1.2, color=PAL[i % len(PAL)], lw=0, rasterized=True)
            ax.scatter([], [], s=22, color=PAL[i % len(PAL)], label=n)
        ax.scatter([], [], s=22, color=L.GRID, label='small pieces')
        ax.set_aspect('equal'); ax.set_title(MBN[k]); ax.set_xlabel('fly right → (µm, posterior view)')
        ax.legend(fontsize=6.8, loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.2, borderaxespad=0)
    axs[0].set_ylabel('dorsal ↑ (µm)')
    fig.tight_layout(); return fig


R.fig(fig_map(d0), 'Calyx KC ↔ APL synapses (a random 8,000 per mushroom body, official ROIs), each drawn at the APL skeleton node it was '
      'placed on, viewed from posterior (x: µm toward the fly\'s right; y: µm toward dorsal; hemibrain and male CNS have different origins, '
      'so only distances within a panel compare). Color: the compartment (sub-arbor) of that node, named and colored as in Report 10; '
      'light gray: small pieces. Neighboring colors overlap in this 2-D view because the sub-arbors interleave in depth.',
      'APL\'s calyx compartments.')

ct = d0['CT']; sp = d0['SP']


def fig_spread(d):
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    for i, k in enumerate(MBS):
        c = d['CT'][d['CT'].mushroom_body == k].copy()
        nm = d['NM'][k]; c['name'] = [nm.get(str(x), 'small pieces') for x in c.compartment]
        g = c.groupby('name').share_APL_to_KC.sum()
        order = sorted([x for x in g.index if x.startswith('S')], key=lambda x: int(x[1:])) + (['small pieces'] if 'small pieces' in g.index else [])
        left = 0
        for n in order:
            col = PAL[(int(n[1:]) - 1) % len(PAL)] if n.startswith('S') else L.GRID
            ax.barh(i, g[n] * 100, left=left, color=col, height=0.62, edgecolor='white', lw=1)
            if g[n] > 0.06: ax.text(left + g[n] * 50, i, n, ha='center', va='center', fontsize=7, color='white')
            left += g[n] * 100
    ax.set_yticks(range(3), [MBN[k] for k in MBS]); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel('share of calyx APL → KC synapses (%)'); L.tag(ax, 'a', x=-0.25)
    ax = axs[1]
    bins = np.linspace(0, 1, 21)
    for k in MBS:
        x = d['SP'][d['SP'].mushroom_body == k].main_share
        ax.hist(x, bins=bins, histtype='step', lw=1.5, color=MBC[k], weights=np.full(len(x), 100 / len(x)), label=MBN[k])
    ax.set_xlabel('share of a KC\'s calyx APL synapses (both directions)\nin its main compartment'); ax.set_ylabel('KCs (% per 0.05 bin)')
    ax.legend(fontsize=7.5, loc='upper left'); L.tag(ax, 'b', x=-0.2)
    fig.tight_layout(); return fig


R.fig(fig_spread(d0), '**a**, how APL\'s calyx output onto KCs (APL → KC synapses) divides among the compartments of each mushroom body '
      '(official ROIs); colors and names as in Figure 1, light gray = all small pieces together. **b**, for each KC, the share of its calyx '
      'synapses with APL (KC → APL and APL → KC together) that lie in the one compartment holding most of them; histogram over KCs with at '
      'least one such synapse, lines = mushroom bodies, y in % of those KCs per 0.05 bin. A value of 1 means all of a KC\'s APL synapses are '
      'in one compartment.', 'How APL\'s output and each KC\'s synapses divide among compartments.')
rows = []
for k in MBS:
    c = ct[ct.mushroom_body == k]; s = sp[sp.mushroom_body == k]
    rows.append([MBN[k], len(c), sum(1 for x in d0['NM'][k].values()), F(1 / np.sum(c.share_APL_to_KC ** 2), 1),
                 L.pct(c.share_APL_to_KC.max()), F(np.median(s.main_share), 2), L.pct((s.main_share == 1).mean())])
R.table(['Mushroom body', 'compartments with KC synapses', 'named (≥ 2%)', 'effective number', 'largest share of APL → KC',
         'KC main-compartment share, median', 'KCs entirely in one compartment'], rows, title='Compartments, official ROIs.',
        caption='Effective number = $1/\\sum_c s_c^2$ with $s_c$ the compartment\'s share of APL → KC synapses (equal to the count if all were '
        'equal). The last two columns use each KC\'s synapses with APL in both directions.', num_cols=[1, 2, 3, 4, 5, 6])
hb_eff = 1 / np.sum(ct[ct.mushroom_body == 'hemibrain R'].share_APL_to_KC ** 2)
R.p(f'Hemibrain R has one dominant compartment (effective number {F(hb_eff, 1)}), so walls change little there; male CNS L is split most '
    'evenly. Most KCs have their APL synapses in more than one compartment, because a KC\'s claws are spread over the calyx, but usually with '
    f'a clear main one (median share {", ".join(F(np.median(sp[sp.mushroom_body == k].main_share), 2) + " in " + MBN[k] for k in MBS)}).')

# ---------------------------------------------------------------- 2 model
R.section('The model, step by step')
R.sub('Step 1: KC input and pinning (unchanged)')
R.p('As in Reports 07 and 11: each KC\'s input is its claws\' PN synapse shares times the glomerular ORN (or GH146) responses; each odor\'s '
    'input gain is set so that with APL blocked the fraction of KCs responding equals that odor\'s TNT fraction; the blocked responses '
    '$y^{\\mathrm{blocked}}_{ko} = [x_{ko} - 1]_+$ are the same for every version.')
R.sub('Step 2: feedback through a coupling matrix')
R.eq('y_{ko} \\;=\\; \\Big[\\, x_{ko} - \\theta - a \\sum_{j} M_{kj}\\, y_{jo} \\Big]_+',
     'The steady state of KC responses with APL intact, solved per odor. Versions differ only in $M$.')
R.p('**Isopotential compartments** ($\\lambda = \\infty$). Inside one compartment the APL potential is the same everywhere; nothing passes '
    'between compartments. KC $j$ raises the potential of compartment $c$ in proportion to its KC → APL synapses there, $I_{cj}$, and KC $k$ is '
    'inhibited by that potential at its APL → KC synapses there, $O_{ck}$:')
R.eq('M_{kj} \\;=\\; \\mathrm{const} \\times \\sum_{c} O_{ck}\\, I_{cj}',
     'A sum over compartments of per-compartment "C" matrices. With a single compartment this is exactly C.')
R.p('**Isopotential, divided by cable length** ("÷ length"). A passive compartment of membrane area $A$ turns an input current into a voltage '
    'change in proportion to its input resistance, which scales as $1/A$. Taking area proportional to cable length $L_c$ (same branch '
    'thickness), the same input moves a small compartment\'s potential more than a large one\'s:')
R.eq('M_{kj} \\;=\\; \\mathrm{const} \\times \\sum_{c} \\frac{O_{ck}\\, I_{cj}}{L_c}', '$L_c$ in µm; a piece with no edge counts as 1 µm.')
R.p('**Decay inside compartments** ($\\lambda$ = 200, 100, 50, 25, 10 µm). The local model of Report 11, with every skeleton edge that leaves '
    'a compartment cut:')
R.eq('M_{kj} \\;=\\; \\mathrm{const} \\times \\sum_{s \\,\\in\\, \\mathrm{out}(k)} \\; \\sum_{t \\,\\in\\, \\mathrm{in}(j)} '
     '\\mathbf{1}[\\,c(s) = c(t)\\,]\\; e^{-d(s,t)/\\lambda}',
     '$\\mathrm{out}(k)$: APL → KC sites on KC $k$; $\\mathrm{in}(j)$: KC → APL sites of KC $j$; $c(s)$: compartment of site $s$.')
R.p('In each version the constant sets the mean of $M$ to that of C; the gain calibration (Step 4) would absorb it anyway.')
R.sub('Step 3: computing M on the tree with walls')
R.p('The two-pass computation of Report 11 (pass 1 from the leaves up, pass 2 from the root down, each edge $v$ weighted by $\\kappa_v$) gives '
    '$\\Phi_{vj} = \\sum_t e^{-d(v,t)/\\lambda}$ exactly on a tree. Walls are added by setting $\\kappa_v = 0$ on every edge whose two ends are '
    'not in the same compartment; a signal can then never cross that edge. Because the path between two nodes of one connected piece stays '
    'inside the piece, an edge of the reduced tree that joins two nodes of the same compartment never leaves it, so this test on the reduced '
    'tree is exact. For $\\lambda = \\infty$ ($\\kappa_v$ = 1 inside), the result was compared with $\\sum_c O_c^{\\top} I_c$ computed directly: '
    + ('identical in all three mushroom bodies (difference exactly 0).' if all(d0["SC"]["checks"][k]["cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference"] == 0 for k in MBS)
       else 'largest relative difference ' + ', '.join(f'{d0["SC"]["checks"][k]["cut kernel at lambda = inf vs sum_c O_c^T I_c: max relative difference"]:.0e}' for k in MBS) + '.'))
R.sub('Step 4: gains, solver, bouton route')
R.p('For every mushroom body, input table and version, the gain $a$ is found by bisection so that the mean fraction of KCs responding '
    'equals TNT3c\'s (6.83% for the 16 odors), with the exact active-set solver of Report 11. The bouton route of Report 11 is kept, with the '
    'APL signal at each bouton taken from the same walled computation ($Q_{bo}$ from $\\Phi$ at its APL → PN synapses): **boutons only** '
    '(all inhibition on boutons) and **KCs + boutons** at the boutons\' real share $f$ of APL\'s calyx output, for $\\lambda$ = ∞ and 50 µm.')
R.sub('Step 5: readouts and null')
R.ul(['**Rank order** $\\rho_{\\mathrm{rank}}$, **pattern** $r$ and **size** (mean ΔS), exactly as in Report 11. Data: '
      f'$\\rho_{{\\mathrm{{rank}}}}$ = {F(DATA_RANK["ORN"], 3)} (ORN odors) and {F(DATA_RANK["GH146"], 3)} (GH146 odors), about {CORR} after '
      f'correcting for recording noise; $r$ can reach about {CEIL} at most; mean ΔS {F(DATA_DS["ORN"], 3)} and {F(DATA_DS["GH146"], 3)}.',
      '**Null**: for the isopotential version (∞ or ∞ ÷ length) with the better $r$, KCs swap their APL synapse sites with other KCs of the '
      'same class (KC $k$ gets KC $\\pi(k)$\'s sites, $M \\to M_{\\pi(k)\\pi(j)}$), 100 times, recalibrating each time. This asks whether it '
      'matters which KC sits in which compartment. One-sided $p$ = (1 + shuffles at least as extreme) / 101.'])

# ---------------------------------------------------------------- 3 coupling
R.section('What walls do to the coupling')
mp = d0['MP']
MV = ['C', 'compartments inf', 'compartments inf, per length', 'local 100 µm', 'compartments 100 µm', 'local 25 µm', 'compartments 25 µm',
      'local 10 µm', 'compartments 10 µm']
MVL = {'C': 'C', 'compartments inf': 'sub-arbors ∞', 'compartments inf, per length': 'sub-arbors ∞ ÷ length', 'local 100 µm': 'no walls 100 µm',
       'compartments 100 µm': 'sub-arbors 100 µm', 'local 25 µm': 'no walls 25 µm', 'compartments 25 µm': 'sub-arbors 25 µm',
       'local 10 µm': 'no walls 10 µm', 'compartments 10 µm': 'sub-arbors 10 µm'}


def fig_M(mp):
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.6))
    x = np.arange(len(MV))
    for ax, col, sc, yl, t in [(axs[0], 'effective_inhibitors_frac_median', 100, 'effective number of KCs inhibiting\na KC, median (% of all KCs)', 'a'),
                               (axs[1], 'zero_coupling_pairs', 100, 'KC pairs with no coupling\n($M_{kj}$ = 0, % of all pairs)', 'b'),
                               (axs[2], 'r_with_C', 1, 'correlation of $M$ with C\n(over KC pairs)', 'c')]:
        for k in MBS:
            y = [mp[(mp.mushroom_body == k) & (mp.version == v)][col].iloc[0] * sc for v in MV]
            ax.scatter(x, y, marker=MK[k], s=22, facecolor='white', edgecolor=[COMP if v.startswith('comp') else (L.PURPLE if v.startswith('local') else L.GRAY) for v in MV],
                       lw=1.0, zorder=3, label=MBN[k])
        ax.set_xticks(x, [MVL[v] for v in MV], rotation=90, fontsize=7); ax.set_ylabel(yl); L.tag(ax, t, x=-0.3)
        ax.yaxis.grid(True, color=L.GRID, lw=0.6); ax.set_axisbelow(True)
    h = [Line2D([], [], marker=MK[k], ls='', mfc='white', mec=L.INK, label=MBN[k]) for k in MBS]
    axs[0].legend(handles=h, fontsize=7, loc='upper right')
    fig.tight_layout(); return fig


R.fig(fig_M(mp), 'Properties of the coupling matrix $M$ (official ROIs), one marker per mushroom body. x-axis: versions; gray = C (one '
      'APL), purple = local without walls (Report 11), gold = confined to sub-arbors, at the same λ side by side. **a**, for each KC $k$ the '
      'participation ratio $(\\sum_j M_{kj})^2 / \\sum_j M_{kj}^2$, the number of KCs that would contribute equally to its inhibition, as a % '
      'of all KCs; median over KCs. **b**, % of all KC pairs $(k, j)$ whose coupling is exactly zero: with walls, two KCs with no '
      'compartment in common cannot inhibit each other. **c**, Pearson correlation of $M$ with C over KC pairs. From '
      '`reports/data/r12_M_properties.csv`.', 'Walls cut couplings between compartments; at short λ they change nothing.')
g = lambda k, v, c: mp[(mp.mushroom_body == k) & (mp.version == v)][c].iloc[0]
R.p('With isopotential compartments ' + ', '.join(f'{L.pct(g(k, "compartments inf", "zero_coupling_pairs"))} of KC pairs in {MBN[k]}' for k in MBS)
    + ' cannot interact at all, yet a typical KC is still inhibited by '
    + ', '.join(L.pct(g(k, 'compartments inf', 'effective_inhibitors_frac_median')) for k in MBS)
    + ' of all KCs (same order), because its synapses lie in the big compartments. At λ = 10 µm the walled and unwalled matrices are '
    'practically the same (the signal has faded long before it would reach another compartment, 200–360 µm away), so from about 25 µm down '
    'this model and Report 11\'s coincide.')

# ---------------------------------------------------------------- 4 results
R.section('Results: rank order and pattern')


def draw(ax, d, rn, kind):
    VV = versions(); x = []; pos = 0; prev = None; vals = []
    for lab, g_, src, key in VV:
        if prev is not None and g_ != prev: pos += 1.1
        x.append(pos); prev = g_; pos += 1
    for xi, (lab, g_, src, key) in zip(x, VV):
        for k in MBS:
            val = per(d, rn, src, key, k, 'spearman_blocked_vs_intact' if kind == 'rank' else 'r_dS_with_data'); vals.append(val)
            ax.scatter(xi, val, marker=MK[k], s=13, facecolor='white', edgecolor=GCOL[g_], lw=0.9, zorder=3)
        m = rk(d, rn, src, key) if kind == 'rank' else rr(d, rn, src, key)
        ax.plot([xi - 0.36, xi + 0.36], [m, m], color=GCOL[g_], lw=2.4, zorder=4, solid_capstyle='butt')
    ax.set_xticks(x, [v[0] for v in VV], rotation=90, fontsize=6.6)
    grp = {}
    for xi, v in zip(x, VV): grp.setdefault(v[1], []).append(xi)
    for g_, xs in grp.items():
        ax.text(np.mean(xs), -0.42, GSHORT[g_], transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.8, color=GCOL[g_],
                fontweight='semibold')
    if kind == 'rank':
        ax.axhline(DATA_RANK[rn], color=L.BLOCKED, lw=1.2); ax.axhline(CORR, color=L.BLOCKED, lw=1.2, ls='--')
        ax.text(x[0] - 0.5, DATA_RANK[rn], f'data {F(DATA_RANK[rn], 3)}', fontsize=6.8, color=L.BLOCKED, va='bottom')
        ax.text(x[0] - 0.5, CORR, f'data, noise-corrected ≈ {CORR}', fontsize=6.8, color=L.BLOCKED, va='bottom')
        ax.set_ylabel('$\\rho_{\\mathrm{rank}}$: Spearman, blocked vs\nintact similarity over odor pairs'); ax.set_ylim(min(0.85, min(vals) - 0.01), 1.0)
    else:
        ax.axhline(CEIL, color=L.BLOCKED, lw=1.2, ls='--'); ax.axhline(0, color=L.GRID, lw=0.8)
        ax.text(x[-1] + 0.5, CEIL, 'data ceiling ≈ 0.84', fontsize=6.8, color=L.BLOCKED, va='bottom', ha='right')
        ax.set_ylabel('$r$: model ΔS vs data ΔS\nover odor pairs'); ax.set_ylim(min(0, min(vals) - 0.04), 0.9)
    ax.set_xlim(-0.8, x[-1] + 0.8)
    ax.yaxis.grid(True, color=L.GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f'{rn} input')


def fig_results(d):
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.4))
    for j, rn in enumerate(RNS):
        draw(axs[0, j], d, rn, 'rank'); draw(axs[1, j], d, rn, 'r')
    h = [Line2D([], [], color=GCOL[g_], lw=2.4, label=GNAME[g_]) for g_ in GCOL]
    h += [Line2D([], [], marker=MK[k], ls='', mfc='white', mec=L.INK, label=MBN[k]) for k in MBS]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    for ax, t in zip(axs.ravel(), 'abcd'): L.tag(ax, t, x=-0.15)
    fig.tight_layout(h_pad=3.0); return fig


R.fig(fig_results(d0), 'Official ROIs. x-axis: model versions in five groups (colors, named in the legend and under the axis): one global APL '
      '(U, C); the local APL without walls of Report 11 (λ from 200 to 2 µm); APL confined to sub-arbors with inhibition on KCs (λ = ∞, '
      'λ = ∞ with each compartment\'s potential divided by its cable length, then λ = 200 to 10 µm inside compartments); confined to '
      'sub-arbors with inhibition on PN boutons only; confined, with KCs and boutons at the real bouton share. Open markers: the three '
      'mushroom bodies; bar: their mean (in **c**–**d** the bar is $r$ of the similarity change averaged over mushroom bodies, so it can lie '
      'above all three markers). **a**–**b**, rank order $\\rho_{\\mathrm{rank}}$, lower = more reordering; orange lines: data, raw (solid) '
      'and noise-corrected (dashed). **c**–**d**, pattern $r$; orange dashed: the data\'s ceiling. Left: ORN input (16 odors, 120 pairs); '
      'right: GH146 input (13 odors, 78 pairs).', 'Confining APL to its sub-arbors does not reorder the pairs more.')

rows = []
SEL = [('U', 'c'), ('C', 'c'), ('compartments inf', 'c'), ('compartments inf, per length', 'c'), ('compartments 50 µm', 'c'),
       ('local 50 µm', 'l'), ('compartments 10 µm', 'c'), ('local 10 µm', 'l'), ('local 2 µm', 'l'),
       ('boutons, compartments inf', 'c'), ('KC + boutons, compartments inf', 'c')]
NICE = {'U': 'uniform (U)', 'C': 'per-KC counts (C)', 'compartments inf': 'sub-arbors, isopotential (λ = ∞)',
        'compartments inf, per length': 'sub-arbors, isopotential ÷ cable length', 'compartments 50 µm': 'sub-arbors, λ = 50 µm inside',
        'local 50 µm': 'no walls, λ = 50 µm (Report 11)', 'compartments 10 µm': 'sub-arbors, λ = 10 µm inside',
        'local 10 µm': 'no walls, λ = 10 µm (Report 11)', 'local 2 µm': 'no walls, λ = 2 µm (Report 11)',
        'boutons, compartments inf': 'sub-arbors, boutons only (λ = ∞)', 'KC + boutons, compartments inf': 'sub-arbors, KCs + boutons (λ = ∞)'}
for rn in RNS:
    for key, src in SEL:
        rows.append([rn, NICE[key], F(rk(d0, rn, src, key), 3), F(rr(d0, rn, src, key), 3), F(ds(d0, rn, src, key), 3)])
    rows.append([rn, '**data** (TNT vs TNT3c)', f'**{F(DATA_RANK[rn], 3)}** (≈ {CORR} noise-corrected)', f'ceiling ≈ {CEIL}', f'**{F(DATA_DS[rn], 3)}**'])
R.table(['Input', 'Version', '$\\rho_{\\mathrm{rank}}$ (mean of MBs)', 'pattern $r$', 'mean ΔS'], rows, title='Main numbers, official ROIs.',
        caption='Selected versions; all are in `APL_compartment_model_summary.json` and, for the versions without walls, '
        '`APL_local_model_summary.json`.', num_cols=[2, 3, 4])


def iso_best(d, rn): return max(['compartments inf', 'compartments inf, per length'], key=lambda v: rr(d, rn, 'c', v))


def answer(d, vname):
    nb = {rn: d['SC'][rn][nkey(d, rn)] for rn in RNS}
    rk_iso = {rn: [rk(d, rn, 'c', v) for v in ['compartments inf', 'compartments inf, per length'] + [f'compartments {l} µm' for l in CL]] for rn in RNS}
    best_noW = {rn: max([f'local {l} µm' for l in LOC], key=lambda v: rr(d, rn, 'l', v)) for rn in RNS}
    pv = {rn: nb[rn]['r real, null mean, null 5-95%, p'][3] for rn in RNS}
    return [
        '**Rank order: no.** Confined to sub-arbors, inhibition on KCs leaves $\\rho_{\\mathrm{rank}}$ at '
        + ' and '.join(f'{F(min(rk_iso[rn]), 3)}–{F(max(rk_iso[rn]), 3)} ({rn})' for rn in RNS)
        + f', next to U ({F(rk(d, "ORN", "c", "U"), 3)}, {F(rk(d, "GH146", "c", "U"), 3)}) and C ({F(rk(d, "ORN", "c", "C"), 3)}, '
        f'{F(rk(d, "GH146", "c", "C"), 3)}), and far above the data\'s ≈ {CORR}. Inhibition on boutons only, confined to sub-arbors, reaches '
        + ', '.join(f'{F(rk(d, rn, "c", "boutons, compartments inf"), 3)} ({rn})' for rn in RNS) + ' at λ = ∞, against '
        + ', '.join(F(d['S2'][rn]['Spearman blocked vs intact, mean over MBs']['boutons inf'], 3) for rn in RNS)
        + ' without walls (Report 11): the walls do not make it reorder more. With KCs and boutons at the real share it is back to '
        + ', '.join(f'{F(rk(d, rn, "c", "KC + boutons, compartments inf"), 3)}' for rn in RNS) + '.',
        '**Pattern: better than one APL only when small compartments count more, and never better than without walls.** Isopotential '
        'compartments give $r$ = ' + ', '.join(f'{F(rr(d, rn, "c", "compartments inf"), 3)} ({rn})' for rn in RNS)
        + ', no better than C; dividing each compartment\'s potential by its cable length gives '
        + ', '.join(f'{F(rr(d, rn, "c", "compartments inf, per length"), 3)}' for rn in RNS)
        + f', against U {F(rr(d, "ORN", "c", "U"), 3)} / {F(rr(d, "GH146", "c", "U"), 3)}, C {F(rr(d, "ORN", "c", "C"), 3)} / {F(rr(d, "GH146", "c", "C"), 3)} '
        'and the best local version without walls ' + ', '.join(f'{F(rr(d, rn, "l", best_noW[rn]), 3)} ({best_noW[rn].replace("local ", "λ = ")}, {rn})' for rn in RNS)
        + '. With decay inside compartments $r$ rises as λ shrinks, but only to the values the same λ gives without walls.',
        '**Placement within compartments:** shuffling KCs within class for ' + ', '.join(f'{rn} ({lam_lab(nkey(d, rn).split("version ")[-1])})' for rn in RNS)
        + ' gives ' + '; '.join(f'{rn}: $r$ {F(nb[rn]["r real, null mean, null 5-95%, p"][0], 3)} vs shuffles {F(nb[rn]["r real, null mean, null 5-95%, p"][1], 3)}, '
                                f'$p$ = {F(pv[rn], 2)}' for rn in RNS) + '.',
        '**Size: no.** Mean ΔS of the KC-route compartment versions is '
        + ', '.join(f'{L.pct(min(ds(d, rn, "c", v) for v in ["compartments inf", "compartments inf, per length"] + [f"compartments {l} µm" for l in CL]) / DATA_DS[rn])}–'
                    f'{L.pct(max(ds(d, rn, "c", v) for v in ["compartments inf", "compartments inf, per length"] + [f"compartments {l} µm" for l in CL]) / DATA_DS[rn])} of the data\'s ({rn})' for rn in RNS)
        + ', smaller than U\'s.']


R.box('Answer, official ROIs', answer(d0, 'official'), 'result')


def fig_walls(d):
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 5.6), sharex=True)
    xl = [200, 100, 50, 25, 10]; xs = np.arange(len(xl) + 1)
    for j, rn in enumerate(RNS):
        for i, (kind, f_) in enumerate([('rank', rk), ('r', rr)]):
            ax = axs[i, j]
            yw = [f_(d, rn, 'c', 'compartments inf')] + [f_(d, rn, 'c', f'compartments {l} µm') for l in xl]
            yn = [f_(d, rn, 'c', 'C')] + [f_(d, rn, 'l', f'local {l} µm') for l in xl]
            ax.plot(xs, yn, marker='o', color=L.PURPLE, label='no walls (Report 11; λ = ∞ is C)')
            ax.plot(xs, yw, marker='s', color=COMP, label='walls between sub-arbors')
            ax.axhline(f_(d, rn, 'c', 'U'), color=L.GRAY, ls='--', lw=1, label='uniform (U)')
            ax.set_xticks(xs, ['∞'] + [str(l) for l in xl]); ax.yaxis.grid(True, color=L.GRID, lw=0.6)
            if i == 1: ax.set_xlabel('length constant λ (µm)')
            ax.set_ylabel('$\\rho_{\\mathrm{rank}}$ (mean of MBs)' if kind == 'rank' else 'pattern $r$')
            ax.set_title(f'{rn} input'); L.tag(ax, 'abcd'[2 * i + j], x=-0.2)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout(); return fig


R.fig(fig_walls(d0), 'The same length constant with and without walls (official ROIs). x-axis: λ, from ∞ (left; without walls this is C, with '
      'walls the isopotential compartments) to 10 µm. Purple circles: Report 11\'s local model; gold squares: the same with no signal crossing '
      'between sub-arbors; gray dashed: uniform APL. **a**–**b**, rank order $\\rho_{\\mathrm{rank}}$ (mean of the three mushroom bodies); '
      '**c**–**d**, pattern $r$ (MB-averaged change). Left: ORN input; right: GH146 input.',
      'Walls versus no walls at the same length constant.')
def _wdiff(d, f_, lams):
    return [f_(d, rn, 'c', f'compartments {l} µm') - f_(d, rn, 'l', f'local {l} µm') for rn in RNS for l in lams if l != 'inf'] + \
        ([f_(d, rn, 'c', 'compartments inf') - f_(d, rn, 'c', 'C') for rn in RNS] if 'inf' in lams else [])


_lr, _lk = _wdiff(d0, rr, ['200', '100', '50', 'inf']), _wdiff(d0, rk, ['200', '100', '50', 'inf'])
_sr = _wdiff(d0, rr, ['10'])
R.p('The walls matter only where the signal would otherwise reach another sub-arbor, at long λ (∞ to 50 µm). There they change '
    f'$\\rho_{{\\mathrm{{rank}}}}$ by {F(min(_lk), 3, True)} to {F(max(_lk), 3, True)} and $r$ by {F(min(_lr), 3, True)} to {F(max(_lr), 3, True)} '
    f'(mostly lowering it, most for GH146). At λ = 10 µm walled and unwalled $r$ '
    + ('agree to three decimals.' if max(abs(x) for x in _sr) < 0.0005 else f'differ by at most {F(max(abs(x) for x in _sr), 3)}.'))


def fig_null(d):
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 2.8))
    for j, rn in enumerate(RNS):
        nk = nkey(d, rn); best = nk.split('version ')[-1]
        n = d['NU'][d['NU'].responses == rn]
        real_r = d['SC'][rn][nk]['r real, null mean, null 5-95%, p'][0]
        real_k = d['SC'][rn][nk]['rank Spearman real, null mean, null 5-95%, p (lower = more reordering)'][0]
        for ax, col, real, xl in [(axs[2 * j], 'r_dS_with_data', real_r, 'pattern $r$ with the data'),
                                  (axs[2 * j + 1], 'rank_spearman_mean_of_MBs', real_k, '$\\rho_{\\mathrm{rank}}$ (mean of MBs)')]:
            ax.hist(n[col], bins=20, color=L.GRAY, alpha=0.8)
            ax.axvline(real, color=COMP, lw=2); ax.set_xlabel(xl); ax.set_ylabel('shuffles (of 100)')
            ax.xaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_title(f'{rn}, {lam_lab(best)}', fontsize=8.5)
    fig.tight_layout(); return fig


R.fig(fig_null(d0), 'Gray: 100 shuffles in which KCs swap their APL synapse sites (and so their compartments) with other KCs of the same '
      'class, in all three mushroom bodies at once; gold line: the real wiring. Shown for the isopotential version with the better pattern $r$ '
      '(panel titles). Left pair: ORN input; right pair: GH146 input; in each pair the pattern $r$ (higher = closer to the data), then the rank '
      'order (lower = more reordering).', 'Does it matter which KC sits in which compartment?')

# ---------------------------------------------------------------- 5 second version
R.section('Second version: synapses outside the official ROIs added')
R.p('The same model on the tables of `with_outside_ROI_synapses/`, which mostly adds the back of the male CNS calyx (Report 02). The '
    'compartments are the same skeleton pieces; more synapses fall into them.')
R.fig(fig_results(d1), 'As Figure 4, with the tables of `with_outside_ROI_synapses/`.', 'Rank order and pattern, second version.')
rows = []
for rn in RNS:
    for key, src in SEL:
        rows.append([rn, NICE[key], F(rk(d0, rn, src, key), 3), F(rk(d1, rn, src, key), 3), F(rr(d0, rn, src, key), 3), F(rr(d1, rn, src, key), 3)])
R.table(['Input', 'Version', '$\\rho_{\\mathrm{rank}}$ official', '$\\rho_{\\mathrm{rank}}$ second version', '$r$ official', '$r$ second version'], rows,
        title='Official ROIs vs the second version.', caption='As Table 3.', num_cols=[2, 3, 4, 5])
R.box('Answer, second version', answer(d1, 'outside'), 'result')

# ---------------------------------------------------------------- 6
R.section('What this means, and caveats')
R.ul(['Confining APL\'s signal to its calyx sub-arbors does **not** make inhibition reorder odor pairs more. The rank order stays at '
      'about 0.98, like a uniform APL. In this model, inhibition that acts on the KCs, whether global, local or split into compartments, '
      'mostly rescales KC responses and keeps the order of similarities (Report 09 explains when that happens).',
      'For **which** pairs change most, walls alone do not help: isopotential compartments match the data no better than one APL. Only '
      'when each compartment\'s potential is divided by its cable length, so that small compartments respond more strongly, does $r$ rise '
      'above U and C, and for GH146 input that version beats within-class shuffles in both versions of the tables (for ORN it does not). '
      'It still stays below Report 11\'s local model without walls at short λ, and at short λ the walls are irrelevant because the signal '
      'fades before reaching another sub-arbor.',
      'The sub-arbors are uneven: hemibrain R is nearly one compartment, so most of any effect would come from male CNS. Most KCs have synapses '
      'in more than one compartment, which spreads each KC\'s inhibition over several compartments and blurs the walls.'])
R.box('Caveats', ['The sub-arbors come from the skeletons\' connectivity. A connection missed by the segmentation, or a false merge, would '
                  'change them (Report 10); worth checking in neuroglancer before relying on them.',
                  'Every small piece is treated as its own compartment. They hold 1–4% of APL\'s synapses, so pooling them differently would '
                  'change little.',
                  '"÷ length" assumes the same branch thickness everywhere; real input resistance also depends on thickness and on the '
                  'connections through the peduncle, which the walls ignore.',
                  'Synapse counts stand in for strength; a steady state is assumed; the shuffle $p$ values do not correct for choosing '
                  'between the two isopotential versions.'], 'caveat')
R.save()
