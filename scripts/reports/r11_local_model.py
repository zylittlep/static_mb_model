"""Report 11: the local, non-spiking APL model (apl_local_model.py, apl_local_boutons.py), step by step.
Rebuilt 2026-11-04 for Yang's revision: main results use the corrected calyx, the new claws and the new ORN table (claws_v2m2); the
earlier official-ROI version is compared at the end. Adds: why theta = 1, every variable of Equation 3 with a mock tree, and what the
C matrix is. Old version in archive_2026-11-03/."""
import sys, json, os
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import report_lib as L
L.style(); L.DATE = '2026-11-04'
F = L.fmt
OWS = {'main': L.B + 'with_outside_ROI_synapses/claws_v2m2/odor_weighted/', 'earlier': L.B + 'odor_weighted/'}
DD = L.B + 'reports/data/'
MBS = ['hemibrain R', 'malecns L', 'malecns R']; MBN = {'hemibrain R': 'hemibrain R', 'malecns L': 'male CNS L', 'malecns R': 'male CNS R'}
MBC = {'hemibrain R': L.DS_COLOR['hemibrain'], 'malecns L': L.DS_COLOR['malecns'], 'malecns R': '#7ccfa9'}
MK = {'hemibrain R': 'o', 'malecns L': 's', 'malecns R': '^'}
RNS = ['ORN', 'GH146']
RO = pd.read_csv(OWS['main'] + 'TNT_rank_order.csv'); _ro = RO[(RO.source == 'data') & (RO.responses == 'raw responses')].set_index('comparison').spearman
CORR = round(float(_ro['half vs half, TNT vs TNT3c'] / np.sqrt(_ro['half vs half, within TNT3c (mean of 1000 splits)'] * _ro['half vs half, within TNT'])), 2)
TVM = json.load(open(OWS['main'] + 'TNT_vs_model_summary.json'))
CEIL = round(TVM['fits']['threshold 1.0, ORN']['ceiling for r with a model (sqrt of that)'], 2)
NKC = TVM['model']['mushroom bodies (KCs)']
FSH = json.load(open(OWS['main'] + 'APL_block_model_summary.json'))['bouton_share_of_APL_calyx_output_synapses']
LAMS = ['200', '100', '50', '25', '10', '5', '2']; LB = ['inf', '50', '25', '10', '5']


def load(ver):
    o = OWS[ver]
    S1 = json.load(open(o + 'APL_local_model_summary.json')); S2 = json.load(open(o + 'APL_local_boutons_summary.json'))
    P1 = {rn: pd.read_csv(o + f'APL_local_model_per_MB_{rn}.csv').set_index(['version', 'mushroom_body']) for rn in RNS}
    P2 = {rn: pd.read_csv(o + f'APL_local_boutons_per_MB_{rn}.csv').set_index(['version', 'mushroom_body']) for rn in RNS}
    NU = pd.read_csv(o + 'APL_local_model_nulls.csv'); PA = pd.read_csv(o + 'APL_local_model_odor_pairs.csv')
    return dict(S1=S1, S2=S2, P1=P1, P2=P2, NU=NU, PA=PA)


V = {v: load(v) for v in OWS}
TAG = '_claws_v2m2'
MP = {'main': pd.read_csv(DD + f'r11_M_properties{TAG}.csv'), 'earlier': pd.read_csv(DD + 'r11_M_properties.csv')}
CK = {'main': json.load(open(DD + f'r11_checks{TAG}.json')), 'earlier': json.load(open(DD + 'r11_checks.json'))}


def versions(d, rn):
    """(label, group, source, key) for the versions present in the outputs."""
    have1 = d['S1'][rn]['model: Spearman blocked vs intact (mean over MBs) by version']; have2 = d['S2'][rn]['Spearman blocked vs intact, mean over MBs']
    v = [('uniform (U)', 'ref', 1, 'U'), ('per-KC counts (C)', 'ref', 1, 'C')]
    v += [(f'{l} µm', 'kc', 1, f'local {l} µm') for l in LAMS if f'local {l} µm' in have1]
    v += [(('∞' if l == 'inf' else f'{l} µm'), 'bo', 2, ('boutons inf' if l == 'inf' else f'boutons {l} µm')) for l in LB
          if ('boutons inf' if l == 'inf' else f'boutons {l} µm') in have2]
    v += [(('∞' if l == 'inf' else f'{l} µm'), 'kb', 2, ('KC + boutons inf' if l == 'inf' else f'KC + boutons {l} µm')) for l in LB
          if ('KC + boutons inf' if l == 'inf' else f'KC + boutons {l} µm') in have2]
    return v


def rank_mean(d, rn, src, key):
    return d['S1'][rn]['model: Spearman blocked vs intact (mean over MBs) by version'][key] if src == 1 else d['S2'][rn]['Spearman blocked vs intact, mean over MBs'][key]


def r_mean(d, rn, src, key):
    return d['S1'][rn]['r over odor pairs, model change vs data change'][key] if src == 1 else \
        d['S2'][rn]['r with the data change, model change averaged over MBs (as tnt_compare.py)'][key]


def per(d, rn, src, key, mb, col):
    return (d['P1'] if src == 1 else d['P2'])[rn].loc[(key, mb), col]


GCOL = {'ref': L.GRAY, 'kc': L.PURPLE, 'bo': L.GREEN, 'kb': L.INK}
GNAME = {'ref': 'one global APL', 'kc': 'local APL on KCs', 'bo': 'local APL on PN boutons only', 'kb': 'local APL on KCs + boutons (real share)'}
d0 = V['main']
DATA_RANK = {rn: d0['S1'][rn]['data: Spearman TNT vs TNT3c similarity over odor pairs'] for rn in RNS}
DATA_DS = {rn: d0['S1'][rn]['mean change in similarity: data, model by version'][0] for rn in RNS}
NKEY = {rn: next(k for k in d0['S1'][rn] if k.startswith('null')) for rn in RNS}

R = L.Report('11', 'If APL is non-spiking, so that its inhibition stays near where it was excited, does it reshape the KC representation '
             '(the rank order of odor-pair similarity with inhibition on versus off) more like TNT vs TNT3c than one uniform APL or an APL '
             'weighted by per-KC synapse counts?',
             inputs=['The pinned KC model of Report 07 (KC input, odor gains, TNT and TNT3c targets; corrected calyx, new claws, new ORN table)',
                     'APL skeletons and geodesic distances (Report 10)', 'Calyx KC ↔ APL and APL → PN synapses with their positions (Reports 01, 02)'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/odor_weighted/`: `APL_local_model_summary.json`, `_fits.csv`, `_per_MB_{ORN,GH146}.csv`, '
                      '`_odor_pairs.csv`, `_nulls.csv`, `APL_local_boutons_summary.json`, `_per_MB_{ORN,GH146}.csv`',
                      'The earlier version (official ROIs, old claws) in `odor_weighted/`'],
             scripts=['`scripts/apl_local_model.py` (inhibition on KCs)', '`scripts/apl_local_boutons.py` (inhibition on PN boutons too)',
                      '`scripts/apl_local_model_figure.py`', '`scripts/reports/r11_prep.py` (matrix properties and checks for this report)',
                      '`scripts/reports/r11_local_model.py` (this report)'])

R.p('Reports 07 and 09 found that blocking APL in the data (TNT vs the TNT3c control) reorders odor pairs by similarity more than the model '
    f'does: the rank correlation between blocked and intact similarity is {DATA_RANK["ORN"]:.2f} in the data (about {CORR} after correcting for '
    f'recording noise) but {rank_mean(d0, "ORN", 1, "U"):.3f} (ORN input) and {rank_mean(d0, "GH146", 1, "U"):.3f} (GH146) in the model with one '
    'uniform APL. That model treats APL as one unit with one activity per odor. APL is thought to be '
    'non-spiking, and its activity can stay local within its arbor (Amin et al. 2020, eLife). This report builds a version of the '
    'model in which inhibition spreads passively along APL\'s actual skeleton and fades with distance, and asks whether that reorders '
    'the pairs more. Hemibrain and male CNS only (3 mushroom bodies), with the corrected calyx and the new claws.')

R.box('What changed in this revision (2026-11-04)', ["The main results use the corrected calyx, the new claws and the new ORN table; the earlier version is compared at the end.", "New: why θ = 1, what the C matrix is, and Equation 3 with every variable defined and worked through on a mock tree."], 'plain')
R.symbols([
    ('k,\\ j', 'Kenyon cells (KCs); $k$ receives inhibition, $j$ excites APL', '$N$ = ' + ', '.join(f'{NKC[k]:,} ({MBN[k]})' for k in MBS)),
    ('o', 'An odor', '16 (ORN input) or 13 (GH146 input)'),
    ('x_{ko}', 'Excitatory input of KC $k$ to odor $o$ (Report 07, pinned)', 'threshold units'),
    ('y_{ko}', 'Response of KC $k$ to odor $o$', '≥ 0'),
    ('\\theta', 'KC threshold. Set to 1, which fixes the units of $x_{ko}$ (Section 2)', '1'),
    ('a', 'APL gain onto KCs, one per mushroom body and version, set so the mean intact fraction responding equals TNT3c\'s', '> 0'),
    ('M_{kj}', 'Coupling: how strongly activity of KC $j$ inhibits KC $k$ through APL', 'mean over $k, j$ fixed (= that of C)'),
    ('u_k,\\ v_j', 'APL → KC synapses on KC $k$ and KC → APL synapses from KC $j$ in the calyx, each divided by its mean over KCs', 'mean 1'),
    ('\\mathrm{in}(j),\\ \\mathrm{out}(k)', 'APL\'s input sites from KC $j$ (KC → APL synapses) and output sites onto KC $k$ (APL → KC synapses), calyx only', 'skeleton nodes'),
    ('d(s, t)', 'Geodesic distance between sites $s$ and $t$ along APL\'s skeleton (Report 10)', 'µm'),
    ('\\lambda', 'Length constant: distance over which a passive signal in APL falls to $1/e$ (37%)', '2–200 µm, and ∞'),
    ('c', 'Scale that sets the mean of the local $M$ equal to the mean of C\'s', ''),
    ('\\ell_v,\\ \\kappa_v', 'Length of the tree edge from node $v$ to its parent, and its decay factor $\\kappa_v = e^{-\\ell_v/\\lambda}$', 'µm; 0 to 1'),
    ('B_{kb}', 'Share of KC $k$\'s claw PN synapses that come from bouton $b$', '0 to 1, rows sum to 1'),
    ('Q_{bo}', 'Local APL signal at bouton $b$ for odor $o$', '≥ 0'),
    ('\\beta', 'Gain of APL\'s divisive inhibition of boutons', '> 0'),
    ('f', 'Boutons\' share of APL\'s calyx output synapses (Report 04)', f'{FSH["hemibrain"]:.3f} hemibrain, {FSH["malecns"]:.3f} male CNS'),
    ('S_{ab}', 'Cosine similarity of the KC population responses to odors $a$ and $b$', '0 to 1'),
    ('\\rho_{\\mathrm{rank}}', 'Spearman correlation over odor pairs between blocked and intact $S_{ab}$', '−1 to 1; lower = more reordering'),
    ('r', 'Pearson correlation over odor pairs between the model\'s and the data\'s $\\Delta S_{ab}$ = blocked − intact', '−1 to 1'),
])

# ---------------------------------------------------------------- 1
R.section('What "non-spiking" means for the model')
R.p('A spiking neuron turns local input into all-or-none action potentials that reach every part of the cell at full size, so "one '
    'activity per odor" is a fair description. A non-spiking neuron like APL instead carries graded voltage changes that spread '
    'passively and shrink with distance. In the simplest passive cable, a steady signal injected at one point falls off with distance $d$ '
    'as $e^{-d/\\lambda}$. The length constant $\\lambda$ of APL\'s fine calyx branches has not been measured, so the model scans it from 2 µm '
    '(very local) to 200 µm (almost global), and $\\lambda = \\infty$ gives back the one-unit APL.')
R.p('Distances are along the skeleton, not straight lines. Report 10 showed that APL\'s calyx arbor comes in several sub-arbors joined '
    'farther down, so two points a few µm apart in space can be over 100 µm apart along APL. For random pairs of a KC → APL site and an '
    'APL → KC site in the calyx the median distance is '
    + ', '.join(f'{F(CK["main"][k]["geodesic distance KC->APL site to APL->KC site, random pairs (µm): median, 10%, 90%"][0], 0)} µm ({MBN[k]})' for k in MBS)
    + ' (Figure 1a).')


def fig_kernel():
    fig, axs = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.15]))
    bins = np.linspace(0, 500, 51)
    for k in MBS:
        dp = np.load(DD + f'r11_pair_distances_{k.replace(" ", "_")}{TAG}.npy')
        axs[0].hist(dp, bins=bins, histtype='step', lw=1.5, color=MBC[k], weights=np.full(len(dp), 1 / len(dp)), label=MBN[k])
    axs[0].set_ylabel('fraction of site pairs\nper 10 µm bin'); axs[0].legend(fontsize=7.5)
    axs[0].set_title('a  Distance along APL between a KC → APL site and an APL → KC site (random pairs, calyx)')
    d = np.linspace(0, 500, 501)
    cols = plt.get_cmap('Purples')(np.linspace(0.35, 1, len(LAMS)))
    for l, c in zip(LAMS, cols):
        axs[1].plot(d, np.exp(-d / float(l)), color=c, label=f'λ = {l} µm')
    axs[1].axhline(1, color=L.GRAY, ls='--', lw=1, label='λ = ∞ (one global APL)')
    axs[1].set_ylabel('relative strength of the\nAPL signal, $e^{-d/\\lambda}$'); axs[1].set_xlabel('geodesic distance along APL, $d$ (µm)')
    axs[1].legend(fontsize=7, ncol=2, loc='upper right'); axs[1].set_xlim(0, 500); axs[1].set_ylim(0, 1.05)
    axs[1].set_title('b  The assumed passive decay for each length constant')
    fig.tight_layout(); return fig


R.fig(fig_kernel(), '**a**, for 300 random KC → APL sites and 300 random APL → KC sites in the calyx of each mushroom body (corrected calyx), '
      'the distance between every input–output pair along the APL skeleton (each synapse placed at its nearest skeleton node). Lines: '
      'mushroom bodies; y-axis: fraction of the 90,000 pairs per 10 µm bin. **b**, the weight $e^{-d/\\lambda}$ that a signal entering APL '
      'at one site has at another site $d$ µm away along APL, for each length constant used. Shared x-axis: distance along APL in µm.',
      'Distances inside APL and the decay that the local model assumes.')

# ---------------------------------------------------------------- 2 model
R.section('The model, step by step')
R.sub('Step 1: KC input and pinning (unchanged from Report 07)')
R.p('Each KC\'s input is its claws\' PN synapse shares times the glomeruli\'s ORN (or GH146) responses, with negative responses set to 0. '
    'Each odor\'s input is scaled by a gain $g_o$ so that, with APL blocked and $\\theta = 1$, the fraction of KCs responding equals that '
    'odor\'s TNT fraction. Report 07 explains this pinning in full; it is the same here, so all versions start from identical blocked responses '
    '$y^{\\mathrm{blocked}}_{ko} = [x_{ko} - 1]_+$.')
R.sub('Step 2: feedback through a coupling matrix')
R.p('With APL intact, each KC is inhibited by an amount that depends on the activity of all KCs, through the coupling matrix $M$:')
R.eq('y_{ko} \\;=\\; \\Big[\\, x_{ko} - \\theta - a \\sum_{j} M_{kj}\\, y_{jo} \\Big]_+',
     'Solved for each odor separately; the responses appear on both sides, so the solution is the self-consistent steady state.')
R.p('**Why θ = 1.** Because each odor\'s input is pinned (Step 1), the threshold can be any positive number: the gains $g_o$ scale the '
    'inputs to it. Setting θ = 1 only fixes the units, so $x_{ko}$ is measured in "threshold units" (1 = exactly at threshold).')
R.p('Every version has this form and differs only in $M$. The versions:')
R.ul(['**U, uniform**: $M_{kj} = 1/N$. APL sums all KC activity and inhibits every KC equally. This is version U of Reports 05 and 07.',
      '**C, per-KC counts**: $M_{kj} = u_k v_j / N$. KC $j$ excites APL in proportion to its KC → APL synapses, and KC $k$ is inhibited in '
      'proportion to its APL → KC synapses, but APL is still one unit with one activity. This "C matrix" is version C of Reports 05 and 07 '
      'written as a matrix: putting it into Equation 1 gives $a\\sum_j M_{kj}\\,y_{jo} = a\\,u_k \\cdot \\frac{1}{N}\\sum_j v_j\\,y_{jo} = '
      'a\\,u_k\\,A_o$, one APL activity $A_o$ (the $v$-weighted mean KC response) applied to KC $k$ in proportion to $u_k$. It is written as '
      'a matrix only to compare it with the local matrices, which have the same form but let the coupling depend on where on APL the synapses are.',
      '**local**, length constant $\\lambda$: Equation 2.'])
R.eq('M_{kj} \\;=\\; c \\sum_{s \\,\\in\\, \\mathrm{out}(k)} \\; \\sum_{t \\,\\in\\, \\mathrm{in}(j)} e^{-d(s,t)/\\lambda}',
     'KC $j$\'s activity enters APL at each of its KC → APL synapses $t$, spreads along APL with decay $e^{-d/\\lambda}$, and inhibits KC $k$ at each APL → KC synapse $s$ on $k$.')
R.p('As $\\lambda \\to \\infty$ every exponential becomes 1, and $M_{kj} = c\\,|\\mathrm{out}(k)|\\,|\\mathrm{in}(j)|$, which is C (checked: correlation '
    '1.000 in all three mushroom bodies). So C is the local model with an infinitely long length constant, and the local versions differ '
    'from C only in where on APL each KC\'s synapses are. The constant $c$ sets the mean of $M$ to that of C, so that the gain $a$ has '
    'comparable values across versions; the calibration in Step 4 would absorb any constant anyway.')
R.box('What counts as a site', ['Only calyx synapses are used, as for C (per-KC counts are calyx only, Yang 2026-10-21). APL\'s side of each synapse is '
                               'placed at the nearest node of the APL skeleton: the T-bar for APL → KC, the PSD for KC → APL (median snap distance '
                               + ', '.join(f'{F(CK["main"][k]["snap distance synapse to node, median (µm)"], 1)} µm in {MBN[k]}' for k in MBS) + ').',
                               'Two sites at the same node have $d = 0$. Snap distances are not added to $d$, so at λ = 2 µm the result depends '
                               'on the skeleton\'s resolution.'], 'plain')

R.sub('Step 3: computing M on the tree')
R.p('A direct sum over all pairs of sites (tens of thousands of each per mushroom body) would need every pairwise distance. Because a skeleton is a '
    'tree, the same sums can be computed exactly in two passes. First the tree is reduced to the part that joins the synapse nodes, with '
    'chains of unbranched nodes merged into single edges of summed length $\\ell_v$ ('
    + ', '.join(f'{CK["main"][k]["nodes of the compressed tree"]:,} nodes remain in {MBN[k]}' for k in MBS) + '). Let $n_{vj}$ be the number of '
    'KC → APL sites of KC $j$ at node $v$, and $\\kappa_v = e^{-\\ell_v/\\lambda}$. Then:')
R.eq('\\text{up}_{vj} = n_{vj} + \\sum_{w \\,\\in\\, \\mathrm{children}(v)} \\kappa_w\\, \\text{up}_{wj}, \\qquad '
     '\\Phi_{vj} = \\text{up}_{vj}\\,(1 - \\kappa_v^2) + \\kappa_v\\, \\Phi_{\\mathrm{parent}(v),\\,j}, \\quad \\Phi_{\\mathrm{root},j} = \\text{up}_{\\mathrm{root},j}',
     'Pass 1 runs from the leaves up; pass 2 from the root down. $\\Phi_{vj} = \\sum_{t \\in \\mathrm{in}(j)} e^{-d(v,t)/\\lambda}$, the signal from KC $j$ at node $v$.')
R.p('Why it works: the path from $v$ to a site $t$ below $v$ stays below $v$, and its weight is the product of the $\\kappa$ of its edges, which '
    'pass 1 collects. A site not below $v$ is reached through $v$\'s parent; pass 2 takes the parent\'s total, multiplies by $\\kappa_v$ for '
    'the edge, and removes the part that came from below $v$ itself (that part crossed the edge twice, hence $\\kappa_v^2$). Finally '
    '$M_{kj} = c\\sum_{s \\in \\mathrm{out}(k)} \\Phi_{s j}$. **Check:** at λ = 10 µm, $\\Phi$ for 40 random KCs at 60 random nodes was also '
    'computed by brute force (shortest paths on the tree); the largest difference relative to the largest value was '
    + ', '.join(f'{CK["main"][k]["two-pass vs brute force at 10 µm: max abs difference / max value"]:.0e}' for k in MBS) + ' (rounding error).')

R.sub('Equation 3, every variable, with a mock tree')


def _mock_eq3():
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


_mock_eq3()

R.sub('Step 4: one APL gain per version, and the solver')
R.p('For each mushroom body, input table and version, the gain $a$ is found by bisection (32 halvings after doubling until it is large '
    f'enough) so that the mean fraction of KCs responding over odors equals TNT3c\'s, 6.83% for the 16 odors. Equation 1 is solved per odor '
    'with an exact **active-set** method: guess which KCs respond, solve the linear equations $(I + aM_{SS})\\,y_S = x_S - \\theta$ for that set $S$, '
    'add KCs whose net input is still above threshold and drop KCs whose solution is negative, and repeat until the set stops changing. If '
    'the set cycles or the result fails a residual check, a damped fixed-point iteration is used instead. Both solvers were run on the same '
    'cases and agree (gains equal to 4 decimals, responses within about 10⁻⁶). With feedback this strong a threshold-linear network could '
    'in principle have more than one steady state; the solver starts from the blocked responses and finds one.')

R.sub('Step 5: readouts')
R.ul(['**Rank order** $\\rho_{\\mathrm{rank}}$: for each mushroom body, the Spearman correlation over odor pairs between blocked and intact '
      'cosine similarity, then the mean over the three mushroom bodies. Data: TNT vs TNT3c, ' + f'{F(DATA_RANK["ORN"], 3)} over the 16 odors, '
      f'{F(DATA_RANK["GH146"], 3)} over the 13 GH146 odors; corrected for recording noise about {CORR} (Report 09).',
      '**Pattern** $r$: the model\'s change in similarity per pair (blocked − intact, similarity averaged over the three mushroom bodies) '
      'correlated with the data\'s change (TNT − TNT3c). The data\'s own reliability limits this to about ' + f'{CEIL} (Report 07).',
      '**Size**: the mean change in similarity over pairs (data: ' + f'{F(DATA_DS["ORN"], 3)} for the 16 odors).'])

R.sub('Step 6: a null for the real placement of each KC on APL')
R.p('Is it the real position of each KC\'s synapses on APL that matters, or only how local the inhibition is? For the local version with '
    'the best pattern $r$, KCs are shuffled within their class (αβ, α′β′, γ): KC $k$ gets the APL sites of KC $\\pi(k)$, which turns $M$ into '
    '$M_{\\pi(k)\\pi(j)}$. This keeps the arbor, the class composition and the locality, and only breaks which KC sits where. 100 shuffles per '
    'mushroom body, each recalibrated. One-sided $p$ = (1 + shuffles at least as extreme) / 101.')
R.box('Caveat on the null', 'The λ tested is the one with the best $r$ among the local versions, picked before the shuffles, so the $p$ value does '
      'not correct for having tried seven λ values.', 'caveat')

R.sub('Step 7: local inhibition of PN boutons too')
R.p('APL also inhibits the PN boutons (about 10% of its calyx output synapses, Report 04). The second script adds this route. Bouton $b$ '
    'receives the local APL signal at its APL → PN synapses, per PN → KC synapse it makes onto claws, and divides its output:')
R.eq('x_{ko} = \\sum_b B_{kb}\\, \\frac{g_o\\, r_{g(b),o}}{1 + \\beta\\, Q_{bo}}, \\qquad '
     'Q_{bo} = \\frac{1}{n^{\\mathrm{claw}}_b}\\sum_{s \\,\\in\\, \\mathrm{APL\\to PN}(b)} \\sum_j \\Phi_{sj}\\, y_{jo}',
     '$g(b)$: the glomerulus of bouton $b$; $n^{\\mathrm{claw}}_b$: PN synapses on the claws whose main bouton is $b$. With $\\beta = 0$ this is exactly the input of Step 1 (checked).')
R.ul(['**boutons only**: all inhibition on boutons ($a = 0$), $\\beta$ calibrated to TNT3c\'s fraction. An upper bound for this route.',
      '**KC + boutons**: KC gain $(1-f)\\,a_\\lambda$ and bouton gain $f\\,\\beta_\\lambda$ (each from its own calibration at the same λ), then '
      'both scaled together to TNT3c\'s fraction. This is the bouton route at its real synapse share.',
      'λ = ∞, 50, 25, 10, 5 µm.'])

# ---------------------------------------------------------------- 3 what M looks like
R.section('What shorter length constants do to the coupling')
mp = MP['main']


def fig_M():
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.1))
    for k in MBS:
        x = mp[(mp.mushroom_body == k) & np.isfinite(mp.lam_um)].sort_values('lam_um')
        for ax, col, sc in zip(axs, ['effective_inhibitors_frac_median', 'r_with_C', 'self_share'], [100, 1, 100]):
            ax.plot(x.lam_um, x[col] * sc, marker=MK[k], ms=4, color=MBC[k], label=MBN[k])
    labs = [('effective number of KCs inhibiting\na KC, median (% of all KCs)', 'a'), ('correlation of $M$ with C\n(over all KC pairs)', 'b'),
            ('share of a KC\'s inhibition that\ncomes from itself (%)', 'c')]
    for ax, (yl, t) in zip(axs, labs):
        ax.set_xscale('log'); ax.set_xlabel('length constant λ (µm, log scale)'); ax.set_ylabel(yl); L.tag(ax, t, x=-0.3)
        ax.yaxis.grid(True, color=L.GRID, lw=0.6); ax.invert_xaxis()
    axs[0].legend(fontsize=7.5)
    fig.tight_layout(); return fig


R.fig(fig_M(), 'Properties of the coupling matrix $M$ of the local model, per mushroom body (lines), as λ shrinks (x-axis, log scale, '
      'long λ on the left). **a**, for each KC $k$ the participation ratio $(\\sum_j M_{kj})^2 / \\sum_j M_{kj}^2$, the number of KCs that '
      'would contribute equally to its inhibition, as a percentage of all KCs; median over KCs. With U it would be 100%. **b**, Pearson '
      'correlation between $M$ and the C matrix over KC pairs (C = λ = ∞). **c**, mean over KCs of $M_{kk} / \\sum_j M_{kj}$: the share of '
      f'inhibition a KC would get from its own activity if all KCs were equally active. From `reports/data/r11_M_properties{TAG}.csv`.',
      'Shorter length constants make each KC\'s inhibition depend on fewer KCs.')
hb = mp[mp.mushroom_body == 'hemibrain R'].set_index('lam_um')
R.p(f'At λ = 100 µm each hemibrain KC is still inhibited by about {L.pct(hb.loc[100, "effective_inhibitors_frac_median"])} of all KCs, '
    f'at 10 µm by about {L.pct(hb.loc[10, "effective_inhibitors_frac_median"])} and at 2 µm by about {L.pct(hb.loc[2, "effective_inhibitors_frac_median"], 1)} '
    f'(about {hb.loc[2, "effective_inhibitors_frac_median"] * NKC['hemibrain R']:.0f} KCs). Self-inhibition stays small even at 2 µm '
    f'({L.pct(hb.loc[2, "self_share"], 1)}), because a KC\'s own APL synapses are only a few of the many near any site. Short λ also shifts '
    f'the coupling toward KCs of the same class ({L.pct(hb.loc[np.inf, "same_class_share"])} of $M$ at λ = ∞, '
    f'{L.pct(hb.loc[2, "same_class_share"])} at 2 µm in hemibrain), since KCs of one class share parts of the calyx.')

# ---------------------------------------------------------------- 4 results
R.section('Results: rank order and pattern')


def draw(ax, d, rn, kind):
    VV = versions(d, rn); x = []; pos = 0; prev = None; vals = []
    for lab, g, src, key in VV:
        if prev is not None and g != prev: pos += 0.9
        x.append(pos); prev = g; pos += 1
    for xi, (lab, g, src, key) in zip(x, VV):
        for k in MBS:
            val = per(d, rn, src, key, k, 'spearman_blocked_vs_intact' if kind == 'rank' else 'r_dS_with_data')
            ax.scatter(xi, val, marker=MK[k], s=14, facecolor='white', edgecolor=GCOL[g], lw=0.9, zorder=3); vals.append(val)
        m = rank_mean(d, rn, src, key) if kind == 'rank' else r_mean(d, rn, src, key)
        ax.plot([xi - 0.36, xi + 0.36], [m, m], color=GCOL[g], lw=2.4, zorder=4, solid_capstyle='butt')
    ax.set_xticks(x, [v[0] for v in VV], rotation=90, fontsize=7)
    if kind == 'rank':
        ax.axhline(DATA_RANK[rn], color=L.BLOCKED, lw=1.2, ls='-'); ax.axhline(CORR, color=L.BLOCKED, lw=1.2, ls='--')
        ax.text(x[-1] + 0.6, DATA_RANK[rn], f'data {F(DATA_RANK[rn], 3)}', fontsize=7, color=L.BLOCKED, va='center')
        ax.text(x[-1] + 0.6, CORR, f'data, noise-\ncorrected ≈ {CORR}', fontsize=7, color=L.BLOCKED, va='center')
        ax.set_ylabel('$\\rho_{\\mathrm{rank}}$: Spearman, blocked vs intact\nsimilarity over odor pairs'); ax.set_ylim(min(0.85, min(vals) - 0.01), 1.0)
    else:
        ax.axhline(CEIL, color=L.BLOCKED, lw=1.2, ls='--'); ax.text(x[-1] + 0.6, CEIL, 'data\nceiling', fontsize=7, color=L.BLOCKED, va='center')
        ax.set_ylabel('$r$: model ΔS vs data ΔS\nover odor pairs'); ax.set_ylim(min(0, min(vals) - 0.04), 0.9); ax.axhline(0, color=L.GRID, lw=0.8)
    ax.set_xlim(-0.7, x[-1] + 2.6)
    ax.yaxis.grid(True, color=L.GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f'{rn} input ({len(d["PA"][d["PA"].responses == rn])} odor pairs)')


def fig_results(d):
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.6))
    for j, rn in enumerate(RNS):
        draw(axs[0, j], d, rn, 'rank'); draw(axs[1, j], d, rn, 'r')
    h = [Line2D([], [], color=GCOL[g], lw=2.4, label=GNAME[g]) for g in GCOL]
    h += [Line2D([], [], marker=MK[k], ls='', mfc='white', mec=L.INK, label=MBN[k]) for k in MBS]
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    for ax, t in zip(axs.ravel(), 'abcd'): L.tag(ax, t, x=-0.16)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); return fig


R.fig(fig_results(d0), 'x-axis: model versions, grouped by color: one global APL (U, C); local APL on KCs (λ from 200 to 2 µm); '
      'local APL on PN boutons only (λ = ∞ to 5 µm); local APL on KCs and boutons with boutons at their real share of APL\'s output. '
      'Open markers: the three mushroom bodies; bar: their mean (for **c**–**d** the bar is $r$ of the similarity averaged over mushroom bodies, '
      'which is how Report 07 computes it, so it can lie above all three markers). **a**–**b**, rank order $\\rho_{\\mathrm{rank}}$; lower means '
      'inhibition reorders the odor pairs more. Orange lines: data, raw (solid) and corrected for recording noise (dashed). **c**–**d**, '
      f'pattern $r$; orange dashed line: the highest $r$ the data\'s reliability allows (about {CEIL}). Left: ORN input, 16 odors; right: GH146 input, 13 odors.',
      'Local inhibition on KCs: rank order and pattern.')
FIG_RES = R.n_fig


def best_local(d, rn):
    loc = [v for v in versions(d, rn) if v[1] == 'kc']
    return max(loc, key=lambda v: r_mean(d, rn, 1, v[3]))


rows = []
for rn in RNS:
    for lab, g, src, key in versions(d0, rn):
        if g == 'kc' and lab not in ('100 µm', '25 µm', '10 µm', '5 µm', '2 µm'): continue
        if g in ('bo', 'kb') and lab not in ('∞', '10 µm', '5 µm'): continue
        md = (d0['S1'][rn]['mean change in similarity: data, model by version'][1].get(key) if src == 1 else
              float(np.mean([per(d0, rn, src, key, k, 'mean_dS') for k in MBS])))
        rows.append([rn, {'ref': 'global', 'kc': 'local, KCs', 'bo': 'local, boutons only', 'kb': 'local, KCs + boutons'}[g] + ('' if g == 'ref' else f', λ = {lab}') if g != 'ref' else lab,
                     F(rank_mean(d0, rn, src, key), 3), F(r_mean(d0, rn, src, key), 3), F(md, 3)])
    rows.append([rn, '**data** (TNT vs TNT3c)', f'**{F(DATA_RANK[rn], 3)}** (≈ {CORR} noise-corrected)', f'ceiling ≈ {CEIL}', f'**{F(DATA_DS[rn], 3)}**'])
R.table(['Input', 'Version', '$\\rho_{\\mathrm{rank}}$ (mean of MBs)', 'pattern $r$', 'mean ΔS'], rows, title='Main numbers.',
        caption='Selected versions; all are in the summary JSON files. Mean ΔS for the bouton versions is the mean over mushroom bodies of each one\'s mean change.',
        num_cols=[2, 3, 4])
nb = {rn: d0['S1'][rn][NKEY[rn]] for rn in RNS}


def pval(n): return n['r real, null mean, null 5-95%, p'][3]


def shuffle_lead(nbx):
    sig = [pval(nbx[rn]) <= 0.05 for rn in RNS]
    if all(sig):
        return 'Shuffling KCs within class lowers $r$, so the real placement of each KC on APL matters, not only locality: '
    if not any(sig):
        return ('Shuffling KCs within class gives about the same $r$, so the gain comes from inhibition being local, not from where each KC '
                'sits on APL: ')
    return ('Whether the real placement of each KC on APL matters beyond locality depends on the input table (it beats within-class shuffles '
            'only where $p$ ≤ 0.05): ')

bl = {rn: best_local(d0, rn) for rn in RNS}


def _ds_ratio(d, rn):
    m = d['S1'][rn]['mean change in similarity: data, model by version']
    return [m[1][v[3]] / m[0] for v in versions(d, rn) if v[1] == 'kc' or v[3] == 'U']

R.box('Answer', [
    '**Rank order: no.** A local APL on the KCs leaves $\\rho_{\\mathrm{rank}}$ at '
    + ' and '.join(f'{F(min(rank_mean(d0, rn, 1, v[3]) for v in versions(d0, rn) if v[1] == "kc"), 3)}–{F(max(rank_mean(d0, rn, 1, v[3]) for v in versions(d0, rn) if v[1] == "kc"), 3)} ({rn})' for rn in RNS)
    + f', like U ({F(rank_mean(d0, "ORN", 1, "U"), 3)}, {F(rank_mean(d0, "GH146", 1, "U"), 3)}) and far from the data\'s ≈ {CORR}. Only inhibition '
    'placed entirely on the PN boutons gets close: ' + ', '.join(f'{F(min(rank_mean(d0, rn, 2, v[3]) for v in versions(d0, rn) if v[1] == "bo"), 3)} ({rn})' for rn in RNS)
    + '. At the boutons\' real share of APL\'s output (KC + boutons) it is back to '
    + ', '.join(f'{F(min(rank_mean(d0, rn, 2, v[3]) for v in versions(d0, rn) if v[1] == "kb"), 3)}–{F(max(rank_mean(d0, rn, 2, v[3]) for v in versions(d0, rn) if v[1] == "kb"), 3)}' for rn in RNS) + '.',
    '**Pattern: somewhat.** Short length constants raise $r$ with the data\'s pattern of changes: best local on KCs '
    + ', '.join(f'{F(r_mean(d0, rn, 1, bl[rn][3]), 3)} at {bl[rn][0]} ({rn})' for rn in RNS)
    + f', vs U {F(r_mean(d0, "ORN", 1, "U"), 3)} / {F(r_mean(d0, "GH146", 1, "U"), 3)} and C {F(r_mean(d0, "ORN", 1, "C"), 3)} / {F(r_mean(d0, "GH146", 1, "C"), 3)}. '
    + shuffle_lead(nb) + '; '.join(
        f'{rn}: $r$ {F(nb[rn]["r real, null mean, null 5-95%, p"][0], 3)} vs shuffles {F(nb[rn]["r real, null mean, null 5-95%, p"][1], 3)} (5–95% '
        f'{F(nb[rn]["r real, null mean, null 5-95%, p"][2][0], 2)}–{F(nb[rn]["r real, null mean, null 5-95%, p"][2][1], 2)}), $p$ = {F(nb[rn]["r real, null mean, null 5-95%, p"][3], 2)}' for rn in RNS) + '.',
    '**Size: no.** The mean change in similarity of U and the local versions on KCs is '
    + ', '.join(f'{L.pct(min(_ds_ratio(d0, rn)))}–{L.pct(max(_ds_ratio(d0, rn)))} of the data\'s ({rn})' for rn in RNS) + '.'], 'result')


def fig_null(d):
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 2.8))
    for j, rn in enumerate(RNS):
        nk = NKEY[rn]; best = nk.split('version ')[-1]
        n = d['NU'][d['NU'].responses == rn]
        real_r = d['S1'][rn][nk]['r real, null mean, null 5-95%, p'][0]; real_k = d['S1'][rn][nk]['rank Spearman real, null mean, null 5-95%, p (lower = more reordering)'][0]
        for ax, col, real, xl in [(axs[2 * j], 'r_dS_with_data', real_r, 'pattern $r$ with the data'),
                                  (axs[2 * j + 1], 'rank_spearman_mean_of_MBs', real_k, '$\\rho_{\\mathrm{rank}}$ (mean of MBs)')]:
            ax.hist(n[col], bins=20, color=L.GRAY, alpha=0.8)
            ax.axvline(real, color=L.PURPLE, lw=2); ax.set_xlabel(xl); ax.set_ylabel('shuffles (of 100)')
            ax.xaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_title(f'{rn}, {best.replace("local ", "λ = ")}', fontsize=8.5)
    fig.tight_layout(); return fig


R.fig(fig_null(d0), 'Gray: 100 shuffles in which KCs swap their APL synapse sites with other KCs of the same class (in all three mushroom bodies at '
      'once); purple line: the real wiring. Shown for the local version with the best pattern $r$ for each input table (named in the panel '
      'title). Left pair: ORN input; right pair: GH146 input. In each pair, the first panel is the pattern $r$ (higher = closer to the data), '
      'the second the rank order (lower = more reordering).', 'Does the real position of each KC on APL matter? Within-class site shuffles.')
_sig = {rn: pval(nb[rn]) <= 0.05 for rn in RNS}
_rk = {rn: nb[rn]['rank Spearman real, null mean, null 5-95%, p (lower = more reordering)'] for rn in RNS}
R.p(('The shuffles show that the improvement in pattern comes partly from where each KC\'s synapses are, not only from inhibition being local. '
     if all(_sig.values()) else
     'For the pattern, the real wiring beats the shuffles for ' + (' and '.join(rn for rn in RNS if _sig[rn]) or 'neither input')
     + ' and not for ' + (' and '.join(rn for rn in RNS if not _sig[rn]) or 'neither') + ', so part of the gain is locality itself. ')
    + 'Rank order with real wiring vs shuffles: ' + '; '.join(f'{rn} {F(_rk[rn][0], 3)} vs {F(_rk[rn][1], 3)}, $p$ = {F(_rk[rn][3], 2)}' for rn in RNS)
    + '. ' + (f'Any difference is a few thousandths, far less than the gap to the data.' if max(_rk[rn][1] - _rk[rn][0] for rn in RNS) < 0.01 else
              f'The real wiring reorders a little more than the shuffles (by up to {max(_rk[rn][1] - _rk[rn][0] for rn in RNS):.3f}), but it still stays '
              f'above the data\'s noise-corrected ≈ {CORR} (gap {min(_rk[rn][0] for rn in RNS) - CORR:.3f} or more).'))


def fig_pairs(d, rn='ORN'):
    pa = d['PA'][d['PA'].responses == rn]; best = best_local(d, rn)[3]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, key, col, nm in [(axs[0], 'U', L.GRAY, 'uniform APL (U)'), (axs[1], best, L.PURPLE, best.replace('local ', 'local APL, λ = '))]:
        dm = pa.S_model_blocked - pa[f'S_model_intact_{key}']
        ax.scatter(dm, pa.dS_data, s=13, color=col, lw=0)
        ax.set_xlabel('model ΔS = blocked − intact similarity'); ax.set_title(f'{nm}: r = {np.corrcoef(dm, pa.dS_data)[0, 1]:.3f}')
        ax.yaxis.grid(True, color=L.GRID, lw=0.6)
    axs[0].set_ylabel('data ΔS = TNT − TNT3c similarity')
    fig.tight_layout(); return fig


R.fig(fig_pairs(d0), 'Each dot is one of the 120 odor pairs (ORN input). x: change in cosine similarity of the model KC responses '
      'when APL is blocked (similarity averaged over the three mushroom bodies); y: the change in the data (TNT minus TNT3c, averaged over '
      'recordings). Left: uniform APL; right: the local version with the best pattern $r$. The x-axes span different ranges: the model\'s changes '
      'are smaller than the data\'s.', 'What the pattern $r$ measures, for the uniform and the best local APL.')

# ---------------------------------------------------------------- 5 outside ROI
R.section('Earlier version: official ROIs and the old claws')
d1 = V['earlier']
R.p('The earlier version of this report used the official ROIs (without the back of the male CNS calyx), the old 2 µm claws and the old '
    f'ORN table. Figure {R.n_fig + 1} is Figure {FIG_RES} for that version, and Table {R.n_tab + 1} compares the main numbers.')
R.fig(fig_results(d1), f'As Figure {FIG_RES}, with the earlier tables (`odor_weighted/`).', 'Rank order and pattern, earlier version.')
rows = []
for rn in RNS:
    keys = [('U', 1), ('C', 1), (best_local(d0, rn)[3], 1), ('local 2 µm', 1), ('boutons 5 µm', 2), ('KC + boutons 5 µm', 2)]
    keys = list(dict.fromkeys(keys))      # best local can be 2 µm itself
    for key, src in keys:
        try:
            rows.append([rn, key.replace('local ', 'local on KCs, λ = ').replace('KC + boutons ', 'KCs + boutons, λ = ').replace('boutons 5', 'boutons only, λ = 5'),
                         F(rank_mean(d0, rn, src, key), 3), F(rank_mean(d1, rn, src, key), 3), F(r_mean(d0, rn, src, key), 3), F(r_mean(d1, rn, src, key), 3)])
        except KeyError:
            pass
R.table(['Input', 'Version', '$\\rho_{\\mathrm{rank}}$ now', '$\\rho_{\\mathrm{rank}}$ earlier', '$r$ now', '$r$ earlier'], rows,
        title='Now vs the earlier version.', caption='Means over the three mushroom bodies ($\\rho_{\\mathrm{rank}}$) and $r$ of the '
        f'MB-averaged change, as in Figure {FIG_RES}. "Now": corrected calyx, new claws, new ORN table.', num_cols=[2, 3, 4, 5])
nb1 = {rn: d1['S1'][rn][next(k for k in d1['S1'][rn] if k.startswith('null'))] for rn in RNS}
R.p('Shuffle test (pattern $r$), earlier version: ' + '; '.join(
        f'{rn} $r$ {F(nb1[rn]["r real, null mean, null 5-95%, p"][0], 3)} vs {F(nb1[rn]["r real, null mean, null 5-95%, p"][1], 3)}, $p$ = {F(nb1[rn]["r real, null mean, null 5-95%, p"][3], 2)}' for rn in RNS) + '.')

# ---------------------------------------------------------------- 6
R.section('What this means, and caveats')
SNAP = [CK['main'][k]['snap distance synapse to node, median (µm)'] for k in MBS]
_all = [(vn, rn, pval(nbv[rn])) for vn, nbv in [('now', nb), ('earlier version', nb1)] for rn in RNS]
_beat = [f'{rn}, {vn} ($p$ = {F(p_, 2)})' for vn, rn, p_ in _all if p_ <= 0.05]
_nobeat = [f'{rn}, {vn} ($p$ = {F(p_, 2)})' for vn, rn, p_ in _all if p_ > 0.05]
R.ul(['A non-spiking APL whose inhibition stays local predicts **which** odor pairs change most when APL is blocked better than a uniform '
      'or count-weighted APL does. ' + ('The real positions of each KC\'s synapses on APL add to this: the real wiring beats within-class '
      'shuffles in every case. ' if not _nobeat else
      'Whether the real positions of each KC\'s synapses add to this beyond locality is mixed: the real wiring beats within-class shuffles for '
      + ('; '.join(_beat) or 'none') + ', but not for ' + '; '.join(_nobeat) + '. ')
      + 'This supports the idea that APL\'s local structure shapes the effect of inhibition on which pairs change.',
      'It does **not** make inhibition reorder the pairs as much as in the data. In this model, inhibition applied at the KCs, whatever its '
      'spatial range, keeps the order of similarities nearly intact (Report 09 shows when that is expected). Inhibition applied before the '
      'KCs, on the PN boutons, changes the input itself and reorders more; at the boutons\' real synapse share it is too weak to matter here.',
      'The model\'s mean changes are ' + ', '.join(f'{L.pct(min(_ds_ratio(dd, rn)))}–{L.pct(max(_ds_ratio(dd, rn)))} ({rn}, {vn})' for vn, dd in [('now', d0), ('earlier version', d1)] for rn in RNS)
      + ' of the data\'s, so something not in the model (e.g. stronger or nonlinear '
      'inhibition, KC-to-KC or other interactions, recording noise that differs between genotypes) also contributes.'])
R.box('Caveats', [f'λ = 2 µm is close to the skeleton\'s resolution (synapses snap {min(SNAP):.1f}–{max(SNAP):.1f} µm to nodes, median), so the shortest λ values are approximate.',
                  'The best λ was picked before the shuffle test; the $p$ values do not correct for that choice.',
                  'Synapse counts stand in for synaptic strength, and the passive decay $e^{-d/\\lambda}$ ignores branch thickness and '
                  'the signal\'s time course (a steady state is assumed).',
                  'A threshold-linear network with strong feedback can have more than one steady state; the solver finds the one reached from the blocked responses.'], 'caveat')
R.save()
