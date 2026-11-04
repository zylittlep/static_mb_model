"""Report 10: geodesic distance along APL between glomeruli's microglomeruli (apl_geodesic.py).
Rebuilt 2026-11-04 for Yang's revision: the main numbers now use the corrected calyx and the new claws (APL_VARIANT=claws_v2m2); the
official-ROI version is compared at the end. Old version in archive_2026-11-03/."""
import sys, json, os
os.environ['APL_VARIANT'] = 'claws_v2m2'
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import report_lib as L
import apl_geodesic as ag
L.style(); L.DATE = '2026-11-04'
F = L.fmt
GD = ag.OUT                                   # with_outside_ROI_synapses/claws_v2m2/apl_geodesic/
assert 'claws_v2m2' in GD, GD
P = pd.read_csv(GD + 'APL_geodesic_glomerulus_pairs.csv'); MG = pd.read_csv(GD + 'APL_geodesic_microglomeruli.csv')
SB = pd.read_csv(GD + 'APL_geodesic_glomerulus_by_subarbor.csv'); SJ = json.load(open(GD + 'APL_geodesic_summary.json'))
SJo = json.load(open(L.B + 'apl_geodesic/APL_geodesic_summary.json'))          # official ROIs, old 2 µm claws
KEYS = list(ag.APLS); LAMS = [10, 25, 50, 100]
G = sorted(P.glomerulus_A.unique()); gi = {g: i for i, g in enumerate(G)}


def mat(k, col):
    Q = P[P.apl == k]; M = np.full((len(G), len(G)), np.nan)
    M[Q.glomerulus_A.map(gi).values, Q.glomerulus_B.map(gi).values] = Q[col].values; return M


MALL = {k: mat(k, 'geodesic_um_all_synapses') for k in KEYS}; MIO = {k: mat(k, 'geodesic_um_inputs_A_to_outputs_B') for k in KEYS}
MEU = {k: mat(k, 'euclidean_um_all_synapses') for k in KEYS}
mean_all = np.mean([MALL[k] for k in KEYS], 0)
Z = linkage(squareform((mean_all + mean_all.T) / 2 - np.diag(np.diag(mean_all)), checks=False), 'average', optimal_ordering=True)
O = leaves_list(Z); NAMES = [G[i] for i in O]; IU = np.triu_indices(len(G), 1)

R = L.Report('10', 'How far apart, measured along APL\'s own arbor, are APL\'s synapses at the microglomeruli of different glomeruli, and what '
             'does that imply for how much a non-spiking APL inhibits a glomerulus\'s own KCs versus other glomeruli\'s?',
             inputs=['APL skeletons (SWC): hemibrain v1.2 body 425790257; male CNS v1.0 bodies 10977 (APL_L) and 10540 (APL_R), in `data/skeletons/`',
                     'Calyx APL synapse tables with bouton and claw assignments (Reports 01, 02)', 'Derived PN boutons (Report 02)'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/apl_geodesic/`: `APL_geodesic_glomerulus_pairs.csv`, `APL_geodesic_microglomeruli.csv`, '
                      '`APL_geodesic_glomerulus_by_subarbor.csv`, `APL_geodesic_summary.json` and figures',
                      'The earlier versions: `apl_geodesic/` (official ROIs, old claws) and `with_outside_ROI_synapses/apl_geodesic/`'],
             scripts=['`scripts/apl_geodesic.py`', '`scripts/reports/r10_geodesic.py` (this report)'])
R.p('APL does not fire action potentials, so a signal entering it at one synapse fades with distance along its neurites. How strongly an '
    'input in one place inhibits KCs elsewhere then depends on the path length along APL, not on the straight-line distance. Yang asked '
    '(2026-10-23) for this distance between every pair of glomeruli, weighted by synapse counts, for hemibrain and male CNS. This report '
    'explains how it is computed and what it shows. Numbers use the corrected calyx (synapses outside the official ROIs added) and the '
    'new claws (Report 02); the earlier official-ROI version is compared in Section 7.')
R.box('What changed in this revision (2026-11-04)', ["The main numbers now use the corrected calyx and the new claws; the earlier official-ROI version is compared at the end."], 'plain')
R.symbols([
    ('s,\\ t', 'Two APL synapses, each placed at the APL skeleton node nearest its APL-side site (APL output: the T-bar; APL input: the PSD)', ''),
    ('d(s, t)', 'Geodesic distance: length of the path between $s$ and $t$ along the APL skeleton', 'µm'),
    ('A,\\ B', 'Two glomeruli (possibly the same)', ''),
    ('\\mathcal{S}_A', 'APL synapses on the microglomeruli of glomerulus $A$', ''),
    ('w_s', 'Weight of synapse $s$: 1 (each synapse counts once; synapse counts stand in for strength)', ''),
    ('D(A, B)', 'Weighted mean geodesic distance between $\\mathcal{S}_A$ and $\\mathcal{S}_B$', 'µm'),
    ('e,\\ \\ell_e', 'An edge of the skeleton (between a node and its parent) and its length', 'µm'),
    ('\\alpha_e,\\ \\beta_e', 'Share of $A$\'s (or $B$\'s) synapse weight on the child side of edge $e$', '0 to 1'),
    ('\\lambda', 'Length constant of an assumed passive decay along APL', 'µm; 10, 25, 50, 100 shown'),
    ('C_\\lambda(A \\to B)', 'Mean of $e^{-d/\\lambda}$ over pairs of an APL input at $A$ and an APL output at $B$', '0 to 1'),
])

# ---------------------------------------------------------------- 1 skeletons
R.section('Step 1: APL skeletons')
sk = {k: SJ['apl'][k]['skeleton: nodes, total length (mm), pieces joined, bridges (µm)'] for k in KEYS}
R.p('A skeleton represents a neuron as a tree of points (nodes) joined by straight segments (edges), each node pointing to a parent. The '
    'files are the public FlyEM skeletons. Distances along a tree are unique, because there is exactly one path between two nodes. A few '
    'skeletons come in several unconnected pieces. These are joined by the shortest bridges between pieces (a minimum spanning tree over the '
    'pieces), and the longest bridge needed is under 3 µm.')
R.table(['APL', 'body ID', 'nodes', 'cable length (mm)', 'pieces joined', 'longest bridge (µm)', 'synapse to nearest node, median (µm)'],
        [[k, ag.APLS[k][2].split('_')[1], f'{sk[k][0]:,}', F(sk[k][1], 1), sk[k][2], F(sk[k][3][0], 2),
          F(SJ['apl'][k]['distance from synapse to nearest skeleton node (µm): median, 95th percentile'][0], 2)] for k in KEYS],
        title='The three APL skeletons.', caption='The male CNS skeletons are coarser (fewer nodes per µm of cable) than hemibrain\'s.', num_cols=[2, 3, 4, 5, 6])

# ---------------------------------------------------------------- 2 synapses
R.section('Step 2: which synapses belong to a glomerulus')
R.p('A **microglomerulus** is one PN bouton plus the KC claws whose main bouton it is. Each new claw belongs to one PN (Report 02), and its '
    'main bouton is the bouton of that PN that gives it most of its PSDs. The microglomerulus\'s glomerulus is the bouton\'s PN\'s '
    'glomerulus, uniglomerular PN types only. The APL synapses on it are:')
R.ul(['APL↔PN synapses on the bouton (partner-side site within 2 µm of the bouton\'s T-bar cluster), both directions;',
      'APL↔KC synapses on those claws, both directions, assigned by the new claw rule (the KC-side site joins the claw whose PN\'s PSDs '
      'are nearest along the KC\'s skeleton, within 8 µm).'])
rows = []
_sh = [SJ['apl'][k]['of those, uniglomerular PN glomerulus'] / SJ['apl'][k]['calyx APL synapses'] for k in KEYS]
SHMG = f'{L.pct(min(_sh))} to {L.pct(max(_sh))}'
for k in KEYS:
    s = SJ['apl'][k]; d = s['by partner and direction']
    rows.append([k, f'{s["calyx APL synapses"]:,}', f'{s["of those, uniglomerular PN glomerulus"]:,}', f'{d["PN ->APL"]:,}', f'{d["PN APL->"]:,}',
                 f'{d["KC ->APL"]:,}', f'{d["KC APL->"]:,}', s['microglomeruli with an APL synapse']])
R.table(['APL', 'calyx APL synapses', 'on microglomeruli of a glomerulus', 'PN→APL', 'APL→PN', 'KC→APL', 'APL→KC', 'microglomeruli'], rows,
        title='Synapses used.', caption=f'{SHMG} of APL\'s calyx synapses sit on a microglomerulus of a uniglomerular PN by this rule; the rest '
        'are on KC dendrites away from claws, on PN axons away from boutons, or on multiglomerular PNs.', num_cols=list(range(1, 8)))
nG = len(G)
R.p(f'Glomeruli with at least 20 such synapses in every APL are kept: {nG} glomeruli.')

# ---------------------------------------------------------------- 3 distance
R.section('Step 3: weighted mean distance between two glomeruli')
R.p('The distance between glomeruli $A$ and $B$ is the mean geodesic distance over all pairs of their synapses, each pair weighted by the '
    'product of the two synapse weights:')
R.eq(r'D(A, B) = \frac{\sum_{s \in \mathcal{S}_A}\sum_{t \in \mathcal{S}_B} w_s\, w_t\, d(s, t)}{\sum_{s \in \mathcal{S}_A} w_s \sum_{t \in \mathcal{S}_B} w_t}')
R.p('Computing every $d(s, t)$ would take 10⁸–10⁹ shortest paths. On a tree there is an exact shortcut. A path between $s$ and $t$ uses '
    'edge $e$ exactly when $s$ and $t$ lie on opposite sides of $e$. Summing over edges instead of over pairs gives')
R.eq(r'D(A, B) = \sum_{e} \ell_e \left[\alpha_e (1 - \beta_e) + (1 - \alpha_e)\, \beta_e\right]',
     note='$\\alpha_e$, $\\beta_e$ = share of $A$\'s and $B$\'s synapse weight in the subtree below edge $e$; one pass over the tree gives all of them')
R.p('$\\alpha_e(1-\\beta_e) + (1-\\alpha_e)\\beta_e$ is the weighted share of $(s, t)$ pairs that sit on opposite sides of $e$. The result was '
    'checked against shortest paths computed directly (Dijkstra) on random synapse sets and agrees to 10⁻¹². Two weightings are reported:')
R.ul(['**All synapses**: every APL synapse on the microglomeruli counts once, both directions together. $D(A, B) = D(B, A)$.',
      '**Inputs → outputs**: $s$ runs over APL\'s inputs at $A$ (KC→APL, PN→APL) and $t$ over its outputs at $B$ (APL→KC, APL→PN). The '
      'pair weight is then input synapses × output synapses, Yang\'s idea of multiplying the two directions, made directional: '
      '$A \\to B$ is how far $A$\'s drive to APL has to travel to reach APL\'s outputs onto $B$.'])


def fig_tree():
    fig, ax = plt.subplots(figsize=(8.5, 3.6)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
    nodes = {'r': (5, 3.6), 'u': (2.8, 2.4), 'v': (7.2, 2.4), 'a1': (1.6, 1.0), 'a2': (3.6, 1.0), 'b1': (6.4, 1.0), 'b2': (8.4, 1.0)}
    edges = [('u', 'r'), ('v', 'r'), ('a1', 'u'), ('a2', 'u'), ('b1', 'v'), ('b2', 'v')]
    for c, p in edges:
        (x0, y0), (x1, y1) = nodes[c], nodes[p]
        ax.plot([x0, x1], [y0, y1], color=L.BLOCKED if (c, p) == ('u', 'r') else L.GRAY, lw=3 if (c, p) == ('u', 'r') else 1.5, zorder=1)
    for n, (x, y) in nodes.items(): ax.scatter(x, y, s=40, color=L.INK, zorder=2)
    for n, wa, wb in [('a1', 2, 0), ('a2', 1, 1), ('b1', 0, 2), ('b2', 1, 1)]:
        x, y = nodes[n]
        ax.text(x, y - 0.45, ('A' * wa) + ('  ' if wa and wb else '') + ('B' * wb), ha='center', fontsize=10, color=L.INK, fontweight='semibold')
    ax.text(3.6, 3.15, 'edge $e$', color=L.BLOCKED, fontsize=9)
    ax.text(0.1, 3.9, 'Toy tree: each letter is one synapse of glomerulus A or B at that node.', fontsize=8.5, color=L.MUTED)
    ax.text(0.1, 0.1, 'Below edge $e$: 3 of A\'s 4 synapses ($\\alpha_e$ = 0.75) and 1 of B\'s 4 ($\\beta_e$ = 0.25). Share of (A, B) pairs on opposite '
                      'sides of $e$ = 0.75 × 0.75 + 0.25 × 0.25 = 0.625,\nso $e$ adds 0.625 × its length to $D(A, B)$.', fontsize=8.2, color=L.INK)
    return fig


R.fig(fig_tree(), 'A toy tree to show Equation 2. Glomerulus A has four synapses and B four (letters at the nodes). The orange edge $e$ separates '
      '3 of A\'s synapses and 1 of B\'s from the rest. A pair (one A synapse, one B synapse) crosses $e$ if its two synapses are on '
      'opposite sides, which is true of 62.5% of the pairs here; $e$ therefore adds 0.625 × $\\ell_e$ to the mean distance.', 'Why the edge formula is exact.')

# ---------------------------------------------------------------- 4 sub-arbors
R.section('Step 4: APL\'s calyx arbor comes in sub-arbors')
R.p('Within the calyx, APL\'s skeleton falls into a few connected pieces that join each other only through the peduncle or the lobes. To '
    'find them, skeleton nodes within 5 µm of a calyx APL synapse are called calyx nodes, and the skeleton is restricted to them. Pieces '
    'that hold at least 2% of the synapses on microglomeruli are named S1, S2, … by size. Two synapses in the same sub-arbor are much closer '
    'along APL than two in different ones.')


def fig_sub():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    cols = [L.INTACT, L.BLOCKED, L.GREEN, L.PURPLE, '#c9a227', '#3fb0c9']
    for a, k in zip(axs, KEYS):
        ds, inst, f = ag.APLS[k]; T = ag.load_tree(ag.SK + f); lab = ag.subarbors(T, ds, inst)
        d, _, _ = ag.synapses(ds, inst); gl = d[d.glom != '']
        node = cKDTree(T['xyz']).query(gl[['x', 'y', 'z']].values)[1]; pc = lab[node]
        vc = pd.Series(pc).value_counts(normalize=True); big = [p for p in vc.index if p >= 0 and vc[p] >= 0.02]
        xyz = T['xyz'] / 1e3; X, Y = (-xyz[:, 0], -xyz[:, 2]) if ds == 'hemibrain' else (-xyz[:, 0], -xyz[:, 1])
        ca = lab >= 0
        a.scatter(X[ca], Y[ca], s=0.3, color=L.GRID, lw=0, rasterized=True)
        for i, p in enumerate(big):
            m = lab == p; a.scatter(X[m], Y[m], s=0.6, color=cols[i % len(cols)], lw=0, rasterized=True, label=f'S{i + 1}: {L.pct(vc[p])} of synapses')
        a.set_aspect('equal', adjustable='datalim'); a.legend(markerscale=10, fontsize=6.8, loc='best')
        a.set_xlabel('fly\'s right → (µm)'); a.set_ylabel('dorsal ↑ (µm)'); a.set_title(f'{k}', fontsize=9)
    fig.tight_layout(); return fig


R.fig(fig_sub(), 'APL\'s skeleton nodes in the calyx, seen from behind the fly (posterior view; hemibrain axes −x, −z; male CNS −x, −y; µm). '
      'Each color is one calyx sub-arbor holding at least 2% of APL\'s synapses on microglomeruli; light grey nodes belong to smaller pieces. '
      'The legend gives each sub-arbor\'s share of those synapses.', 'APL\'s calyx sub-arbors.')
R.table(['APL', 'sub-arbors (≥ 2%)', 'largest share', 'same sub-arbor: mean distance (µm)', 'different sub-arbors: mean distance (µm)'],
        [[k, len([x for x in SJ['apl'][k]['calyx sub-arbors holding >= 2% of these synapses, and their shares'] if x.startswith('S')]),
          L.pct(SJ['apl'][k]['calyx sub-arbors holding >= 2% of these synapses, and their shares']['S1']),
          F(SJ['apl'][k]['sub-arbors: mean geodesic distance (µm) between two synapses in the same / different named sub-arbors'][0], 0),
          F(SJ['apl'][k]['sub-arbors: mean geodesic distance (µm) between two synapses in the same / different named sub-arbors'][1], 0)] for k in KEYS],
        title='Sub-arbors.', caption='The number of sub-arbors and the share of the largest depend on the skeleton, not on the synapse '
        'tables, apart from the 2% cut.', num_cols=[1, 2, 3, 4])

# ---------------------------------------------------------------- 5 heatmaps
R.section('Step 5: the glomerulus × glomerulus distances')


def fig_heat():
    fig, axs = plt.subplots(2, 3, figsize=(13, 9.2))
    vmax = np.nanpercentile(np.concatenate([MALL[k].ravel() for k in KEYS] + [MIO[k].ravel() for k in KEYS]), 98)
    for j, k in enumerate(KEYS):
        for i, (M, t) in enumerate([(MALL[k], 'all synapses'), (MIO[k], 'inputs at A → outputs at B')]):
            a = axs[i, j]; im = a.imshow(M[np.ix_(O, O)], cmap=L.SEQ, vmin=0, vmax=vmax, interpolation='nearest')
            a.set_xticks(range(len(O))); a.set_xticklabels(NAMES, rotation=90, fontsize=4.6); a.set_yticks(range(len(O))); a.set_yticklabels(NAMES, fontsize=4.6)
            a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]
            a.set_title(f'{"abcdef"[i * 3 + j]}  {k}, {t}', fontsize=8.5)
            a.set_ylabel('glomerulus A'); a.set_xlabel('glomerulus B')
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label('weighted mean geodesic distance D(A, B), µm', fontsize=9); cb.outline.set_visible(False)
    return fig


R.fig(fig_heat(), f'Each cell is $D(A, B)$ for glomerulus $A$ (row) and $B$ (column), in µm along APL. Top row: all synapses (symmetric). Bottom row: '
      f'APL inputs at $A$ to APL outputs at $B$ (not symmetric). Columns: the three APLs. All panels share one color scale (white 0, dark '
      f'blue at the 98th percentile, {np.nanpercentile(np.concatenate([MALL[k].ravel() for k in KEYS]), 98):.0f} µm). The {nG} glomeruli are in the '
      'same order everywhere, from average-linkage clustering of the mean of the three APLs\' all-synapse distances. The diagonal is the '
      'same glomerulus.', 'Distance along APL between every pair of glomeruli.')
med = SJ['weighted mean geodesic distance (µm), all synapses: median over glomerulus pairs (off-diagonal), median of diagonal']
ASY = {k: float(np.nanmedian(np.abs(MIO[k] - MIO[k].T)[IU])) for k in KEYS}
spm = SJ['spearman between APLs over glomerulus pairs (all synapses, off-diagonal)']
ge = SJ['geodesic / euclidean, median over glomerulus pairs; pearson r between them']
R.ul(['**Long paths.** Median $D$ between different glomeruli: ' + ', '.join(f'{med[k][0]:.0f} µm ({k})' for k in KEYS) + '. That is '
      + ', '.join(f'{ge[k][0]:.1f}' for k in KEYS) + ' times the straight-line distance between the same synapses.',
      '**Bands, not blocks.** A glomerulus\'s distance to all others mostly tracks how much of it sits in the main sub-arbor, which makes '
      'whole rows and columns light or dark.',
      '**Not conserved.** Over glomerulus pairs, Spearman between APLs: ' + '; '.join(f'{a.replace(" vs ", " vs ")} {v:.2f}' for a, v in spm.items()) + '.',
      f'**Inputs → outputs** is close to the all-synapse version; $A \\to B$ and $B \\to A$ differ by {min(ASY.values()):.0f}–{max(ASY.values()):.0f} µm (median over pairs).'])

# ---------------------------------------------------------------- 6 self vs cross; passive
R.section('Step 6: self versus cross, and an illustration with passive decay')


def fig_self():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1))
    a = axs[0]
    for k, mk, col in zip(KEYS, ['o', 's', '^'], [L.BLOCKED, L.GREEN, L.INTACT]):
        M = MALL[k]; diag = np.diag(M); cross = (np.nansum(M, 1) - diag) / (M.shape[1] - 1)
        a.scatter(cross, diag, marker=mk, s=18, facecolor='white', edgecolor=col, lw=1, label=f'{k}: self shorter in {L.pct(np.mean(diag < cross))}')
    lim = [0, a.get_xlim()[1]]; a.plot(lim, lim, color=L.GRID, lw=0.8); a.set_xlim(lim); a.set_ylim(0, None)
    a.set_xlabel('mean D to the other glomeruli (µm)'); a.set_ylabel('D to itself (diagonal, µm)'); a.legend(fontsize=7)
    a.set_title('a  Self against cross, one point per glomerulus')
    a = axs[1]
    for k, col in zip(KEYS, [L.BLOCKED, L.GREEN, L.INTACT]):
        med_, lo_, hi_ = [], [], []
        for l in LAMS:
            C = mat(k, f'passive_spread_inputs_A_to_outputs_B_lambda_{l}um'); diag = np.diag(C); cross = (np.nansum(C, 1) - diag) / (C.shape[1] - 1)
            r_ = diag / cross; med_.append(np.median(r_)); lo_.append(np.quantile(r_, 0.25)); hi_.append(np.quantile(r_, 0.75))
        a.plot(LAMS, med_, color=col, marker='o', ms=4, label=k); a.fill_between(LAMS, lo_, hi_, color=col, alpha=0.15, lw=0)
    a.axhline(1, color=L.INK, lw=0.8, ls=':'); a.set_xscale('log'); a.set_xticks(LAMS); a.set_xticklabels([str(x) for x in LAMS])
    a.set_xlabel('assumed length constant λ along APL (µm, log scale)'); a.set_ylabel('self ÷ mean cross coupling, $C_λ(A→A) / C_λ(A→B)$')
    a.legend(fontsize=7.5); a.set_title('b  How much more a glomerulus inhibits itself')
    fig.tight_layout(); return fig


R.p('If APL\'s signal decayed passively as $e^{-d/\\lambda}$ with path length, the coupling from APL\'s inputs at $A$ to its outputs at $B$ would be')
R.eq(r'C_\lambda(A \to B) = \frac{\sum_{s \in \mathrm{in}(A)} \sum_{t \in \mathrm{out}(B)} e^{-d(s,t)/\lambda}}{|\mathrm{in}(A)|\;|\mathrm{out}(B)|}',
     note='computed exactly on the tree in two passes (children to parents, then parents to children); 1 would mean every input sits on every output')
R.p('This is an illustration only. APL\'s length constant is not known, and a real cable also loses signal at branch points and depends '
    'on neurite diameter, which skeletons do not carry. The ratio of the self term to the mean cross term says how much more a '
    'glomerulus\'s KCs inhibit their own glomerulus than another (Figure 4b).')
R.fig(fig_self(), '**a**, Each point is one glomerulus in one APL: its mean distance to the other glomeruli (horizontal) against its distance to '
      'itself (vertical; all synapses). Points below the grey line have a shorter self distance. **b**, The self coupling divided by the mean '
      'cross coupling, $C_\\lambda(A \\to A)$ over the mean of $C_\\lambda(A \\to B)$, for four assumed length constants. Lines are medians over '
      'glomeruli and shaded bands the 25th–75th percentiles. The dotted line at 1 means no preference for self.', 'Self versus cross inhibition.')

# ---------------------------------------------------------------- 7 second version
R.section('Earlier version: official ROIs and the old claws')
mo = SJo['weighted mean geodesic distance (µm), all synapses: median over glomerulus pairs (off-diagonal), median of diagonal']
R.p('The earlier version of this report used the official ROIs and the old 2 µm claws. Median distances between glomeruli there: '
    + ', '.join(f'{mo[k][0]:.0f} µm ({k})' for k in KEYS) + ' (now: ' + ', '.join(f'{med[k][0]:.0f}' for k in KEYS) + ' µm). Synapses on '
    'microglomeruli: ' + ', '.join(f'{SJo["apl"][k]["of those, uniglomerular PN glomerulus"]:,}' for k in KEYS) + ' then, '
    + ', '.join(f'{SJ["apl"][k]["of those, uniglomerular PN glomerulus"]:,}' for k in KEYS) + ' now (same order). Its files are in `apl_geodesic/`.')
R.box('Caveats', ['The sub-arbor split depends on the skeleton\'s connectivity. A connection missed by the segmentation, or a false merge, would '
                  'change it, so the sub-arbors are worth checking in neuroglancer before relying on them.',
                  'Synapse counts stand in for synaptic strength, and the passive decay ignores branching and diameter.',
                  'The skeleton nodes are 0.4 µm (hemibrain) and 0.7 µm (male CNS) apart on median; distances are exact on the skeleton, not on the membrane.'], 'caveat')
R.save()
