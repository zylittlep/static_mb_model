"""Report 00: overview, roadmap and glossary of the step-by-step reports."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import report_lib as L
import pandas as pd
L.style(); L.DATE = '2026-11-04'
F = L.fmt
OWV = L.B + 'with_outside_ROI_synapses/claws_v2m2/odor_weighted/'
TVM = json.load(open(OWV + 'TNT_vs_model_summary.json')); FO = TVM['fits']['threshold 1.0, ORN']
FIT = pd.read_csv(OWV + 'TNT_vs_model_fits.csv'); FIT = FIT[FIT.threshold == 1.0].set_index(['responses', 'version'])
RO = pd.read_csv(OWV + 'TNT_rank_order.csv'); _ro = RO[(RO.source == 'data') & (RO.responses == 'raw responses')].set_index('comparison').spearman
CORR = _ro['half vs half, TNT vs TNT3c'] / np.sqrt(_ro['half vs half, within TNT3c (mean of 1000 splits)'] * _ro['half vs half, within TNT'])
PR = pd.read_csv(OWV + 'TNT_vs_model_odor_pairs.csv'); PR = PR[(PR.threshold == 1.0) & (PR.responses == 'ORN')]
from scipy.stats import spearmanr
RHO = [spearmanr(PR.S_model_blocked, PR.S_model_blocked - PR[c])[0] for c in PR.columns if c.startswith('dS_model_')]
SL = json.load(open(OWV + 'APL_local_model_summary.json')); SC = json.load(open(OWV + 'APL_compartment_model_summary.json'))
NUL = {rn: SL[rn][next(k for k in SL[rn] if k.startswith('null'))]['r real, null mean, null 5-95%, p'][3] for rn in ['ORN', 'GH146']}
RKC = [v for rn in ['ORN', 'GH146'] for k, v in SC[rn]['model: Spearman blocked vs intact (mean over MBs) by version'].items()
       if k.startswith('compartments')]

R = L.Report('00', 'What was done in this project, in which order, where each step is explained, and what every recurring term and symbol means.')

R.box('Revision of 2026-11-04', [
    'All reports were rebuilt for Yang\'s revision list of 2026-11-03. They now use the **corrected calyx** (synapses outside the official '
    'ROIs added), the **new claws** (Yang\'s rule of 2026-11-04: a KC\'s calyx sites clustered at 8 µm along its skeleton, one claw per PN '
    'with at least 2 PSDs) and the **new ORN table** (from `responses_wide.csv`, 2026-11-03). The model uses hemibrain R and male CNS L and R.',
    'Each report opens with what changed. The earlier versions are kept in `reports/archive_2026-11-03_before_revision/` (their scripts in `scripts/reports/archive_2026-11-03/`), and the official-ROI files are unchanged.',
    'New in the model (Report 05): a PN-level transformation of the ORN responses fitted to GH146, runs with negative responses kept, and '
    'version MC (per-KC counts plus bouton inhibition per glomerulus). The questions of the revision list are answered in the reports '
    'themselves (Reports 05–11).'], 'plain')
R.p('The project asks how the APL neuron, the single inhibitory neuron of each mushroom body, is wired with the projection neurons (PNs) '
    'and Kenyon cells (KCs) in the calyx, and whether that wiring can explain how blocking APL changes the KC odor code in Yang\'s '
    'recordings (TNT, APL blocked, vs the TNT3c control). The work falls into three parts: building the synapse data (Reports 01–03), '
    'a model of KCs with APL feedback compared with the recordings (Reports 04–09), and APL\'s geometry with local, non-spiking versions of '
    'the model (Reports 10–12). Each report states its question, inputs, outputs and scripts at the top, defines every symbol in a table, '
    'and explains each figure in its caption. All reports exist as HTML (links work, best on screen) and PDF.')


def fig_road():
    fig, ax = plt.subplots(figsize=(10.4, 7.3)); ax.axis('off')
    W, H = 2.85, 1.3
    X = [0.2, 3.45, 6.7, 9.95]; Y = {1: 7.25, 2: 5.0, 3: 2.75, 4: 0.5}
    boxes = {'01': (0, 1, 'Connectome synapse\ntables (4 datasets)'), '02': (1, 1, 'Claws, boutons, calyx view,\noutside-ROI version'),
             '03': (2, 1, 'Odor responses per\nglomerulus, odor maps'),
             '04': (0, 2, 'APL wiring per\nglomerulus'), '05': (1, 2, 'KC model with APL\nfeedback, step by step'),
             '07': (2, 2, 'Model pinned to the\nrecordings, compared'), '06': (3, 2, 'TNT and TNT3c\nrecordings'),
             '10': (0, 4, 'Distances along APL\nbetween glomeruli'), '11': (1, 4, 'Local, non-spiking APL\nin the KC model'),
             '12': (2, 4, 'APL confined to its\ncalyx sub-arbors'),
             '08': (2, 3, 'Why feedback makes the\neffect uneven'), '09': (3, 3, 'Rank order of\nsimilarity, toy models')}
    col = {'01': L.GRAY, '02': L.GRAY, '03': L.GRAY, '04': L.INTACT, '05': L.INTACT, '06': L.BLOCKED, '07': L.INTACT, '08': L.INTACT,
           '09': L.INTACT, '10': L.PURPLE, '11': L.PURPLE, '12': L.PURPLE}
    P = {k: (X[c], Y[r]) for k, (c, r, _) in boxes.items()}
    for k, (c, r, t) in boxes.items():
        x, y = P[k]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0.02,rounding_size=0.12', fc='white', ec=col[k], lw=1.6))
        ax.text(x + 0.12, y + H - 0.12, k, fontsize=10, fontweight='bold', color=col[k], va='top')
        ax.text(x + W / 2 + 0.15, y + H / 2 - 0.08, t, fontsize=8, ha='center', va='center', color=L.INK)

    def arrow(a, b):
        (xa, ya), (xb, yb) = P[a], P[b]
        if abs(ya - yb) < 1e-6:
            p0, p1 = ((xa + W, ya + H / 2), (xb, yb + H / 2)) if xb > xa else ((xa, ya + H / 2), (xb + W, yb + H / 2))
        elif abs(xa - xb) < 1e-6:
            fx = 0.88 if xa == X[0] else 0.5      # keep clear of the row labels at the left
            p0, p1 = (xa + W * fx, ya), (xb + W * fx, yb + H)
        else:
            p0, p1 = (xa + W * 0.25, ya), (xb + W * 0.75, yb + H)
        ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='-|>', color=L.MUTED, lw=1.1, shrinkA=2, shrinkB=2))
    for a, b in [('01', '02'), ('02', '03'), ('01', '04'), ('02', '05'), ('03', '05'), ('04', '05'), ('05', '07'), ('06', '07'),
                 ('07', '08'), ('08', '09'), ('05', '11'), ('10', '11'), ('11', '12')]:
        arrow(a, b)
    for r, t in [(1, 'Synapse data'), (2, 'Model and recordings'), (3, 'Explanations'), (4, 'APL geometry and local models')]:
        ax.text(0.2, Y[r] + H + 0.18, t, fontsize=9, fontweight='bold', color=L.MUTED)
    ax.set_xlim(0, 13); ax.set_ylim(0.3, 9.0)
    return fig


R.fig(fig_road(), 'Each box is one report (number top left). Arrows point from a report to a report that builds on it; reading in number '
      'order also works. Not drawn, to keep the figure readable: Report 10 uses the claws and boutons of Report 02, and Report 11 uses the '
      'distances of Report 10, the pinning of Report 07 and the rank-order readout of Report 09; Report 12 uses the sub-arbors of Report 10. Box outlines: gray = synapse data; blue = the '
      'model and its comparison with the recordings; orange = the recordings; purple = APL geometry and the local models.', 'Roadmap of the reports.')

# ---------------------------------------------------------------- summaries
R.section('The reports in order')
S = [
    ('01', 'One table per connectome (FlyWire, hemibrain, male CNS, BANC) with every APL synapse in the mushroom body: who the partner is, '
            'where the synapse is, and its region and compartment. Synapses are counted as T-bar → PSD pairs. KCs are APL\'s main partner '
            'everywhere; in the calyx PNs come second. Per KC, synapses onto APL and from APL go together (Spearman 0.62–0.75, BANC 0.46).'),
    ('02', 'Boutons (on PN axons) are clusters of a PN\'s T-bars within 2 µm. Claws (on KC dendrites) follow Yang\'s rule of 2026-11-04: '
            'a KC\'s calyx sites are clustered at 8 µm along its skeleton and every PN with at least 2 PSDs in a cluster gets its own claw; this '
            'matches Prat\'s hemibrain claw counts best of the rules tried. About 56–58% of KC → APL and 66–69% of APL → KC calyx synapses '
            'now sit on claws (hemibrain, male CNS). The corrected calyx adds synapses just outside the official ROIs, mostly the back of '
            'the male CNS calyx. The posterior view used in every map is defined here.'),
    ('03', 'Yang\'s ORN (new table, 2026-11-03) and GH146 responses are averaged over flies per glomerulus and attached to each APL synapse '
            'through its PN or claw, in 2D and 3D maps. An odor\'s strongest boutons are scattered over the whole calyx. Weighting odor '
            'similarity by APL\'s synapses barely changes it.'),
    ('04', 'Per glomerulus: APL → PN synapses per PN → KC synapse (bouton weight $p_g$), APL → KC synapses on its claws per claw PN synapse '
            '(claw weight $q_g$), and the boutons\' share $f$ of APL\'s calyx output (about 10%). These are the wiring numbers the model uses.'),
    ('05', 'The KC model: KC input from claws × glomerulus responses, a threshold, and APL feedback solved self-consistently, in a uniform '
            'version and versions using per-KC counts, per-glomerulus bouton and claw weights, and both together (MC). Built and explained '
            'step by step, with a PN-level transformation of the ORN input, negative responses, every version\'s similarity maps and the '
            'recordings\' maps for comparison.'),
    ('06', 'The recordings (FC z-scores): 8 TNT and 11 TNT3c recordings (each treated as an independent hemisphere), 16 odors. With APL '
            'blocked more units respond and every odor pair becomes more similar (mean change in cosine similarity 0.18). Most of the '
            'variation between pairs is per odor, and it follows chance overlap: odors that drive many units in TNT overlap more with everything.'),
    ('07', 'The model is pinned to the recordings (each odor\'s blocked fraction matches TNT; one APL gain matches TNT3c\'s mean). Uniform '
            f'feedback predicts which pairs gain similarity (r {FIT.loc[("ORN", "U"), "r_pairs"]:.2f} with ORN input, '
            f'{FIT.loc[("GH146", "U"), "r_pairs"]:.2f} with GH146; data ceiling about {FO["ceiling for r with a model (sqrt of that)"]:.2f}). '
            'Per-KC counts and bouton weights change $r$ by a few hundredths at most. The model\'s change is '
            f'{FIT.loc[("ORN", "U"), "model_mean_dS"] / FIT.loc[("ORN", "U"), "data_mean_dS"]:.2f} times the data\'s.'),
    ('08', 'Why: feedback holds down odors that drive many KCs more than odors that drive few, so blocking APL helps strong odors most, '
            'and pairs of strong odors gain the most similarity by chance overlap. Feedback fits better than a fixed threshold raise with the '
            'same mean inhibition, and as well as the data\'s own per-odor fractions. The recordings show this signature in response size but only weakly in the number of responding units.'),
    ('09', 'Whether inhibition changes the **order** of odor-pair similarities. Global inhibition keeps the order exactly only in special cases; '
            f'the model keeps it almost (Spearman {min(RHO):.2f}–{max(RHO):.2f} over versions), the data less ({_ro["TNT vs TNT3c, all recordings"]:.2f}, '
            f'about {CORR:.2f} after correcting for noise). Toy simulations show each case.'),
    ('10', 'Distances along APL\'s skeleton between the microglomeruli of every pair of glomeruli, hemibrain and male CNS. APL\'s calyx arbor '
            'comes in sub-arbors that join only farther down, so glomeruli in the same sub-arbor are much closer along APL.'),
    ('11', 'APL as a non-spiking neuron: inhibition spreads along its skeleton and fades as $e^{-d/\\lambda}$. Short-range inhibition on KCs '
            'predicts which odor pairs change most better than a uniform APL (the real wiring beats within-class shuffles for '
            + (' and '.join(rn for rn in NUL if NUL[rn] <= 0.05) or 'neither input') + ' input'
            + ('' if all(v <= 0.05 for v in NUL.values()) else ', not for ' + ' and '.join(rn for rn in NUL if NUL[rn] > 0.05)) + '), '
            'but barely changes the rank order; inhibition on PN boutons alone comes closest to the data\'s reordering, and is too weak at '
            'its real share.'),
    ('12', 'APL with its signal confined to each calyx sub-arbor of Report 10 (no crossing between them). Rank order stays at '
            f'{min(RKC):.2f}–{max(RKC):.2f}, '
            'like a uniform APL; the pattern of which pairs change is no better than one APL unless each sub-arbor\'s potential is divided by '
            'its cable length, and never better than the local model without walls.'),
]
for num, txt in S:
    R.p(f'**{R.ref(num)}.** ' + txt)

# ---------------------------------------------------------------- glossary
R.section('Glossary of terms')
G = [
    ('APL', 'Anterior paired lateral neuron: one large GABAergic (inhibitory) neuron per mushroom body that receives input from KCs and inhibits them and the PN boutons. Thought to be non-spiking (graded potentials), like the locust GGN.'),
    ('Kenyon cell (KC)', 'The intrinsic neurons of the mushroom body (about 2,000 per side). Classes αβ, α′β′ and γ by the lobes their axons enter. KCs get odor input in the calyx and send output in the lobes.'),
    ('Projection neuron (PN)', 'An antennal-lobe output neuron that carries odor information from one glomerulus (uniglomerular) or several to the calyx and the lateral horn.'),
    ('ORN', 'Olfactory receptor neuron. ORNs expressing the same receptor converge on one glomerulus.'),
    ('Glomerulus', 'A unit of the antennal lobe (e.g. DM1, VA1v); names follow FlyWire. Each uniglomerular PN belongs to one.'),
    ('Calyx, peduncle, lobes', 'Parts of the mushroom body: dendritic input region, axon bundle, and axon output region (α, α′, β, β′, γ lobes).'),
    ('Compartment', 'Subdivision of a lobe (α1 … γ5) defined by its dopaminergic and output neurons.'),
    ('Bouton', 'A swelling of a PN axon in the calyx holding many release sites; here a cluster of the PN\'s T-bars within 2 µm.'),
    ('Claw', 'A KC dendritic ending that wraps a bouton. Here (from 2026-11-04): within an 8 µm cluster of the KC\'s calyx sites along its skeleton, the sites of one PN with at least 2 PSDs. Its glomerulus is that PN\'s.'),
    ('Microglomerulus', 'One bouton with the claws around it, and APL\'s synapses there.'),
    ('T-bar, PSD', 'The presynaptic release site and a postsynaptic density facing it. One T-bar can face several PSDs.'),
    ('Synaptic weight', 'Number of T-bar → PSD pairs from one neuron onto another in a region (neuPrint "weight", FlyWire "syn_count").'),
    ('ROI', 'Region of interest: a dataset\'s official 3D volume for a brain region. The corrected calyx adds synapses just outside the official ROIs; it is used everywhere from 2026-11-03.'),
    ('Hemibrain, male CNS, FlyWire, BANC', 'The four electron-microscopy connectomes (Report 01). The model uses hemibrain (right side) and male CNS (both sides).'),
    ('Posterior view', 'Calyx seen from behind: fly\'s right to the right, dorsal up (Report 02, Table 2).'),
    ('ORN table, GH146', 'Yang\'s glomerular responses: ORN imaging (new table from `responses_wide.csv`, fly means; 16 odors used by the model) and GH146 PN imaging (peak ΔF/F; 13 of those odors).'),
    ('FC z-score', 'The unit of the TNT and TNT3c responses (Yang, 2026-11-03).'),
    ('TNT, TNT3c', 'Yang\'s KC recordings with APL blocked by tetanus toxin (TNT) and the control with an inactive toxin (TNT3c, APL intact).'),
    ('Recording, unit', 'One imaged hemisphere (treated as independent), and one imaged KC region of interest in it.'),
    ('Responding, fraction responding', 'A unit responds to an odor when its response exceeds 1 (data); a model KC when its response is above 0. The fraction responding is also called the coding level.'),
    ('Cosine similarity', 'Similarity of two odors\' population responses: dot product divided by the product of lengths. 1 − cosine is the distance in Yang\'s RDMs.'),
    ('ΔS', 'Change in similarity of an odor pair from intact to blocked (TNT − TNT3c in the data).'),
    ('Rank order', 'The ordering of odor pairs by similarity. Measured by the Spearman correlation between blocked and intact similarities over pairs.'),
    ('Spearman, Pearson', 'Rank correlation, and linear correlation.'),
    ('Threshold-linear', 'A response that is 0 below a threshold and grows linearly above it: $[x - \\theta]_+$.'),
    ('Feedback, self-consistent', 'APL\'s activity depends on the KC responses it inhibits, so responses and APL activity are solved together as a steady state.'),
    ('Pinning', 'Setting each odor\'s input gain so that the model\'s blocked fraction responding equals the TNT fraction (Report 07).'),
    ('Calibration', 'Choosing one gain (APL strength) by bisection so that a target is met, here the mean intact fraction responding of TNT3c.'),
    ('Shuffle, null', 'Rewiring that keeps totals but breaks the real assignment (e.g. KCs swap synapse counts within their class), to test whether the real wiring matters.'),
    ('Bootstrap', 'Resampling recordings with replacement to get the spread of a statistic.'),
    ('Noise ceiling', 'How well any prediction could match the data given the data\'s own variability (split-half reliability).'),
    ('Geodesic distance', 'Distance along a neuron\'s skeleton between two points, as opposed to a straight line.'),
    ('Length constant λ', 'In a passive neurite, the distance over which a steady signal falls to 37% ($1/e$).'),
    ('Sub-arbor', 'A piece of APL\'s calyx arbor that connects to the others only outside the calyx (Report 10); a compartment in Report 12.'),
]
R.table(['Term', 'Meaning'], [[a, b] for a, b in G], title='Terms.', caption='Symbols are defined anew in each report\'s own table; the most common ones are below.')
R.symbols([
    ('k,\\ j', 'Kenyon cells', ''), ('g', 'Glomerulus', ''), ('o,\\ a,\\ b', 'Odors (a, b as a pair)', ''),
    ('W_{kg}', 'Share of KC $k$\'s claw PN input from glomerulus $g$', '0 to 1'),
    ('r_{go}', 'Response of glomerulus $g$ to odor $o$ (negatives set to 0 in the model)', 'ORN or GH146 units'),
    ('g_o', 'Input gain of odor $o$ (pinning)', ''),
    ('x_{ko},\\ y_{ko}', 'Input and response of KC $k$ to odor $o$', ''),
    ('\\theta', 'KC threshold', '1 in Reports 07–12'),
    ('A_o', 'APL activity for odor $o$ (one-unit APL)', ''),
    ('a', 'APL gain onto KCs', 'calibrated'),
    ('u_k,\\ v_k', 'APL → KC and KC → APL calyx synapses of KC $k$, each ÷ its mean', 'mean 1'),
    ('p_g,\\ q_g', 'Per-glomerulus bouton and claw weights', 'Report 04'),
    ('f', 'Boutons\' share of APL\'s calyx output synapses', 'about 0.1'),
    ('M_{kj}', 'Coupling from KC $j$ to KC $k$ through APL', 'Reports 11–12'),
    ('d(s,t),\\ \\lambda', 'Distance along APL between sites $s$ and $t$; length constant', 'µm'),
    ('\\phi_o', 'Fraction of KCs (or units) responding to odor $o$', '0 to 1'),
    ('S_{ab},\\ \\Delta S_{ab}', 'Cosine similarity of odors $a$ and $b$; its change when APL is blocked', ''),
    ('\\rho_{\\mathrm{rank}}', 'Spearman correlation over pairs between blocked and intact similarity', ''),
    ('r', 'Pearson correlation over pairs between model and data $\\Delta S$', ''),
], title='Recurring symbols')

R.section('Decisions that apply to every report')
R.ul(['Synapse counts are T-bar → PSD pairs and stand in for synaptic strength.',
      'Two versions of every table and plot are kept: official ROIs, and with synapses outside them added (Yang, 2026-10-14). From '
      '2026-11-03 the reports use the corrected calyx (synapses outside the ROIs added), and from 2026-11-04 the new claws.',
      'The model and the comparison with the recordings use only hemibrain and male CNS (Yang, 2026-10-21).',
      'TNT3c is the control (APL intact) and TNT has APL blocked; each recording is an independent hemisphere (Yang, 2026-10-15).',
      'Odor-pair similarity is cosine similarity, as in Yang\'s RDMs (from 2026-10-21).',
      'Per-KC KC → APL and APL → KC counts in the model are calyx only (from 2026-10-21).'])
R.save()
