"""Report 01: the connectome data. Where the APL synapse tables come from, what one row is, the filters, the region and
compartment labels, and the per-KC and per-PN counts that later reports use."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import report_lib as L
L.style()
F = L.fmt
B = L.B
DS4 = ['flywire', 'hemibrain', 'malecns', 'banc']
LAB = L.DS_LABEL
LOBES = {'VL', 'ML', 'aL', "a'L", 'bL', "b'L", 'gL'}
C_IN, C_OUT = L.GREEN, L.PURPLE        # partner -> APL, APL -> partner (this report only)

# ---------------------------------------------------------------- data
T = {}
for ds in DS4:
    t = pd.read_csv(f'{B}{ds}_APL_MB_synapses.csv.gz', usecols=['direction', 'apl_id', 'apl_side', 'apl_instance', 'mb_region', 'partner_type',
                                                              'partner_class', 'mb_compartment', 'mb_compartment_source', 'claw_id', 'bouton_id'],
                    low_memory=False)
    t['out'] = t.direction.str.startswith('APL_output')
    t['region3'] = np.where(t.mb_region == 'CA', 'calyx', np.where(t.mb_region == 'PED', 'peduncle', np.where(t.mb_region.isin(LOBES), 'lobes', 'other')))
    pc = t.partner_class.fillna('').str.lower(); pt = t.partner_type.fillna('')
    t['pgroup'] = np.select([pc.str.contains('kenyon') | pt.str.startswith('KC'), pc.isin(['alpn', 'antennal_lobe_projection_neuron']),
                             pc.isin(['dan', 'mushroom_body_dopaminergic_neuron']), pc.isin(['mbon', 'mushroom_body_output_neuron']),
                             pt.eq('') & pc.eq('')],
                            ['Kenyon cells', 'Antennal-lobe PNs', 'Dopaminergic neurons', 'MB output neurons', 'Unlabeled partner'], 'Other labeled')
    T[ds] = t
KCW = pd.read_csv(B + 'KC_weights_APL_PN_per_KC.csv')
PNW = pd.read_csv(B + 'PN_calyx_weights_APL_vs_KC.csv')
VAL = json.load(open(B + 'malecns_compartment_validation.json'))
LVAL = json.load(open(B + 'lobe_compartment_validation.json'))
N = {ds: len(T[ds]) for ds in DS4}
assert N == {'flywire': 277252, 'hemibrain': 241853, 'malecns': 439589, 'banc': 67241}, N

R = L.Report('01', 'Where do the APL synapse tables come from, what does one row mean, how are synapses counted, and how many synapses does '
             'APL make with each kind of partner in each part of the mushroom body?',
             inputs=['Public synapse tables of four adult fly connectomes (downloaded 2026-10-07; sources in Table 2)'],
             outputs=['`<dataset>_APL_MB_synapses.csv.gz`: one row per APL synapse in the mushroom body (four datasets)',
                      '`KC_weights_APL_PN_per_KC.csv`: per Kenyon cell, its PN input and its synapses with APL (calyx and whole mushroom body)',
                      '`PN_calyx_weights_APL_vs_KC.csv`, `PN_MB_weights_APL_vs_KC.csv`: per projection neuron, its synapses onto KCs and with APL',
                      '`summary_counts_by_compartment_and_partner_class.csv`, the scatter plots `scatter*.png`'],
             scripts=['`scripts/hemibrain.py`, `malecns.py`, `malecns2.py`, `fafb_princeton.py`, `banc_extract.py` (download and filter)',
                      '`scripts/prep.py`, `run.py`, `export.py` (regions, compartments, claws, boutons, export)',
                      '`scripts/scatter.py`, `mb_plots.py`, `pnmb_*.py` (per-cell counts and scatter plots)',
                      '`scripts/reports/r01_connectome_tables.py` (this report)'])

R.p('Every later step starts from the tables described here. Reports 02 and 03 add structure to them (claws, boutons, odor responses); '
    'Reports 04 to 11 use the hemibrain and male CNS parts of them to build and test a model. This report uses the tables with the official '
    'regions of interest (ROIs) only. Report 02 describes the second version, which adds synapses that lie just outside the official ROIs.')

R.symbols([
    ('i,\\ j', 'Two neurons; $i$ presynaptic, $j$ postsynaptic', 'neuron IDs of each dataset'),
    ('w_{ij}', 'Synaptic weight from $i$ to $j$: the number of synaptic connections (T-bar → postsynaptic density pairs) from $i$ onto $j$, '
     'counted inside the region named', 'whole number ≥ 0; same as neuPrint "weight" and FlyWire "syn_count"'),
    ('\\mathcal{R}', 'A region of the mushroom body: calyx (CA), peduncle (PED) or a lobe', 'official ROI of each dataset'),
    ('n^{\\mathrm{PN\\to KC}}_k', 'PN input of Kenyon cell $k$: $\\sum_{i\\in\\mathrm{PNs}} w_{ik}$ in the calyx', 'synapses per KC'),
    ('n^{\\mathrm{KC\\to APL}}_k', 'Synapses from KC $k$ onto APL, $w_{k,\\mathrm{APL}}$, in the calyx (or the whole mushroom body where stated)', 'synapses per KC'),
    ('n^{\\mathrm{APL\\to KC}}_k', 'Synapses from APL onto KC $k$, $w_{\\mathrm{APL},k}$, in the calyx (or the whole mushroom body)', 'synapses per KC'),
    ('n^{\\mathrm{PN\\to APL}}_i,\\ n^{\\mathrm{APL\\to PN}}_i', 'Synapses between projection neuron $i$ and APL in the calyx, each direction', 'synapses per PN'),
    ('\\rho_s', 'Spearman rank correlation over cells', '−1 to 1'),
    ('k_{\\mathrm{NN}}', 'Number of nearest dopaminergic T-bars that vote on a lobe synapse\'s compartment (Section 4)', '15'),
])

# ---------------------------------------------------------------- 1
R.section('The four connectomes and how each synapse is counted')
R.p('A connectome is a map of every neuron and synapse in a piece of nervous tissue, traced from electron microscopy. A chemical synapse '
    'is seen as one presynaptic release site (the **T-bar**) facing one or more postsynaptic densities (**PSDs**) on other neurons. All '
    'tables here count **T-bar → PSD pairs**: one T-bar facing three PSDs on three cells counts once for each cell, so the weight $w_{ij}$ '
    'is the number of PSDs on $j$ that face T-bars of $i$. This is the convention of neuPrint (hemibrain, male CNS) and FlyWire.')
R.eq('w_{ij}(\\mathcal{R}) \\;=\\; \\#\\{(t, s)\\ :\\ t \\text{ is a T-bar of } i,\\ s \\text{ is a PSD of } j \\text{ facing } t,\\ s \\in \\mathcal{R}\\}',
     'A synapse belongs to region $\\mathcal{R}$ by the position of its postsynaptic site (hemibrain, male CNS) or by the dataset\'s own neuropil label (FlyWire, BANC).')
R.p('The APL neuron (anterior paired lateral) is one large GABAergic neuron per brain hemisphere. It covers the whole mushroom body, '
    'receives input from the Kenyon cells and inhibits them back. Every APL synapse inside a mushroom body ROI was kept, in both '
    'directions and with every partner.')
R.table(['Dataset', 'Animal', 'Source table', 'Filter', 'APLs', 'Rows'],
        [['FlyWire 783 (FAFB)', 'female, whole brain', 'Codex "Princeton" synapse table, v783',
          'proofread neurons, no autapses, ≥5 synapses per neuron pair (Codex default)', '2 (left, right)', f'{N["flywire"]:,}'],
         ['Hemibrain v1.2', 'female, central brain, right side', 'neuPrint release feather files (synapse partners and points)',
          'release confidence thresholds', '1 (APL_R; the left APL is outside the volume)', f'{N["hemibrain"]:,}'],
         ['Male CNS v1.0', 'male, brain and nerve cord', 'neuPrint release, `minconf-0.5` partner table', 'synapse confidence ≥ 0.5',
          '2 (APL_L, APL_R)', f'{N["malecns"]:,}'],
         ['BANC 888', 'female, brain and nerve cord', 'BANC v2 enriched synapse table', 'synapse size ≥ 5 (as in the BANC paper)', '2 (left, right)',
          f'{N["banc"]:,}']],
        title='The four connectomes.', caption='Rows = APL synapses inside the official mushroom body ROIs. Every source is a public, '
        'no-login download. Hemibrain and male CNS come from the FlyEM team (Janelia), FlyWire from the FlyWire consortium (Princeton), '
        'BANC from the BANC consortium. The model in Reports 05 to 11 uses only hemibrain and male CNS (Yang, 2026-10-21).',
        num_cols=[5])
R.box('Why BANC counts are low', 'BANC has 3–5 times fewer APL synapses than the other datasets (median KC↔APL synapses per KC in the calyx: '
      f'{F(KCW[KCW.dataset == "banc"].KC_to_APL_calyx.median(), 0)} vs {F(KCW[KCW.dataset == "hemibrain"].KC_to_APL_calyx.median(), 0)} in hemibrain). '
      'This follows from its synapse detector and the size ≥ 5 filter, and it is why BANC numbers are treated as rough throughout.', 'caveat')
R.box('Two FlyWire versions', 'The FlyWire rows in the main table keep only neuron pairs with ≥5 synapses (Codex default). The per-KC and '
      'per-PN counts (Section 5) instead use every Princeton synapse between proofread neurons, without that threshold, because a KC often '
      'makes fewer than 5 synapses with APL in the calyx. So FlyWire per-cell counts are larger than counts from the main table.', 'caveat')

# ---------------------------------------------------------------- 2
R.section('What one row of the table is')
R.p('Each row is one T-bar → PSD pair where APL is on one side. The columns that the later steps use:')
R.table(['Column', 'Meaning'],
        [['`direction`', '`APL_output (APL presynaptic)`: APL → partner. `APL_input (APL postsynaptic)`: partner → APL.'],
         ['`apl_id`, `apl_side`', 'Which APL. Each mushroom body has its own APL, so "APL side" = which mushroom body.'],
         ['`partner_id`, `partner_type`, `partner_class`', 'The other neuron, its cell type and its broad class (blank = untyped fragment)'],
         ['`pre_*_nm`, `post_*_nm`', 'Positions of the T-bar and the PSD, in nanometres, in the dataset\'s own coordinate space'],
         ['`partner_side_*_nm`', 'The partner\'s own site (the PSD when APL is presynaptic, the T-bar when APL is postsynaptic); used to place a synapse on a claw or bouton'],
         ['`roi_official`, `mb_region`', 'The dataset\'s own ROI label, and the mushroom body region: CA, PED, or a lobe'],
         ['`mb_compartment`, `mb_compartment_source`', 'The finest compartment label (α1 … γ5 in the lobes) and whether it is official or derived (Section 4)'],
         ['`claw_id`, `bouton_id` (and `nearest_*`, `*_dist_nm`)', 'The KC claw or PN bouton the synapse sits on, if any (Report 02)']],
        title='Main columns of `<dataset>_APL_MB_synapses.csv.gz`.', caption='The README lists every column.')

# ---------------------------------------------------------------- 3
R.section('Regions of the mushroom body and how many APL synapses each has')
R.p('The mushroom body has three parts. The **calyx** (CA) is where projection neurons (PNs) from the antennal lobe pass odor information '
    'to the dendritic claws of the Kenyon cells (KCs). The **peduncle** (PED) is the bundle of KC axons leaving the calyx. The **lobes** are '
    'where KC axons end and talk to output neurons (MBONs) and dopaminergic neurons (DANs). Hemibrain and male CNS split the lobes into '
    'α, α′, β, β′ and γ lobes; FlyWire and BANC only into a vertical lobe (VL: α + α′) and a medial lobe (ML: β + β′ + γ).')


def fig_regions():
    fig, axs = plt.subplots(1, 4, figsize=(10.2, 3.0), sharey=False)
    for ax, ds in zip(axs, DS4):
        t = T[ds]; nap = t.apl_id.nunique()
        g = t.groupby(['region3', 'out']).size().unstack().reindex(['calyx', 'peduncle', 'lobes']) / nap / 1000
        x = np.arange(3)
        ax.bar(x - 0.2, g[False], 0.38, color=C_IN, label='partner → APL (APL input)')
        ax.bar(x + 0.2, g[True], 0.38, color=C_OUT, label='APL → partner (APL output)')
        ax.set_xticks(x, ['calyx', 'peduncle', 'lobes']); ax.set_title(f'{LAB[ds]} ({nap} APL{"s" if nap > 1 else ""})')
        ax.yaxis.grid(True, color=L.GRID, lw=0.6); ax.set_axisbelow(True)
        if ds == 'flywire': ax.set_ylabel('APL synapses per APL (thousands)')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); return fig


def share(ds, reg, out):
    t = T[ds]; return (t[(t.region3 == reg)].out == out).sum() / max(1, (t.region3 == reg).sum())


R.fig(fig_regions(), 'Each panel is one connectome. Bars count the rows of `<dataset>_APL_MB_synapses.csv.gz` in each region, divided by '
      'the number of APLs in the dataset (so the height is synapses of one average APL). Green = synapses onto APL (APL is postsynaptic), '
      'purple = synapses made by APL (APL is presynaptic). Lobes = all lobe ROIs together. y-axis: thousands of T-bar → PSD pairs. '
      'The y-axes differ between panels: BANC has far fewer synapses (Section 1).',
      'APL synapses per region and direction, official ROIs.')
R.p(f'In every dataset APL makes more synapses than it receives in the calyx (APL output share of calyx APL synapses: '
    + ', '.join(f'{LAB[ds]} {L.pct(share(ds, "calyx", True))}' for ds in DS4) + '). In the lobes the two directions are closer to equal.')

# ---------------------------------------------------------------- 3.1 partners
R.sub('Who APL talks to')
GROUPS = ['Kenyon cells', 'Antennal-lobe PNs', 'Dopaminergic neurons', 'MB output neurons', 'Other labeled', 'Unlabeled partner']
GCOL = ['#2a78d6', '#1f9e6e', '#eb6834', '#8a5cc2', '#b5b3ab', '#e1e0d9']


def fig_partners():
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 2.9), sharey=True)
    for ax, reg in zip(axs, ['calyx', 'peduncle', 'lobes']):
        yl = []
        for yi, ds in enumerate(DS4):
            t = T[ds]; t = t[t.region3 == reg]
            s = t.pgroup.value_counts(normalize=True).reindex(GROUPS).fillna(0).values
            left = 0
            for v, c, gname in zip(s, GCOL, GROUPS):
                ax.barh(yi, v * 100, left=left, color=c, height=0.62, edgecolor='white', lw=0.8, label=gname if (reg == 'calyx' and yi == 0) else None)
                if v > 0.06: ax.text(left + v * 50, yi, f'{100 * v:.0f}', ha='center', va='center', fontsize=7, color='white' if c not in ('#b5b3ab', '#e1e0d9') else L.INK)
                left += v * 100
            yl.append(LAB[ds])
        ax.set_yticks(range(4), yl); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_title(reg)
        ax.set_xlabel('share of APL synapses in this region (%)')
    fig.legend(loc='lower center', ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); return fig


R.fig(fig_partners(), 'For each region (panel) and dataset (row), the bar splits all APL synapses there (both directions) by the partner\'s '
      'class: Kenyon cells, antennal-lobe projection neurons (PNs), dopaminergic neurons (DANs), mushroom body output neurons (MBONs), '
      'other neurons with a class label, and unlabeled partners (untyped fragments). Numbers inside bars are percentages (shown when '
      'above 6%). The class labels come from each dataset\'s own annotation: hemibrain from the BANC team\'s compiled table, the others '
      'from their own releases. BANC has many unlabeled partners (39% of rows).', 'Partner classes of APL synapses, per region.')
R.p('Kenyon cells are by far APL\'s main partner everywhere. In the calyx, PNs are the second partner (9–12% of APL synapses there): APL inhibits the PN '
    'boutons as well as the KC claws, which Reports 04 and 05 use.')

# ---------------------------------------------------------------- 4 compartments
R.section('Lobe compartments')
R.p('Each lobe is divided into compartments (α1–3, α′1–3, β1–2, β′1–2, γ1–5), each defined by which DAN and MBON types innervate it. Only the '
    'male CNS publishes compartment volumes (`malecns-subcompartments-v3`); there the official label is used. For the other datasets the '
    'compartment of a lobe synapse is **derived** from the dopaminergic neurons around it:')
R.ul(['In male CNS, measure for each DAN family (e.g. PAM12, PPL101) what fraction of its T-bars falls in each compartment: '
      '$P(c\\mid\\phi)$ for compartment $c$ and family $\\phi$.',
      'For a synapse $s$ in a lobe, take the $k_{\\mathrm{NN}}=15$ nearest DAN T-bars inside the same lobe ROI.',
      'Each of them votes with its family\'s distribution. The compartment with the largest summed vote wins (Equation 2); '
      '`compartment_vote_share` is that vote divided by the total, a confidence between 0 and 1.',
      'A synapse whose nearest DAN T-bar is more than 8 µm away gets no compartment.'], ordered=True)
R.eq('\\hat c(s) \\;=\\; \\arg\\max_{c\\,\\in\\,\\mathrm{lobe}(s)} \\sum_{d\\,\\in\\,\\mathrm{kNN}(s)} P\\big(c \\mid \\phi(d)\\big)',
     '$\\hat c(s)$: derived compartment of synapse $s$; $\\mathrm{kNN}(s)$: its 15 nearest DAN T-bars in the same lobe; $\\phi(d)$: the family of DAN T-bar $d$.')
pc = VAL['per_comp']
R.p(f'**Check against the official volumes.** Applied to the male CNS itself, the derived label matches the official one for '
    f'{L.pct(VAL["agreement"], 1)} of {VAL["n"]:,} lobe synapses when the lobe is known (the hemibrain situation). The weakest compartment '
    f'is γ3 ({L.pct(pc["g3"])}). With only the coarse VL/ML lobes (FlyWire, BANC) the agreement was 88%, mainly because α′2 is then '
    f'called α2. For the per-KC lobe plots a refined vote (DAN T-bars restricted to the KC\'s own lobe) reaches '
    f'{L.pct(LVAL["fine_lobe_DANs"]["agreement"], 1)}.')
R.box('Not used by the model', 'Compartments matter only for the lobe scatter plots (`scatter4`–`scatter7`). The model in Reports 05–11 uses calyx '
      'synapses only.', 'plain')

# ---------------------------------------------------------------- 5 per-cell counts
R.section('Per-cell counts: Kenyon cells and projection neurons')
R.p('Two tables sum the synapses per cell. `KC_weights_APL_PN_per_KC.csv` has one row per KC with its calyx PN input '
    '$n^{\\mathrm{PN\\to KC}}_k$, its calyx synapses onto APL $n^{\\mathrm{KC\\to APL}}_k$ and from APL $n^{\\mathrm{APL\\to KC}}_k$, '
    'and the same three counts over the whole mushroom body (`*_MB` columns). KC class (αβ, α′β′, γ) comes from the KC type name. '
    '`PN_calyx_weights_APL_vs_KC.csv` has one row per antennal-lobe PN with its calyx synapses onto KCs and to and from APL.')
R.eq('n^{\\mathrm{PN\\to KC}}_k = \\sum_{i\\,\\in\\,\\mathrm{PNs}} w_{ik}(\\mathrm{CA}),\\qquad '
     'n^{\\mathrm{KC\\to APL}}_k = w_{k,\\mathrm{APL}}(\\mathrm{CA}),\\qquad n^{\\mathrm{APL\\to KC}}_k = w_{\\mathrm{APL},k}(\\mathrm{CA})')
cls = KCW.groupby(['dataset', 'kc_class']).size().unstack().reindex(DS4).fillna(0).astype(int)
R.table(['Dataset', 'αβ', 'α′β′', 'γ', 'no subtype', 'all KCs'],
        [[LAB[ds]] + [f'{cls.loc[ds].get(c, 0):,}' for c in ['αβ', 'α′β′', 'γ', 'no subtype']] + [f'{cls.loc[ds].sum():,}'] for ds in DS4],
        title='Kenyon cells per class.', caption='Counted from `KC_weights_APL_PN_per_KC.csv` (KCs with at least one PN or APL synapse in the '
        'mushroom body). FlyWire, male CNS and BANC have both hemispheres; hemibrain has one. FlyWire has about twice as many γ KCs per side '
        'as hemibrain (a known difference, also reported by Schlegel et al. 2024). "No subtype" = BANC KCs without a class in their type name.',
        num_cols=[1, 2, 3, 4, 5])


def fig_kc_hist():
    cols = [('PN_to_KC_calyx', 'PN → KC synapses per KC, calyx\n$n^{\\mathrm{PN\\to KC}}_k$', 300),
            ('KC_to_APL_calyx', 'KC → APL synapses per KC, calyx\n$n^{\\mathrm{KC\\to APL}}_k$', 50),
            ('APL_to_KC_calyx', 'APL → KC synapses per KC, calyx\n$n^{\\mathrm{APL\\to KC}}_k$', 50)]
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for ax, (c, lab, mx) in zip(axs, cols):
        bins = np.linspace(0, mx, 51) if mx > 60 else np.arange(-0.5, mx + 1, 1)
        for ds in DS4:
            v = KCW.loc[KCW.dataset == ds, c].clip(upper=mx)
            ax.hist(v, bins=bins, histtype='step', lw=1.5, color=L.DS_COLOR[ds], density=True, label=f'{LAB[ds]} (median {v.median():.0f})')
        ax.set_xlabel(lab); ax.set_ylabel('fraction of KCs per bin'); ax.legend(fontsize=6.8, loc='upper right')
        ax.set_xlim(0, mx)
    fig.tight_layout(); return fig


R.fig(fig_kc_hist(), 'Distributions over all KCs of a dataset (both hemispheres where present) of the three calyx counts defined in Equation 3. '
      'Each line is one connectome (colors in the legend, with the median). x-axis: synapses per KC (T-bar → PSD pairs); values above the '
      'axis end (300 or 50) are piled into the last bin. y-axis: fraction of the dataset\'s KCs per bin (bin width 6 synapses in the first '
      'panel, 1 in the others). The peaks at 0 are mostly γd and αβp KCs, whose dendrites sit in the accessory calyces outside the calyx ROI: '
      'they have no PN input and no APL synapse in the main calyx (2–5% of KCs in FlyWire, hemibrain and male CNS; 12% in BANC).',
      'How many PN and APL synapses a Kenyon cell has in the calyx.')


def fig_kc_recip():
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 2.9))
    for ax, ds in zip(axs, DS4):
        k = KCW[KCW.dataset == ds]
        x, y = k.KC_to_APL_calyx.values, k.APL_to_KC_calyx.values
        mx = np.percentile(np.r_[x, y], 99.5)
        hb = ax.hexbin(x, y, gridsize=26, extent=(0, mx, 0, mx), cmap=L.SEQ, mincnt=1, bins='log', linewidths=0)
        ax.plot([0, mx], [0, mx], color=L.GRAY, lw=0.8, ls='--')
        r = spearmanr(x, y)[0]
        ax.set_title(f'{LAB[ds]}\n$\\rho_s$ = {r:.2f}, {len(k):,} KCs')
        ax.set_xlabel('KC → APL synapses per KC (calyx)')
        if ds == 'flywire': ax.set_ylabel('APL → KC synapses per KC (calyx)')
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        L.cbar(fig, hb, ax, 'KCs per hexagon (log scale)')
    fig.tight_layout(); return fig


R.fig(fig_kc_recip(), 'Each panel is one connectome; each KC is placed by its calyx synapses onto APL (x) and from APL (y). Hexagons are colored '
      'by how many KCs fall in them, on a log color scale (color bar right of each panel; white = no KC). Dashed line: equal counts. Axes '
      'end at the 99.5th percentile. $\\rho_s$ = Spearman rank correlation over KCs. KCs that give APL many synapses also receive many '
      'from it: the connection is reciprocal at the level of single cells.', 'Reciprocity between KCs and APL in the calyx.')


def fig_pn():
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 2.9))
    for ax, ds in zip(axs, DS4):
        p = PNW[PNW.dataset == ds]
        ax.scatter(p.PN_to_KC, p.APL_to_PN, s=10, color=L.DS_COLOR[ds], alpha=0.75, lw=0)
        r = spearmanr(p.PN_to_KC, p.APL_to_PN)[0]
        ax.set_title(f'{LAB[ds]}\n$\\rho_s$ = {r:.2f}, {len(p)} PNs')
        ax.set_xlabel('PN → KC synapses per PN (calyx)')
        if ds == 'flywire': ax.set_ylabel('APL → PN synapses per PN (calyx)')
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout(); return fig


R.fig(fig_pn(), 'Each dot is one antennal-lobe PN with at least one calyx synapse onto a KC. x: its synapses onto KCs in the calyx; y: '
      'synapses it receives from APL in the calyx. $\\rho_s$ = Spearman rank correlation over PNs. PNs with more output to KCs (more or '
      'bigger boutons) get more APL inhibition, roughly in proportion. Report 04 uses the ratio of the two, per glomerulus, as the '
      'bouton weight $p_g$.', 'APL input to each projection neuron vs its output to Kenyon cells, calyx.')
R.p('Table 5 gives the means behind these figures.')
m = KCW.groupby('dataset')[['PN_to_KC_calyx', 'KC_to_APL_calyx', 'APL_to_KC_calyx', 'KC_to_APL_MB', 'APL_to_KC_MB']].mean().reindex(DS4)
pm = PNW.groupby('dataset')[['PN_to_KC', 'PN_to_APL', 'APL_to_PN']].mean().reindex(DS4)
R.table(['Dataset', 'PN→KC per KC', 'KC→APL per KC', 'APL→KC per KC', 'KC→APL per KC, whole MB', 'APL→KC per KC, whole MB',
         'PN→KC per PN', 'PN→APL per PN', 'APL→PN per PN'],
        [[LAB[ds]] + [F(v, 1) for v in m.loc[ds]] + [F(v, 0) for v in pm.loc[ds]] for ds in DS4],
        title='Mean synapses per cell.', caption='First five columns: per KC (calyx unless "whole MB"); last three: per antennal-lobe PN, calyx. '
        'FlyWire counts here have no per-pair threshold (Section 1). Male CNS calyx counts per KC are lower than hemibrain\'s partly because '
        'its calyx ROI misses the back of the calyx (Report 02).', num_cols=list(range(1, 9)))

# ---------------------------------------------------------------- 6 caveats
R.section('Caveats')
R.ul(['**Detection differs between datasets.** Each connectome has its own synapse detector and thresholds, so absolute counts are not '
      'comparable across datasets; ratios and ranks within a dataset are safer. The model (Reports 05–11) works with ratios and ranks.',
      '**Male CNS calyx ROI.** The official CA(L)/CA(R) ROIs leave out the posterior part of each calyx: 18% (left) and 9% (right) of PN↔APL '
      'and 33% / 14% of KC↔APL calyx synapses fall outside them. The second version adds them back (Report 02).',
      '**Untyped partners.** Partners without a type are 9% of hemibrain rows, 10% of male CNS rows and 39% of BANC rows.',
      '**Hemibrain types.** `partner_type` is the official neuPrint v1.2 type. A compiled table had labelled about 200 α′β′ KCs as γ; '
      'that was corrected on 2026-10-10.',
      '**Summary table.** `summary_counts_by_compartment_and_partner_class.csv` was made from an earlier hemibrain table; 5 hemibrain γ3–γ5 rows '
      'differ by 1–4 synapses from the main table.'])

R.save()
