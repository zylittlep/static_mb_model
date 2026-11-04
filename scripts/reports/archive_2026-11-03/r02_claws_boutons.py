"""Report 02: PN boutons and KC claws (how they are found from synapse positions), placing APL synapses on them, the
posterior view and calyx geometry, and the second version of the tables with synapses outside the official ROIs."""
import sys
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import report_lib as L
L.style()
F = L.fmt
B = L.B; BO = B + 'with_outside_ROI_synapses/'
DS4 = ['flywire', 'hemibrain', 'malecns', 'banc']
LAB = L.DS_LABEL
LOBES = {'VL', 'ML', 'aL', "a'L", 'bL', "b'L", 'gL'}
ADDED_C = '#eb6834'; OFFICIAL_C = '#b5b3ab'


def to_rda(ds, x, y, z):
    """native nm -> (fly's right, dorsal, anterior-posterior with native sign) in µm; the posterior-view axes of the maps."""
    x, y, z = x / 1e3, y / 1e3, z / 1e3
    return {'flywire': np.c_[x, -y, z], 'hemibrain': np.c_[-x, -z, y], 'malecns': np.c_[-x, -y, z], 'banc': np.c_[-x, z, y]}[ds]


CL = {ds: pd.read_csv(f'{B}{ds}_KC_claws_derived.csv.gz', dtype={'kc_id': str, 'main_PN_id': str}) for ds in DS4}
BT = {ds: pd.read_csv(f'{B}{ds}_PN_boutons_derived.csv.gz', dtype={'pn_id': str}) for ds in DS4}
A = {}
for ds in DS4:
    a = pd.read_csv(f'{B}{ds}_APL_MB_synapses.csv.gz', usecols=['direction', 'apl_side', 'apl_id', 'mb_region', 'partner_class', 'partner_type',
                                                              'claw_id', 'claw_dist_nm', 'bouton_id', 'bouton_dist_nm'], low_memory=False)
    A[ds] = a[a.mb_region == 'CA'].copy()
PUR = pd.read_csv(B + 'reports/data/r02_claw_purity.csv')
EX = pd.read_csv(B + 'reports/data/r02_example_points.csv', dtype={'cell': str})
SH = pd.read_csv(B + 'odor_weighted/calyx_shape_by_dataset.csv')
ADD = pd.read_csv(BO + 'added_synapses_summary.csv')


def kc_rows(ds):
    a = A[ds]; return a[a.claw_dist_nm.notna()]


def pn_rows(ds):
    a = A[ds]; return a[a.bouton_dist_nm.notna()]


R = L.Report('02', 'How are PN boutons and KC claws found from synapse positions, how are APL synapses placed on them, how is the calyx '
             'drawn, and which synapses does the second version of the tables add?',
             inputs=['Raw PN → KC synapses in the calyx of each connectome (positions of T-bars and PSDs)', 'The APL synapse tables of Report 01'],
             outputs=['`<dataset>_PN_boutons_derived.csv.gz`, `<dataset>_KC_claws_derived.csv.gz`',
                      'columns `claw_id`, `nearest_claw_id`, `claw_dist_nm`, `bouton_id`, `nearest_bouton_id`, `bouton_dist_nm` of the APL tables',
                      '`with_outside_ROI_synapses/` (the second version of every table and plot)', '`odor_weighted/calyx_shape_by_dataset.csv/.png`'],
             scripts=['`scripts/core.py` (`single_linkage`, `cluster_by_owner`, `assign_nearest`)', '`scripts/run.py` (claws, boutons, assignment)',
                      '`scripts/offroi/assign.py` (synapses outside the official ROIs)', '`scripts/calyx_shape.py`',
                      '`scripts/reports/r02_prep.py` (claw purity and the example cells for this report)', '`scripts/reports/r02_claws_boutons.py` (this report)'])

R.p('None of the four connectomes publishes which synapses form a PN bouton or a KC claw, so both are derived here from synapse positions. '
    'They matter because odor information enters the calyx bouton by bouton: each bouton belongs to one PN and so to one glomerulus, and '
    'each claw takes most of its input from one bouton. Reports 03, 04, 05 and 10 use claws and boutons to give APL synapses a glomerulus.')

R.symbols([
    ('d_{\\mathrm{link}}', 'Linking distance for clustering: two sites closer than this are in the same unit', '2 µm (2,000 nm)'),
    ('d_{\\mathrm{assign}}', 'Largest distance from an APL synapse\'s partner-side site to a claw or bouton site for the synapse to be placed on it', '2 µm'),
    ('\\mathbf{x}_s', 'Position of site $s$ (T-bar or PSD)', 'nm, native coordinates of the dataset'),
    ('\\mathrm{claw}(c)', 'The set of PN → KC PSDs on one KC that form claw $c$', ''),
    ('\\pi(c)', 'Main PN of claw $c$: the PN that makes most of the claw\'s PN synapses', 'neuron ID'),
    ('s_{\\mathrm{main}}(c)', 'Main-PN share of claw $c$: synapses from $\\pi(c)$ ÷ all PN synapses on $c$', '0 to 1'),
    ('\\delta(e)', 'Distance from the KC-side (or PN-side) site of APL synapse $e$ to the nearest PN→KC site of that same KC (or T-bar of that same PN)', 'nm, column `claw_dist_nm` / `bouton_dist_nm`'),
    ('(R,\\ D,\\ A)', 'Posterior-view axes: fly\'s right, dorsal, anterior–posterior', 'µm'),
])

# ---------------------------------------------------------------- 1 clustering
R.section('Clustering sites into boutons and claws')
R.p('Both units are found with the same rule, **single-linkage clustering** at $d_{\\mathrm{link}} = 2$ µm, applied to one neuron at a time:')
R.ul(['**Boutons.** Take every T-bar of one antennal-lobe PN that has a KC or other partner in the calyx (one point per T-bar, however many '
      'PSDs it faces). Link any two T-bars closer than 2 µm. Each connected group is one bouton.',
      '**Claws.** Take every PSD on one KC that faces a PN T-bar in the calyx. Link any two closer than 2 µm. Each connected group is one claw.'])
R.eq('s \\sim s\' \\iff \\exists\\ s=s_0, s_1, \\dots, s_m=s\' \\text{ (sites of one neuron)}:\\ '
     '\\lVert \\mathbf{x}_{s_{l}} - \\mathbf{x}_{s_{l-1}} \\rVert < d_{\\mathrm{link}}\\ \\ \\forall\\, l',
     'Two sites are in the same unit when they are linked by such a chain; a unit can therefore be longer than 2 µm.')
R.p('Units are numbered by size within each neuron (`_bouton1` and `_claw1` are the largest). For each claw the table also stores its '
    'number of PN synapses, its **main PN** $\\pi(c)$ (the PN with most synapses on it), that PN\'s type, its main bouton (the bouton holding '
    'most of those T-bars) and its centroid. Each claw then has a glomerulus: that of its main PN. Figure 1 shows one PN and one KC.')


def fig_example():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
    cm = plt.get_cmap('tab20')
    for ax, kind in zip(axs, ['PN T-bars by bouton', 'KC PN-input sites by claw']):
        e = EX[EX.kind == kind]
        P = to_rda('hemibrain', e.x.values, e.y.values, e.z.values)
        units = e.unit.str.extract(r'(\d+)$')[0].astype(int).values
        order = np.unique(units)
        for j, u in enumerate(order):
            m = units == u
            ax.scatter(P[m, 0], P[m, 1], s=9, color=cm(j % 20), lw=0, label=f'{"bouton" if kind.startswith("PN") else "claw"} {u} ({m.sum()} sites)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('fly\'s right → (µm, hemibrain $-x$)'); ax.set_ylabel('dorsal ↑ (µm, hemibrain $-z$)')
        if kind.startswith('PN'):
            ax.set_title(f'a  One DM1 PN ({e.cell_type.iloc[0]}): {len(e)} calyx T-bars in {len(order)} boutons')
        else:
            pnt = e.pre_type.fillna('?').str.replace(r'_.*', '', regex=True)
            ax.set_title(f'b  One αβ KC (KCab-m): {len(e)} PN synapses in {len(order)} claws')
            for u in order:
                m = units == u; top = pnt[m].value_counts()
                ax.annotate(f'{top.index[0]} {top.iloc[0]}/{m.sum()}', P[m].mean(0)[:2], xytext=(6, 4), textcoords='offset points', fontsize=6.5, color=L.MUTED)
        ax.legend(fontsize=6.3, loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.2, markerscale=1.2)
    fig.tight_layout(); return fig


R.fig(fig_example(), 'Hemibrain, posterior view (fly\'s right to the right, dorsal up; axes in µm of the dataset\'s own coordinates). '
      '**a**, every calyx T-bar of one DM1 PN, colored by the bouton it was assigned to (single linkage at 2 µm); the legend gives each '
      'bouton\'s number and its T-bar count. **b**, every PSD on one αβ KC that faces a PN T-bar in the calyx, colored by claw. The gray label '
      'next to each claw is its main glomerulus and how many of the claw\'s PN synapses come from PNs of that glomerulus. Colors only '
      'separate units; they have no other meaning. A T-bar more than 2 µm from every other T-bar of its PN forms a bouton of its own '
      '(the 1-site boutons in **a**). Both cells were picked as typical: the DM1 PN with the median T-bar count, and an '
      'αβ KC with 8 claws (the hemibrain median).', 'One PN split into boutons and one KC split into claws.')

# ---------------------------------------------------------------- 1.1 stats
R.sub('How many, and how pure')
cpk = {ds: CL[ds].groupby('kc_id').size() for ds in DS4}
bpp = {ds: BT[ds].groupby('pn_id').size() for ds in DS4}


def fig_stats():
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 2.8))
    for ds in DS4:
        c = L.DS_COLOR[ds]
        axs[0].hist(bpp[ds].clip(upper=30), bins=np.arange(0.5, 31), histtype='step', lw=1.5, color=c, density=True, label=f'{LAB[ds]} ({bpp[ds].median():.0f})')
        axs[1].hist(cpk[ds].clip(upper=20), bins=np.arange(0.5, 21), histtype='step', lw=1.5, color=c, density=True, label=f'{LAB[ds]} ({cpk[ds].median():.0f})')
        axs[2].hist(CL[ds].n_PN_synapses.clip(upper=40), bins=np.arange(0.5, 41), histtype='step', lw=1.5, color=c, density=True,
                    label=f'{LAB[ds]} ({CL[ds].n_PN_synapses.median():.0f})')
    for ds in ['hemibrain', 'malecns']:
        v = PUR[PUR.dataset == ds].main_PN_share
        axs[3].hist(v, bins=np.linspace(0, 1, 21), histtype='step', lw=1.5, color=L.DS_COLOR[ds], weights=np.full(len(v), 1 / len(v)),
                    label=f'{LAB[ds]} ({L.pct((v >= 0.9).mean())} ≥ 0.9)')
    labs = [('PN boutons per PN (calyx)', 'fraction of PNs'), ('claws per KC', 'fraction of KCs'),
            ('PN synapses per claw', 'fraction of claws'), ('main-PN share of a claw, $s_{\\mathrm{main}}(c)$', 'fraction of claws (log scale)')]
    axs[3].set_yscale('log')
    for ax, (xl, yl), t in zip(axs, labs, 'abcd'):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=6.3, title='median' if t != 'd' else None, title_fontsize=6.5); L.tag(ax, t, x=-0.3)
    fig.tight_layout(); return fig


R.fig(fig_stats(), 'Distributions per connectome (line colors in the legends; the number in brackets is the median, or in **d** the share of '
      'claws with $s_{\\mathrm{main}} \\ge 0.9$). **a**, boutons per antennal-lobe PN in the calyx. **b**, claws per KC (KCs with at least one '
      'PN synapse in the calyx). FlyWire PNs have more boutons (median 8 vs 4) and more single-site boutons (29% vs 17–22% in hemibrain and male CNS); its synapse table gives each connection its own presynaptic point instead of one point per T-bar, which likely explains part of this. **c**, PN synapses per claw (PSDs facing PN T-bars). **d**, for hemibrain and male CNS: the main-PN share of '
      'each claw (Equation 2), from `reports/data/r02_claw_purity.csv`; y-axis: fraction of claws per bin of width 0.05, log scale. Values beyond the axis end are piled into the last bin.',
      'Boutons per PN, claws per KC, synapses per claw, and how much of a claw comes from one PN.')
R.eq('s_{\\mathrm{main}}(c) \\;=\\; \\frac{\\#\\{s \\in \\mathrm{claw}(c) : s \\text{ faces a T-bar of } \\pi(c)\\}}{|\\mathrm{claw}(c)|}')
pm = PUR.groupby('dataset').main_PN_share.apply(lambda v: (v >= 0.9).mean())
R.p(f'Hemibrain KCs have {cpk["hemibrain"].median():.0f} claws (median) and male CNS KCs {cpk["malecns"].median():.0f}; FlyWire '
    f'{cpk["flywire"].median():.0f}, BANC {cpk["banc"].median():.0f}. Published counts from light microscopy and EM are about 5–7 claws per KC; '
    f'a 2 µm linking distance can split a large claw in two, so these counts are a little high. Claws are pure: '
    f'{L.pct(pm["hemibrain"])} (hemibrain) and {L.pct(pm["malecns"])} (male CNS) of claws get at least 90% of their PN synapses from one PN, '
    'so naming a claw by its main PN\'s glomerulus loses little.')

# ---------------------------------------------------------------- 2 assign
R.section('Placing APL synapses on claws and boutons')
R.p('For each APL synapse $e$ with a KC in the calyx, take the KC-side site (the PSD when APL is presynaptic, the T-bar when the KC is), '
    'and find the nearest PN → KC PSD **of that same KC**. Its distance is $\\delta(e)$ (column `claw_dist_nm`), and its claw is '
    '`nearest_claw_id`. The synapse is **on a claw** (`claw_id` filled) when $\\delta(e) \\le d_{\\mathrm{assign}} = 2$ µm. APL synapses with '
    'PNs are placed on boutons the same way, with the PN-side site and the T-bars of that PN.')
R.eq('\\delta(e) = \\min_{s \\,\\in\\, \\mathrm{PSDs\\ of\\ PN\\ input\\ on\\ KC}(e)} \\lVert \\mathbf{x}_{e}^{\\mathrm{KC}} - \\mathbf{x}_s \\rVert,\\qquad '
     'e \\text{ is on claw } \\mathrm{claw}\\big(\\arg\\min_s\\big) \\iff \\delta(e) \\le 2\\ \\mu\\mathrm{m}')


def fig_dist():
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.0))
    grid = np.linspace(0, 10, 201)
    for ds in DS4:
        a = kc_rows(ds)
        for ax, out in zip(axs[:2], [False, True]):
            d = a[a.direction.str.startswith('APL_output') == out].claw_dist_nm.values / 1e3
            ax.plot(grid, [(d <= g).mean() for g in grid], color=L.DS_COLOR[ds], label=f'{LAB[ds]} ({L.pct((d <= 2).mean())} at 2 µm)')
        p = pn_rows(ds).bouton_dist_nm.values / 1e3
        axs[2].plot(grid, [(p <= g).mean() for g in grid], color=L.DS_COLOR[ds], label=f'{LAB[ds]} ({L.pct((p <= 2).mean())})')
    tl = ['KC → APL synapses in the calyx', 'APL → KC synapses in the calyx', 'APL ↔ PN synapses in the calyx (both directions)']
    xl = ['$\\delta$: distance to the nearest\nPN-input site on the same KC (µm)'] * 2 + ['distance to the nearest\nT-bar of the same PN (µm)']
    for ax, t, x, tg in zip(axs, tl, xl, 'abc'):
        ax.axvline(2, color=L.INK, lw=0.8, ls='--'); ax.set_xlim(0, 10); ax.set_ylim(0, 1.01)
        ax.set_title(t); ax.set_xlabel(x); ax.set_ylabel('cumulative fraction of synapses'); ax.legend(fontsize=6.5, loc='lower right')
        ax.yaxis.grid(True, color=L.GRID, lw=0.6); L.tag(ax, tg, x=-0.22)
    fig.tight_layout(); return fig


R.fig(fig_dist(), 'Cumulative distributions over all calyx APL synapses of a given kind, one line per connectome: the fraction whose '
      'partner-side site lies within the distance on the x-axis of that same partner\'s PN-input sites (**a**, **b**: KC partners, nearest '
      'PSD facing a PN on that KC) or T-bars (**c**: PN partners, nearest T-bar of that PN). Dashed line: the 2 µm cut-off used to place a '
      'synapse on a claw or bouton. The legend gives the fraction at 2 µm. FlyWire rows here are the main-table rows (≥5 synapses per pair).',
      'How far APL synapses lie from claws and boutons.')
oc = {ds: kc_rows(ds) for ds in DS4}
R.p('Almost every APL ↔ PN synapse in the calyx sits on a bouton, as expected: APL contacts the boutons. Only a minority of KC ↔ APL '
    'synapses sit on a claw, more so for APL → KC than for KC → APL: in the calyx most KC ↔ APL synapses are on the KC dendrite between '
    'claws, or on its fine branches. The column `claw_dist_nm` lets anyone re-cut at another distance.')
R.box('Where this is used', ['Report 03 colors KC ↔ APL synapses by the odor response of their claw\'s glomerulus, using only synapses on a claw.',
                              'Report 04 counts APL → KC synapses on the claws of each glomerulus (the claw weight $q_g$).',
                              'Report 10 measures distances along APL between the APL sites of the claws and boutons of different glomeruli.'], 'plain')

# ---------------------------------------------------------------- 3 posterior view
R.section('Drawing the calyx: the posterior view')
R.p('Every map in these reports shows the calyx as seen from behind the fly (posterior view), with the fly\'s right to the right and '
    'dorsal up, so that all datasets look alike. Each dataset has its own axes, so the view is built from its native coordinates '
    '(Table 2), checked against mushroom-body landmarks (the lobes are anterior and ventral of the calyx, the vertical lobe is dorsal).')
R.table(['Dataset', 'Right (x of the map)', 'Dorsal (y of the map)', 'Anterior–posterior (depth)', 'Note'],
        [['FlyWire 783', '$x$', '$-y$', '$z$', 'FlyWire coordinates are mirror-imaged relative to its side labels, so sides come from the labels'],
         ['Hemibrain v1.2', '$-x$', '$-z$', '$y$', 'the hemibrain $y$ axis is anterior–posterior'],
         ['Male CNS v1.0', '$-x$', '$-y$', '$z$', ''], ['BANC 888', '$-x$', '$z$', '$y$', '']],
        title='Native axes used for the posterior view.', caption='Coordinates are in nm in the tables and µm in the maps.')
sh = SH.set_index('calyx')
R.p('The calyx outlines still differ between datasets in these maps, mainly because each brain sits at a slightly different angle. The '
    'calyx is a flattened cup; its thinnest axis is tilted from the anterior–posterior axis by '
    + ', '.join(f'{F(sh.loc[c, "tilt_of_calyx_axis_from_AP_deg"], 0)}° ({c})' for c in sh.index if ('Hemibrain' in c or 'Male' in c))
    + f', and by {F(sh[sh.index.str.startswith("BANC")].tilt_of_calyx_axis_from_AP_deg.min(), 0)}–'
      f'{F(sh[sh.index.str.startswith("BANC")].tilt_of_calyx_axis_from_AP_deg.max(), 0)}° in BANC, '
      'so BANC\'s calyx is seen more obliquely and looks flatter. `odor_weighted/calyx_shape_by_dataset.png` shows every calyx from three sides.')

# ---------------------------------------------------------------- 4 outside ROI
R.section('The second version: synapses just outside the official ROIs')
R.p('Each connectome defines its mushroom body regions with its own ROI meshes. These meshes do not always cover the whole structure. '
    'In the male CNS the calyx ROIs leave out the posterior part of each calyx; in FlyWire and BANC the meshes of nearby neuropils '
    '(CRE, SIP, SMP) take in parts of the lobes. Yang decided (2026-10-14) to keep both versions: the official-ROI tables stay as they '
    'are, and `with_outside_ROI_synapses/` holds the same tables, plots and maps with these synapses added back. The rule, in order:')
R.ul(['**Anchors.** Calyx: the PN → KC synapses inside the official calyx ROI, extended through PN → KC synapses outside every MB ROI '
      'that connect to them in steps of ≤ 3 µm (claws the ROI cut off). Peduncle and lobes: the APL ↔ KC synapses inside the official '
      'peduncle and lobe ROIs.',
      '**Growing through KC synapses.** APL ↔ KC synapses outside every MB ROI join the first anchor that reaches them in steps of ≤ 3 µm, '
      'and take that anchor\'s region. KC dendrites and axons exist only in the mushroom body, so these chains stay inside it.',
      '**Joining.** Any other synapse outside every MB ROI joins when its postsynaptic site lies within 3 µm of an anchor point, with the '
      'region of the nearest one (the calyx wins ties).',
      '**Accessory calyces stay out.** No chain runs through synapses with γd or αβp KCs, whose dendrites form the accessory calyces.',
      '**PN synapses** next to the calyx join only if they are PN → KC or their T-bar belongs to a calyx bouton.'], ordered=True)
R.p('Every added row is flagged in `mb_region_source`; keeping the rows marked `official ROI` gives the main tables back exactly '
    '(checked when the second version was built).')

ap = ADD[ADD.synapses.str.startswith('APL')].copy()
ap['reg'] = np.where(ap.mb_region == 'CA', 'calyx', np.where(ap.mb_region == 'PED', 'peduncle', 'lobes'))
g = ap.groupby(['dataset', 'reg'])[['n_official_ROI', 'n_added']].sum()
kcca = ap[(ap.reg == 'calyx') & (ap.partner_group == 'KC')].groupby('dataset')[['n_official_ROI', 'n_added']].sum()
pnkc = ADD[ADD.synapses.str.startswith('ALPN')].groupby('dataset')[['n_official_ROI', 'n_added']].sum()


def pc(n, a): return f'+{100 * a / n:.1f}%'


R.table(['Dataset', 'APL synapses, calyx', 'peduncle', 'lobes', 'KC ↔ APL, calyx', 'PN → KC, calyx'],
        [[LAB[ds]] + [f'+{int(g.loc[(ds, r), "n_added"]):,} ({pc(*g.loc[(ds, r)].values)})' for r in ['calyx', 'peduncle', 'lobes']]
         + [pc(*kcca.loc[ds].values), pc(*pnkc.loc[ds].values)] for ds in DS4],
        title='Synapses added in the second version.', caption='Added synapses as a count and as a share of the official-ROI count, from '
        '`with_outside_ROI_synapses/added_synapses_summary.csv`. The male CNS calyx gains the most (+31% of APL synapses).',
        num_cols=[1, 2, 3, 4, 5])


def fig_added():
    fig, axs = plt.subplots(2, 3, figsize=(10.4, 6.6))
    cal = [('hemibrain', 'R'), ('malecns', 'L'), ('malecns', 'R')]
    for j, (ds, side) in enumerate(cal):
        t = pd.read_csv(f'{BO}{ds}_APL_MB_synapses.csv.gz', usecols=['apl_side', 'apl_instance', 'mb_region', 'mb_region_source', 'post_x_nm', 'post_y_nm', 'post_z_nm'], low_memory=False)
        sd = t.apl_side.fillna(t.apl_instance.str[-1])
        t = t[sd == side]
        P = to_rda(ds, t.post_x_nm.values, t.post_y_nm.values, t.post_z_nm.values)
        lobe = t.mb_region.isin(LOBES).values; ca = (t.mb_region == 'CA').values
        sgn = 1 if P[lobe, 2].mean() > P[ca, 2].mean() else -1      # anterior positive
        P[:, 2] *= sgn
        off = ca & (t.mb_region_source == 'official ROI').values; add = ca & ~off
        rng = np.random.default_rng(0)
        io = rng.choice(np.flatnonzero(off), min(25000, off.sum()), replace=False); ia = rng.choice(np.flatnonzero(add), min(12000, add.sum()), replace=False)
        for i, (cx, cy, xl, yl) in enumerate([(0, 1, 'fly\'s right → (µm)', 'dorsal ↑ (µm)'), (2, 1, 'anterior → (µm)', 'dorsal ↑ (µm)')]):
            ax = axs[i, j]
            ax.scatter(P[io, cx], P[io, cy], s=0.6, color=OFFICIAL_C, lw=0, rasterized=True)
            ax.scatter(P[ia, cx], P[ia, cy], s=0.9, color=ADDED_C, lw=0, rasterized=True)
            ax.set_aspect('equal', adjustable='datalim'); ax.set_xlabel(xl); ax.set_ylabel(yl)
            v = 'posterior view' if i == 0 else 'side view'
            ax.set_title(f'{LAB[ds]}, {side} calyx, {v}\n{off.sum():,} in ROI, {add.sum():,} added ({pc(off.sum(), add.sum())})' if i == 0 else f'{LAB[ds]}, {side} calyx, {v}')
    h = [Line2D([], [], marker='o', ls='', color=OFFICIAL_C, label='APL synapse inside the official calyx ROI'),
         Line2D([], [], marker='o', ls='', color=ADDED_C, label='APL synapse added to the calyx (outside every official MB ROI)')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1)); return fig


R.fig(fig_added(), 'Every APL synapse (both directions, all partners) that belongs to the calyx in the second version, for the three calyces '
      'used by the model. Gray = inside the dataset\'s official calyx ROI; orange = added by the rule above. Top row: posterior view '
      '(axes of Table 2, µm). Bottom row: side view, anterior to the right (the sign of the anterior–posterior axis set so that the lobes '
      'are anterior of the calyx), dorsal up. Up to 25,000 gray and 12,000 orange points are drawn per panel (random subsample); the '
      'counts in the titles are of all synapses. In the left male CNS calyx the added synapses form the back (posterior) part of the calyx; '
      'in the right one they lie mostly along its ventral and posterior edge. In hemibrain the few additions sit at the calyx\'s edges.',
      'Where the second version adds calyx synapses: hemibrain and male CNS.')
R.p('With the added synapses, male CNS PN input per KC matches hemibrain (96 synapses per KC in both), and its calyx KC ↔ APL synapses per '
    'KC go from 38% below hemibrain\'s to 20% below. Every analysis from Report 03 on is done in both versions.')
R.box('Caveats', ['Where an accessory calyx touches the calyx or peduncle, some synapses with γd and αβp KCs within 3 µm of an anchor still '
                  'join (male CNS 1,291; hemibrain 1,487). They are flagged rows with `partner_type` KCg-d or KCab-p.',
                  'The 3 µm steps are a choice. They were checked by eye on maps of the added synapses '
                  '(`with_outside_ROI_synapses/where_synapses_were_added.png`), not against a ground truth.'], 'caveat')

R.save()
