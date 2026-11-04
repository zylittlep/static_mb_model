"""Report 02, rebuilt 2026-11-04 on the corrected calyx and the new claws (Yang's rule of 2026-11-04, >= 2 PSDs per PN).
PN boutons and KC claws, how APL synapses are placed on them, where off-claw APL synapses sit on the KC, the posterior view,
and the synapses the corrected calyx adds. The 2026-10-23 version is in scripts/reports/archive_2026-11-03/.
Run with python3 -I (it reads Prat's uploaded claw counts)."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import report_lib as L
L.style()
L.DATE = '2026-11-04'
F = L.fmt
B = L.B; BO = B + 'with_outside_ROI_synapses/'; CV = BO + 'claws_v2m2/'
CH = B + 'claws_geodesic/Yang_rule_2026-11-04/chosen_min2PSD/'
SKD = B + 'kc_skeletons/claws_v2m2/'
PRAT = '/mnt/project-files/uploads/hearth/384d39ba-cbbd-49a8-ad93-99fb5e95b2be'
DS4 = ['flywire', 'hemibrain', 'malecns', 'banc']
DS2 = ['hemibrain', 'malecns']
LAB = L.DS_LABEL
LOBES = {'VL', 'ML', 'aL', "a'L", 'bL', "b'L", 'gL'}
ADDED_C = '#eb6834'; OFFICIAL_C = '#b5b3ab'
C3 = {'old': '#eda100', 'new': '#1baf7a', 'Prat': '#6250d6'}          # validated three-method palette (claw comparison)
ON_C, OFF_C = '#1baf7a', '#b5b3ab'
UNIT = ['#2a78d6', '#eb6834', '#1f9e6e', '#8a5cc2', '#d1477f', '#a87a0c', '#2b84b0', '#6b6a65', '#c0392b', '#5a8f29', '#9b59b6', '#16a085']


def to_rda(ds, x, y, z):
    """native nm -> (fly's right, dorsal, anterior-posterior with native sign) in µm; the posterior-view axes of the maps."""
    x, y, z = x / 1e3, y / 1e3, z / 1e3
    return {'flywire': np.c_[x, -y, z], 'hemibrain': np.c_[-x, -z, y], 'malecns': np.c_[-x, -y, z], 'banc': np.c_[-x, z, y]}[ds]


# ---------------------------------------------------------------- data
CL = {ds: pd.read_csv(f'{CV}{ds}_KC_claws_derived.csv.gz', dtype={'kc_id': str, 'main_PN_id': str}) for ds in DS2}
CLO = {ds: pd.read_csv(f'{BO}{ds}_KC_claws_derived.csv.gz', dtype={'kc_id': str, 'main_PN_id': str}) for ds in DS4}
BT = {ds: pd.read_csv(f'{BO}{ds}_PN_boutons_derived.csv.gz', dtype={'pn_id': str}) for ds in DS4}
A = {}
for ds in DS2:
    a = pd.read_csv(f'{CV}{ds}_APL_MB_synapses.csv.gz', usecols=['direction', 'apl_side', 'apl_instance', 'mb_region', 'mb_region_source',
                                                                 'partner_id', 'partner_type', 'claw_id', 'claw_dist_nm', 'bouton_id', 'bouton_dist_nm',
                                                                 'post_x_nm', 'post_y_nm', 'post_z_nm', 'partner_side_x_nm', 'partner_side_y_nm',
                                                                 'partner_side_z_nm'], low_memory=False)
    a = a[a.mb_region == 'CA'].copy()
    a['side'] = a.apl_side.fillna(a.apl_instance.str[-1]).astype(str).str[0].str.upper()
    A[ds] = a
KA = pd.read_csv(CV + 'KC_APL_calyx_synapses_claws_v2m2.csv.gz', dtype={'kc_id': str}, low_memory=False)
EXP = json.load(open(CV + 'export_summary.json'))
COMP = pd.read_csv(B + 'reports/data/r02v2_claw_composition.csv')
SHR = pd.read_csv(B + 'reports/data/r02v2_site_shares.csv')
SEG = pd.read_csv(B + 'reports/data/r02v2_fig1_segments.csv')
PTS = pd.read_csv(B + 'reports/data/r02v2_fig1_points.csv', dtype={'unit': str})
CELLS = pd.read_csv(B + 'reports/data/r02v2_fig1_cells.csv')
SH = pd.read_csv(B + 'odor_weighted/calyx_shape_by_dataset.csv')
ADD = pd.read_csv(BO + 'added_synapses_summary.csv')
CPK = pd.read_csv(CH + 'claws_per_KC.csv'); CPK = CPK[CPK.mb == 'hemibrain R'].set_index('kc_id')
PR = pd.read_csv(PRAT).set_index('bodyId')
NEWCOL = 'new v2, all sites, split by PN (PN claw >= 2 PSDs)'
H3 = pd.DataFrame({'old': CPK['old (2 µm, PN PSDs)'], 'new': CPK[NEWCOL], 'type': CPK.kc_type}).join(
    PR[['KC_subtype', 'n_claws', 'n_claws_corrected']], how='inner').rename(columns={'n_claws_corrected': 'Prat', 'n_claws': 'Prat_before'})
PR_MISSING = PR.loc[~PR.index.isin(CPK.index)]
POP = json.load(open(SKD + 'summary.json'))['all right KCs']
SKKC = pd.read_csv(SKD + 'KC_APL_synapses_by_location.csv')


def agree(a, b):
    d = H3[a] - H3[b]
    return dict(r=float(np.corrcoef(H3[a], H3[b])[0, 1]), md=float(d.mean()), same=float((d == 0).mean()), w1=float((d.abs() <= 1).mean()),
                w2=float((d.abs() <= 2).mean()))


AG = {k: agree(*k) for k in [('new', 'Prat'), ('old', 'Prat'), ('new', 'old'), ('new', 'Prat_before'), ('old', 'Prat_before')]}

R = L.Report('02', 'How are PN boutons and KC claws found, how are APL synapses placed on them, where on the KC do the other APL '
             'synapses sit, and which synapses does the corrected calyx add?',
             inputs=['PN → KC synapses in the corrected calyx (positions of T-bars and PSDs)', 'Every synapse site of every KC near the calyx',
                     'KC skeletons: hemibrain v1.2 (release archive) and male CNS v1.0 (public SWC files)',
                     'Prat\'s claw counts per hemibrain KC (uploaded `KC_Claw_Counts_Hemibrain` file)', 'The APL synapse tables of Report 01'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/` (claws, the APL tables with the new claw columns, KC ↔ APL on claws)',
                      '`claws_geodesic/` (the claw comparisons and the 40 example KCs)', '`kc_skeletons/claws_v2m2/` (KC skeletons with APL synapses)',
                      '`<dataset>_PN_boutons_derived.csv.gz`', '`with_outside_ROI_synapses/` (the corrected calyx)'],
             scripts=['`scripts/claws_geodesic/extract_kc_sites.py`, `segment_claws.py`, `claws_v2.py` (claws)',
                      '`scripts/claws_geodesic/compare_claws.py`, `compare_three.py` (comparisons)',
                      '`scripts/claws_geodesic/export_claws_v2.py`, `export_claws_v2_tables.py` (tables for the other scripts)',
                      '`scripts/kc_skeleton_apl_synapses_v2.py` (KC skeletons)', '`scripts/run.py`, `scripts/core.py` (boutons)',
                      '`scripts/offroi/assign.py` (corrected calyx)', '`scripts/reports/r02_prep_v2.py`, `scripts/reports/r02_claws_boutons.py` (this report)'])

R.p('None of the four connectomes publishes which synapses form a PN bouton or a KC claw, so both are derived here. They matter because odor '
    'information enters the calyx bouton by bouton: each bouton belongs to one PN and so to one glomerulus, and each claw takes its input '
    'from one PN. Reports 03, 04, 05 and 10–12 use claws and boutons to give APL synapses a glomerulus.')
R.box('What changed in this version (2026-11-04)', [
    'All tables are the **corrected calyx** (synapses just outside the official ROIs added; last section).',
    '**New claws**, found along each KC\'s skeleton with the rule Yang gave on 2026-11-04 and the variant Yang chose (a PN needs at least 2 PSDs '
    'to get its own claw). The old claws were 2 µm straight-line clusters of PN → KC PSDs.',
    'The new claws exist for **hemibrain and male CNS**, the two datasets with KC skeletons. FlyWire and BANC keep the old 2 µm claws where '
    'they appear (Report 04); the model (Reports 05–12) uses only hemibrain and male CNS.',
    'New: Figure 1 (3 random PNs and 10 random KCs with their skeletons), the comparison with Prat\'s claw counts, APL ↔ KC figures in the '
    'calyx, and KC skeletons showing where the off-claw APL synapses sit.'], 'plain')

R.symbols([
    ('d_{\\mathrm{geo}}(s, s\')', 'Distance between sites $s$ and $s\'$ of one KC measured along its skeleton', 'µm'),
    ('d_{\\mathrm{cut}}', 'Cut for the claw clusters and for joining a claw', '8 µm, along the skeleton'),
    ('m', 'Fewest PSDs a PN needs in a cluster to get its own claw', '2 (Yang\'s choice)'),
    ('d_{\\mathrm{link}}', 'Linking distance for boutons: two T-bars of one PN closer than this are in the same bouton', '2 µm, straight line'),
    ('\\mathbf{x}_s', 'Position of site $s$ (T-bar or PSD)', 'nm, native coordinates'),
    ('\\pi(c)', 'The PN of claw $c$ (each new claw has exactly one)', 'neuron ID'),
    ('\\delta(e)', 'Distance along the KC from the KC-side site of APL synapse $e$ to the nearest PSD of a claw\'s own PN', 'µm, column `claw_dist_nm`'),
    ('(R,\\ D,\\ A)', 'Posterior-view axes: fly\'s right, dorsal, anterior–posterior', 'µm'),
])

# ================================================================ 1 boutons and claws
R.section('Boutons and claws')
R.sub('Boutons: straight-line clusters of a PN\'s T-bars')
R.p('Take every T-bar of one antennal-lobe PN that has a partner in the calyx (one point per T-bar, however many PSDs it faces). Link any '
    'two T-bars closer than $d_{\\mathrm{link}} = 2$ µm. Each connected group is one bouton. This rule is unchanged.')
R.eq('s \\sim s\' \\iff \\exists\\ s=s_0, s_1, \\dots, s_m=s\' \\text{ (T-bars of one PN)}:\\ '
     '\\lVert \\mathbf{x}_{s_{l}} - \\mathbf{x}_{s_{l-1}} \\rVert < d_{\\mathrm{link}}\\ \\ \\forall\\, l',
     'Two T-bars are in the same bouton when such a chain links them, so a bouton can be longer than 2 µm.')
R.sub('Claws: clusters along the KC skeleton, split by PN')
R.p('The new claws follow the rule Yang gave on 2026-11-04. They use every synapse site of the KC in the calyx: PSDs from any partner and the '
    'KC\'s own T-bars. A site is in the calyx if it is labelled CA or lies within 3 µm of a corrected-calyx PN → KC PSD. Each site is '
    'placed on the nearest node of the KC\'s skeleton, and distances are measured along the skeleton.')
R.ul(['**Cluster.** Link any two sites of the KC whose distance along the skeleton is at most $d_{\\mathrm{cut}} = 8$ µm. The connected '
      'groups are the clusters. This is the same as building the minimum spanning tree of the along-skeleton distances and cutting every '
      'edge longer than 8 µm.',
      '**Split by PN.** In each cluster, every PN that makes at least $m = 2$ PSDs there gets its own claw, made of its PSDs.',
      '**Join.** Every other site in the cluster joins the claw whose own PN\'s PSDs are nearest along the skeleton, if they are within '
      '8 µm. This covers PSDs from APL and other partners, the KC\'s T-bars, and PSDs of a PN with a single PSD there. A site further '
      'than 8 µm from every claw is not on a claw.'], ordered=True)
R.eq('\\mathrm{claw}(e) = \\arg\\min_{c}\\ \\min_{s \\in \\mathrm{PSDs\\ of\\ }\\pi(c)\\ \\mathrm{in}\\ c} d_{\\mathrm{geo}}(e, s),'
     '\\qquad e \\text{ is on a claw} \\iff \\delta(e) = \\min_c \\min_s d_{\\mathrm{geo}}(e, s) \\le 8\\ \\mu\\mathrm{m}',
     'Rule 3 for a site $e$ that is not a PSD of a claw\'s own PN.')
R.p('**Why $m = 2$.** With $m = 1$, the rule as first written, a single stray PSD from a PN makes a claw of its own: 13% of hemibrain claws '
    'had one PSD, and hemibrain KCs had 7.3 claws against 6.7 in Prat\'s counts. Requiring at least 2 PSDs gave the closest match to Prat\'s '
    'counts of every version tried, and Yang chose it on 2026-11-04 (`claws_geodesic/README.md` has $m$ = 1, 2, 3 and 5).')
R.p('**Skeletons.** Hemibrain: the v1.2 release archive. Male CNS: the public v1.0 SWC files (coarse, about 0.7 µm between nodes). Pieces '
    'of a skeleton that are not connected are joined by their shortest gaps first. A site sits a median of '
    '0.19 µm from its skeleton node in hemibrain and 0.63 µm in male CNS (90%: 0.42 and 1.18 µm).')


def fig_cells():
    fig = plt.figure(figsize=(12, 11.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1, 1], hspace=0.42)
    top = gs[0].subgridspec(1, 3, wspace=0.25); rows = [gs[1].subgridspec(1, 5, wspace=0.18), gs[2].subgridspec(1, 5, wspace=0.18)]
    pcells = CELLS[CELLS.kind == 'PN']; kcells = CELLS[CELLS.kind == 'KC']
    for i, r in enumerate(pcells.itertuples()):
        ax = fig.add_subplot(top[0, i])
        s = SEG[SEG.cell == r.cell]
        ax.add_collection(LineCollection(np.stack([s[['r1', 'd1']].values, s[['r2', 'd2']].values], 1), colors='#a9a8a2', linewidths=0.6, zorder=1))
        p = PTS[PTS.cell == r.cell]
        units = p.unit.str.extract(r'(\d+)$')[0].astype(int)
        for j, u in enumerate(sorted(units.unique())):
            m = (units == u).values
            ax.scatter(p.right[m], p.dorsal[m], s=7, color=UNIT[j % len(UNIT)], marker='osD^v'[j // len(UNIT) % 5], lw=0, zorder=3)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'{"abc"[i]}  PN {r.cell}, {r.type}\n{int(r.calyx_tbars)} calyx T-bars in {int(r.boutons)} boutons', fontsize=8.5)
        ax.set_xlabel('fly\'s right → (µm)', fontsize=7.5); ax.set_ylabel('dorsal ↑ (µm)', fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    for i, r in enumerate(kcells.itertuples()):
        ax = fig.add_subplot(rows[i // 5][0, i % 5])
        s = SEG[SEG.cell == r.cell]
        ax.add_collection(LineCollection(np.stack([s[['r1', 'd1']].values, s[['r2', 'd2']].values], 1), colors='#a9a8a2', linewidths=0.6, zorder=1))
        p = PTS[PTS.cell == r.cell]
        u = p.unit.astype(float).astype(int)
        claws = sorted(u[u >= 0].unique())
        col = {c: UNIT[j % len(UNIT)] for j, c in enumerate(claws)}
        rest = p[u < 0]
        ax.scatter(rest.right, rest.dorsal, s=2.5, color='#d3d2cc', lw=0, zorder=2)
        for c in claws:
            q = p[(u == c).values]
            pn = q[q.kind == 'PN->KC PSD']; ot = q[q.kind != 'PN->KC PSD']
            ax.scatter(ot.right, ot.dorsal, s=3, color=col[c], alpha=0.55, lw=0, zorder=2)
            ax.scatter(pn.right, pn.dorsal, s=9, color=col[c], lw=0, zorder=3)
            g = q.claw_glomerulus.dropna()
            if len(pn):
                ax.annotate(g.iloc[0] if len(g) else '?', (pn.right.mean(), pn.dorsal.mean()), xytext=(4, 3), textcoords='offset points',
                            fontsize=5.8, color=L.INK, zorder=5)
        ax.set_aspect('equal', adjustable='datalim')
        pr = 'n/a' if np.isnan(r.Prat_claws) else int(r.Prat_claws)
        ax.set_title(f'{"defghijklm"[i]}  {r.type} {r.cell}\nnew {int(r.new_claws)} · old {int(r.old_claws)} · Prat\'s {pr}', fontsize=7.8)
        ax.tick_params(labelsize=6)
        if i % 5 == 0: ax.set_ylabel('dorsal ↑ (µm)', fontsize=7.5)
        if i >= 5: ax.set_xlabel('fly\'s right → (µm)', fontsize=7.5)
    return fig


R.fig(fig_cells(), 'Hemibrain right calyx, posterior view (fly\'s right to the right, dorsal up, µm). Cells drawn at random among those '
      'with a skeleton (seed 1103): **a–c**, three uniglomerular PNs; **d–m**, ten KCs. Grey lines: the part of each skeleton inside the '
      'calyx (nodes within 3–4 µm of calyx synapse sites). **a–c**, every calyx T-bar of the PN, colored by its bouton (colors only separate '
      'boutons). **d–m**, the KC\'s calyx synapse sites. Large dots are PN → KC PSDs, small dots the other sites (PSDs from other partners, '
      'the KC\'s T-bars), both colored by their new claw; light grey = not on a claw. The label next to each claw is its PN\'s glomerulus. '
      'Each title gives the claw count from the new rule, the old 2 µm rule and Prat\'s segmentation (corrected count).',
      'Three random PNs and ten random KCs with their skeletons.')

# ================================================================ 2 comparison
R.section('New claws against the old ones and Prat\'s segmentation')
R.p('Prat\'s segmentation is the uploaded hemibrain file of claw counts per KC (`n_claws` and, after correction, `n_claws_corrected`; the '
    'corrected count is used). It has counts only, no claw positions, so it can be compared per KC but not drawn. '
    f'{len(H3):,} hemibrain KCs are in all three. {len(PR_MISSING)} of Prat\'s KCs are left out because they have no PN input in our '
    f'corrected calyx ({int((PR_MISSING.KC_subtype == "KC part due to gap").sum())} are labelled "KC part due to gap").')


def fig_three():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (a, b) in zip(axs[:2], [('new', 'Prat'), ('old', 'Prat')]):
        x, y = H3[b].astype(int).values, H3[a].astype(int).values
        M = np.zeros((23, 23)); np.add.at(M, (np.clip(y, 0, 22), np.clip(x, 0, 22)), 1)
        cm = L.SEQ.copy(); cm.set_bad('white')
        im = ax.imshow(np.where(M > 0, M, np.nan), origin='lower', cmap=cm, aspect='equal', interpolation='nearest')
        ax.plot([0, 22], [0, 22], color=L.MUTED, lw=0.8, ls='--')
        g = AG[(a, b)]
        ax.set_title(f'{"ab"[a == "old"]}  {a} vs Prat\'s: Pearson r = {g["r"]:.2f}\nsame count {L.pct(g["same"])}, within 1 {L.pct(g["w1"])}', fontsize=8.8)
        ax.set_xlabel('claws per KC, Prat\'s (corrected)'); ax.set_ylabel(f'claws per KC, {a} rule')
        ax.set_xlim(-0.5, 20.5); ax.set_ylim(-0.5, 20.5)
        L.cbar(fig, im, ax, 'KCs')
    ax = axs[2]
    order = ["KCg-m", "KCab-c", "KCab-s", "KCab-m", "KCa'b'-ap1", "KCa'b'-ap2", "KCa'b'-m", "KCg-d", "KCab-p"]
    BY = H3.groupby('type')[['old', 'new', 'Prat']].mean().reindex(order).dropna()
    n = H3.groupby('type').size()
    yy = np.arange(len(BY))
    for i, m in enumerate(['old', 'new', 'Prat']):
        ax.scatter(BY[m], yy + (i - 1) * 0.22, s=30, color=C3[m], label={'old': 'old (2 µm)', 'new': 'new (8 µm, split by PN)', 'Prat': 'Prat\'s'}[m], zorder=3)
    for j in yy: ax.axhline(j, color=L.GRID, lw=0.5, zorder=1)
    ax.set_yticks(yy, [f'{t} ({n[t]})' for t in BY.index], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel('mean claws per KC'); ax.set_xlim(0, 12); ax.legend(fontsize=7, loc='lower right')
    ax.set_title('c  Mean per KC subtype (number of KCs)', fontsize=8.8)
    fig.tight_layout(); return fig


R.fig(fig_three(), '**a**, **b**, Each cell counts the KCs with a given pair of claw counts (color, white = none); the dashed line is equality. '
      '**a**, new rule against Prat\'s; **b**, old 2 µm rule against Prat\'s. **c**, mean claws per KC for each hemibrain KC subtype under the three '
      'segmentations (yellow old, green new, violet Prat\'s). The full comparison, with the uncorrected counts and difference histograms, is '
      '`claws_geodesic/Yang_rule_2026-11-04/chosen_min2PSD/three_way_claw_comparison.png`.', 'Claws per hemibrain KC: new, old and Prat\'s.')
R.table(['pair', 'Pearson r', 'mean difference', 'same count', 'within 1', 'within 2'],
        [[f'{a} vs {b.replace("Prat_before", "Prat (uncorrected)").replace("Prat", "Prat")}', F(g['r']), F(g['md'], 2, True), L.pct(g['same']), L.pct(g['w1']), L.pct(g['w2'])]
         for (a, b), g in AG.items()],
        f'Per-KC agreement over {len(H3):,} hemibrain KCs. Means: old {H3.old.mean():.1f}, new {H3.new.mean():.1f}, Prat\'s {H3.Prat.mean():.1f} '
        f'(uncorrected {H3.Prat_before.mean():.1f}) claws per KC.', 'How well the claw counts agree.', num_cols=[1, 2, 3, 4, 5])
R.p(f'The new rule matches Prat\'s counts much better than the old one (Pearson r {AG[("new", "Prat")]["r"]:.2f} against '
    f'{AG[("old", "Prat")]["r"]:.2f}; {L.pct(AG[("new", "Prat")]["w1"])} of KCs within one claw, against {L.pct(AG[("old", "Prat")]["w1"])}). '
    'The old rule overcounted most where a claw is long or branched (α′β′-m and γ-m KCs): 2 µm in a straight line cuts such a claw '
    'into pieces, while 8 µm along the skeleton keeps it whole. The new rule gives slightly fewer claws than Prat\'s on average.')
R.p('**Examples.** `claws_geodesic/Yang_rule_2026-11-04/chosen_min2PSD/claw_disparity_KCs.pdf` (and a rotatable 3D view, '
    '`claw_disparity_KCs_3D.html`) shows the 40 KCs where the old and new claws differ most, old claws on the left and new on the right, '
    'Prat\'s count in each title. There, PN → KC PSDs are colored and shaped by the glomerulus of their presynaptic PN, the same in both '
    'panels, so the presynaptic partners can be seen not to change between segmentations; only their grouping into claws does.')

# ================================================================ 3 statistics
R.section('How many claws and boutons, and what a claw holds')
cpk = {ds: CL[ds].groupby('kc_id').size() for ds in DS2}
cpk_old = {ds: CLO[ds].groupby('kc_id').size() for ds in DS4}
bpp = {ds: BT[ds].groupby('pn_id').size() for ds in DS4}
COMP['pn'] = COMP.own_PN_PSDs + COMP.other_PN_PSDs
COMP['own_share'] = COMP.own_PN_PSDs / COMP.pn


def fig_stats():
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.0))
    for ds in DS4:
        axs[0].hist(bpp[ds].clip(upper=30), bins=np.arange(0.5, 31), histtype='step', lw=1.5, color=L.DS_COLOR[ds], density=True,
                    label=f'{LAB[ds]} ({bpp[ds].median():.0f})')
    for ds in DS2:
        c = L.DS_COLOR[ds]
        axs[1].hist(cpk[ds].clip(upper=20), bins=np.arange(0.5, 21), histtype='step', lw=1.6, color=c, density=True, label=f'{LAB[ds]}, new ({cpk[ds].median():.0f})')
        axs[1].hist(cpk_old[ds].clip(upper=20), bins=np.arange(0.5, 21), histtype='step', lw=1.0, ls='--', color=c, density=True, label=f'{LAB[ds]}, old ({cpk_old[ds].median():.0f})')
        v = COMP[COMP.dataset == ds]
        axs[2].hist(v.own_PN_PSDs.clip(upper=60), bins=np.arange(0.5, 61), histtype='step', lw=1.5, color=c, density=True,
                    label=f'{LAB[ds]} ({v.own_PN_PSDs.median():.0f})')
        sz = v.own_PN_PSDs + v.other_PN_PSDs + v.non_PN_sites
        axs[3].hist((v.non_PN_sites / sz), bins=np.linspace(0, 1, 21), histtype='step', lw=1.5, color=c,
                    weights=np.full(len(v), 1 / len(v)), label=f'{LAB[ds]} ({(v.non_PN_sites / sz).median():.2f})')
    labs = [('PN boutons per PN (calyx)', 'fraction of PNs'), ('claws per KC', 'fraction of KCs'),
            ('PSDs of the claw\'s own PN', 'fraction of claws'), ('share of a claw\'s sites that are not PN PSDs', 'fraction of claws')]
    for ax, (xl, yl), t in zip(axs, labs, 'abcd'):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=6.0, title='median', title_fontsize=6.3); L.tag(ax, t, x=-0.3)
    fig.tight_layout(); return fig


R.fig(fig_stats(), 'Line color = dataset; the number in brackets is the median. **a**, boutons per antennal-lobe PN in the calyx (all four '
      'datasets; FlyWire\'s table gives each connection its own presynaptic point, which likely explains its higher count). **b**, claws per KC '
      'with the new rule (solid) and the old 2 µm rule (dashed), KCs with at least one claw. **c**, PSDs from the claw\'s own PN. **d**, the '
      'share of a claw\'s sites that are not PN → KC PSDs (APL and other partners, the KC\'s T-bars).', 'Boutons per PN, claws per KC, and what a claw holds.')
sh = SHR.set_index(['dataset', 'sites'])['share on a claw']
pur = COMP.groupby('dataset').own_share.apply(lambda v: (v >= 0.9).mean())
R.p(f'Hemibrain KCs have a median of {cpk["hemibrain"].median():.0f} new claws ({cpk_old["hemibrain"].median():.0f} with the old rule) and '
    f'male CNS KCs {cpk["malecns"].median():.0f} ({cpk_old["malecns"].median():.0f}). Each new claw has one PN by construction; PSDs of a PN '
    f'with a single PSD in the cluster can still join it as "other sites", so {L.pct(pur["hemibrain"])} (hemibrain) and '
    f'{L.pct(pur["malecns"])} (male CNS) of claws get at least 90% of their PN PSDs from their own PN. '
    f'{L.pct(COMP[COMP.dataset == "hemibrain"].shares_cluster.mean())} of hemibrain claws share their 8 µm cluster with another PN\'s claw, '
    'which is where the split by PN matters.')
R.table(['sites', 'hemibrain', 'male CNS'],
        [[k, L.pct(sh[('hemibrain', k)]), L.pct(sh[('malecns', k)])] for k in ['all calyx sites', 'PN->KC PSDs', 'other PSDs (non-PN partner)', 'KC T-bars']],
        'Share of each kind of KC calyx site that ends up on a claw.', 'Which sites are on claws.', num_cols=[1, 2])

# ================================================================ 4 APL synapses
R.section('Placing APL synapses on claws and boutons')
R.p('**KC partners.** An APL ↔ KC synapse in the calyx is a site of the KC: a PSD when APL is presynaptic, a T-bar of the KC when the KC '
    'is. That site already has its claw from the rule above (step 3), so the synapse is on a claw exactly when its KC-side site is. Its '
    'distance $\\delta(e)$ along the KC to the nearest PSD of a claw\'s own PN is stored as `claw_dist_nm`.')
mc = EXP['malecns']
R.p(f'{L.pct(EXP["hemibrain"]["matched to a claw site"], 1)} of hemibrain and {L.pct(mc["matched to a claw site"], 1)} of male CNS KC ↔ APL '
    'calyx synapses are claw-segmentation sites. The rest, almost all in the back of the male CNS calyx, lie in the corrected calyx but more '
    'than 3 µm from any PN → KC PSD; they got the same rule straight on the skeleton, and '
    f'{L.pct(mc["of those on a claw"])} of them landed on a claw.')
R.p('**PN partners.** APL ↔ PN synapses are placed on boutons as before: on the bouton of the nearest T-bar of that same PN, if within 2 µm.')


def fig_dist():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    grid = np.linspace(0, 8, 161)
    for ax, dr, t in zip(axs[:2], ['KC_to_APL', 'APL_to_KC'], ['a  KC → APL synapses in the calyx', 'b  APL → KC synapses in the calyx']):
        for ds in DS2:
            k = KA[(KA.dataset == ds) & (KA.direction == dr)]
            d = k.claw_dist_nm.fillna(np.inf).values / 1e3
            ax.plot(grid, [(d <= g).mean() for g in grid], color=L.DS_COLOR[ds], label=f'{LAB[ds]} ({L.pct(k.claw_id.notna().mean())} on a claw)')
        ax.set_xlim(0, 8); ax.set_ylim(0, 1.01)
        ax.set_title(t); ax.set_xlabel('$\\delta$: distance along the KC to the nearest\nPSD of a claw\'s own PN (µm)')
        ax.set_ylabel('cumulative fraction of synapses'); ax.legend(fontsize=6.8, loc='lower right'); ax.yaxis.grid(True, color=L.GRID, lw=0.6)
    ax = axs[2]
    rows = []
    for ds in DS2:
        for dr in ['KC_to_APL', 'APL_to_KC']:
            k = KA[(KA.dataset == ds) & (KA.direction == dr)]
            rows.append((f'{"hemibrain" if ds == "hemibrain" else "male CNS"}\n{dr.replace("_to_", " → ")}', k.old_claw_id_2um.notna().mean(), k.claw_id.notna().mean()))
    y = np.arange(len(rows))
    ax.barh(y + 0.19, [r[2] * 100 for r in rows], height=0.36, color=C3['new'], label='new claws')
    ax.barh(y - 0.19, [r[1] * 100 for r in rows], height=0.36, color=C3['old'], label='old 2 µm claws')
    for yi, r in zip(y, rows):
        ax.text(r[2] * 100 + 1, yi + 0.19, L.pct(r[2]), va='center', fontsize=7, color=L.INK)
        ax.text(r[1] * 100 + 1, yi - 0.19, L.pct(r[1]), va='center', fontsize=7, color=L.INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel('% of KC ↔ APL calyx synapses on a claw'); ax.legend(fontsize=7, loc='upper right'); ax.set_title('c  On a claw: new vs old')
    fig.tight_layout(); return fig


R.fig(fig_dist(), '**a**, **b**, Cumulative distributions over all KC ↔ APL synapses in the corrected calyx, one line per dataset: the '
      'fraction whose KC-side site lies within the distance on the x-axis, measured along the KC skeleton, of a PSD of one of the KC\'s claws\' '
      'own PNs. The axis ends at the 8 µm cut: distances were computed only up to 8 µm, so each curve ends at the share of synapses on a '
      'claw, and the rest are further than 8 µm along the KC (or in a cluster with no claw). **c**, Share of these synapses on a claw with the new rule (green) and with the old 2 µm rule (yellow).',
      'How far KC ↔ APL synapses lie from claws, and how many are on one.')
R.p('With the new claws, most KC ↔ APL calyx synapses are on a claw: '
    + ', '.join(f'{LAB[ds]} {L.pct(KA[(KA.dataset == ds) & (KA.direction == "KC_to_APL")].claw_id.notna().mean())} of KC → APL and '
                f'{L.pct(KA[(KA.dataset == ds) & (KA.direction == "APL_to_KC")].claw_id.notna().mean())} of APL → KC' for ds in DS2)
    + '. The old 2 µm rule put only 21–49% on claws, because it measured straight-line distance to the PN PSDs and APL synapses often sit '
    'a few micrometres along the claw from them.')


def fig_where():
    cal = [('hemibrain', 'R'), ('malecns', 'L'), ('malecns', 'R')]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.4))
    for j, (ds, side) in enumerate(cal):
        a = A[ds]; a = a[(a.side == side) & a.partner_type.astype(str).str.startswith('KC')]
        P = to_rda(ds, a.partner_side_x_nm.values, a.partner_side_y_nm.values, a.partner_side_z_nm.values)
        lobe_ap = {'hemibrain': 1, 'malecns': -1}[ds]       # sign that puts the lobes (anterior) to the right in the side view
        P[:, 2] *= lobe_ap
        on = a.claw_id.notna().values
        rng = np.random.default_rng(0)
        for i, (cx, cy, xl) in enumerate([(0, 1, 'fly\'s right → (µm)'), (2, 1, 'anterior → (µm)')]):
            ax = axs[i, j]
            for m, c, s in [(~on, OFF_C, 0.9), (on, ON_C, 0.9)]:
                idx = np.flatnonzero(m); idx = rng.choice(idx, min(15000, len(idx)), replace=False)
                ax.scatter(P[idx, cx], P[idx, cy], s=s, color=c, lw=0, rasterized=True)
            ax.set_aspect('equal', adjustable='datalim'); ax.set_xlabel(xl); ax.set_ylabel('dorsal ↑ (µm)')
            if i == 0:
                ax.set_title(f'{LAB[ds]}, {side} calyx, posterior view\n{len(a):,} KC ↔ APL synapses, {L.pct(on.mean())} on a claw')
            else:
                ax.set_title(f'{LAB[ds]}, {side} calyx, side view')
    h = [Line2D([], [], marker='o', ls='', color=ON_C, label='on a claw (new rule)'), Line2D([], [], marker='o', ls='', color=OFF_C, label='not on a claw')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1)); return fig


R.fig(fig_where(), 'Every KC ↔ APL synapse (both directions) in the three corrected calyces used by the model, drawn at its KC-side site. '
      'Green = on a claw, grey = not. Top: posterior view (axes of Table 4, µm). Bottom: side view, anterior to the right (sign set so that the '
      'lobes lie anterior of the calyx), dorsal up. At most 15,000 points of each color are drawn per panel (random subsample).',
      'APL ↔ KC synapses in the calyx, on and off claws.')

# ================================================================ 5 off-claw on the skeleton
R.section('Where the off-claw APL synapses sit on the KC')
R.p('For male CNS KCs of the right calyx, every calyx KC ↔ APL synapse was placed on the KC\'s skeleton and classified. The skeleton is '
    'rooted at its most anterior node (a lobe tip), so "downstream" means away from the axon. For the synapse\'s node, count the claws whose '
    'PSDs lie at or downstream of it:')
R.ul(['**on a claw**: the synapse is on a claw (rule above);',
      '**dendritic trunk**: not on a claw, upstream of two or more claws (a larger branch, before it splits into claws);',
      '**claw stalk**: upstream of exactly one claw (the branch leading to one claw, more than 8 µm from its PSDs);',
      '**branch without claw**: no claw downstream (a side branch, or the soma neurite).'])
sp = POP['share by location (calyx)']; ko = POP['share by location, KC->APL']; ao = POP['share by location, APL->KC']
LOC = ['on claw', 'dendritic trunk', 'claw stalk', 'branch without claw']
R.table(['location', 'all', 'KC → APL', 'APL → KC'], [[l, L.pct(sp[l]), L.pct(ko[l]), L.pct(ao[l])] for l in LOC],
        f'All {POP["KCs"]:,} right-calyx male CNS KCs with a skeleton, {POP["calyx KC<->APL synapses"]:,} calyx KC ↔ APL synapses. '
        f'Off-claw synapses lie a median of {POP["off-claw: distance along the KC to the nearest PN->KC site (um), median / 90%"][0]:.0f} µm '
        f'along the KC from the nearest PN → KC PSD (90% within {POP["off-claw: distance along the KC to the nearest PN->KC site (um), median / 90%"][1]:.0f} µm).',
        'Where calyx KC ↔ APL synapses sit on the KC.', num_cols=[1, 2, 3])


def fig_skel():
    img = plt.imread(SKD + 'KC_skeletons_APL_synapses_calyx.png')
    fig = plt.figure(figsize=(12, 12 * img.shape[0] / img.shape[1]))
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis('off')
    return fig


R.fig(fig_skel(), 'The ten right-calyx male CNS KCs with the most calyx KC ↔ APL synapses on the dendritic trunk (at most two per KC '
      'type), so the panels show the case Yang asked about: APL synapses on the larger branches before they split into claws. For each KC, '
      'the posterior view (left) and the side view seen from the fly\'s right (right) of its calyx part. Grey lines: skeleton; small grey '
      'dots: PN → KC PSDs; dashed circles: claw centroids. KC ↔ APL synapses: circles KC → APL, triangles APL → KC, colored by location '
      '(green on a claw, purple dendritic trunk, brown claw stalk, pink branch without claw); black outline = outside the official ROIs. '
      'Background: the corrected right calyx (grey inside the official ROI, orange only in the corrected calyx). Whole KCs and a rotatable '
      '3D view are in `kc_skeletons/claws_v2m2/`.', 'Ten KCs with off-claw APL synapses on their larger branches.')
R.p(f'Across all right-calyx KCs, {L.pct(sp["on claw"])} of calyx KC ↔ APL synapses are on claws. Of the rest, the largest group sits on the '
    f'dendritic trunk ({L.pct(sp["dendritic trunk"])} of all), then on branches without claws ({L.pct(sp["branch without claw"])}) and on '
    f'claw stalks ({L.pct(sp["claw stalk"])}). KC → APL synapses are more often off-claw than APL → KC synapses '
    f'({L.pct(1 - ko["on claw"])} against {L.pct(1 - ao["on claw"])}), and they sit on the trunk more often ({L.pct(ko["dendritic trunk"])} '
    f'against {L.pct(ao["dendritic trunk"])}).')

# ================================================================ 6 posterior view
R.section('Drawing the calyx: the posterior view')
R.p('Every 2D map in these reports shows the calyx as seen from behind the fly (posterior view), with the fly\'s right to the right and '
    'dorsal up, so that all datasets look alike. Each dataset has its own axes, so the view is built from its native coordinates '
    '(Table 4), checked against mushroom-body landmarks (the lobes are anterior and ventral of the calyx, the vertical lobe is dorsal).')
R.table(['Dataset', 'Right (x of the map)', 'Dorsal (y of the map)', 'Anterior–posterior (depth)', 'Note'],
        [['FlyWire 783', '$x$', '$-y$', '$z$', 'FlyWire coordinates are mirror-imaged relative to its side labels, so sides come from the labels'],
         ['Hemibrain v1.2', '$-x$', '$-z$', '$y$', 'the hemibrain $y$ axis is anterior–posterior'],
         ['Male CNS v1.0', '$-x$', '$-y$', '$z$', ''], ['BANC 888', '$-x$', '$z$', '$y$', '']],
        title='Native axes used for the posterior view.', caption='Coordinates are in nm in the tables and µm in the maps. The 3D maps of '
        'Report 03 use the same right and dorsal axes, with anterior signed so that the peduncle lies anterior of the calyx.')
sh2 = SH.set_index('calyx')
R.p('The calyx outlines still differ between datasets in these maps, mainly because each brain sits at a slightly different angle. The '
    'calyx is a flattened cup; its thinnest axis is tilted from the anterior–posterior axis by '
    + ', '.join(f'{F(sh2.loc[c, "tilt_of_calyx_axis_from_AP_deg"], 0)}° ({c})' for c in sh2.index if ('Hemibrain' in c or 'Male' in c))
    + f', and by {F(sh2[sh2.index.str.startswith("BANC")].tilt_of_calyx_axis_from_AP_deg.min(), 0)}–'
      f'{F(sh2[sh2.index.str.startswith("BANC")].tilt_of_calyx_axis_from_AP_deg.max(), 0)}° in BANC, '
      'so BANC\'s calyx is seen more obliquely and looks flatter. `odor_weighted/calyx_shape_by_dataset.png` shows every calyx from three sides.')

# ================================================================ 7 corrected calyx
R.section('The corrected calyx: synapses just outside the official ROIs')
R.p('Each connectome defines its mushroom body regions with its own ROI meshes, and these do not always cover the whole structure. In the '
    'male CNS the calyx ROIs leave out the posterior part of each calyx; in FlyWire and BANC the meshes of nearby neuropils (CRE, SIP, SMP) '
    'take in parts of the lobes. `with_outside_ROI_synapses/` holds the tables with these synapses added back, and since 2026-11-03 it is the '
    'main version of every regenerated report (the official-ROI tables are kept unchanged). The rule, in order:')
R.ul(['**Anchors.** Calyx: the PN → KC synapses inside the official calyx ROI, extended through PN → KC synapses outside every MB ROI '
      'that connect to them in steps of ≤ 3 µm (claws the ROI cut off). Peduncle and lobes: the APL ↔ KC synapses inside the official '
      'peduncle and lobe ROIs.',
      '**Growing through KC synapses.** APL ↔ KC synapses outside every MB ROI join the first anchor that reaches them in steps of ≤ 3 µm, '
      'and take that anchor\'s region. KC dendrites and axons exist only in the mushroom body, so these chains stay inside it.',
      '**Joining.** Any other synapse outside every MB ROI joins when its postsynaptic site lies within 3 µm of an anchor point, with the '
      'region of the nearest one (the calyx wins ties).',
      '**Accessory calyces stay out.** No chain runs through synapses with γd or αβp KCs, whose dendrites form the accessory calyces.',
      '**PN synapses** next to the calyx join only if they are PN → KC or their T-bar belongs to a calyx bouton.'], ordered=True)
R.p('Every added row is flagged in `mb_region_source`; keeping the rows marked `official ROI` gives the official tables back exactly.')
ap = ADD[ADD.synapses.str.startswith('APL')].copy()
ap['reg'] = np.where(ap.mb_region == 'CA', 'calyx', np.where(ap.mb_region == 'PED', 'peduncle', 'lobes'))
g = ap.groupby(['dataset', 'reg'])[['n_official_ROI', 'n_added']].sum()
kcca = ap[(ap.reg == 'calyx') & (ap.partner_group == 'KC')].groupby('dataset')[['n_official_ROI', 'n_added']].sum()
pnkc = ADD[ADD.synapses.str.startswith('ALPN')].groupby('dataset')[['n_official_ROI', 'n_added']].sum()


def pc(n, a): return f'+{100 * a / n:.1f}%'


R.table(['Dataset', 'APL synapses, calyx', 'peduncle', 'lobes', 'KC ↔ APL, calyx', 'PN → KC, calyx'],
        [[LAB[ds]] + [f'+{int(g.loc[(ds, r), "n_added"]):,} ({pc(*g.loc[(ds, r)].values)})' for r in ['calyx', 'peduncle', 'lobes']]
         + [pc(*kcca.loc[ds].values), pc(*pnkc.loc[ds].values)] for ds in DS4],
        title='Synapses added in the corrected version.', caption='Added synapses as a count and as a share of the official-ROI count, from '
        '`with_outside_ROI_synapses/added_synapses_summary.csv`. The male CNS calyx gains the most (+31% of APL synapses).', num_cols=[1, 2, 3, 4, 5])


def fig_added():
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.4))
    cal = [('hemibrain', 'R'), ('malecns', 'L'), ('malecns', 'R')]
    for j, (ds, side) in enumerate(cal):
        a = A[ds]; a = a[(a.side == side) & a.partner_type.astype(str).str.startswith('KC')]
        P = to_rda(ds, a.partner_side_x_nm.values, a.partner_side_y_nm.values, a.partner_side_z_nm.values)
        P[:, 2] *= {'hemibrain': 1, 'malecns': -1}[ds]
        off = (a.mb_region_source == 'official ROI').values; add = ~off
        rng = np.random.default_rng(0)
        io = rng.choice(np.flatnonzero(off), min(20000, off.sum()), replace=False); ia = rng.choice(np.flatnonzero(add), min(12000, add.sum()), replace=False)
        for i, (cx, cy, xl) in enumerate([(0, 1, 'fly\'s right → (µm)'), (2, 1, 'anterior → (µm)')]):
            ax = axs[i, j]
            ax.scatter(P[io, cx], P[io, cy], s=0.7, color=OFFICIAL_C, lw=0, rasterized=True)
            ax.scatter(P[ia, cx], P[ia, cy], s=0.9, color=ADDED_C, lw=0, rasterized=True)
            ax.set_aspect('equal', adjustable='datalim'); ax.set_xlabel(xl); ax.set_ylabel('dorsal ↑ (µm)')
            ax.set_title(f'{LAB[ds]}, {side} calyx, posterior view\n{off.sum():,} in ROI, {add.sum():,} added ({pc(off.sum(), add.sum())})' if i == 0
                         else f'{LAB[ds]}, {side} calyx, side view')
    h = [Line2D([], [], marker='o', ls='', color=OFFICIAL_C, label='KC ↔ APL synapse inside the official calyx ROI'),
         Line2D([], [], marker='o', ls='', color=ADDED_C, label='KC ↔ APL synapse added to the calyx (outside every official MB ROI)')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1)); return fig


R.fig(fig_added(), 'Every KC ↔ APL synapse (both directions) in the corrected calyx, at its KC-side site, for the three calyces used by the '
      'model. Grey = inside the dataset\'s official calyx ROI; orange = added by the rule above. Top row: posterior view (axes of Table 4, '
      'µm). Bottom row: side view, anterior to the right, dorsal up. Up to 20,000 grey and 12,000 orange points are drawn per panel (random '
      'subsample); the counts in the titles are of all KC ↔ APL calyx synapses. In the left male CNS calyx the added synapses form the back '
      '(posterior) part of the calyx; in the right one they lie mostly along its ventral and posterior edge.',
      'Where the corrected calyx adds KC ↔ APL synapses.')
R.box('Caveats', ['Where an accessory calyx touches the calyx or peduncle, some synapses with γd and αβp KCs within 3 µm of an anchor still '
                  'join (male CNS 1,291; hemibrain 1,487). They are flagged rows with `partner_type` KCg-d or KCab-p.',
                  'The 3 µm steps of the corrected calyx and the 8 µm cut of the claws are choices; the claws were checked against Prat\'s '
                  'counts, the corrected calyx by eye on maps.',
                  'Male CNS skeletons are coarse (about 0.7 µm between nodes), so distances along them are approximate at that scale.'], 'caveat')
R.box('Where this is used', ['Report 03 colors KC ↔ APL synapses by the odor response of their claw\'s glomerulus, using synapses on a claw.',
                              'Report 04 counts APL → KC synapses on the claws of each glomerulus (the claw weight $q_g$).',
                              'Reports 05–12 build each KC\'s input from its claws.'], 'plain')
R.save()
