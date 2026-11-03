"""Report 03: odor-weighted maps. Which odor response data are used, how each APL synapse gets a glomerulus and so an odor
response, how the color scale is built, and what the odor-pair similarity of on-claw synapses shows."""
import sys
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import report_lib as L
L.style()
F = L.fmt
OW = L.OW
ORN_TABLE = '/mnt/project-files/uploads/hearth/5d477ed3-345f-4332-8b48-1ad3e2e0a0c1'   # ORN_ovlp16_resp_flyave_merge_dropna.csv
RDBU = matplotlib.colormaps['RdBu_r']
NODATA = '#9a9994'; PANEL_BG = '#e4e3dd'
DSN = {'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'flywire': 'FlyWire 783', 'banc': 'BANC 888'}

orn = pd.read_csv(ORN_TABLE, index_col=0)
ODORS = list(orn.columns)
ofm = pd.read_csv(OW + 'ORN_fly_means_by_glomerulus.csv', index_col=0)
col16 = {o: next(c for c in ofm.columns if c.split(' @ ')[0] == o) for o in ODORS}
assert np.allclose(ofm[[col16[o] for o in ODORS]].reindex(orn.index).values, orn.values, atol=1e-4)   # per-fly means reproduce the table
gh = pd.read_csv(OW + 'GH146_mean_by_glomerulus.csv', index_col=0)
GH_COL = {o: f'megamat: {o} @ -3' for o in ODORS} | {'paa': 'glomeruli_diagnostics: paa @ -5'}
gh16 = pd.DataFrame({o: gh[c] for o, c in GH_COL.items() if c in gh.columns})
GHM = gh.drop(columns='n_flies')
PN = pd.read_csv(OW + 'PN_APL_synapses_ORN_per_odor_weighted.csv', usecols=['dataset', 'apl_side', 'direction', 'glomerulus', 'mb_region',
                                                                          'view_right_um', 'view_dorsal_um', 'pn_type'], low_memory=False)
PN = PN[PN.mb_region == 'CA']
KC = pd.read_csv(OW + 'KC_APL_synapses_on_claws_by_glomerulus.csv', usecols=['dataset', 'apl_side', 'direction', 'glomeruli', 'view_right_um', 'view_dorsal_um'])
SIM = pd.read_csv(OW + 'odor_similarity_KC_APL_on_claws.csv')
NORM_ORN = TwoSlopeNorm(vcenter=0, vmin=ofm.min().min(), vmax=ofm.max().max())
NORM_GH = TwoSlopeNorm(vcenter=0, vmin=GHM.min().min(), vmax=GHM.max().max())

R = L.Report('03', 'How does each APL synapse in the calyx get an odor response, how are the odor-weighted maps drawn, and does weighting '
             'by APL\'s synapses on claws change how similar two odors look?',
             inputs=['Yang\'s ORN table `ORN_ovlp16_resp_flyave_merge_dropna.csv` (36 glomeruli × 16 odors, fly means)',
                     'Yang\'s per-fly ORN pickles `orn_resp_megamat_trialmean_resp.pkl` (9 flies, 17 odors) and `orn_resp_validation2_trialmean_resp.pkl` (5 flies, 22 odors)',
                     'Yang\'s GH146 PN imaging `GH146_ij_certain-roi_stats.csv` (7 flies, 29 glomeruli, 41 odor presentations, 3 repeats)',
                     'The APL tables, claws and boutons of Reports 01 and 02'],
             outputs=['`odor_weighted/PN_APL_synapses_*_weighted.csv`, `KC_APL_synapses_on_claws_by_glomerulus.csv` (synapse tables with weights)',
                      '`odor_weighted/*_map_*_posterior_view.png` (the maps)', '`ORN_fly_means_by_glomerulus.csv`, `GH146_mean_by_glomerulus.csv`',
                      '`odor_weighted/odor_similarity_KC_APL_on_claws.csv/.png`'],
             scripts=['`scripts/odor_meta.py`, `odor_maps.py`, `gh146_maps.py`, `glom_maps.py`, `orn_load.py`, `orn_maps.py`, `claw_glom_maps.py`, '
                      '`gh146_per_odor_maps.py`, `odor_similarity.py`', '`scripts/reports/r03_odor_maps.py` (this report)'])

R.p('The maps were made to see where in the calyx APL meets the PNs and claws that a given odor drives. The same glomerulus-to-response '
    'step is what feeds odors into the model (Report 05), so it is described here in full. All maps are in both versions of the tables '
    '(official ROIs, and with the synapses outside them added; Report 02).')

R.symbols([
    ('g', 'An antennal-lobe glomerulus (FlyWire names; hemibrain VC3l, VC3m, VC5 renamed VC3, VC5, VM6)', '36 in the ORN table, 29 in GH146'),
    ('o', 'An odor presentation: odor name and log10 dilution, e.g. "pa @ −3"', ''),
    ('f', 'A fly (ORN or GH146 imaging)', ''),
    ('r^{(f)}_{g,o}', 'Response of glomerulus $g$ to odor $o$ in fly $f$: ORN = trial-mean response as given in Yang\'s pickles; GH146 = peak ΔF/F of one trial',
     'units as in Yang\'s files; ΔF/F for GH146'),
    ('\\mathcal{F}_{g,o}', 'The flies in which glomerulus $g$ was imaged with odor $o$', ''),
    ('\\bar r_{g,o}', 'Fly-mean response (Equation 1)', 'same units'),
    ('g(e)', 'Glomerulus of APL synapse $e$: of its PN (PN↔APL) or of its claw\'s main PN (KC↔APL on a claw)', ''),
    ('w_o(e)', 'Weight of synapse $e$ for odor $o$: $\\bar r_{g(e),o}$', 'blank if $g(e)$ has no response'),
    ('v_{\\min},\\ v_{\\max}', 'Smallest and largest value of the whole mean matrix; the ends of the color scale', 'ORN −0.476 / 3.503; GH146 −0.158 / 1.962'),
    ('\\rho_{ab}', 'Similarity of odors $a$ and $b$ in Section 5: Pearson correlation over synapses (or over glomeruli)', '−1 to 1'),
])

# ---------------------------------------------------------------- 1
R.section('Odor responses per glomerulus')
R.p('Three response tables from Yang are used. The **ORN table** gives the response of the olfactory receptor neurons of each glomerulus '
    '(pebbled imaging) to the 16 odors of the TNT / TNT3c experiment; it is the input to the model. The **per-fly ORN pickles** give the same '
    'kind of response per fly, for 39 odors in two panels (megamat: 17 odors at 10⁻³; validation2: 22 odors at several dilutions). The '
    '**GH146 table** gives peak ΔF/F of the projection neurons (GH146 line) in 29 glomeruli, per fly and trial, for 41 odor presentations '
    '(24 in a "diagnostics" panel and 17 in megamat). Each is averaged over flies:')
R.eq('\\bar r_{g,o} \\;=\\; \\frac{1}{|\\mathcal{F}_{g,o}|} \\sum_{f \\,\\in\\, \\mathcal{F}_{g,o}} r^{(f)}_{g,o}',
     'ORN: flies where the glomerulus was imaged (missing values skipped). GH146: every fly has 3 repeats, so the mean over flies and repeats equals the mean of per-fly means.')
R.p('The fly means of the ORN pickles reproduce the ORN table exactly (largest difference 2 × 10⁻¹⁶), so the table is the same data. '
    'For the 16 model odors GH146 has 13: the 12 megamat odors at 10⁻³ and paa at 10⁻⁵ (diagnostics panel); −aPine, euc and B-myr were not presented.')


def fig_heat():
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 6.4), gridspec_kw=dict(width_ratios=[16, 13]))
    for ax, M, nrm, t, lab in [(axs[0], orn, TwoSlopeNorm(vcenter=0, vmin=orn.min().min(), vmax=orn.max().max()),
                                'a  ORN responses (36 glomeruli × 16 odors)', 'ORN response, fly mean $\\bar r_{g,o}$\n(units as in Yang\'s table)'),
                               (axs[1], gh16.reindex(columns=[o for o in ODORS if o in gh16.columns]), None,
                                'b  GH146 PN responses (29 glomeruli × 13 odors)', 'GH146 peak ΔF/F, mean over\n7 flies × 3 repeats')]:
        if nrm is None: nrm = TwoSlopeNorm(vcenter=0, vmin=min(-1e-3, M.min().min()), vmax=M.max().max())
        im = ax.imshow(M.values, cmap=RDBU, norm=nrm, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(M.shape[1]), M.columns, rotation=90, fontsize=7); ax.set_yticks(range(M.shape[0]), M.index, fontsize=6.5)
        ax.set_xlabel('odor (ORN: paa @ −4, −aPine @ −1.5, euc @ −1.75, B-myr @ −2, others @ −3)' if 'ORN' in t else 'odor (paa @ −5, others @ −3)')
        ax.set_ylabel('glomerulus'); ax.set_title(t); ax.spines[:].set_visible(False)
        cb = L.cbar(fig, im, ax, lab, fraction=0.05); cb.ax.set_yscale('linear')
    fig.tight_layout(); return fig


R.fig(fig_heat(), 'Rows: glomeruli; columns: the 16 odors used by the model (GH146 lacks three of them). Color: fly-mean response '
      '(Equation 1) on a red–white–blue scale whose white is exactly 0, blue end the smallest value and red end the largest value of each '
      'panel\'s matrix (two-slope scale: negative and positive values are stretched separately, so the colorbar is linear in value on each '
      'side of 0). The two panels have their own scales. Odor abbreviations follow Yang\'s files (pa: pentyl acetate, paa: phenylacetaldehyde, '
      'ms: methyl salicylate, eb: ethyl butyrate, ep: ethyl propionate, 2h: 2-heptanone, IaA: isoamyl acetate, 2-but: 2-butanone, 6al: hexanal, '
      't2h: trans-2-hexenal, 1-5ol / 1-6ol / 1-8ol: pentanol / hexanol / octanol, −aPine: (−)-α-pinene, euc: eucalyptol, B-myr: β-myrcene).',
      'The odor response tables used throughout.')

# ---------------------------------------------------------------- 2
R.section('Giving each synapse a glomerulus')
R.ul(['**PN ↔ APL synapses.** The PN\'s cell type names its glomerulus (e.g. `DM1_lPN` → DM1). Only uniglomerular types count. Types that name '
      'two glomeruli (VP1d+VP4, VP1m+VP5, VP3+VP1l) count for both; "X+" types (VP2+, VP3+) count for X. Multiglomerular (M_) and '
      'untyped PNs get no glomerulus: 1.2–1.4% of PN ↔ APL calyx synapses.',
      '**KC ↔ APL synapses.** Only synapses **on a claw** (KC-side site within 2 µm of the claw, Report 02) get a glomerulus: that of the '
      'claw\'s main PN. 18–27% of KC → APL and 30–49% of APL → KC calyx synapses are on a claw (lowest in BANC; Report 02, Figure 3).',
      '**Names.** Hemibrain uses older names for three glomeruli; they are renamed to FlyWire\'s (VC3l → VC3, VC3m → VC5, VC5 → VM6), which '
      'the response tables use.'])
R.p('Each synapse $e$ then carries, for each odor, the response of its glomerulus:')
R.eq('w_o(e) \\;=\\; \\bar r_{g(e),\\,o}',
     'Blank (drawn gray) when $g(e)$ is not in the response table, or $e$ has no glomerulus.')


def cover(tab, gloms):
    return tab.glomerulus.isin(gloms).mean()


rows = []
for ds in ['flywire', 'hemibrain', 'malecns', 'banc']:
    p = PN[PN.dataset == ds]
    rows.append([DSN[ds], f'{len(p):,}', L.pct(p.glomerulus.notna().mean()), L.pct(cover(p, set(orn.index))), L.pct(cover(p, set(GHM.index)))])
R.table(['Dataset', 'PN ↔ APL synapses in the calyx', 'with a glomerulus', 'glomerulus in the ORN table (36)', 'glomerulus in GH146 (29)'], rows,
        title='How many PN ↔ APL calyx synapses can be colored.', caption='Both directions and both APLs together, official ROIs. The '
        'rest belong to glomeruli that were not imaged (mostly VP thermo- and hygrosensory glomeruli and a few olfactory ones).',
        num_cols=[1, 2, 3, 4])

# ---------------------------------------------------------------- 3
R.section('Drawing the maps')
R.p('Each panel shows one calyx in posterior view (Report 02, Table 2): every calyx synapse of the chosen kind is a dot at its position, '
    'colored by $w_o(e)$; synapses without a response are gray. The color scale follows Yang\'s GH146 heatmap PDF: the red–blue map '
    '`RdBu_r` with a two-slope normalization, so 0 is white, $v_{\\min}$ is the darkest blue and $v_{\\max}$ the darkest red:')
R.eq('c(v) \\;=\\; \\mathrm{RdBu\\_r}\\big(t(v)\\big),\\qquad t(v) = \\begin{cases} \\tfrac12 + \\tfrac12\\, v / v_{\\max} & v \\ge 0 \\\\ '
     '\\tfrac12 - \\tfrac12\\, v / v_{\\min} & v < 0 \\end{cases}',
     '$t \\in [0, 1]$ is the position along the color map. One scale is used for all odors of a map, set by the whole mean matrix.')
R.p('The first maps (PA, PAA, MS with the ORN table) drew negative responses as 0; all later maps use the signed two-slope scale above. '
    'Within a panel, dots are drawn in order of increasing $|w_o|$, so strong responses sit on top.')


def fig_map():
    odor = 'pa @ -3'
    om = ofm[odor]
    cal = [('hemibrain', 'R'), ('malecns', 'L'), ('malecns', 'R')]
    kinds = [('PN', 'PN_to_APL', 'PN → APL'), ('PN', 'APL_to_PN', 'APL → PN'), ('KC', 'KC_to_APL', 'KC → APL on claws'), ('KC', 'APL_to_KC', 'APL → KC on claws')]
    fig, axs = plt.subplots(len(kinds), 3, figsize=(9.6, 12.6))
    fig.subplots_adjust(hspace=0.42, wspace=0.22)
    SH = {'hemibrain': 'Hemibrain', 'malecns': 'Male CNS'}
    for j, (ds, side) in enumerate(cal):
        allp = PN[(PN.dataset == ds) & (PN.apl_side == side)]
        cu = allp.view_right_um.quantile([.002, .998]).mean(); cv = allp.view_dorsal_um.quantile([.002, .998]).mean()
        for i, (src, d, lab) in enumerate(kinds):
            ax = axs[i, j]
            t = (PN if src == 'PN' else KC); t = t[(t.dataset == ds) & (t.apl_side == side) & (t.direction == d)]
            g = t.glomerulus if src == 'PN' else t.glomeruli.where(~t.glomeruli.fillna('').str.contains(r'\+'))
            w = g.map(om).values
            u, v = t.view_right_um.values - cu, t.view_dorsal_um.values - cv
            ax.set_facecolor(PANEL_BG)
            nan = np.isnan(w)
            ax.scatter(u[nan], v[nan], s=1.2, color=NODATA, lw=0, rasterized=True)
            o = np.argsort(np.abs(w[~nan]))
            sc = ax.scatter(u[~nan][o], v[~nan][o], c=w[~nan][o], cmap=RDBU, norm=NORM_ORN, s=2.2 if src == 'PN' else 1.6, lw=0, rasterized=True)
            ax.set_xlim(-27, 27); ax.set_ylim(-24, 24); ax.set_aspect('equal')
            ax.set_title(f'{SH[ds]} {side}: {lab}\n{(~nan).sum():,} colored, {nan.sum():,} gray', fontsize=8.3)
            if i == len(kinds) - 1: ax.set_xlabel('fly\'s right → (µm from calyx center)')
            if j == 0: ax.set_ylabel('dorsal ↑ (µm from calyx center)')
    cb = fig.colorbar(sc, ax=axs, fraction=0.025, pad=0.02, shrink=0.5)
    cb.set_label('ORN response of the synapse\'s glomerulus to pa @ −3, $w_o(e)$\n(fly mean; scale ends = min and max of all 39 odors)', fontsize=8.5)
    cb.ax.set_yscale('linear'); cb.outline.set_visible(False)
    return fig


R.fig(fig_map(), 'Odor = pentyl acetate at 10⁻³ (pa @ −3), ORN fly means. Columns: the three calyces used by the model; rows: the four kinds '
      'of APL synapse (PN → APL and APL → PN: all calyx synapses of uniglomerular PNs; KC → APL and APL → KC: only synapses on a claw). '
      'Each dot is one synapse in posterior view, in µm from the calyx center (center = midpoint of the 0.2–99.8% range of PN ↔ APL '
      'positions). Color = response of the synapse\'s glomerulus (Equation 2) on the two-slope scale of Equation 3, with ends '
      f'$v_{{\\min}}$ = {F(NORM_ORN.vmin)} and $v_{{\\max}}$ = {F(NORM_ORN.vmax)} (the whole 40-glomerulus × 39-odor ORN matrix); gray = glomerulus '
      'not imaged, or no glomerulus. Panel background is light gray so that white (0) stays visible. The delivered maps '
      '(`odor_weighted/ORN_per_odor_map_*.png` and the GH146 ones) show every odor and all APLs of all four datasets in this layout.',
      'One odor drawn on APL\'s calyx synapses.')
R.p('The strongest boutons and claws of an odor are scattered over the whole calyx rather than gathered in one place: for PA, PAA and MS '
    'the response-weighted center of mass lies within 3 µm of the calyx center in FlyWire, hemibrain and male CNS. Some glomeruli do sit in '
    'typical places (DM1, the strongest PAA glomerulus, on the ventral rim; DL1, the strongest MS glomerulus, near the center).')
R.box('Per-glomerulus maps', 'Two more map sets show, for each of 58 glomeruli in turn, that glomerulus\'s synapses in color and all others in '
      'gray: `glomerulus_map_{PN_to_APL,APL_to_PN}_posterior_view.png` (PN boutons) and `claw_glomerulus_map_{KC_to_APL,APL_to_KC}_posterior_view.png` '
      '(claws). They use no odor data.', 'plain')

# ---------------------------------------------------------------- 4 similarity
R.section('Odor-pair similarity of on-claw APL synapses')
R.p('If APL\'s synapses on claws favored some glomeruli, two odors could look more or less alike "through APL\'s eyes" than through the '
    'glomeruli alone. To test this, each odor becomes a vector over all on-claw KC ↔ APL synapses of a dataset (both APLs), with entry '
    '$w_o(e)$; synapses whose glomerulus has no response are left out. The similarity of two odors is the Pearson correlation of their '
    'vectors. Because all synapses of a glomerulus carry the same value, this equals a correlation over glomeruli in which glomerulus $g$ '
    'counts $n_g$ times, $n_g$ being its number of on-claw synapses:')
R.eq('\\rho_{ab} \\;=\\; \\frac{\\sum_g n_g\\,(\\bar r_{g,a} - \\mu_a)(\\bar r_{g,b} - \\mu_b)}'
     '{\\sqrt{\\sum_g n_g\\,(\\bar r_{g,a} - \\mu_a)^2}\\;\\sqrt{\\sum_g n_g\\,(\\bar r_{g,b} - \\mu_b)^2}},\\qquad '
     '\\mu_a = \\frac{\\sum_g n_g \\bar r_{g,a}}{\\sum_g n_g}',
     'The "responses only" reference sets every $n_g = 1$.')


def fig_sim():
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 2.9))
    for ax, (resp, ds, d) in zip(axs, [('ORN', 'hemibrain', 'KC_to_APL'), ('ORN', 'hemibrain', 'APL_to_KC'),
                                       ('ORN', 'malecns', 'APL_to_KC'), ('GH146', 'malecns', 'APL_to_KC')]):
        ref = SIM[(SIM.responses == resp) & (SIM.dataset == 'responses only')].set_index(['odor_a', 'odor_b']).pearson_r
        s = SIM[(SIM.responses == resp) & (SIM.dataset == ds) & (SIM.direction == d)].set_index(['odor_a', 'odor_b']).pearson_r
        j = pd.concat([ref.rename('x'), s.rename('y')], axis=1).dropna()
        j = j[[a < b for a, b in j.index]]
        ax.scatter(j.x, j.y, s=12, color=L.DS_COLOR[ds], lw=0, alpha=0.85)
        ax.plot([-0.4, 1], [-0.4, 1], color=L.GRAY, lw=0.8, ls='--')
        r = np.corrcoef(j.x, j.y)[0, 1]
        ax.set_title(f'{resp}, {DSN[ds]}\n{d.replace("_to_", " → ")} on claws; r = {r:.3f}', fontsize=8.3)
        ax.set_xlabel('$\\rho_{ab}$, glomeruli counted once'); ax.set_xlim(-0.4, 1); ax.set_ylim(-0.4, 1); ax.set_aspect('equal')
        if ax is axs[0]: ax.set_ylabel('$\\rho_{ab}$, weighted by on-claw\nAPL synapses ($n_g$)')
    fig.tight_layout(); return fig


R.fig(fig_sim(), 'Each dot is one pair of odors (ORN: 120 pairs of the 16 odors; GH146: 78 pairs of 13). x: Pearson correlation of the two '
      'odors\' glomerular responses with every glomerulus counted once; y: the same with glomerulus $g$ counted $n_g$ times, $n_g$ = its '
      'number of on-claw KC → APL or APL → KC synapses (Equation 4). Dashed line: equal. r in each title = correlation between x and y over '
      'pairs. Four of the 16 dataset × direction × response combinations are shown; all are in `odor_weighted/odor_similarity_KC_APL_on_claws.csv`.',
      'Weighting by APL\'s on-claw synapses barely changes odor similarity.')
R.p('In every dataset, direction and response table the weighted similarities track the unweighted ones closely (r 0.945–0.986 over the 16 '
    'combinations), and the four connectomes agree with each other (r ≥ 0.947 between any two). So the glomerular distribution of APL\'s on-claw synapses does not by itself reshape '
    'odor similarity. Whether APL\'s **activity** reshapes the KC representation is a different question, taken up with a model from '
    'Report 05 on.')
R.box('Pearson here, cosine later', 'This section uses the Pearson correlation (built 2026-10-14). From 2026-10-21 on, Yang asked for cosine '
      'similarity, as in their RDMs; the model and the TNT comparison (Reports 05–11) use cosine. `odor_similarity.py` was not switched.', 'caveat')

R.save()
