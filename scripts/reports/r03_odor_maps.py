"""Report 03 (rebuilt 2026-11-04 for Yang's revision of 2026-11-03): odor-weighted maps of APL's calyx synapses with the new ORN
tables, the corrected calyx and the new claws; 2D and 3D maps, the APL -> KC + PN map, weighted vs unweighted odor similarity
for PN and KC synapses, and odor-by-odor distance heat maps. The old version is in scripts/reports/archive_2026-11-03/."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/odor_maps_2026_11')
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
import report_lib as L
import common as c
L.style(); L.DATE = '2026-11-04'
F = L.fmt
NM = c.OUT                                                     # with_outside_ROI_synapses/odor_weighted/new_ORN_2026-11-03/
NMr = 'with_outside_ROI_synapses/odor_weighted/new_ORN_2026-11-03/'
NEW = c.NEW
RS = c.response_sets(); NORMS = c.norms(RS)
PN = c.pn_synapses(); KC = c.kc_synapses()
SIMS = json.load(open(NM + 'odor_similarity_summary.json')); SIMK = json.load(open(NM + 'odor_similarity_KC_summary.json'))
AG_PN = pd.read_csv(NM + 'odor_similarity_PN_APL_weighted_vs_once_summary.csv')
AG_KC = pd.read_csv(NM + 'odor_similarity_KC_APL_weighted_vs_once_summary.csv')
KC2D = json.load(open(NM + 'kc_maps2d_summary.json'))
DSN = c.DSN
CAL = c.CALYCES; MB3 = c.MODEL_MBS
calname = lambda ds, s: f'{DSN[ds].split(" v")[0]} {s}'
SETN = {'ORN_megamat_diag': 'ORN, dataset 1 (megamat + diagnostics)', 'ORN_valid2_diag': 'ORN, dataset 4 (validation2 + diagnostics)',
        'GH146': 'GH146 PNs (diagnostics + megamat)'}

R = L.Report('03', 'How does each APL synapse in the calyx get an odor response, what do the odor-weighted maps look like in 2D and 3D, '
             'and does weighting by APL\'s synapses change how similar two odors look?',
             inputs=['Yang\'s ORN responses `responses_wide.csv` (2026-11-03), turned into fly-mean tables in `orn_inputs_2026-11-03/`: '
                     'dataset 1 (megamat + diagnostics, 9 flies) and dataset 4 (validation2 + diagnostics, 6 flies)',
                     'Yang\'s GH146 PN imaging `GH146_ij_certain-roi_stats.csv` (7 flies, 29 glomeruli, 41 odor presentations)',
                     'PN ↔ APL calyx synapses of the corrected calyx (`with_outside_ROI_synapses/`, Report 02)',
                     'KC ↔ APL synapses on the new claws (`with_outside_ROI_synapses/claws_v2m2/`, Report 02); hemibrain and male CNS only'],
             outputs=[f'`{NMr}`: 2D maps `<set>_map_<kind>_posterior_view.png`, 3D maps `<set>_3D_maps_{{PN,KC}}_APL.pdf`, '
                      'similarity figures and tables `odor_similarity_{PN,KC}_APL_weighted_vs_once.*`, heat maps '
                      '`<set>_odor_distance_heatmaps_{PN,KC}_APL.png` with `odor_distance_{PN,KC}_APL_maps.csv`; `README.md` there'],
             scripts=['`scripts/orn_inputs_2026_11_03.py` (ORN tables, consistency check)',
                      '`scripts/odor_maps_2026_11/`: `common.py`, `maps2d.py`, `maps3d.py`, `kc_maps.py` (KC and KC + PN maps), '
                      '`similarity.py` (PN), `kc_similarity.py` (KC)', '`scripts/reports/r03_odor_maps.py` (this report)'])

R.p('The maps show where in the calyx APL meets the PN boutons and KC claws that a given odor drives. The step that gives each synapse a '
    'glomerulus and so an odor response is also how odors enter the model (Report 05), so it is described here in full.')
R.box('What changed in this version (2026-11-04)', [
    '**New ORN data.** The ORN responses come from Yang\'s `responses_wide.csv`. Dataset 1 (megamat + diagnostics) and dataset 4 '
    '(validation2 + diagnostics) are kept separate, because their diagnostic odors failed the consistency check (Section 1.2).',
    '**Corrected calyx** everywhere: the synapses outside the official calyx ROI are included (Report 02).',
    '**New claws** (8 µm along the KC skeleton, split by PN, ≥ 2 PSDs per PN; Report 02). They exist for hemibrain and male CNS, so the '
    'KC maps cover the three calyces of the model.',
    '**New maps.** There is an odor-weighted APL → KC + PN map, and every map now also comes in 3D (two angles per odor and calyx; BANC left out).',
    '**New analyses.** Odor similarity weighted by synapses is now cosine (Section 5), and odor-by-odor distance heat maps have been '
    'added (Section 6).'], 'plain')

R.symbols([
    ('g', 'An antennal-lobe glomerulus (FlyWire names; hemibrain VC3l, VC3m, VC5 renamed VC3, VC5, VM6)', '40 in the ORN tables, 29 in GH146'),
    ('o', 'An odor presentation: odor name and log10 dilution, e.g. "pa @ −3"', ''),
    ('f', 'A fly', ''),
    ('r^{(f)}_{g,o}', 'Response of glomerulus $g$ to odor $o$ in fly $f$, averaged over that fly\'s trials. ORN: as given in `responses_wide.csv`; '
     'GH146: peak ΔF/F', 'ORN: units as in Yang\'s file (not stated there); GH146: ΔF/F'),
    ('\\bar r_{g,o}', 'Fly-mean response (Equation 1)', 'same units'),
    ('e', 'One APL synapse in the calyx', ''),
    ('g(e)', 'Glomerulus of synapse $e$: of its PN (PN ↔ APL), or of its claw\'s PN (KC ↔ APL on a claw)', ''),
    ('w_o(e)', 'Weight of synapse $e$ for odor $o$: $\\bar r_{g(e),o}$', 'blank if $g(e)$ has no response'),
    ('n_g', 'Number of synapses (of one kind, in one calyx) whose glomerulus is $g$', 'count'),
    ('S_{ab}', 'Cosine similarity of odors $a$ and $b$ (Equations 4, 7)', '−1 to 1'),
    ('\\sigma', 'Width of the 3D Gaussian that smooths a synapse map (Section 6)', '0, 2 (main) or 5 µm'),
    ('K_{ab}', 'Overlap of the smoothed maps of odors $a$ and $b$ (Equation 5)', 'response² units'),
    ('d_{ab}', 'Euclidean distance between the smoothed maps of odors $a$ and $b$ (Equation 6)', 'response units'),
])

# ---------------------------------------------------------------- 1 responses
R.section('Odor responses per glomerulus')
R.sub('Three response tables')
R.p('Each response table is averaged in two steps. First each fly\'s trials are averaged, then the flies:')
R.eq('\\bar r_{g,o} \\;=\\; \\frac{1}{|\\mathcal{F}_{g,o}|} \\sum_{f \\,\\in\\, \\mathcal{F}_{g,o}} r^{(f)}_{g,o}',
     '$\\mathcal{F}_{g,o}$ = the flies in which glomerulus $g$ was imaged with odor $o$ (missing values are skipped).')
rows = []
for rs in ['ORN_megamat_diag', 'ORN_valid2_diag', 'GH146']:
    M, panel, _ = RS[rs]
    nfly = {'ORN_megamat_diag': 9, 'ORN_valid2_diag': 6, 'GH146': 7}[rs]
    main = panel[0]
    od = (f'{sum(p == main for p in panel)} {main} + {sum(p == "diagnostics" for p in panel)} diagnostics' if rs != 'GH146'
          else f'{sum(p == "diagnostics" for p in panel)} diagnostics + {sum(p == "megamat" for p in panel)} megamat')
    rows.append([SETN[rs], nfly, M.shape[0], od, f'{F(np.nanmin(M.values))} to {F(np.nanmax(M.values))}', L.pct(np.isnan(M.values).mean(), 1)])
R.table(['Response set', 'flies', 'glomeruli', 'odor presentations', 'range of $\\bar r_{g,o}$', 'missing'], rows,
        title='The three response sets used for the maps.',
        caption='In dataset 1, 2h, ms and va @ −3 were given in both megamat and diagnostics, and the megamat presentation is used. GH146 '
                'has ms @ −3 in both panels; the two are kept and marked "(diag)" and "(megamat)". The model uses a 16-odor ORN table built '
                'from the same two datasets (`ORN_TNT16_flymean.csv`: 12 megamat odors from dataset 1, and paa @ −4, −aPine, euc and B-myr '
                'from dataset 4).', num_cols=[1, 2])


def fig_tables():
    fig = plt.figure(figsize=(12, 15.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.78], hspace=0.62)
    for i, rs in enumerate(['ORN_megamat_diag', 'ORN_valid2_diag', 'GH146']):
        M, panel, title = RS[rs]
        ax = fig.add_subplot(gs[i])
        nrm = NORMS['GH146' if rs == 'GH146' else 'ORN']
        cm = c.CMAP.copy(); cm.set_bad('#d6d6d4')
        im = ax.imshow(np.ma.masked_invalid(M.values), cmap=cm, norm=nrm, aspect='auto', interpolation='nearest')
        nmain = sum(p != 'diagnostics' for p in panel) if rs != 'GH146' else sum(p == 'diagnostics' for p in panel)
        ax.axvline(nmain - 0.5, color=c.INK, lw=0.8)
        ax.set_xticks(range(M.shape[1]), [o.replace(' -', ' −') for o in M.columns], rotation=90, fontsize=6.4)
        ax.set_yticks(range(M.shape[0]), M.index, fontsize=5.6)
        ax.set_title(f'{"abc"[i]}  {SETN[rs]}: {M.shape[0]} glomeruli × {M.shape[1]} odors', loc='left', fontsize=10)
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
        cb = L.cbar(fig, im, ax, 'GH146 peak ΔF/F' if rs == 'GH146' else 'ORN response\n(units as in Yang\'s file)', fraction=0.025)
        cb.ax.set_yscale('linear')
    return fig


R.fig(fig_tables(), 'Rows are glomeruli and columns are odor presentations. The vertical line separates the first panel (megamat or '
      'validation2; diagnostics in GH146) from the second. Color is the fly mean (Equation 1) on a red–white–blue scale whose white is exactly 0. '
      'The blue end is the smallest value and the red end the largest: of both ORN tables for panels a and b (one shared scale), and of the '
      'GH146 table for panel c. Grey cells were not measured (DL4 and VM4 were imaged only in dataset 4).', 'The response tables.')

R.sub('Are the two ORN diagnostic sets the same? No, so they are kept apart')
R.p('Yang asked to merge the diagnostic odors of the two ORN datasets only if they are consistent. The check compares 3 flies of '
    'dataset 1 with 3 flies of dataset 4, and 3 flies with 3 other flies of the same dataset. It does this for the response pattern and '
    'for the odor-by-odor similarity matrix. The two count as the same only if (1) the between-dataset correlation is not below the 5th '
    'percentile of the within-dataset ones, and (2) a fly-label permutation test finds no difference.')
R.img(NEW + 'diagnostics_consistency.png', 'Panel a: correlations within dataset 1, within dataset 4 and between the datasets, for the response pattern '
      'and for the odor-similarity matrix (3 flies against 3 flies, many random splits). Panels b and c: the glomeruli and odors where the two '
      'datasets differ most. Details in `orn_inputs_2026-11-03/README.md`.', 'The two diagnostic sets fail the consistency check.')
R.ul(['Response pattern: median r 0.76 between datasets, against 0.86–0.87 within. Odor similarity: 0.53 between, against 0.82 within. '
      'The fly-label permutation gives p = 0.0002.',
      'The largest differences: HCl @ −1 drives DC4 strongly in dataset 1 (1.24) but hardly in dataset 4 (0.17); DM5 responds to a-terp and '
      '3mtp only in dataset 4; DC4, DP1m and DP1l are tuned differently. Leaving HCl out does not rescue the check (odor similarity r 0.60).',
      'So every map and heat map below is made separately for **dataset 1** (megamat + diagnostics) and **dataset 4** (validation2 + diagnostics).'])

# ---------------------------------------------------------------- 2 glomerulus of a synapse
R.section('Giving each synapse a glomerulus')
R.ul(['**PN ↔ APL synapses.** The PN\'s cell type names its glomerulus (e.g. `DM1_lPN` → DM1). Only uniglomerular types count. Types that '
      'name two glomeruli (VP1d+VP4, VP1m+VP5, VP3+VP1l) count for both. "X+" types (VP2+, VP3+) count for X. Multiglomerular (M_) and '
      'untyped PNs get no glomerulus.',
      '**KC ↔ APL synapses.** Only synapses **on a claw** get a glomerulus: that of the claw\'s PN. With the new claws (Report 02) a KC ↔ APL '
      'synapse is on a claw when it lies within 8 µm, along the KC, of a PN synapse of that claw. 56–58% of KC → APL and 66–69% of APL → KC '
      'calyx synapses are on a claw. A claw whose PN type names two glomeruli has no single glomerulus and stays grey.',
      '**APL → KC + PN** (Yang\'s new map) puts APL\'s two outputs in the calyx together: APL → KC synapses on claws and APL → PN synapses on boutons.'])
R.p('Each synapse $e$ then carries, for each odor, the response of its glomerulus:')
R.eq('w_o(e) \\;=\\; \\bar r_{g(e),\\,o}', 'Blank (drawn grey) when $g(e)$ is not in the response table, or $e$ has no single glomerulus.')
rows = []
for ds, s in CAL:
    p = PN[(PN.dataset == ds) & (PN.apl_side == s)]
    rows.append([calname(ds, s), 'PN ↔ APL', f'{len(p):,}'] + [L.pct(p.glomerulus.isin(RS[rs][0].index).mean()) for rs in ['ORN_megamat_diag', 'ORN_valid2_diag', 'GH146']])
for ds, s in MB3:
    for kk, kl in [('KC_to_APL', 'KC → APL on claws'), ('APL_to_KC', 'APL → KC on claws'), ('APL_to_KC_PN', 'APL → KC + PN')]:
        rows.append([calname(ds, s), kl, f'{KC2D["GH146 " + kk][f"{ds} {s}"]["synapses"]:,}'] +
                    [L.pct(KC2D[f'{rs} {kk}'][f'{ds} {s}']['share_colored']) for rs in ['ORN_megamat_diag', 'ORN_valid2_diag', 'GH146']])
R.table(['Calyx', 'Synapses', 'number', 'colored, ORN dataset 1', 'colored, ORN dataset 4', 'colored, GH146'], rows,
        title='How many synapses can be colored.',
        caption='Corrected calyx; both directions together for PN ↔ APL. "Colored" = the synapse\'s glomerulus has a response in that table. '
                'The rest belong to glomeruli that were not imaged (mostly thermo- and hygrosensory VP glomeruli), to multiglomerular PNs, or to '
                'claws whose PN names two glomeruli. GH146 covers fewer glomeruli (29) than the ORN tables (40).', num_cols=[2, 3, 4, 5])

# ---------------------------------------------------------------- 3 2D maps
R.section('2D maps')
R.p('Each panel shows one calyx in posterior view (Report 02, Table 4). Every calyx synapse of the chosen kind is a dot at its position, '
    'colored by $w_o(e)$; synapses without a response are grey. The color map is `RdBu_r` with a two-slope normalization, so 0 is white, the '
    'smallest value $v_{\\min}$ is the darkest blue and the largest value $v_{\\max}$ the darkest red:')
R.eq('c(v) \\;=\\; \\mathrm{RdBu\\_r}\\big(t(v)\\big),\\qquad t(v) = \\begin{cases} \\tfrac12 + \\tfrac12\\, v / v_{\\max} & v \\ge 0 \\\\ '
     '\\tfrac12 - \\tfrac12\\, v / v_{\\min} & v < 0 \\end{cases}',
     f'One scale for both ORN tables ($v_{{\\min}}$ = {F(NORMS["ORN"].vmin)}, $v_{{\\max}}$ = {F(NORMS["ORN"].vmax)}) and one for GH146 '
     f'({F(NORMS["GH146"].vmin)}, {F(NORMS["GH146"].vmax)}). Within a panel, the strongest colors are drawn on top.')


def fig_map(odor='pa @ -3', rs='ORN_megamat_diag'):
    om = RS[rs][0][odor]; nrm = NORMS['ORN']
    kinds = [('PN', 'PN_to_APL', 'PN → APL'), ('PN', 'APL_to_PN', 'APL → PN'), ('KC', 'KC_to_APL', 'KC → APL on claws'),
             ('KC', 'APL_to_KC', 'APL → KC on claws'), ('both', None, 'APL → KC + PN')]
    fig, axs = plt.subplots(len(kinds), 3, figsize=(9.6, 15.4))
    fig.subplots_adjust(hspace=0.36, wspace=0.12)
    nkc = {}
    for j, (ds, side) in enumerate(MB3):
        allp = PN[(PN.dataset == ds) & (PN.apl_side == side)]
        cu = allp.view_right_um.quantile([.002, .998]).mean(); cv = allp.view_dorsal_um.quantile([.002, .998]).mean()
        for i, (src, d, lab) in enumerate(kinds):
            ax = axs[i, j]
            if src == 'both':
                a = KC[(KC.dataset == ds) & (KC.apl_side == side) & (KC.direction == 'APL_to_KC')]
                b = PN[(PN.dataset == ds) & (PN.apl_side == side) & (PN.direction == 'APL_to_PN')]
                t = pd.concat([a, b]); nkc[(ds, side)] = len(a) / len(t)
            else:
                t = (PN if src == 'PN' else KC); t = t[(t.dataset == ds) & (t.apl_side == side) & (t.direction == d)]
            w = t.glomerulus.map(om).values.astype(float)
            u, v = t.view_right_um.values - cu, t.view_dorsal_um.values - cv
            ax.set_facecolor(c.PANEL_BG); nan = np.isnan(w)
            ax.scatter(u[nan], v[nan], s=1.0, color=c.NODATA, lw=0, rasterized=True)
            o = np.argsort(np.abs(w[~nan]))
            sc = ax.scatter(u[~nan][o], v[~nan][o], c=w[~nan][o], cmap=c.CMAP, norm=nrm, s=2.2 if src == 'PN' else 1.4, lw=0, rasterized=True)
            ax.set_xlim(-28, 28); ax.set_ylim(-24, 24); ax.set_aspect('equal'); ax.set_xticks([-20, 0, 20]); ax.set_yticks([-20, 0, 20])
            ax.set_title(f'{calname(ds, side)}: {lab}\n{(~nan).sum():,} colored, {nan.sum():,} grey', fontsize=8.3)
            if i == len(kinds) - 1: ax.set_xlabel('fly\'s right → (µm)')
            if j == 0: ax.set_ylabel('dorsal ↑ (µm)')
    cb = fig.colorbar(sc, ax=axs, fraction=0.025, pad=0.02, shrink=0.45)
    cb.set_label(f'ORN response of the synapse\'s glomerulus to {odor.replace(" -", " −")}, $w_o(e)$\n(dataset 1; scale ends = min and max of both ORN tables)', fontsize=8.5)
    cb.ax.set_yscale('linear'); cb.outline.set_visible(False)
    return fig, nkc


fg, NKC = fig_map()
R.fig(fg, 'Odor = pentyl acetate at 10⁻³ (pa @ −3), ORN dataset 1. Columns are the three calyces of the model. Rows are the kinds of '
      'APL synapse: PN → APL and APL → PN (all calyx synapses of uniglomerular PNs); KC → APL and APL → KC (only synapses on a new claw); '
      'and APL → KC + PN (rows 2 and 4 together). Each dot is one synapse in posterior view, in µm from the calyx center (the midpoint of the '
      '0.2–99.8% range of PN ↔ APL positions). Color = response of the synapse\'s glomerulus (Equation 2) on the scale of Equation 3; grey = '
      'no response or no single glomerulus. The panel background is light grey so that white (0) stays visible.',
      'One odor drawn on APL\'s calyx synapses.')
files = [['PN → APL, APL → PN', 'all five calyces of FlyWire, hemibrain and male CNS', '`<set>_map_{PN_to_APL,APL_to_PN}_posterior_view.png`'],
         ['KC → APL, APL → KC (on claws)', 'hemibrain R, male CNS L and R', '`<set>_map_{KC_to_APL,APL_to_KC}_posterior_view.png`'],
         ['APL → KC + PN', 'hemibrain R, male CNS L and R', '`<set>_map_APL_to_KC_PN_posterior_view.png`']]
R.table(['Kind', 'Calyces (columns)', 'File'], files, title='The full 2D map sets.',
        caption=f'In `{NMr}`. `<set>` is `ORN_megamat_diag` (dataset 1, {RS["ORN_megamat_diag"][0].shape[1]} odors), `ORN_valid2_diag` '
                f'(dataset 4, {RS["ORN_valid2_diag"][0].shape[1]} odors) or `GH146` ({RS["GH146"][0].shape[1]} odors). Each file has one row per '
                'odor in Yang\'s order (first panel, then diagnostics marked "·d").')
R.p('As before, the strongest boutons and claws of an odor are spread over the whole calyx rather than gathered in one place. The KC maps '
    'look like the PN maps. Each claw takes the response of its PN, and claws sit around their PN\'s boutons, so both kinds of synapse '
    'carry the same glomerular pattern at the same places. The APL → KC + PN map is dominated by APL → KC synapses '
    f'({L.pct(min(NKC.values()))}–{L.pct(max(NKC.values()))} of the dots).')

# ---------------------------------------------------------------- 4 3D
R.section('3D maps')
R.p('The 3D maps place every synapse at its position in anatomical axes: the fly\'s right, anterior and dorsal, in µm from the calyx '
    'center. "Anterior" is signed so that the peduncle lies anterior of the calyx in every dataset. Each page is one odor. Every calyx is shown from '
    'two angles: **A**, posterior-lateral (from behind, turned 35° toward the calyx\'s lateral side, 20° up), and **B**, lateral (from the '
    'calyx\'s lateral side, 20° up). The projection is orthographic, so sizes do not shrink with depth. Grey dots are drawn first and colored '
    'ones on top. BANC is left out, as Yang asked.')
R.img(NM + 'ORN_megamat_diag_3D_example_KC_pa.png', 'pa @ −3, ORN dataset 1. Rows: hemibrain R, male CNS L, male CNS R. Columns: KC → APL '
      'on claws, APL → KC on claws, and APL → KC + PN, each from angles A and B. The small arrows are 5 µm long and point lateral, anterior '
      'and dorsal. One page of `ORN_megamat_diag_3D_maps_KC_APL.pdf`.', 'A 3D map of KC ↔ APL synapses on claws, and of APL → KC + PN.')
R.img(NM + 'ORN_megamat_diag_3D_example_pa.png', 'The same odor for PN → APL and APL → PN synapses in the five calyces of FlyWire, hemibrain and '
      'male CNS (rows), each from angles A and B. One page of `ORN_megamat_diag_3D_maps_PN_APL.pdf`.', 'A 3D map of PN ↔ APL synapses.')
R.p(f'The PDFs (`<set>_3D_maps_PN_APL.pdf` and `<set>_3D_maps_KC_APL.pdf` in `{NMr}`) have one page per odor for each of the three '
    'response sets.')

# ---------------------------------------------------------------- 5 weighted vs once
R.section('Does weighting by APL\'s synapses change odor similarity?')
R.p('If APL\'s synapses favored some glomeruli, two odors could look more or less alike "through APL\'s synapses" than through the '
    'glomeruli alone. Each odor becomes a vector over the synapses of one kind in one calyx, with entry $w_o(e)$; synapses without a response '
    'are left out. The similarity of two odors is the cosine of their vectors (as in Yang\'s RDMs). All synapses of a glomerulus carry the '
    'same value, so this equals a cosine over glomeruli in which glomerulus $g$ counts $n_g$ times:')
R.eq('S^{\\mathrm{weighted}}_{ab} \\;=\\; \\frac{\\sum_g n_g\\,\\bar r_{g,a}\\,\\bar r_{g,b}}{\\sqrt{\\sum_g n_g\\,\\bar r_{g,a}^2}\\;\\sqrt{\\sum_g n_g\\,\\bar r_{g,b}^2}}',
     'Setting every $n_g = 1$ gives $S^{\\mathrm{once}}_{ab}$, glomeruli counted once. ORN: the 16 TNT odors of `ORN_TNT16_flymean.csv` '
     '(120 pairs); GH146: the 13 of them that GH146 has (78 pairs). Only glomeruli with a value for every odor are used (DL4 and VM4 '
     'lack 12 of the 16 ORN odors and are left out).')
R.p('Yang asked for six panels, "3 ORN + 3 GH146". These are the three calyces of the model (hemibrain R, male CNS L, male CNS R), '
    'for the ORN and the GH146 responses. The analysis is done once for PN ↔ APL synapses and once for KC ↔ APL synapses on claws.')
f_pn = R.img(NM + 'odor_similarity_PN_APL_weighted_vs_once.png', 'Each dot is one pair of odors. x: cosine similarity with every glomerulus '
             'counted once. y: the same, each glomerulus weighted by its PN → APL synapses (filled) or APL → PN synapses (open) in that calyx. '
             'Dashed line: equal. r in the titles = correlation of x and y over pairs. The two FlyWire and two BANC calyces are in '
             '`odor_similarity_PN_APL_weighted_vs_once.csv`.', 'Odor similarity weighted by PN ↔ APL synapses, against glomeruli counted once.', width=88)
R.img(NM + 'odor_similarity_KC_APL_weighted_vs_once.png', f'As Figure {f_pn}, with the glomerulus weights $n_g$ = KC → APL synapses (filled) or '
      'APL → KC synapses (open) on that glomerulus\'s claws (new claws). The APL → KC + PN weighting is in '
      '`odor_similarity_KC_APL_weighted_vs_once.csv`.', 'Odor similarity weighted by KC ↔ APL synapses on claws.', width=88)


def ag_row(A, resp, kind, kc=True):
    t = A[(A.responses == resp) & (A['kind' if kc else 'direction'] == kind)]
    t = t[[(d, s) in MB3 for d, s in zip(t.dataset, t.apl_side)]]
    return [f'{F(t.r.min(), 3)}–{F(t.r.max(), 3)}', f'{F(t.mean_change.min(), 3, True)} to {F(t.mean_change.max(), 3, True)}', F(t.max_abs_change.max(), 2)]


rows = []
for resp in ['ORN', 'GH146']:
    for kind, lab, kc in [('PN_to_APL', 'PN → APL', False), ('APL_to_PN', 'APL → PN', False), ('KC_to_APL', 'KC → APL on claws', True),
                          ('APL_to_KC', 'APL → KC on claws', True), ('APL_to_KC_PN', 'APL → KC + PN', True)]:
        rows.append([resp, lab] + ag_row(AG_KC if kc else AG_PN, resp, kind, kc))
R.table(['Responses', 'Weights $n_g$ from', 'r, weighted vs once', 'mean change of $S_{ab}$', 'largest change of one pair'], rows,
        title='How much weighting changes odor similarity (three model calyces).',
        caption='Ranges over hemibrain R, male CNS L and male CNS R. "Mean change" = mean over pairs of $S^{\\mathrm{weighted}} - S^{\\mathrm{once}}$.',
        num_cols=[2, 3, 4])
pk = AG_KC[[(d, s) in MB3 for d, s in zip(AG_KC.dataset, AG_KC.apl_side)]]
pp = AG_PN[[(d, s) in MB3 for d, s in zip(AG_PN.dataset, AG_PN.apl_side)]]
R.ul([f'**Weighting barely changes odor similarity**, for PN and KC synapses alike: weighted and unweighted similarities correlate at '
      f'r ≥ {F(min(pk.r.min(), pp.r.min()), 2)} over odor pairs in every case (≥ {F(pk.r.min(), 2)} for KC synapses on claws).',
      '**The direction of the small shifts depends on the responses.** With ORN responses, weighting by APL\'s outputs (APL → PN, APL → KC) '
      'raises similarities slightly. With GH146 every weighting lowers them slightly. Weighting by PN → APL lowers them with both.',
      'So the glomerular distribution of APL\'s synapses does not by itself reshape odor similarity. Whether APL\'s **activity** reshapes the KC '
      'representation is the question of the model, from Report 05 on.'])

# ---------------------------------------------------------------- 6 heat maps
R.section('Odor-by-odor distance between response-weighted synapse maps')
R.p('Yang asked for the "correlation (euclidean) between odors by weighing the APL ↔ KC/PN synaptic coordinates by responses". The '
    'reading used here: each odor turns APL\'s synapses into a map over the calyx, every synapse $e$ at position $\\mathbf{x}_e$ carrying weight '
    '$w_o(e)$, smoothed by a 3D Gaussian of width $\\sigma$. Two odors\' maps are then compared by their overlap:')
R.eq('K_{ab} \\;=\\; \\frac{1}{N} \\sum_{e} \\sum_{e\'} w_a(e)\\, G_{ee\'}\\, w_b(e\'),\\qquad G_{ee\'} = \\exp\\!\\Big(-\\frac{|\\mathbf{x}_e - \\mathbf{x}_{e\'}|^2}{4\\sigma^2}\\Big)',
     '$N$ = number of synapses with a response. $G_{ee\'}$ is the overlap of two Gaussians of width $\\sigma$, scaled to 1 at distance 0. '
     '$\\sigma$ = 0 means $G$ = identity: each synapse compared only with itself.')
R.eq('d_{ab} \\;=\\; \\sqrt{K_{aa} + K_{bb} - 2K_{ab}}', 'Euclidean distance between the two smoothed maps.')
R.eq('S_{ab} \\;=\\; \\frac{K_{ab}}{\\sqrt{K_{aa}\\,K_{bb}}}', 'Cosine similarity of the two smoothed maps.')
R.p('The reference is "responses only": the same quantities with glomeruli counted once and no positions. For KC synapses (up to about '
    '21,000 per calyx) the matrix $G$ is too large to hold at once, so $K$ is summed in blocks of rows.')
f_h = R.img(NM + 'ORN_megamat_diag_odor_distance_heatmaps_PN_APL.png', 'ORN dataset 1 (megamat + diagnostics). Top row: responses only. Then one row '
            'per calyx (FlyWire L and R, hemibrain R, male CNS L and R). Columns: PN → APL and APL → PN, each as Euclidean distance $d_{ab}$ '
            '(purple) and cosine $S_{ab}$ (blue), with $\\sigma$ = 2 µm. Lines separate the megamat odors (top left) from the diagnostics. Each '
            f'panel has its own color scale. Dataset 4 is in `{NMr}ORN_valid2_diag_odor_distance_heatmaps_PN_APL.png`.',
            'Odor-by-odor distance and similarity of PN ↔ APL synapse maps.', max_w=1600)
R.img(NM + 'ORN_megamat_diag_odor_distance_heatmaps_KC_APL.png', f'As Figure {f_h} for the three model calyces, with columns KC → APL on claws, '
      'APL → KC on claws and APL → KC + PN. Dataset 4 is in `ORN_valid2_diag_odor_distance_heatmaps_KC_APL.png`.',
      'Odor-by-odor distance and similarity of KC ↔ APL and APL → KC + PN synapse maps.')


def vals(D, rs, meas, sg, kinds, cals):
    return [x for k, x in D['vs_responses_only_r'].items() if k.startswith(f'{rs} {meas} sigma{sg:g} ')
            and any(k.endswith(f' {kk}') for kk in kinds) and any(f' {ds} {s} ' in k for ds, s in cals)]


rng_ = lambda *a: (lambda v: f'{F(min(v), 2)}–{F(max(v), 2)}')(vals(*a))
rows = []
for rs in ['ORN_megamat_diag', 'ORN_valid2_diag']:
    for D, kinds, cals, lab in [(SIMS, ['PN_to_APL', 'APL_to_PN'], CAL, 'PN ↔ APL (5 calyces)'),
                                (SIMK, ['KC_to_APL', 'APL_to_KC'], MB3, 'KC ↔ APL on claws (3 calyces)'),
                                (SIMK, ['APL_to_KC_PN'], MB3, 'APL → KC + PN (3 calyces)')]:
        rows.append([f'dataset {1 if "megamat" in rs else 4}', lab] + [rng_(D, rs, m, sg, kinds, cals) for m in ['euclid', 'cosine'] for sg in [0, 2, 5]])
R.table(['ORN', 'Synapses', 'distance, σ 0', 'σ 2 µm', 'σ 5 µm', 'cosine, σ 0', 'σ 2 µm', 'σ 5 µm'], rows,
        title='How closely each calyx\'s odor matrix follows the responses-only one.',
        caption='Correlation over odor pairs between a calyx\'s matrix and the responses-only matrix; range over calyces and directions.',
        num_cols=list(range(2, 8)))
eu = vals(SIMS, 'ORN_megamat_diag', 'euclid', 2, ['PN_to_APL', 'APL_to_PN'], CAL) + vals(SIMK, 'ORN_megamat_diag', 'euclid', 2, ['KC_to_APL', 'APL_to_KC', 'APL_to_KC_PN'], MB3)
co5 = vals(SIMS, 'ORN_megamat_diag', 'cosine', 5, ['PN_to_APL', 'APL_to_PN'], CAL) + vals(SIMK, 'ORN_megamat_diag', 'cosine', 5, ['KC_to_APL', 'APL_to_KC', 'APL_to_KC_PN'], MB3)
R.ul([f'**Euclidean distances mostly follow response strength.** An odor that drives many glomeruli strongly is far from every other odor. '
      f'At σ = 2 µm the calyx matrices match the responses-only one at r {F(min(eu), 2)}–{F(max(eu), 2)} (dataset 1).',
      f'**Cosine similarities depend more on where the synapses sit.** Without smoothing they follow the responses-only matrix. With smoothing '
      f'they drift away from it (r {F(min(co5), 2)}–{F(max(co5), 2)} at 5 µm, dataset 1), because the Gaussian mixes neighboring boutons and '
      'claws of different glomeruli.',
      'KC synapse maps follow the responses-only matrix more closely than PN maps, already without smoothing (σ = 0), where only how the '
      'synapses are spread over glomeruli matters.'])
R.box('Open interpretation', ['"Correlation (euclidean)" was read as Euclidean distance and cosine between smoothed maps (both are shown). '
      'If Yang meant something else, for example a correlation over a voxel grid, the same inputs can be reused.'], 'caveat')
R.box('Caveats', ['Synapse counts stand in for strength; nothing here measures how strong a single synapse is.',
                  'The ORN units of `responses_wide.csv` are not stated in the file. They are used as given.',
                  'FlyWire and BANC KCs have no new claws (the new rule needs KC skeletons, done for hemibrain and male CNS), so the KC maps '
                  'cover the three calyces of the model.'], 'caveat')
R.save()
