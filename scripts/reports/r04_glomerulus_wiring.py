"""Report 04: APL wiring per glomerulus, the connectome numbers the block model uses (apl_block_model.py, section 1).
Rebuilt 2026-11-04 (Yang's revision): corrected calyx, new claws (APL_VARIANT=claws_v2m2), row names on Figure 2, and p_g per
calyx for every connectome (r04_prep_v2.py). The old version is in archive_2026-11-03/."""
import sys, os
os.environ['APL_VARIANT'] = 'claws_v2m2'
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts'); sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
import report_lib as L
import apl_block_model as abm
L.style(); L.DATE = '2026-11-04'
F = L.fmt
DS = abm.DS; DSN = {'hemibrain': 'hemibrain', 'malecns': 'male CNS'}
GW = pd.read_csv(abm.OUT + 'APL_wiring_by_glomerulus.csv'); GWi = GW.set_index(['dataset', 'glomerulus'])
GL = abm.GL_OK; KW = abm.kw; PNW = abm.PNW
assert np.allclose(GWi.APL_to_PN_per_PN_to_KC.values, abm.GWi.reindex(GWi.index).APL_to_PN_per_PN_to_KC.values, atol=1e-5, equal_nan=True)

QUANT = [  # column, short name, symbol, plain description
    ('APL_to_PN_per_PN_to_KC', 'APL→PN per PN→KC', 'p_g', 'APL→PN synapses on the glomerulus\'s PNs in the calyx, per PN→KC synapse of those PNs'),
    ('PN_to_APL_per_PN_to_KC', 'PN→APL per PN→KC', 'p^{\\mathrm{in}}_g', 'PN→APL synapses from the glomerulus\'s PNs in the calyx, per PN→KC synapse of those PNs'),
    ('APL_to_KC_per_claw_PN_synapse', 'APL→KC on claws per claw PN synapse', 'q_g', 'APL→KC synapses on the glomerulus\'s claws, per PN synapse on those claws'),
    ('KC_to_APL_per_claw_PN_synapse', 'KC→APL on claws per claw PN synapse', 'q^{\\mathrm{in}}_g', 'KC→APL synapses on the glomerulus\'s claws, per PN synapse on those claws'),
    ('APL_to_KC_per_claw', 'APL→KC per claw', 'c^{\\mathrm{out}}_g', 'APL→KC synapses on the glomerulus\'s claws, per claw'),
    ('KC_to_APL_per_claw', 'KC→APL per claw', 'c^{\\mathrm{in}}_g', 'KC→APL synapses on the glomerulus\'s claws, per claw')]

R = L.Report('04', 'How much APL wiring does each antennal-lobe glomerulus get in the calyx, measured per unit of its input to the KCs, '
             'and is that pattern the same in hemibrain and male CNS?',
             inputs=['All in `with_outside_ROI_synapses/claws_v2m2/` (corrected calyx, new claws; Report 02):',
                     '`PN_calyx_weights_APL_vs_KC.csv`: per PN, its calyx synapses onto KCs and to and from APL (Report 01)',
                     '`odor_weighted/KC_APL_synapses_on_claws_by_glomerulus.csv`: KC↔APL synapses that sit on a claw, with the claw\'s glomerulus (Report 02)',
                     '`hemibrain_KC_claws_derived.csv.gz`, `malecns_KC_claws_derived.csv.gz`: the claws (Report 02)',
                     '`KC_weights_APL_PN_per_KC.csv`: per KC, its calyx APL synapses (Report 01)',
                     'For Section 5: the corrected-calyx PN ↔ KC and PN ↔ APL synapses of all four connectomes'],
             outputs=['`with_outside_ROI_synapses/claws_v2m2/odor_weighted/APL_wiring_by_glomerulus.csv` (one row per dataset × glomerulus)',
                      'the weights $p_g$, $q_g$ and the bouton share $f$ used in Report 05',
                      '`reports/data/r04v2_pg_per_calyx.csv`, `r04v2_PN_calyx_weights_per_side.csv` ($p_g$ per calyx, all connectomes)'],
             scripts=['`scripts/apl_block_model.py`, section "1. APL wiring per glomerulus" (`APL_VARIANT=claws_v2m2`)',
                      '`scripts/reports/r04_prep_v2.py` (per-calyx counts)', '`scripts/reports/r04_glomerulus_wiring.py` (this report)'])
R.p('Report 05 lets APL\'s inhibition differ between glomeruli in three of its versions (P, L and M). This report explains where those '
    'per-glomerulus numbers come from. The model uses only hemibrain and male CNS (Yang, 2026-10-21), with the corrected calyx and the new '
    'claws (Report 02). Section 5 adds the bouton weight for every calyx of all four connectomes, as Yang asked. In the male CNS '
    'the two calyces are pooled here, because the per-glomerulus numbers are computed per dataset.')
R.symbols([
    ('g', 'An antennal-lobe glomerulus. Only uniglomerular PN types count (one glomerulus in the type name).', 'hemibrain names VC3l, VC3m, VC5 are renamed VC3, VC5, VM6 to match the other datasets'),
    ('\\mathrm{PN}(g)', 'The uniglomerular PNs of glomerulus $g$', ''),
    ('\\mathrm{claws}(g)', 'Claws whose main PN belongs to $g$ (Report 02)', ''),
    ('n^{\\mathrm{PN\\to KC}}_g', 'PN→KC synapses of $\\mathrm{PN}(g)$ in the calyx (onto any KC)', 'synapse count'),
    ('n^{\\mathrm{claw}}_g', 'PN synapses on $\\mathrm{claws}(g)$ (from any PN)', 'synapse count'),
    ('|\\mathrm{claws}(g)|', 'Number of claws of $g$', 'count'),
    ('p_g', 'Bouton weight: APL→PN synapses on $\\mathrm{PN}(g)$ in the calyx ÷ $n^{\\mathrm{PN\\to KC}}_g$', 'synapses per synapse'),
    ('q_g', 'Claw weight: APL→KC synapses on $\\mathrm{claws}(g)$ ÷ $n^{\\mathrm{claw}}_g$', 'synapses per synapse'),
    ('f', 'Boutons\' share of APL\'s calyx output: APL→PN ÷ (APL→PN + APL→KC), all calyx synapses', 'per dataset'),
    ('\\rho_s', 'Spearman rank correlation, here between datasets over glomeruli', '−1 to 1'),
])

# ---------------------------------------------------------------- 1 units
R.section('The units: PN boutons, claws and microglomeruli')
R.p('In the calyx, each PN axon forms **boutons**: swellings that carry its output synapses. Several KC dendritic **claws** wrap around '
    'one bouton, and the bouton with its claws is called a **microglomerulus**. APL makes synapses in both directions with both partners: '
    'APL→PN and PN→APL on the boutons, and APL→KC and KC→APL on the claws (Figure 1). Report 02 explains how they are found. Boutons are '
    'clusters of a PN\'s calyx T-bars within 2 µm. Claws follow Yang\'s corrected rule: a KC\'s calyx synapse sites are clustered at '
    '8 µm along the KC skeleton, and each cluster is split by presynaptic PN (each PN with ≥ 2 PSDs gets its own claw). A KC↔APL synapse '
    'counts as **on a claw** if it lies within 8 µm, along the KC, of a PN synapse of that claw. Every claw belongs to one glomerulus: that '
    'of its PN.')


def schematic():
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off'); ax.set_aspect('equal')
    ax.add_patch(Ellipse((3.2, 2.5), 1.9, 1.3, fc='#e7f5ef', ec=L.GREEN, lw=1.4))
    ax.text(3.2, 2.62, 'PN bouton', ha='center', fontsize=9, color=L.INK); ax.text(3.2, 2.25, 'glomerulus $g$', ha='center', fontsize=8, color=L.MUTED)
    ax.plot([0.3, 2.25], [2.5, 2.5], color=L.GREEN, lw=2.2); ax.text(0.3, 2.7, 'PN axon', fontsize=8, color=L.GREEN)
    for ang in [60, 140, 220, 300]:
        x, y = 3.2 + 1.35 * np.cos(np.radians(ang)), 2.5 + 1.05 * np.sin(np.radians(ang))
        ax.add_patch(Circle((x, y), 0.32, fc='white', ec=L.INK, lw=1.1)); ax.text(x, y, 'claw', ha='center', va='center', fontsize=6.5)
    ax.add_patch(FancyBboxPatch((7.3, 1.4), 1.9, 2.2, boxstyle='round,pad=0.02,rounding_size=0.25', fc='#e8f1fc', ec=L.INTACT, lw=1.3))
    ax.text(8.25, 2.5, 'APL', ha='center', va='center', fontsize=12, fontweight='bold', color=L.INTACT)
    arrows = [((7.28, 3.25), (4.2, 2.95), 'APL→PN (on the bouton)', L.INTACT, 3.45),
              ((4.2, 2.75), (7.28, 2.95), 'PN→APL', L.GREEN, 2.98),
              ((7.28, 2.1), (4.45, 1.75), 'APL→KC (on a claw)', L.INTACT, 1.6),
              ((4.45, 1.45), (7.28, 1.75), 'KC→APL (from a claw)', L.INK, 1.18)]
    for (x0, y0), (x1, y1), lab, col, ty in arrows:
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.2))
        ax.text(5.85, ty, lab, ha='center', fontsize=7.8, color=col)
    ax.text(3.2, 0.35, 'microglomerulus = one bouton + the claws around it', ha='center', fontsize=8, color=L.MUTED)
    return fig


R.fig(schematic(), 'One PN bouton (green, belonging to glomerulus $g$) with KC claws around it, forming a microglomerulus. APL (blue) '
      'makes synapses in four directions there. APL→PN and PN→APL are on the bouton; APL→KC and KC→APL are on the claws. Arrows point '
      'from the presynaptic to the postsynaptic cell. The drawing is schematic and not to scale.', 'The four kinds of APL synapse at a microglomerulus.')

# ---------------------------------------------------------------- 2 quantities
R.section('Six per-glomerulus numbers')
R.p('Raw synapse counts mostly reflect how big a glomerulus is: how many PNs it has and how many claws. To compare glomeruli, each '
    'count is divided by a measure of the glomerulus\'s input to the KCs. For glomerulus $g$:')
R.eq(r'p_g = \frac{\#\,\mathrm{APL{\to}PN}\ \text{synapses on } \mathrm{PN}(g)}{n^{\mathrm{PN\to KC}}_g}, \qquad '
     r'p^{\mathrm{in}}_g = \frac{\#\,\mathrm{PN{\to}APL}\ \text{synapses from } \mathrm{PN}(g)}{n^{\mathrm{PN\to KC}}_g}',
     note='all synapses in the calyx; $n^{\\mathrm{PN\\to KC}}_g$ = PN→KC synapses of the same PNs')
R.eq(r'q_g = \frac{\#\,\mathrm{APL{\to}KC}\ \text{on } \mathrm{claws}(g)}{n^{\mathrm{claw}}_g}, \qquad '
     r'q^{\mathrm{in}}_g = \frac{\#\,\mathrm{KC{\to}APL}\ \text{on } \mathrm{claws}(g)}{n^{\mathrm{claw}}_g}',
     note='$n^{\\mathrm{claw}}_g$ = PN synapses on those claws')
R.p('Two more divide the claw counts by the number of claws instead ($c^{\\mathrm{out}}_g$ and $c^{\\mathrm{in}}_g$, APL→KC and KC→APL '
    'synapses per claw). The model uses only $p_g$ (version P, and M) and $q_g$ (version L). $p_g$ measures how much APL inhibition '
    'a glomerulus\'s boutons receive for each synapse they pass on to KCs. $q_g$ measures how much APL inhibition its claws receive for '
    'each PN synapse they get.')
tot = GW.groupby('dataset')[['claws', 'PN_synapses_on_claws', 'KC_to_APL_on_claws', 'APL_to_KC_on_claws', 'PN_to_KC', 'PN_to_APL', 'APL_to_PN', 'n_PNs']].sum()
R.table(['dataset', 'glomeruli', 'PNs', 'PN→KC synapses', 'APL→PN', 'PN→APL', 'claws', 'claw PN synapses', 'APL→KC on claws', 'KC→APL on claws'],
        [[DSN[ds], int((GW.dataset == ds).sum()), f'{int(tot.loc[ds, "n_PNs"]):,}', f'{int(tot.loc[ds, "PN_to_KC"]):,}', f'{int(tot.loc[ds, "APL_to_PN"]):,}',
          f'{int(tot.loc[ds, "PN_to_APL"]):,}', f'{int(tot.loc[ds, "claws"]):,}', f'{int(tot.loc[ds, "PN_synapses_on_claws"]):,}',
          f'{int(tot.loc[ds, "APL_to_KC_on_claws"]):,}', f'{int(tot.loc[ds, "KC_to_APL_on_claws"]):,}'] for ds in DS],
        title='Totals over uniglomerular glomeruli.', caption='Calyx synapses only; male CNS includes both calyces.', num_cols=list(range(1, 10)))
R.p(f'Small glomeruli give noisy ratios, so the comparisons below use the {len(GL)} glomeruli with at least 40 claws and at least 500 '
    'PN→KC synapses in both datasets. The model itself uses every glomerulus that has a value.')


def fig_heat():
    rows, labs = [], []
    for col, nm, sym, _ in QUANT:
        for ds in DS:
            x = GWi[col].loc[ds].reindex(GL); rows.append(np.log2(np.maximum(x / x.median(), 2 ** -1.6)).values)   # zeros shown as the lowest color
            labs.append(f'${sym}$  {nm}, {DSN[ds]}')
    M = np.array(rows)
    order = np.argsort(np.nanmean(M[:2], 0))
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    im = ax.imshow(M[:, order], aspect='auto', cmap=L.DIV, norm=TwoSlopeNorm(0, -1.6, 1.6), interpolation='nearest')
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7.4)
    ax.set_xticks(range(len(GL))); ax.set_xticklabels(np.array(GL)[order], rotation=90, fontsize=6.6)
    for y in range(2, len(labs), 2): ax.axhline(y - 0.5, color='white', lw=2)
    ax.set_xlabel(f'glomerulus ({len(GL)} with ≥ 40 claws and ≥ 500 PN→KC synapses in both datasets), sorted by $p_g$ (mean of the two datasets)')
    ax.tick_params(length=0)
    cb = L.cbar(fig, im, ax, 'log₂( value ÷ median over glomeruli )\n+1 = twice the median, −1 = half', fraction=0.03)
    cb.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    return fig


R.fig(fig_heat(), 'Each row is one of the six numbers in one dataset, and each column a glomerulus. Color is the value divided by that row\'s '
      'median over glomeruli, on a log₂ scale: orange above the median, blue below, grey-white at the median. The scale is clipped at '
      '±1.6, about 3× above or below, and a value of 0 (hemibrain DC4 has no APL→PN synapse) takes the darkest blue. Columns are sorted by the bouton weight $p_g$. The top two rows ($p_g$) show the clearest '
      'glomerulus-to-glomerulus differences, in the same order in both datasets.', 'APL wiring per glomerulus, relative to the median.')

# ---------------------------------------------------------------- 3 conserved?
R.section('Which numbers vary, and are they conserved?')
W = abm.wiring
R.table(['number', 'symbol', 'hemibrain: 10th–90th percentile (× median)', 'male CNS: 10th–90th percentile (× median)', 'Spearman $\\rho_s$, hemibrain vs male CNS'],
        [[nm, f'${sym}$', '–'.join(F(x, 2) for x in W[col]['10th-90th percentile, x median']['hemibrain']),
          '–'.join(F(x, 2) for x in W[col]['10th-90th percentile, x median']['malecns']), F(W[col]['spearman between datasets: min-max'][0], 2)]
         for col, nm, sym, _ in QUANT],
        title=f'Spread across glomeruli and agreement between datasets ({len(GL)} glomeruli).',
        caption='The spread is the 10th and 90th percentile over glomeruli divided by the median: "0.5–1.6" means the middle 80% of glomeruli '
                'lie between half and 1.6 times the median. The last column is the rank correlation of the glomerulus values between the two datasets.',
        num_cols=[4])


def fig_scatter():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for a, (col, nm, sym, desc) in zip(axs, [QUANT[0], QUANT[2]]):
        x = GWi[col].loc['hemibrain'].reindex(GL); y = GWi[col].loc['malecns'].reindex(GL)
        a.scatter(x, y, s=22, color=L.GREEN if sym == 'p_g' else L.PURPLE, edgecolor='white', lw=0.5, zorder=3)
        rel = (x / x.median() + y / y.median()) / 2
        for g in list(rel.sort_values().index[:4]) + list(rel.sort_values().index[-5:]):
            a.annotate(g, (x[g], y[g]), xytext=(3, 2), textcoords='offset points', fontsize=7, color=L.MUTED)
        lim = [0, max(x.max(), y.max()) * 1.08]; a.plot(lim, lim, color=L.GRID, lw=1, zorder=1); a.set_xlim(lim); a.set_ylim(lim)
        rho = spearmanr(x, y)[0]
        a.set_xlabel(f'hemibrain: ${sym}$ (synapses per synapse)'); a.set_ylabel(f'male CNS: ${sym}$ (synapses per synapse)')
        a.set_title(f'{"a" if sym == "p_g" else "b"}  ${sym}$: {nm}  (ρ = {rho:.2f})')
    fig.tight_layout(); return fig


R.fig(fig_scatter(), f'Each point is one of the {len(GL)} glomeruli. Its value in hemibrain is on the horizontal axis, in male CNS on the '
      'vertical axis, and the grey line marks equal values. **a**, Bouton weight $p_g$: APL→PN synapses per PN→KC synapse. **b**, Claw weight '
      '$q_g$: APL→KC synapses on the claws per claw PN synapse. The labelled glomeruli are the four lowest and five highest (mean of the two '
      'datasets, each relative to its median). ρ is the Spearman correlation over glomeruli.', 'The two weights the model uses, hemibrain against male CNS.')
p = W['APL_to_PN_per_PN_to_KC']
fold = lambda col: '–'.join(F(W[col]['10th-90th percentile, x median'][ds][1] / W[col]['10th-90th percentile, x median'][ds][0], 1) for ds in DS)
R.ul([f'**Bouton weight $p_g$** varies {fold("APL_to_PN_per_PN_to_KC")}-fold across glomeruli (10th to 90th percentile; hemibrain, male CNS), in a similar order in both datasets '
      f'($\\rho_s$ = {F(p["spearman between datasets: min-max"][0], 2)}). Lowest: {", ".join(p["lowest (mean x median)"])}. '
      f'Highest: {", ".join(p["highest (mean x median)"])}.',
      f'**Claw weight $q_g$** varies only {fold("APL_to_KC_per_claw_PN_synapse")}-fold ($\\rho_s$ = {F(W["APL_to_KC_per_claw_PN_synapse"]["spearman between datasets: min-max"][0], 2)}).',
      f'The other four numbers agree less between the datasets ($\\rho_s$ {F(min(W[c]["spearman between datasets: min-max"][0] for c, *_ in QUANT[1:2] + QUANT[3:]), 2)}–'
      f'{F(max(W[c]["spearman between datasets: min-max"][0] for c, *_ in QUANT[1:2] + QUANT[3:]), 2)}). KC→APL on claws per claw PN synapse agrees least.'])

# ---------------------------------------------------------------- 4 bouton share
R.section('How much of APL\'s calyx output goes to boutons')
R.p('Versions M0 and M of Report 05 put only part of the inhibition on the boutons, the same share as APL\'s synapses:')
R.eq(r'f = \frac{\sum_{\mathrm{PNs}} \#\,\mathrm{APL{\to}PN}}{\sum_{\mathrm{PNs}} \#\,\mathrm{APL{\to}PN} + \sum_{\mathrm{KCs}} \#\,\mathrm{APL{\to}KC}}',
     note='all calyx synapses of the dataset, not only those on claws or uniglomerular PNs')
fs = abm.F_SHARE
ap = PNW.groupby('dataset').APL_to_PN.sum(); ak = KW.groupby('dataset').APL_to_KC_calyx.sum()
R.table(['dataset', 'APL→PN in the calyx', 'APL→KC in the calyx', '$f$'],
        [[DSN[ds], f'{int(ap[ds]):,}', f'{int(ak[ds]):,}', F(fs[ds], 3)] for ds in DS], title='The bouton share $f$.',
        caption='About one in ten of APL\'s calyx output synapses is onto a PN; the rest are onto KCs.', num_cols=[1, 2, 3])
# ---------------------------------------------------------------- 5 p_g per calyx, all connectomes
R.section('The bouton weight in every connectome, per calyx')
R.p('Yang asked for $p_g$ in each connectome, one row per calyx. $p_g$ needs no claws, so it can be computed for all four connectomes. '
    'Each calyx is counted on its own: an APL↔PN synapse belongs to the calyx of its APL, and a PN→KC synapse to the calyx of the nearest '
    'APL calyx synapse (each APL covers only its own calyx). The counts of the two calyces add up exactly to the per-PN table '
    '`PN_calyx_weights_APL_vs_KC.csv` (corrected calyx). A few PN types reach both calyces (V_ilPN, VL1_ilPN, VP1d_il2PN and '
    'some multiglomerular types); their synapses are split between the calyces accordingly.')
GC = pd.read_csv(L.B + 'reports/data/r04v2_pg_per_calyx.csv')
CALS = [('flywire', 'left'), ('flywire', 'right'), ('hemibrain', 'right'), ('malecns', 'left'), ('malecns', 'right'), ('banc', 'left'), ('banc', 'right')]
CN = {'flywire': 'FlyWire', 'hemibrain': 'hemibrain', 'malecns': 'male CNS', 'banc': 'BANC'}
cname = lambda d, sd: f'{CN[d]} {sd[0].upper()}'
GC['cal'] = [cname(d, sd) for d, sd in zip(GC.dataset, GC.side)]
ORDER = [cname(d, sd) for d, sd in CALS]
PK = GC.pivot(index='glomerulus', columns='cal', values='PN_to_KC').reindex(columns=ORDER)
G7 = PK.index[(PK.fillna(0) >= 200).all(1)].tolist()
PG = GC.pivot(index='glomerulus', columns='cal', values='p_g').reindex(index=G7, columns=ORDER)
SP = PG.corr(method='spearman')


def fig_pg_cal():
    rel = np.log2(np.maximum(PG / PG.median(), 2 ** -2))
    order = rel.mean(1).sort_values().index
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1.25], hspace=0.62, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(rel.loc[order].T.values, aspect='auto', cmap=L.DIV, norm=TwoSlopeNorm(0, -2, 2), interpolation='nearest')
    ax.set_yticks(range(len(ORDER)), ORDER, fontsize=8.5)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6.8)
    for y in [1.5, 2.5, 4.5]: ax.axhline(y, color='white', lw=2)
    ax.tick_params(length=0); ax.set_title(f'a  $p_g$ per calyx ({len(G7)} glomeruli with ≥ 200 PN→KC synapses in every calyx), sorted by the mean over calyces', loc='left', fontsize=10)
    cb = L.cbar(fig, im, ax, 'log₂( $p_g$ ÷ median of the row )', fraction=0.02); cb.set_ticks([-2, -1, 0, 1, 2])
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(SP.values, cmap=L.SEQ, vmin=0, vmax=1)
    ax2.set_xticks(range(7), ORDER, rotation=60, ha='right', fontsize=8); ax2.set_yticks(range(7), ORDER, fontsize=8)
    for i in range(7):
        for j in range(7):
            ax2.text(j, i, f'{SP.values[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if SP.values[i, j] > 0.6 else L.INK)
    ax2.set_title('b  Spearman ρ between calyces, over glomeruli', loc='left', fontsize=10); ax2.tick_params(length=0)
    ax3 = fig.add_subplot(gs[1, 1])
    for k, cl in enumerate(ORDER):
        v = PG[cl].values
        ax3.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.18, 0.18, len(v)), v * 100, s=9, color=L.DS_COLOR[GC[GC.cal == cl].dataset.iloc[0]], lw=0, alpha=0.8)
        ax3.plot([k - 0.3, k + 0.3], [np.median(v) * 100] * 2, color=L.INK, lw=1.5)
    ax3.set_xticks(range(7), ORDER, rotation=60, ha='right', fontsize=8)
    ax3.set_ylabel('$p_g$ × 100 (APL→PN per 100 PN→KC synapses)'); ax3.set_title('c  values per calyx (bar = median)', loc='left', fontsize=10)
    ax3.set_ylim(0, None)
    return fig


R.fig(fig_pg_cal(), '**a**, Each row is one calyx and each column one glomerulus. Color is $p_g$ (APL→PN synapses on the glomerulus\'s PNs in '
      'that calyx, per PN→KC synapse of those PNs there) divided by the row\'s median, on a log₂ scale clipped at ±2 (4× above or below); a '
      'value of 0 takes the darkest blue. White lines separate the connectomes. **b**, Rank correlation of the rows of a over the same '
      'glomeruli. **c**, The raw values: each dot is one glomerulus; the dot color marks the connectome.', '$p_g$ per calyx in every connectome.')
same = [SP.loc['FlyWire L', 'FlyWire R'], SP.loc['male CNS L', 'male CNS R'], SP.loc['BANC L', 'BANC R']]
iu = np.triu_indices(7, 1)
pairs = [(ORDER[i], ORDER[j]) for i, j in zip(*iu)]
cross = [SP.loc[a, b] for a, b in pairs if a.split(' ')[0] != b.split(' ')[0]]
rows = []
for cl in ORDER:
    t = GC[GC.cal == cl]
    rel = PG[cl] / PG[cl].median()
    rows.append([cl, len(t), f'{int(t.PN_to_KC.sum()):,}', f'{int(t.APL_to_PN.sum()):,}', F(100 * t.APL_to_PN.sum() / t.PN_to_KC.sum(), 2),
                 F(100 * PG[cl].median(), 2), f'{F(rel.quantile(.1), 2)}–{F(rel.quantile(.9), 2)}'])
R.table(['Calyx', 'glomeruli', 'PN→KC', 'APL→PN', 'all glomeruli: APL→PN per 100 PN→KC', f'median $p_g$ × 100 ({len(G7)} glomeruli)',
         '10th–90th percentile (× median)'], rows, title='The bouton weight per calyx.',
        caption='Uniglomerular PNs only, corrected calyx. Columns 2–5 use every glomerulus of the calyx; the last two use the glomeruli of Figure 4.',
        num_cols=[1, 2, 3, 4, 5, 6])
lev = {cl: 100 * GC[GC.cal == cl].APL_to_PN.sum() / GC[GC.cal == cl].PN_to_KC.sum() for cl in ORDER}
R.ul([f'**The pattern is shared across connectomes.** Over the {len(G7)} glomeruli the calyces correlate at $\\rho_s$ {F(min(cross), 2)}–{F(max(cross), 2)} '
      f'between connectomes, and {", ".join(F(x, 2) for x in same)} between the two calyces of FlyWire, male CNS and BANC.',
      f'**The level differs between calyces**, even within one brain: APL→PN per 100 PN→KC synapses ranges {F(min(lev.values()), 2)}–'
      f'{F(max(lev.values()), 2)} (Table 5, all glomeruli), and BANC has the fewest. The pattern across glomeruli, not the level, is what '
      'the model uses ($p_g$ enters relative to its mean).',
      'The glomeruli at the two ends are mostly the same as in Section 3 (Figure 3).'])
R.box('Caveats', ['Synapse counts stand in for synaptic strength. Nothing here measures how strong a single APL→PN or APL→KC synapse is.',
                  'The model numbers (Sections 2–4) pool the two male CNS calyces, because the per-glomerulus weights are computed per dataset.',
                  'Glomeruli missing from the response tables still have weights, but they carry no odor input in the model.'], 'caveat')
R.save()
