"""PN<->APL synapse maps weighted by GH146 PN responses, colored with the scale of Yang's PDF.

Weights: Yang's GH146_ij_certain-roi_stats.csv (7 flies, 29 ImageJ glomerulus ROIs, 3 repeats per odor). Each
glomerulus gets its mean peak dF/F over all flies and repeats (identical to the mean of per-fly means here):
  PA  = 'pa @ -3'  (megamat panel)
  PAA = 'paa @ -5' (glomeruli_diagnostics panel; the only phenylacetaldehyde presentation)
  MS  = 'ms @ -3'  (megamat panel; the diagnostics-panel 'ms @ -3' is kept as w_MS_diagnostics)
Every PN<->APL synapse of a uniglomerular PN of that glomerulus gets that weight; other synapses get none (grey).
Color: the PDF's own scale, matplotlib RdBu_r with TwoSlopeNorm(vcenter=0, vmin/vmax = min/max of the full
29 x 41 mean matrix = -0.158 / 1.962). This reproduces the PDF colorbar to within 3.5/255 per channel, and the means
computed here match the PDF cells (median |difference| 0.006 dF/F after decoding the PDF colors).
Synapses, posterior view and panel windows: same as odor_maps.py (reads its synapse table)."""
import json, sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

GH = '/mnt/project-files/uploads/hearth/125c6882-0aab-4f2f-a9c9-5144fd7eacc3'  # GH146_ij_certain-roi_stats.csv
sys.path.insert(0, 'scripts')
from variant import OUT as VOUT, TAG
OUT = VOUT + 'odor_weighted/'
ODORS = [('PA', ('megamat', 'pa @ -3')), ('PAA', ('glomeruli_diagnostics', 'paa @ -5')), ('MS', ('megamat', 'ms @ -3'))]
EXTRA = [('MS_diagnostics', ('glomeruli_diagnostics', 'ms @ -3'))]
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
INK, MUTED, AXIS = '#0b0b0b', '#52514e', '#c9c8c2'
PANEL_BG, NODATA = '#d6d6d4', '#8f8e8a'
CMAP = matplotlib.colormaps['RdBu_r']

# ---------------- GH146 means ----------------
g = pd.read_csv(GH, header=[0, 1, 2], index_col=[0, 1, 2, 3, 4])
g.columns.names = ['date', 'fly_num', 'roi']; g.index.names = ['panel', 'is_pair', 'odor1', 'odor2', 'repeat']
long = g.stack(['date', 'fly_num', 'roi'], future_stack=True).rename('dff').reset_index().dropna(subset=['dff'])
M = long.groupby(['panel', 'odor1', 'roi']).dff.mean().unstack('roi')
NF = long.groupby(['panel', 'odor1', 'roi'])[['date', 'fly_num']].apply(lambda x: len(x.drop_duplicates())).unstack('roi')
NORM = TwoSlopeNorm(vcenter=0, vmin=M.min().min(), vmax=M.max().max())
Wt = pd.DataFrame({name: M.loc[key] for name, key in ODORS + EXTRA})
Wt['n_flies'] = NF.loc[ODORS[0][1]]
Wt.index.name = 'glomerulus'
Wt.round(4).to_csv(OUT + 'GH146_weights_used.csv')

# ---------------- synapses (from odor_maps.py) ----------------
S = pd.read_csv(OUT + 'PN_APL_synapses_odor_weighted.csv', dtype={'apl_id': str, 'pn_id': str})
S = S.drop(columns=[c for c in S.columns if c.startswith('w_')])
for name, _ in ODORS + EXTRA:
    S[f'w_{name}'] = S.glomerulus.map(Wt[name])
S.to_csv(OUT + 'PN_APL_synapses_GH146_weighted.csv', index=False)

C = S[S.mb_region == 'CA'].copy()
box = C.groupby(['dataset', 'apl_side'])[['view_right_um', 'view_dorsal_um']].quantile([.002, .998]).unstack()
mid_u = (box[('view_right_um', .002)] + box[('view_right_um', .998)]) / 2
mid_v = (box[('view_dorsal_um', .002)] + box[('view_dorsal_um', .998)]) / 2
WU = np.ceil(((box[('view_right_um', .998)] - box[('view_right_um', .002)]) / 2).max() + 3)
WV = np.ceil(((box[('view_dorsal_um', .998)] - box[('view_dorsal_um', .002)]) / 2).max() + 3)
key = list(zip(C.dataset, C.apl_side))
C['du'] = C.view_right_um.values - mid_u.loc[key].values
C['dv'] = C.view_dorsal_um.values - mid_v.loc[key].values
PANELS = [(ds, s_) for ds in DS for s_ in ['L', 'R'] if (ds, s_) in mid_u.index]
ROWLAB = {'PA': 'PA\npa @ −3', 'PAA': 'PAA\npaa @ −5', 'MS': 'MS\nms @ −3'}


def draw(direction, fname, title):
    D = C[C.direction == direction]
    pw = 2.35; ph = pw * WV / WU
    fig, ax = plt.subplots(3, len(PANELS), figsize=(pw * len(PANELS) + 1.3, 3 * ph + 1.3), constrained_layout=True, squeeze=False)
    for i, (name, _) in enumerate(ODORS):
        for j, (ds, s_) in enumerate(PANELS):
            a = ax[i, j]
            t = D[(D.dataset == ds) & (D.apl_side == s_)]
            nd = t[t[f'w_{name}'].isna()]
            wd = t[t[f'w_{name}'].notna()].copy()
            wd['f'] = NORM(wd[f'w_{name}'].values)
            wd = wd.iloc[np.argsort(np.abs(wd.f.values - 0.5), kind='stable')]   # most saturated on top
            a.set_facecolor(PANEL_BG)
            a.scatter(nd.du, nd.dv, s=2.5, color=NODATA, linewidth=0, zorder=1)
            a.scatter(wd.du, wd.dv, s=7, c=CMAP(wd.f.values), linewidth=0, zorder=2)
            a.set_xlim(-WU, WU); a.set_ylim(-WV, WV); a.set_aspect('equal')
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(False)
            a.plot([-WU + 2, -WU + 12], [-WV + 2, -WV + 2], color=INK, lw=1.5, solid_capstyle='butt')
            a.text(-WU + 7, -WV + 3.2, '10 µm', ha='center', va='bottom', fontsize=6.5, color=INK)
            if i == 0:
                med = 'medial →' if s_ == 'L' else '← medial'
                a.set_title(f'{LABEL[ds]} · {"left" if s_ == "L" else "right"} APL\n{len(t):,} synapses, '
                            f'{t[f"w_{name}"].notna().mean():.0%} weighted · {med}', fontsize=8, color=INK, loc='left')
            if j == 0:
                a.set_ylabel(ROWLAB[name], fontsize=11, color=INK, rotation=0, ha='right', va='center', labelpad=8)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, shrink=.55, aspect=22, pad=.01)
    cb.ax.set_yscale('linear')   # value-linear colorbar axis, as in the PDF (blue arm compressed below 0)
    cb.set_ticks(np.arange(0, 1.76, .25)); cb.set_label('mean peak ΔF/F (GH146, same scale as the PDF)', fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED); cb.outline.set_visible(False)
    h = [Line2D([], [], marker='o', ls='', color=NODATA, markersize=5,
                label='glomerulus not in the GH146 table, multiglomerular or untyped PN'),
         Line2D([], [], marker='s', ls='', color=PANEL_BG, markersize=7, label='grey panel background, so a zero response (white) stays visible')]
    fig.legend(handles=h, loc='lower left', ncol=2, frameon=False, fontsize=8, labelcolor=MUTED, bbox_to_anchor=(0.0, -0.035))
    fig.suptitle(title + TAG, fontsize=12, color=INK, x=0.01, ha='left')
    fig.savefig(OUT + fname, dpi=170, bbox_inches='tight'); plt.close(fig)


view = 'calyx, posterior view (fly\'s right on the right, dorsal up); each dot one synapse'
draw('PN_to_APL', 'GH146_map_PN_to_APL_posterior_view.png', f'PN → APL synapses colored by the PN glomerulus\'s GH146 response · {view}')
draw('APL_to_PN', 'GH146_map_APL_to_PN_posterior_view.png', f'APL → PN synapses colored by the PN glomerulus\'s GH146 response · {view}')

# ---------------- numbers for the reply ----------------
summ = {'norm': [float(NORM.vmin), 0.0, float(NORM.vmax)],
        'weighted_share': S.groupby(['dataset', 'direction']).w_PA.apply(lambda s: round(s.notna().mean(), 3)).unstack().to_dict(),
        'odor_max': {n: [Wt[n].idxmax(), round(Wt[n].max(), 3)] for n, _ in ODORS + EXTRA}}
top = {}
for name, _ in ODORS:
    for d in ['PN_to_APL', 'APL_to_PN']:
        t = S[(S.direction == d) & S[f'w_{name}'].notna()]
        c = t.groupby(['dataset', 'glomerulus'])[f'w_{name}'].sum().clip(lower=0)
        sh = c / c.groupby('dataset').transform('sum')
        top[f'{name} {d}'] = {ds: sh[ds].nlargest(4).round(2).to_dict() for ds in DS}
summ['top_glomeruli_share_of_weight'] = top
json.dump(summ, open(OUT + 'GH146_maps_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
