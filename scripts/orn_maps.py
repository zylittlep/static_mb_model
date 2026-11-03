"""PN<->APL synapse maps colored by fly-averaged ORN responses, one row per odor (all 39 odors of Yang's two panels).

Responses: Yang's pebbled ORN imaging pickles, each a dict {(date, fly_num): glomerulus x odor table of trial-mean
responses}: orn_resp_megamat_trialmean_resp.pkl (9 flies, 17 odors @ -3; aa and va in 7 flies) and
orn_resp_validation2_trialmean_resp.pkl (5 flies, 22 odors at -1.5 to -5). Loaded with an unpickler that only allows
pandas/numpy classes (orn_load.py).
Mean per odor and glomerulus = mean over the flies in which that glomerulus was imaged (NaN skipped). This reproduces
Yang's ORN_ovlp16_resp_flyave_merge_dropna.csv exactly (max |diff| 2e-16 over its 36 glomeruli x 16 odors).
Every PN<->APL synapse of a uniglomerular PN gets its glomerulus's mean (glomerulus column of
PN_APL_synapses_odor_weighted.csv: FlyWire names, hemibrain VC3l -> VC3, VC3m -> VC5, VC5 -> VM6); other synapses
(glomerulus not imaged in that panel, multiglomerular or untyped PN) get none and are drawn dark grey.
Colors: the construction of Yang's GH146 PDF, RdBu_r with TwoSlopeNorm(vcenter 0, vmin/vmax = min/max of the full
mean matrix, 40 glomeruli x 39 odors), one scale for every odor, value-linear colorbar.
Synapses, posterior view and panel windows as in odor_maps.py / gh146_maps.py (calyx only)."""
import sys, json; sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from orn_load import load, FILES
sys.path.insert(0, 'scripts')
from variant import OUT as VOUT, TAG

OUT = VOUT + 'odor_weighted/'
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
INK, MUTED = '#0b0b0b', '#52514e'
PANEL_BG, NODATA = '#d6d6d4', '#8f8e8a'
CMAP = matplotlib.colormaps['RdBu_r']
PNL = ['megamat', 'validation2']

# ---------------- fly means ----------------
rows = []
for panel in PNL:
    for (date, fly), df in load(FILES[panel]).items():
        l = df.stack(future_stack=True).rename('resp').reset_index().rename(columns={'level_1': 'odor'})
        rows.append(l.assign(panel=panel, fly=f'{date}_{fly}'))
L = pd.concat(rows).dropna(subset=['resp'])
ODORS = {p: list(load(FILES[p])[next(iter(load(FILES[p])))].columns) for p in PNL}   # Yang's column order
ALL = ODORS['megamat'] + ODORS['validation2']
assert len(ALL) == len(set(ALL)) == 39
long = (L.groupby(['panel', 'odor', 'glomerulus'])
         .agg(mean_resp=('resp', 'mean'), n_flies=('fly', 'nunique')).reset_index())
M = long.pivot(index='glomerulus', columns='odor', values='mean_resp')[ALL]
NFLY = L.groupby(['panel', 'odor']).fly.nunique()
long['odor'] = pd.Categorical(long.odor, ALL)
long.sort_values(['odor', 'glomerulus']).round(5).to_csv(OUT + 'ORN_fly_means_long.csv', index=False)
M.round(5).to_csv(OUT + 'ORN_fly_means_by_glomerulus.csv')
NORM = TwoSlopeNorm(vcenter=0, vmin=M.min().min(), vmax=M.max().max())

# check against Yang's earlier fly-averaged table (16 odors)
old = pd.read_csv('/mnt/project-files/uploads/hearth/5d477ed3-345f-4332-8b48-1ad3e2e0a0c1', index_col=0)
short = {o.split(' @ ')[0]: o for o in ALL}
chk = float(np.nanmax(np.abs(M.loc[old.index, [short[c] for c in old.columns]].values - old.values)))

# ---------------- synapses ----------------
S = pd.read_csv(OUT + 'PN_APL_synapses_odor_weighted.csv', dtype={'apl_id': str, 'pn_id': str})
S = S.drop(columns=[c for c in S.columns if c.startswith('w_')])
W = pd.DataFrame({f'w_{o}': S.glomerulus.map(M[o]) for o in ALL})
S = pd.concat([S, W], axis=1)
S.round({c: 5 for c in W.columns}).to_csv(OUT + 'PN_APL_synapses_ORN_per_odor_weighted.csv', index=False)

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
ROWS = max(len(v) for v in ODORS.values())


def flies_txt(p):
    n = NFLY[p]; c = n.value_counts()
    if len(c) == 1: return f'{n.iloc[0]} flies'
    few = [o.split(' @ ')[0] for o in n.index if n[o] != n.max()]
    return f'{n.max()} flies ({", ".join(few)}: {n.min()})'


def draw(direction, fname, arrow):
    D = C[C.direction == direction]
    pw = 1.6; ph = pw * WV / WU
    H = ph * ROWS * 1.08 + 2.1
    fig = plt.figure(figsize=(2 * (pw * len(PANELS) + .95) + .3, H))
    outer = fig.add_gridspec(1, 2, wspace=.1, top=1 - 1.75 / H, bottom=.12 / H, left=.04, right=.998)
    for b, p in enumerate(PNL):
        gs = outer[b].subgridspec(ROWS, len(PANELS), hspace=.08, wspace=.04)
        for j, (ds, s_) in enumerate(PANELS):
            t = D[(D.dataset == ds) & (D.apl_side == s_)]
            for r, o in enumerate(ODORS[p]):
                a = fig.add_subplot(gs[r, j])
                nd = t[t[f'w_{o}'].isna()]
                wd = t[t[f'w_{o}'].notna()]
                f = NORM(wd[f'w_{o}'].values)
                order = np.argsort(np.abs(f - 0.5), kind='stable')     # most saturated on top
                a.set_facecolor(PANEL_BG)
                a.scatter(nd.du, nd.dv, s=1.4, color=NODATA, linewidth=0, zorder=1, rasterized=True)
                a.scatter(wd.du.values[order], wd.dv.values[order], s=4.5, c=CMAP(f[order]), linewidth=0, zorder=2, rasterized=True)
                a.set_xlim(-WU, WU); a.set_ylim(-WV, WV); a.set_aspect('equal')
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values(): sp.set_visible(False)
                if j == 0:
                    a.set_ylabel(o.replace(' -', ' −'), fontsize=10, color=INK, rotation=0, ha='right', va='center', labelpad=5)
                if r == 0:
                    med = 'medial →' if s_ == 'L' else '← medial'
                    a.set_title(f'{LABEL[ds]}\n{"left" if s_ == "L" else "right"} APL · {med}\n'
                                f'{len(t):,} syn · {wd.shape[0] / max(len(t), 1):.0%} colored', fontsize=7.2, color=INK, loc='left')
                    if j == 0:
                        x0 = a.get_position().x0
                        fig.text(x0 - .035, a.get_position().y1 + .62 / H, f'{p} panel · {flies_txt(p)}',
                                 fontsize=12, color=INK, ha='left', va='bottom', weight='bold')
                if r == len(ODORS[p]) - 1:
                    a.plot([-WU + 2, -WU + 12], [-WV + 2, -WV + 2], color=INK, lw=1.2, solid_capstyle='butt')
                    a.text(-WU + 7, -WV + 3.2, '10 µm', ha='center', va='bottom', fontsize=6, color=INK)
        if len(ODORS[p]) < ROWS - 3:                      # free rows under the shorter block: colorbar + legend
            r0 = len(ODORS[p])
            bb = gs[r0, 1:len(PANELS) - 1].get_position(fig)
            cax = fig.add_axes([bb.x0, bb.y0 + bb.height * .3, bb.width, bb.height * .2])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation='horizontal')
            cb.ax.set_xscale('linear')     # value-linear colorbar axis, as in Yang's PDF (blue arm compressed below 0)
            cb.set_ticks([-0.4, 0, .5, 1, 1.5, 2, 2.5, 3, 3.5])
            cb.ax.tick_params(labelsize=8, colors=MUTED); cb.outline.set_visible(False)
            cb.set_label('mean ORN response (trial means averaged over flies); one scale for all 39 odors',
                         fontsize=9, color=MUTED)
            bb2 = gs[r0 + 1, 1:len(PANELS)].get_position(fig)
            h = [Line2D([], [], marker='o', ls='', color=NODATA, markersize=5,
                        label='glomerulus not imaged in this panel, multiglomerular or untyped PN'),
                 Line2D([], [], marker='s', ls='', color=PANEL_BG, markersize=8,
                        label='grey panel background, so a zero response (white) stays visible')]
            fig.legend(handles=h, loc='upper left', bbox_to_anchor=(bb2.x0, bb2.y1), ncol=1, frameon=False,
                       fontsize=9, labelcolor=MUTED)
    fig.suptitle(f'{arrow} synapses colored by the PN glomerulus\'s mean ORN response, one row per odor · calyx, posterior '
                 f'view (fly\'s right on the right, dorsal up){TAG}\nColor scale as in the GH146 PDF: RdBu_r centred at 0, ends = min and '
                 f'max of the full mean matrix ({NORM.vmin:.2f}, {NORM.vmax:.2f}). Each dot one synapse; strongest colors drawn on top.',
                 fontsize=11, color=INK, x=0.005, ha='left', y=1 - .1 / H, va='top')
    fig.savefig(OUT + fname, dpi=140, bbox_inches='tight'); plt.close(fig)


draw('PN_to_APL', 'ORN_per_odor_map_PN_to_APL_posterior_view.png', 'PN → APL')
draw('APL_to_PN', 'ORN_per_odor_map_APL_to_PN_posterior_view.png', 'APL → PN')

# ---------------- numbers for the reply ----------------
colored = {p: S.groupby(['dataset', 'direction'])[f'w_{ODORS[p][0]}'].apply(lambda s: round(s.notna().mean(), 3)).unstack().to_dict()
           for p in PNL}
summ = {'check_vs_ORN_ovlp16_table_max_abs_diff': chk, 'norm': [float(NORM.vmin), 0.0, float(NORM.vmax)],
        'norm_ends': [list(M.stack().idxmin()), list(M.stack().idxmax())],
        'flies_per_odor': {p: NFLY[p].to_dict() for p in PNL},
        'glomeruli_per_panel': {p: sorted(long[long.panel == p].glomerulus.unique()) for p in PNL},
        'share_of_PN_APL_synapses_colored': colored,
        'odor_max': {o: [M[o].idxmax(), round(float(M[o].max()), 3)] for o in ALL}}
# share of each odor's summed positive weight per dataset over PN->APL calyx synapses, top 3 glomeruli
top = {}
for o in ALL:
    t = C[(C.direction == 'PN_to_APL') & C[f'w_{o}'].notna()]
    c = t.groupby(['dataset', 'glomerulus'])[f'w_{o}'].sum().clip(lower=0)
    sh = c / c.groupby('dataset').transform('sum')
    top[o] = {ds: sh[ds].nlargest(3).round(2).to_dict() for ds in DS}
summ['top_glomeruli_share_of_positive_weight_PN_to_APL'] = top
json.dump(summ, open(OUT + 'ORN_per_odor_maps_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k not in ('top_glomeruli_share_of_positive_weight_PN_to_APL', 'odor_max')}, indent=1))
print('odor max', summ['odor_max'])
