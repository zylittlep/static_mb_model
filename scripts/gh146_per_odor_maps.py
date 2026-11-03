"""PN<->APL and claw KC<->APL synapse maps colored by GH146 PN responses, one row per odor: all 41 odor presentations of
Yang's GH146 table (glomeruli_diagnostics panel, 24 odors, then megamat panel, 17 odors; 'ms @ -3' is in both).

Responses: Yang's GH146_ij_certain-roi_stats.csv (7 flies, 29 ImageJ glomerulus ROIs, 3 repeats per odor); each glomerulus
gets its mean peak dF/F over all flies and repeats, as in gh146_maps.py.
PN<->APL: every calyx synapse of a uniglomerular PN gets its glomerulus's mean (glomerulus column of
PN_APL_synapses_odor_weighted.csv: FlyWire names, hemibrain VC3l -> VC3, VC3m -> VC5, VC5 -> VM6).
KC<->APL: only synapses on KC claws (KC_APL_synapses_on_claws_by_glomerulus.csv from claw_glom_maps.py: KC-side site
within 2 µm of one of that KC's claws). Each gets the response of the glomerulus of the claw's main PN, the PN that makes
most of the claw's PN synapses (91-95% of claws get >= 90% of their PN synapses from it).
Synapses without a response (PN or claw main PN from a glomerulus without a GH146 ROI, biglomerular, multiglomerular or
untyped) are drawn dark grey.
Colors: Yang's GH146 PDF scale, RdBu_r with TwoSlopeNorm(vcenter=0, vmin/vmax = min/max of the full 29 x 41 mean matrix
= -0.158 / 1.962), one scale for every odor, value-linear colorbar. Posterior view and panel windows as in the other
calyx maps (windows from the PN<->APL calyx synapses)."""
import json, sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
sys.path.insert(0, 'scripts')
from variant import OUT as VOUT, TAG

GH = '/mnt/project-files/uploads/hearth/125c6882-0aab-4f2f-a9c9-5144fd7eacc3'  # GH146_ij_certain-roi_stats.csv
OUT = VOUT + 'odor_weighted/'
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
INK, MUTED = '#0b0b0b', '#52514e'
PANEL_BG, NODATA = '#d6d6d4', '#8f8e8a'
CMAP = matplotlib.colormaps['RdBu_r']
PNL = ['glomeruli_diagnostics', 'megamat']

# ---------------- GH146 means ----------------
g = pd.read_csv(GH, header=[0, 1, 2], index_col=[0, 1, 2, 3, 4])
g.columns.names = ['date', 'fly_num', 'roi']; g.index.names = ['panel', 'is_pair', 'odor1', 'odor2', 'repeat']
long = g.stack(['date', 'fly_num', 'roi'], future_stack=True).rename('dff').reset_index().dropna(subset=['dff'])
M = long.groupby(['panel', 'odor1', 'roi']).dff.mean().unstack('roi')
NFLY = long.groupby('roi')[['date', 'fly_num']].apply(lambda x: len(x.drop_duplicates()))
ix = g.index.to_frame(index=False).drop_duplicates(['panel', 'odor1'])
ODORS = {p: list(ix.odor1[ix.panel == p]) for p in PNL}           # Yang's order, as in the PDF
COLS = [(p, o) for p in PNL for o in ODORS[p]]
assert len(COLS) == 41 and set(COLS) == set(M.index)
NORM = TwoSlopeNorm(vcenter=0, vmin=M.min().min(), vmax=M.max().max())
Wide = M.loc[COLS].T
Wide.columns = [f'{p}: {o}' for p, o in COLS]
Wide.insert(0, 'n_flies', NFLY)
Wide.index.name = 'glomerulus'
Wide.round(5).to_csv(OUT + 'GH146_mean_by_glomerulus.csv')
RESP = {(p, o): M.loc[(p, o)] for p, o in COLS}

# ---------------- synapses ----------------
S = pd.read_csv(OUT + 'PN_APL_synapses_odor_weighted.csv', dtype={'apl_id': str, 'pn_id': str},
                usecols=['dataset', 'apl_id', 'apl_side', 'direction', 'pn_id', 'pn_type', 'glomerulus', 'mb_region',
                         'view_right_um', 'view_dorsal_um'])
C = S[S.mb_region == 'CA'].copy()
K = pd.read_csv(OUT + 'KC_APL_synapses_on_claws_by_glomerulus.csv', dtype={'apl_id': str, 'kc_id': str})
K['glomerulus'] = K.glomeruli.where(~K.glomeruli.fillna('').str.contains(r'\+'), '')   # biglomerular main PN: no response
box = C.groupby(['dataset', 'apl_side'])[['view_right_um', 'view_dorsal_um']].quantile([.002, .998]).unstack()
mid_u = (box[('view_right_um', .002)] + box[('view_right_um', .998)]) / 2
mid_v = (box[('view_dorsal_um', .002)] + box[('view_dorsal_um', .998)]) / 2
WU = np.ceil(((box[('view_right_um', .998)] - box[('view_right_um', .002)]) / 2).max() + 3)
WV = np.ceil(((box[('view_dorsal_um', .998)] - box[('view_dorsal_um', .002)]) / 2).max() + 3)
for t_ in (C, K):
    key = list(zip(t_.dataset, t_.apl_side))
    t_['du'] = t_.view_right_um.values - mid_u.loc[key].values
    t_['dv'] = t_.view_dorsal_um.values - mid_v.loc[key].values
    t_['has_resp'] = t_.glomerulus.isin(M.columns)
PANELS = [(ds, s_) for ds in DS for s_ in ['L', 'R'] if (ds, s_) in mid_u.index]
ROWS = max(len(v) for v in ODORS.values())
NF = long.groupby('panel')[['date', 'fly_num']].apply(lambda x: len(x.drop_duplicates()))


def draw(D, fname, title, nodata_label, what):
    pw = 1.6; ph = pw * WV / WU
    H = ph * ROWS * 1.08 + 2.1
    fig = plt.figure(figsize=(2 * (pw * len(PANELS) + .95) + .3, H))
    outer = fig.add_gridspec(1, 2, wspace=.1, top=1 - 1.75 / H, bottom=.12 / H, left=.04, right=.998)
    for b, p in enumerate(PNL):
        gs = outer[b].subgridspec(ROWS, len(PANELS), hspace=.08, wspace=.04)
        for j, (ds, s_) in enumerate(PANELS):
            t = D[(D.dataset == ds) & (D.apl_side == s_)]
            nd, wd = t[~t.has_resp], t[t.has_resp]
            for r, o in enumerate(ODORS[p]):
                a = fig.add_subplot(gs[r, j])
                f = NORM(wd.glomerulus.map(RESP[(p, o)]).values)
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
                                f'{len(t):,} {what} · {len(wd) / max(len(t), 1):.0%} colored', fontsize=7.2, color=INK, loc='left')
                    if j == 0:
                        x0 = a.get_position().x0
                        fig.text(x0 - .035, a.get_position().y1 + .62 / H, f'{p} panel · {len(ODORS[p])} odors · {NF[p]} flies',
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
            cb.set_ticks(np.arange(0, 1.76, .25))
            cb.ax.tick_params(labelsize=8, colors=MUTED); cb.outline.set_visible(False)
            cb.set_label('mean peak ΔF/F (GH146, over flies and repeats); one scale for all 41 odors', fontsize=9, color=MUTED)
            bb2 = gs[r0 + 1, 1:len(PANELS)].get_position(fig)
            h = [Line2D([], [], marker='o', ls='', color=NODATA, markersize=5, label=nodata_label),
                 Line2D([], [], marker='s', ls='', color=PANEL_BG, markersize=8,
                        label='grey panel background, so a zero response (white) stays visible')]
            fig.legend(handles=h, loc='upper left', bbox_to_anchor=(bb2.x0, bb2.y1), ncol=1, frameon=False,
                       fontsize=9, labelcolor=MUTED)
    fig.suptitle(f'{title}, one row per odor · calyx, posterior view (fly\'s right on the right, dorsal up){TAG}\n'
                 f'Color scale as in the GH146 PDF: RdBu_r centred at 0, ends = min and max of the full mean matrix '
                 f'({NORM.vmin:.3f}, {NORM.vmax:.3f}). Each dot one synapse; strongest colors drawn on top.',
                 fontsize=11, color=INK, x=0.005, ha='left', y=1 - .1 / H, va='top')
    fig.savefig(OUT + fname, dpi=140, bbox_inches='tight'); plt.close(fig)


PN_ND = 'PN from a glomerulus without a GH146 ROI, multiglomerular or untyped PN'
KC_ND = "claw whose main PN is from a glomerulus without a GH146 ROI, bi- or multiglomerular, or untyped"
KC_T = 'synapses on KC claws, colored by the GH146 response of the glomerulus of the claw\'s main PN'
draw(C[C.direction == 'PN_to_APL'], 'GH146_per_odor_map_PN_to_APL_posterior_view.png',
     'PN → APL synapses colored by the PN glomerulus\'s GH146 response', PN_ND, 'syn')
draw(C[C.direction == 'APL_to_PN'], 'GH146_per_odor_map_APL_to_PN_posterior_view.png',
     'APL → PN synapses colored by the PN glomerulus\'s GH146 response', PN_ND, 'syn')
draw(K[K.direction == 'KC_to_APL'], 'GH146_per_odor_map_KC_to_APL_on_claws_posterior_view.png', f'KC → APL {KC_T}', KC_ND, 'on claws')
draw(K[K.direction == 'APL_to_KC'], 'GH146_per_odor_map_APL_to_KC_on_claws_posterior_view.png', f'APL → KC {KC_T}', KC_ND, 'on claws')

# ---------------- numbers for the reply ----------------
gl_data = {ds: sorted(set(M.columns) - set(C.glomerulus[C.dataset == ds])) for ds in DS}
summ = {'norm': [float(NORM.vmin), 0.0, float(NORM.vmax)], 'norm_ends': [list(M.stack().idxmin()), list(M.stack().idxmax())],
        'odors': {p: ODORS[p] for p in PNL}, 'flies': NF.to_dict(), 'flies_per_glomerulus': NFLY.to_dict(),
        'GH146_glomeruli_without_PN_APL_calyx_synapses': gl_data,
        'share_colored_PN_APL': C.groupby(['dataset', 'direction']).has_resp.mean().round(3).unstack().to_dict('index'),
        'share_colored_KC_APL_on_claws': K.groupby(['dataset', 'direction']).has_resp.mean().round(3).unstack().to_dict('index'),
        'n_KC_APL_on_claws': K.groupby(['dataset', 'direction']).size().unstack().to_dict('index'),
        'odor_max': {f'{p}: {o}': [M.loc[(p, o)].idxmax(), round(float(M.loc[(p, o)].max()), 3)] for p, o in COLS}}
json.dump(summ, open(OUT + 'GH146_per_odor_maps_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k not in ('odor_max', 'odors', 'flies_per_glomerulus')}, indent=1))
