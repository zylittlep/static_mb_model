"""2D posterior-view maps of PN<->APL calyx synapses colored by the new ORN fly means (Yang's responses_wide.csv, 2026-11-03),
one PNG per response set and direction, one row per odor, one column per calyx (all four datasets, as the earlier maps).
Same drawing rules as scripts/orn_maps.py: RdBu_r, two-slope scale centred at 0 with ends = min and max of both new ORN
tables together, grey = glomerulus not imaged / multiglomerular / untyped PN, strongest colors on top."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common as c

os.makedirs(c.OUT, exist_ok=True)
RS = c.response_sets(); NORMS = c.norms(RS); NORM = NORMS['ORN']
S = c.pn_synapses()
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
box = S.groupby(['dataset', 'apl_side'])[['view_right_um', 'view_dorsal_um']].quantile([.002, .998]).unstack()
mid_u = (box[('view_right_um', .002)] + box[('view_right_um', .998)]) / 2
mid_v = (box[('view_dorsal_um', .002)] + box[('view_dorsal_um', .998)]) / 2
WU = np.ceil(((box[('view_right_um', .998)] - box[('view_right_um', .002)]) / 2).max() + 3)
WV = np.ceil(((box[('view_dorsal_um', .998)] - box[('view_dorsal_um', .002)]) / 2).max() + 3)
key = list(zip(S.dataset, S.apl_side))
S['du'] = S.view_right_um.values - mid_u.loc[key].values
S['dv'] = S.view_dorsal_um.values - mid_v.loc[key].values
PANELS = [(ds, s_) for ds in DS for s_ in ['L', 'R'] if (ds, s_) in mid_u.index]
summ = {}


def draw(rs, direction, arrow):
    M, panel, title = RS[rs]
    NORM = NORMS['GH146' if rs == 'GH146' else 'ORN']
    src = 'GH146 table' if rs == 'GH146' else 'both new ORN tables'
    D = S[S.direction == direction]
    odors = list(M.columns); rows = len(odors)
    pw = 1.6; ph = pw * WV / WU
    H = ph * (rows + 2) * 1.08 + 2.3
    fig = plt.figure(figsize=(pw * len(PANELS) + 1.6, H))
    gs = fig.add_gridspec(rows + 2, len(PANELS), hspace=.08, wspace=.04, top=1 - 1.9 / H, bottom=.1 / H, left=.11, right=.995)
    for j, (ds, s_) in enumerate(PANELS):
        t = D[(D.dataset == ds) & (D.apl_side == s_)]
        for r, o in enumerate(odors):
            w = t.glomerulus.map(M[o]).values
            a = fig.add_subplot(gs[r, j])
            nan = np.isnan(w)
            f = NORM(w[~nan]); order = np.argsort(np.abs(f - 0.5), kind='stable')
            a.set_facecolor(c.PANEL_BG)
            a.scatter(t.du.values[nan], t.dv.values[nan], s=1.4, color=c.NODATA, linewidth=0, zorder=1, rasterized=True)
            a.scatter(t.du.values[~nan][order], t.dv.values[~nan][order], s=4.5, c=c.CMAP(f[order]), linewidth=0, zorder=2, rasterized=True)
            a.set_xlim(-WU, WU); a.set_ylim(-WV, WV); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(False)
            if j == 0:
                lab = o.replace(' -', ' −') + ('' if panel[r] != 'diagnostics' else '  ·d')
                a.set_ylabel(lab, fontsize=10, color=c.INK, rotation=0, ha='right', va='center', labelpad=5)
            if r == 0:
                med = 'medial →' if s_ == 'L' else '← medial'
                a.set_title(f'{c.DSN[ds]}\n{"left" if s_ == "L" else "right"} APL · {med}\n{len(t):,} syn · {(~nan).mean():.0%} colored',
                            fontsize=7.2, color=c.INK, loc='left')
                summ.setdefault(rs, {})[f'{ds} {s_} {direction}'] = {'synapses': len(t), 'share_colored': round(float((~nan).mean()), 3)}
            if r == rows - 1:
                a.plot([-WU + 2, -WU + 12], [-WV + 2, -WV + 2], color=c.INK, lw=1.2, solid_capstyle='butt')
                a.text(-WU + 7, -WV + 3.2, '10 µm', ha='center', va='bottom', fontsize=6, color=c.INK)
    bb = gs[rows, 1:len(PANELS) - 1].get_position(fig)
    cax = fig.add_axes([bb.x0, bb.y0 + bb.height * .45, bb.width, bb.height * .2])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=c.CMAP), cax=cax, orientation='horizontal')
    cb.ax.set_xscale('linear'); cb.set_ticks([-0.3, 0, .5, 1, 1.5, 2, 2.4])
    cb.ax.tick_params(labelsize=8, colors=c.MUTED); cb.outline.set_visible(False)
    cb.set_label('mean GH146 PN response (peak ΔF/F, mean over flies and repeats)' if rs == 'GH146' else
                 'mean ORN response (fly means of trial responses; units as in Yang\'s responses_wide.csv); one scale for both new ORN tables', fontsize=9, color=c.MUTED)
    bb2 = gs[rows + 1, 1:].get_position(fig)
    h = [Line2D([], [], marker='o', ls='', color=c.NODATA, markersize=5, label='glomerulus not imaged, multiglomerular or untyped PN'),
         Line2D([], [], marker='s', ls='', color=c.PANEL_BG, markersize=8, label='grey panel background, so a zero response (white) stays visible')]
    fig.legend(handles=h, loc='upper left', bbox_to_anchor=(bb2.x0, bb2.y1), ncol=2, frameon=False, fontsize=9, labelcolor=c.MUTED)
    fig.suptitle(f'{arrow} synapses colored by the PN glomerulus\'s mean ORN response · {title}\nOne row per odor (·d = diagnostics panel); '
                 f'calyx (corrected, with synapses outside the official ROI), posterior view (fly\'s right on the right, dorsal up).\n'
                 f'Color: RdBu_r centred at 0, ends = min and max of the {src} ({NORM.vmin:.2f}, {NORM.vmax:.2f}). '
                 'Each dot one synapse; strongest colors drawn on top.', fontsize=10.5, color=c.INK, x=0.005, ha='left', y=1 - .1 / H, va='top')
    fn = f'{rs}_map_{direction}_posterior_view.png'
    fig.savefig(c.OUT + fn, dpi=110); plt.close(fig)
    print(fn, flush=True)


import sys
for rs in sys.argv[1:] or ['ORN_megamat_diag', 'ORN_valid2_diag', 'GH146']:
    for d, arrow in [('PN_to_APL', 'PN → APL'), ('APL_to_PN', 'APL → PN')]:
        draw(rs, d, arrow)
summ['norm'] = {k: [v.vmin, v.vmax] for k, v in NORMS.items()}
json.dump(summ, open(c.OUT + 'maps2d_summary.json', 'w'), indent=1)
