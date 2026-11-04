"""3D views of the odor-weighted PN<->APL calyx maps (Yang 2026-11-03: "try a 3D version, 2 viewing angles per odor per
connectome, ignore BANC"). One PDF page per odor: rows = the five calyces of FlyWire, hemibrain and male CNS; columns =
PN -> APL and APL -> PN, each from two angles:
  A  posterior-lateral: from behind, turned 35° toward the calyx's lateral side, 20° above the horizontal;
  B  lateral: from the calyx's lateral side, 20° above the horizontal (shows the calyx's anterior-posterior depth).
Left and right calyces are viewed as mirror images, so medial is always toward the panel's middle in A.
Axes are anatomical (common.AXES): right, anterior, dorsal in µm, centered on the calyx's median synapse position.
Color as the 2D maps (RdBu_r, two-slope, one scale per response type); grey = no response for that glomerulus.
Usage: python3 maps3d.py <response set: ORN_megamat_diag | ORN_valid2_diag | GH146>"""
import sys, os, io, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
import common as c

RSN = sys.argv[1]
RS = c.response_sets(); M, PANEL, TITLE = RS[RSN]
NORM = c.norms(RS)['GH146' if RSN == 'GH146' else 'ORN']
S = c.pn_synapses()
S = S[S.dataset != 'banc'].copy()
ctr = S.groupby(['dataset', 'apl_side'])[['right_um', 'anterior_um', 'dorsal_um']].median()
key = list(zip(S.dataset, S.apl_side))
for a in ['right_um', 'anterior_um', 'dorsal_um']:
    S[a[0]] = S[a].values - ctr.loc[key, a].values
LIM = {k: float(np.ceil(np.nanpercentile(np.abs(S[k]), 99.8)) + 1) for k in 'rad'}
VIEWS = [('A', 'posterior-lateral', 20, lambda s: -90 + (35 if s == 'R' else -35)), ('B', 'lateral', 20, lambda s: 0 if s == 'R' else 180)]
KINDS = [('PN_to_APL', 'PN → APL'), ('APL_to_PN', 'APL → PN')]
UNIT = 'units as in Yang\'s responses_wide.csv' if RSN != 'GH146' else 'GH146 peak ΔF/F'


def panel(ax, t, w, side, elev, azim):
    nan = np.isnan(w)
    lat = 1 if side == 'R' else -1
    ax.scatter(t.r.values[nan], t.a.values[nan], t.d.values[nan], s=0.6, color=c.NODATA, alpha=0.6, lw=0, depthshade=False, zorder=1)
    ax.scatter(t.r.values[~nan], t.a.values[~nan], t.d.values[~nan], s=2.2, c=c.CMAP(NORM(w[~nan])), lw=0, depthshade=False, zorder=2)
    ax.set_xlim(-LIM['r'], LIM['r']); ax.set_ylim(-LIM['a'], LIM['a']); ax.set_zlim(-LIM['d'], LIM['d'])
    ax.computed_zorder = False; ax.set_box_aspect((LIM['r'], LIM['a'], LIM['d']))
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    o = np.array([-lat * (LIM['r'] - 2), -LIM['a'] + 2, -LIM['d'] + 2])       # 5 µm orientation arrows at the medial-posterior-ventral corner
    for vec, lab in [((lat * 5, 0, 0), 'lat'), ((0, 5, 0), 'ant'), ((0, 0, 5), 'dor')]:
        ax.quiver(*o, *vec, color=c.INK, lw=0.8, arrow_length_ratio=0.25)
        ax.text(*(o + np.array(vec) * 1.4), lab, fontsize=6, color=c.INK, ha='center', va='center', clip_on=False)
    ax.patch.set_alpha(0)


pages = []
summ = {'set': RSN, 'title': TITLE, 'lim_um': LIM, 'odors': list(M.columns)}
for oi, o in enumerate(M.columns):
    fig = plt.figure(figsize=(10, 12.6))
    gs = fig.add_gridspec(len(c.CALYCES), 4, hspace=0.0, wspace=0.0, top=0.9, bottom=0.06, left=0.12, right=0.99)
    for i, (ds, side) in enumerate(c.CALYCES):
        for k, (d, dl) in enumerate(KINDS):
            t = S[(S.dataset == ds) & (S.apl_side == side) & (S.direction == d)]
            w = t.glomerulus.map(M[o]).values.astype(float)
            for v, (vk, vn, el, az) in enumerate(VIEWS):
                ax = fig.add_subplot(gs[i, 2 * k + v], projection='3d')
                ax.set_proj_type('ortho')
                panel(ax, t, w, side, el, az(side))
                ax.set_position(ax.get_position().expanded(1.3, 1.3))
                if i == 0:
                    bb = gs[0, 2 * k + v].get_position(fig)
                    fig.text(bb.x0 + bb.width / 2, 0.915, f'{dl}\nview {vk}: {vn}', fontsize=8.5, color=c.INK, ha='center', va='bottom')
        fig.text(0.01, gs[i, 0].get_position(fig).y0 + gs[i, 0].get_position(fig).height / 2,
                 f'{c.DSN[ds]}\n{"left" if side == "L" else "right"} calyx', fontsize=8.5, color=c.INK, va='center')
    pn = PANEL[oi]
    fig.suptitle(f'{o.replace(" -", " −")}  ({pn} panel)  ·  {TITLE}\nPN ↔ APL calyx synapses (corrected calyx) colored by the PN '
                 f'glomerulus\'s mean response; grey = glomerulus without data, multiglomerular or untyped PN.\nOrthographic 3D views; arrows = '
                 f'5 µm toward lateral (lat), anterior (ant) and dorsal (dor). Page {oi + 1} of {M.shape[1]}.',
                 fontsize=9.5, color=c.INK, x=0.01, ha='left', y=0.985, va='top')
    cax = fig.add_axes([0.3, 0.035, 0.45, 0.01])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=c.CMAP), cax=cax, orientation='horizontal')
    cb.ax.set_xscale('linear'); cb.ax.tick_params(labelsize=7, colors=c.MUTED); cb.outline.set_visible(False)
    cb.set_label(f'mean response of the synapse\'s glomerulus ({UNIT}); ends = min and max of '
                 + ('both new ORN tables' if RSN != 'GH146' else 'the GH146 matrix'), fontsize=7.5, color=c.MUTED)
    buf = io.BytesIO(); fig.savefig(buf, dpi=100, format='png'); plt.close(fig)
    im = Image.open(buf).convert('RGB'); pages.append(im)
    if oi == 0 or o.startswith('pa @') or o.startswith('-aPine'):
        im.save(c.OUT + f'{RSN}_3D_example_{o.split(" @")[0].replace("-", "m")}.png')
    print(oi, o, flush=True)
pages[0].save(c.OUT + f'{RSN}_3D_maps_PN_APL.pdf', save_all=True, append_images=pages[1:], quality=82, resolution=100)
json.dump(summ, open(c.OUT + f'{RSN}_3D_maps_summary.json', 'w'), indent=1)
