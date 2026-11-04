"""Odor-weighted maps of KC <-> APL synapses on the new claws (Report 03 revision, 2026-11-04), and the combined map Yang asked
for ("odor weighted APL -> KC + PN map"). Hemibrain R and male CNS L/R only: the new claws (claw_v2m2) exist where KC skeletons
were segmented. A synapse on a claw carries the response of its claw's PN glomerulus; grey = claw whose PN type names two
glomeruli or none, or a glomerulus without data.
Kinds: KC -> APL (on claws), APL -> KC (on claws), APL -> KC + PN (APL -> KC on claws and APL -> PN on boutons, together).
2D: posterior view, one row per odor, one column per calyx (same drawing rules as maps2d.py), one PNG per response set and kind.
3D: one PDF page per odor, rows = the three calyces, columns = the three kinds x two angles (as maps3d.py).
Usage: python3 kc_maps.py 2d | 3d <response set>"""
import sys, os, io, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common as c
from PIL import Image

MODE = sys.argv[1]
RS = c.response_sets(); NORMS = c.norms(RS)
K = c.kc_synapses()
P = c.pn_synapses()
P = P[P.dataset.isin(['hemibrain', 'malecns'])]
cols = ['dataset', 'apl_side', 'direction', 'glomerulus', 'right_um', 'anterior_um', 'dorsal_um', 'view_right_um', 'view_dorsal_um']
S = pd.concat([K[cols].assign(partner='KC'), P[cols].assign(partner='PN')], ignore_index=True)
KINDS = [('KC_to_APL', 'KC → APL (on claws)', lambda t: t[(t.partner == 'KC') & (t.direction == 'KC_to_APL')]),
         ('APL_to_KC', 'APL → KC (on claws)', lambda t: t[(t.partner == 'KC') & (t.direction == 'APL_to_KC')]),
         ('APL_to_KC_PN', 'APL → KC + PN', lambda t: t[t.direction.isin(['APL_to_KC', 'APL_to_PN'])])]
UNIT = {'GH146': 'GH146 peak ΔF/F'}
summ = {}


def unit(rs): return UNIT.get(rs, 'units as in Yang\'s responses_wide.csv')


def norm(rs): return NORMS['GH146' if rs == 'GH146' else 'ORN']


if MODE == '2d':
    box = S.groupby(['dataset', 'apl_side'])[['view_right_um', 'view_dorsal_um']].quantile([.002, .998]).unstack()
    mid_u = (box[('view_right_um', .002)] + box[('view_right_um', .998)]) / 2
    mid_v = (box[('view_dorsal_um', .002)] + box[('view_dorsal_um', .998)]) / 2
    WU = np.ceil(((box[('view_right_um', .998)] - box[('view_right_um', .002)]) / 2).max() + 3)
    WV = np.ceil(((box[('view_dorsal_um', .998)] - box[('view_dorsal_um', .002)]) / 2).max() + 3)
    key = list(zip(S.dataset, S.apl_side))
    S['du'] = S.view_right_um.values - mid_u.loc[key].values
    S['dv'] = S.view_dorsal_um.values - mid_v.loc[key].values
    for rs in sys.argv[2:] or ['ORN_megamat_diag', 'ORN_valid2_diag', 'GH146']:
        M, panel, title = RS[rs]; NORM = norm(rs)
        for kk, kl, sel in KINDS:
            D = sel(S); odors = list(M.columns); rows = len(odors)
            pw = 2.0; ph = pw * WV / WU
            H = ph * (rows + 2) * 1.08 + 2.4
            fig = plt.figure(figsize=(pw * 3 + 1.6, H))
            gs = fig.add_gridspec(rows + 2, 3, hspace=.08, wspace=.04, top=1 - 1.7 / H, bottom=.1 / H, left=.2, right=.995)
            for j, (ds, s_) in enumerate(c.MODEL_MBS):
                t = D[(D.dataset == ds) & (D.apl_side == s_)]
                for r, o in enumerate(odors):
                    w = t.glomerulus.map(M[o]).values.astype(float)
                    a = fig.add_subplot(gs[r, j]); nan = np.isnan(w)
                    f = NORM(w[~nan]); order = np.argsort(np.abs(f - 0.5), kind='stable')
                    a.set_facecolor(c.PANEL_BG)
                    a.scatter(t.du.values[nan], t.dv.values[nan], s=1.0, color=c.NODATA, linewidth=0, zorder=1, rasterized=True)
                    a.scatter(t.du.values[~nan][order], t.dv.values[~nan][order], s=2.6, c=c.CMAP(f[order]), linewidth=0, zorder=2, rasterized=True)
                    a.set_xlim(-WU, WU); a.set_ylim(-WV, WV); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
                    for sp in a.spines.values(): sp.set_visible(False)
                    if j == 0:
                        a.set_ylabel(o.replace(' -', ' −') + ('' if panel[r] != 'diagnostics' else '  ·d'), fontsize=10, color=c.INK,
                                     rotation=0, ha='right', va='center', labelpad=5)
                    if r == 0:
                        med = 'medial →' if s_ == 'L' else '← medial'
                        a.set_title(f'{c.DSN[ds]}\n{"left" if s_ == "L" else "right"} APL · {med}\n{len(t):,} syn · {(~nan).mean():.0%} colored',
                                    fontsize=7.6, color=c.INK, loc='left')
                        summ.setdefault(f'{rs} {kk}', {})[f'{ds} {s_}'] = {'synapses': len(t), 'share_colored': round(float((~nan).mean()), 3)}
                    if r == rows - 1:
                        a.plot([-WU + 2, -WU + 12], [-WV + 2, -WV + 2], color=c.INK, lw=1.2, solid_capstyle='butt')
                        a.text(-WU + 7, -WV + 3.2, '10 µm', ha='center', va='bottom', fontsize=6, color=c.INK)
            bb = gs[rows, :].get_position(fig)
            cax = fig.add_axes([bb.x0 + bb.width * .1, bb.y0 + bb.height * .45, bb.width * .8, bb.height * .2])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=c.CMAP), cax=cax, orientation='horizontal')
            cb.ax.set_xscale('linear'); cb.ax.tick_params(labelsize=8, colors=c.MUTED); cb.outline.set_visible(False)
            cb.set_label(f'mean response of the glomerulus ({unit(rs)})', fontsize=9, color=c.MUTED)
            bb2 = gs[rows + 1, :].get_position(fig)
            h = [Line2D([], [], marker='o', ls='', color=c.NODATA, markersize=5, label='claw/PN with two or no glomeruli, or no data')]
            fig.legend(handles=h, loc='upper left', bbox_to_anchor=(bb2.x0, bb2.y1), frameon=False, fontsize=9, labelcolor=c.MUTED)
            fig.suptitle(f'{kl} synapses colored by the glomerulus\'s mean response\n{title}\nOne row per odor (·d = diagnostics). New claws '
                         f'(8 µm along the KC,\nsplit by PN), corrected calyx, posterior view.\nColor: RdBu_r centred at 0, ends '
                         f'{NORM.vmin:.2f} and {NORM.vmax:.2f}.\nStrongest colors on top.', fontsize=10, color=c.INK, x=0.005, ha='left', y=1 - .1 / H, va='top')
            fn = f'{rs}_map_{kk}_posterior_view.png'
            fig.savefig(c.OUT + fn, dpi=110); plt.close(fig)
            print(fn, flush=True)
    json.dump(summ, open(c.OUT + 'kc_maps2d_summary.json', 'w'), indent=1)

else:
    rs = sys.argv[2]
    M, PANEL, TITLE = RS[rs]; NORM = norm(rs)
    ctr = S.groupby(['dataset', 'apl_side'])[['right_um', 'anterior_um', 'dorsal_um']].median()
    key = list(zip(S.dataset, S.apl_side))
    for a in ['right_um', 'anterior_um', 'dorsal_um']:
        S[a[0]] = S[a].values - ctr.loc[key, a].values
    LIM = {k: float(np.ceil(np.nanpercentile(np.abs(S[k]), 99.8)) + 1) for k in 'rad'}
    VIEWS = [('A', 'posterior-lateral', 20, lambda s: -90 + (35 if s == 'R' else -35)), ('B', 'lateral', 20, lambda s: 0 if s == 'R' else 180)]

    def panel3d(ax, t, w, side, elev, azim):
        nan = np.isnan(w); lat = 1 if side == 'R' else -1
        ax.scatter(t.r.values[nan], t.a.values[nan], t.d.values[nan], s=0.4, color=c.NODATA, alpha=0.6, lw=0, depthshade=False, zorder=1)
        ax.scatter(t.r.values[~nan], t.a.values[~nan], t.d.values[~nan], s=1.4, c=c.CMAP(NORM(w[~nan])), lw=0, depthshade=False, zorder=2)
        ax.set_xlim(-LIM['r'], LIM['r']); ax.set_ylim(-LIM['a'], LIM['a']); ax.set_zlim(-LIM['d'], LIM['d'])
        ax.computed_zorder = False; ax.set_box_aspect((LIM['r'], LIM['a'], LIM['d'])); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
        o = np.array([-lat * (LIM['r'] - 2), -LIM['a'] + 2, -LIM['d'] + 2])
        for vec, lab in [((lat * 5, 0, 0), 'lat'), ((0, 5, 0), 'ant'), ((0, 0, 5), 'dor')]:
            ax.quiver(*o, *vec, color=c.INK, lw=0.8, arrow_length_ratio=0.25)
            ax.text(*(o + np.array(vec) * 1.4), lab, fontsize=6, color=c.INK, ha='center', va='center', clip_on=False)
        ax.patch.set_alpha(0)

    pages = []
    for oi, o in enumerate(M.columns):
        fig = plt.figure(figsize=(14, 8.4))
        gs = fig.add_gridspec(3, 6, hspace=0.0, wspace=0.0, top=0.86, bottom=0.08, left=0.09, right=0.995)
        for i, (ds, side) in enumerate(c.MODEL_MBS):
            for k, (kk, kl, sel) in enumerate(KINDS):
                t = sel(S); t = t[(t.dataset == ds) & (t.apl_side == side)]
                w = t.glomerulus.map(M[o]).values.astype(float)
                for v, (vk, vn, el, az) in enumerate(VIEWS):
                    ax = fig.add_subplot(gs[i, 2 * k + v], projection='3d'); ax.set_proj_type('ortho')
                    panel3d(ax, t, w, side, el, az(side))
                    ax.set_position(ax.get_position().expanded(1.3, 1.3))
                    if i == 0:
                        bb = gs[0, 2 * k + v].get_position(fig)
                        fig.text(bb.x0 + bb.width / 2, 0.87, f'{kl}\nview {vk}: {vn}', fontsize=8.5, color=c.INK, ha='center', va='bottom')
            bb = gs[i, 0].get_position(fig)
            fig.text(0.005, bb.y0 + bb.height / 2, f'{c.DSN[ds]}\n{"left" if side == "L" else "right"} calyx', fontsize=8.5, color=c.INK, va='center')
        fig.suptitle(f'{o.replace(" -", " −")}  ({PANEL[oi]} panel)  ·  {TITLE}\nKC ↔ APL synapses on the new claws and APL → PN synapses, '
                     f'colored by the glomerulus\'s mean response; grey = two or no glomeruli, or no data. Orthographic 3D views; arrows = 5 µm '
                     f'toward lateral, anterior and dorsal. Page {oi + 1} of {M.shape[1]}.', fontsize=9.5, color=c.INK, x=0.01, ha='left', y=0.985, va='top')
        cax = fig.add_axes([0.3, 0.045, 0.45, 0.012])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=c.CMAP), cax=cax, orientation='horizontal')
        cb.ax.set_xscale('linear'); cb.ax.tick_params(labelsize=7, colors=c.MUTED); cb.outline.set_visible(False)
        cb.set_label(f'mean response of the glomerulus ({unit(rs)})', fontsize=7.5, color=c.MUTED)
        buf = io.BytesIO(); fig.savefig(buf, dpi=95, format='png'); plt.close(fig)
        im = Image.open(buf).convert('RGB'); pages.append(im)
        if oi == 0 or o.startswith('pa @') or o.startswith('-aPine'):
            im.save(c.OUT + f'{rs}_3D_example_KC_{o.split(" @")[0].replace("-", "m")}.png')
        print(oi, o, flush=True)
    pages[0].save(c.OUT + f'{rs}_3D_maps_KC_APL.pdf', save_all=True, append_images=pages[1:], quality=82, resolution=95)
