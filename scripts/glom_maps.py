"""Per-glomerulus PN<->APL synapse maps for all 58 antennal-lobe glomeruli (51 olfactory + 7 thermo/hygro VP).
One panel per glomerulus x APL: synapses of PNs innervating that glomerulus are colored, every other PN<->APL synapse
of the same direction is grey. Uniglomerular PNs in blue; PNs whose type names two glomeruli (VP1d+VP4, VP1m+VP5,
VP3+VP1l) are orange and appear in both rows; 'X+' types (X plus other regions, e.g. VP2+, VP5+Z) count for X.
Multiglomerular M_ types and untyped PNs stay grey. Hemibrain names use FlyWire's (VC3l -> VC3, VC3m -> VC5, VC5 -> VM6).
Synapses, posterior view and panel windows as in odor_maps.py / gh146_maps.py (calyx only)."""
import re, sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, 'scripts')
from variant import OUT as VOUT, TAG
OUT = VOUT + 'odor_weighted/'
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
INK, MUTED = '#0b0b0b', '#52514e'
REST, UNI, BI, FRAME = '#d3d2cc', '#2a78d6', '#eb6834', '#e1e0d9'
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'

S = pd.read_csv(OUT + 'PN_APL_synapses_GH146_weighted.csv', dtype={'apl_id': str, 'pn_id': str})
C = S[S.mb_region == 'CA'].copy()


def gloms_of(t, ds):
    """Glomeruli named by a PN type: 'DM1_lPN' -> [DM1]; 'VP1d+VP4_l2PN1' -> [VP1d, VP4]; 'VP2+_adPN' -> [VP2]."""
    t = str(t)
    if t.startswith('M_') or '_' not in t: return []
    head = t.split('_')[0]
    names = [n for n in head.split('+') if re.fullmatch(GPAT, n)]
    return [HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in names]


C['gl_list'] = [gloms_of(t, ds) for t, ds in zip(C.pn_type, C.dataset)]
C['n_gl'] = C.gl_list.str.len()
E = C[C.n_gl > 0].explode('gl_list').rename(columns={'gl_list': 'glom'})
GLOMS = sorted(E.glom.unique())
assert len(GLOMS) == 58, (len(GLOMS), GLOMS)

box = C.groupby(['dataset', 'apl_side'])[['view_right_um', 'view_dorsal_um']].quantile([.002, .998]).unstack()
mid_u = (box[('view_right_um', .002)] + box[('view_right_um', .998)]) / 2
mid_v = (box[('view_dorsal_um', .002)] + box[('view_dorsal_um', .998)]) / 2
WU = np.ceil(((box[('view_right_um', .998)] - box[('view_right_um', .002)]) / 2).max() + 3)
WV = np.ceil(((box[('view_dorsal_um', .998)] - box[('view_dorsal_um', .002)]) / 2).max() + 3)
for T in (C, E):
    key = list(zip(T.dataset, T.apl_side))
    T['du'] = T.view_right_um.values - mid_u.loc[key].values
    T['dv'] = T.view_dorsal_um.values - mid_v.loc[key].values
PANELS = [(ds, s_) for ds in DS for s_ in ['L', 'R'] if (ds, s_) in mid_u.index]
NB = 2                                   # glomeruli split over two side-by-side blocks
ROWS = int(np.ceil(len(GLOMS) / NB))


def draw(direction, fname, arrow):
    D = C[C.direction == direction]; G = E[E.direction == direction]
    pw = 1.5; ph = pw * WV / WU
    H = ph * ROWS * 1.08 + 1.5
    fig = plt.figure(figsize=(NB * (pw * len(PANELS) + .75) + .3, H))
    outer = fig.add_gridspec(1, NB, wspace=.07, top=1 - 1.2 / H, bottom=.15 / H, left=.035, right=.998)
    for b in range(NB):
        gs = outer[b].subgridspec(ROWS, len(PANELS), hspace=.08, wspace=.04)
        for j, (ds, s_) in enumerate(PANELS):
            t = D[(D.dataset == ds) & (D.apl_side == s_)]
            g_ = G[(G.dataset == ds) & (G.apl_side == s_)]
            for r in range(ROWS):
                k = b * ROWS + r
                if k >= len(GLOMS): continue
                gl = GLOMS[k]
                a = fig.add_subplot(gs[r, j])
                h = g_[g_.glom == gl]
                a.scatter(t.du, t.dv, s=1.1, color=REST, linewidth=0, zorder=1, rasterized=True)
                a.scatter(h.du, h.dv, s=5, c=np.where(h.n_gl > 1, BI, UNI), linewidth=0, zorder=2, rasterized=True)
                a.set_xlim(-WU, WU); a.set_ylim(-WV, WV); a.set_aspect('equal')
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values(): sp.set_color(FRAME); sp.set_linewidth(.5)
                a.text(.98, .96, f'{len(h)}', transform=a.transAxes, ha='right', va='top', fontsize=6.5, color=MUTED)
                if j == 0:
                    a.set_ylabel(gl, fontsize=10, color=INK, rotation=0, ha='right', va='center', labelpad=5)
                if r == 0:
                    med = 'medial →' if s_ == 'L' else '← medial'
                    a.set_title(f'{LABEL[ds]}\n{"left" if s_ == "L" else "right"} APL · {med}', fontsize=7.5, color=INK, loc='left')
                if r == ROWS - 1 or k == len(GLOMS) - 1:
                    a.plot([-WU + 2, -WU + 12], [-WV + 2, -WV + 2], color=INK, lw=1.2, solid_capstyle='butt')
                    a.text(-WU + 7, -WV + 3.2, '10 µm', ha='center', va='bottom', fontsize=6, color=INK)
    fig.suptitle(f'{arrow} synapses by glomerulus, all 58 glomeruli · calyx, posterior view (fly\'s right on the right, dorsal up){TAG}\n'
                 f'Blue = synapses of uniglomerular PNs of that glomerulus; orange = PNs innervating that glomerulus and one other '
                 f'(VP1d+VP4, VP1m+VP5, VP3+VP1l; shown in both rows); grey = every other {arrow} synapse. '
                 f'Number = colored synapses in the panel.', fontsize=10, color=INK, x=0.005, ha='left', y=1 - .12 / H, va='top')
    fig.savefig(OUT + fname, dpi=130, bbox_inches='tight'); plt.close(fig)


draw('PN_to_APL', 'glomerulus_map_PN_to_APL_posterior_view.png', 'PN → APL')
draw('APL_to_PN', 'glomerulus_map_APL_to_PN_posterior_view.png', 'APL → PN')
cov = C.n_gl.gt(0).groupby(C.direction).mean().round(3).to_dict()
print('glomeruli', len(GLOMS), GLOMS, '\nshare of PN<->APL calyx synapses colored in some row:', cov)
print(E.groupby(['glom', 'dataset']).size().unstack(fill_value=0).loc[['VP1d', 'VP1l', 'VP1m', 'VP2', 'VP3', 'VP4', 'VP5', 'V']])
