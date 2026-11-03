"""Odor-weighted PN<->APL synapse maps, posterior view.

Each PN<->APL synapse in the mushroom body gets a weight equal to its PN's glomerulus response to an odor
(Yang's ORN_ovlp16_resp_flyave_merge_dropna.csv: fly-averaged ORN responses, 36 glomeruli x 16 odors).
Synapses of PNs whose glomerulus is not in that table (other glomeruli, multiglomerular or untyped PNs) get no weight.
Only uniglomerular PN types are matched; hemibrain names are converted to FlyWire's (VC3l -> VC3, VC3m -> VC5, VC5 -> VM6).

Posterior view: the fly's right on the viewer's right, dorsal up, anterior-posterior collapsed. Native axes, checked
with MB landmarks (calyx posterior, vertical lobe dorsal, left/right APL):
  FlyWire  x = left-right (fly's right = +x, FlyWire's corrected side labels), y = dorsal-ventral (dorsal = -y), z = A-P
  hemibrain x = left-right (right = -x), y = A-P, z = dorsal-ventral (dorsal = -z)
  male CNS x = left-right (right = -x), y = dorsal-ventral (dorsal = -y), z = A-P
  BANC     x = left-right (right = -x), y = A-P, z = dorsal-ventral (dorsal = +z)
Synapse position = postsynaptic site. Counts follow the delivered tables; FlyWire uses all Princeton synapses
between proofread neurons (no per-pair threshold)."""
import sys, re, json; sys.path.insert(0, 'odor')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from odor_meta import meta_types, apl_rows
sys.path.insert(0, 'scripts')
from variant import OUT as VOUT, FINAL, TAG

RESP = '/mnt/project-files/uploads/hearth/5d477ed3-345f-4332-8b48-1ad3e2e0a0c1'  # ORN_ovlp16_resp_flyave_merge_dropna.csv
OUT = VOUT + 'odor_weighted/'
ODORS = [('pa', 'PA'), ('paa', 'PAA'), ('ms', 'MS')]
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
VIEW = {'flywire': lambda x, y, z: (x, -y), 'hemibrain': lambda x, y, z: (-x, -z),
        'malecns': lambda x, y, z: (-x, -y), 'banc': lambda x, y, z: (-x, z)}
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
INK, MUTED, AXIS = '#0b0b0b', '#52514e', '#c9c8c2'
NODATA = '#dddcd6'
# sequential blue ramp (dataviz reference palette, steps 150 -> 700)
CMAP = LinearSegmentedColormap.from_list('blue', ['#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
                                                  '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'])

resp = pd.read_csv(RESP).set_index('glomerulus')


def glom(t, ds):
    t = str(t); m = re.match(r'^([A-Z]+\d*[a-z]*)_(.*)$', t)
    if not m or '+' in t or t.startswith('M_'): return None
    g = m.group(1)
    return HB_RENAME.get(g, g) if ds == 'hemibrain' else g


rows, centers = [], {}
for ds in DS:
    typ, pns, proof, side = meta_types(ds)
    t = apl_rows(ds, proof)
    if ds == 'flywire':
        t['apl_side'] = t.apl_id.map(side)
    else:
        a = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet', columns=['apl_id', 'apl_side', 'apl_instance']).drop_duplicates('apl_id')
        a['apl_id'] = a.apl_id.astype(str)
        sd = a.apl_side.where(a.apl_side.notna(), a.apl_instance.str[-1])
        t['apl_side'] = t.apl_id.map(pd.Series(sd.values, index=a.apl_id))
    t['apl_side'] = t.apl_side.astype(str).str[0].str.upper()
    u, v = VIEW[ds](t.post_x.values / 1e3, t.post_y.values / 1e3, t.post_z.values / 1e3)
    t['view_right_um'], t['view_dorsal_um'] = u, v
    for (a_id, s_), g in t[t.region == 'CA'].groupby(['apl_id', 'apl_side']):
        centers[(ds, s_)] = (g.view_right_um.median(), g.view_dorsal_um.median())
    p = t[t.partner_id.isin(pns)].copy()
    p['pn_type'] = p.partner_id.map(typ)
    p['glomerulus'] = p.pn_type.map(lambda s: glom(s, ds))
    p['direction'] = np.where(p.apl_is_pre, 'APL_to_PN', 'PN_to_APL')
    for col, name in ODORS:
        p[f'w_{name}'] = p.glomerulus.map(resp[col])
    p.insert(0, 'dataset', ds)
    rows.append(p)
S = pd.concat(rows, ignore_index=True)
S = S.rename(columns={'partner_id': 'pn_id', 'region': 'mb_region', 'post_x': 'x_nm', 'post_y': 'y_nm', 'post_z': 'z_nm'})
S = S[['dataset', 'apl_id', 'apl_side', 'direction', 'pn_id', 'pn_type', 'glomerulus', 'mb_region', 'x_nm', 'y_nm', 'z_nm',
       'view_right_um', 'view_dorsal_um'] + [f'w_{n}' for _, n in ODORS]]
for c in ['view_right_um', 'view_dorsal_um']: S[c] = S[c].round(3)
S.to_csv(OUT + 'PN_APL_synapses_odor_weighted.csv', index=False)

# ---------------- figures ----------------
PANELS = [(ds, s_) for ds in DS for s_ in ['L', 'R'] if (ds, s_) in centers]
C = S[S.mb_region == 'CA'].copy()
# each panel is centred on the middle of its PN<->APL calyx synapses (0.2-99.8 percentile range); one window size for all
box = C.groupby(['dataset', 'apl_side'])[['view_right_um', 'view_dorsal_um']].quantile([.002, .998]).unstack()
mid_u = (box[('view_right_um', .002)] + box[('view_right_um', .998)]) / 2
mid_v = (box[('view_dorsal_um', .002)] + box[('view_dorsal_um', .998)]) / 2
WU = np.ceil(((box[('view_right_um', .998)] - box[('view_right_um', .002)]) / 2).max() + 3)
WV = np.ceil(((box[('view_dorsal_um', .998)] - box[('view_dorsal_um', .002)]) / 2).max() + 3)
key = list(zip(C.dataset, C.apl_side))
C['du'] = C.view_right_um.values - mid_u.loc[key].values
C['dv'] = C.view_dorsal_um.values - mid_v.loc[key].values
VMAX = {n: resp[c].max() for c, n in ODORS}
TOP = {n: resp[c].idxmax() for c, n in ODORS}


def draw(direction, fname, title):
    D = C[C.direction == direction]
    pw = 2.35; ph = pw * WV / WU
    fig, ax = plt.subplots(3, len(PANELS), figsize=(pw * len(PANELS) + 1.3, 3 * ph + 1.3), constrained_layout=True, squeeze=False)
    for i, (_, name) in enumerate(ODORS):
        norm = Normalize(0, VMAX[name])
        for j, (ds, s_) in enumerate(PANELS):
            a = ax[i, j]
            g = D[(D.dataset == ds) & (D.apl_side == s_)]
            nd = g[g[f'w_{name}'].isna()]
            wd = g[g[f'w_{name}'].notna()].sort_values(f'w_{name}')
            f = norm(wd[f'w_{name}'].clip(lower=0).values).clip(0, 1)
            a.scatter(nd.du, nd.dv, s=2.5, color=NODATA, linewidth=0, zorder=1)
            a.scatter(wd.du, wd.dv, s=2.5 + 12 * f, c=CMAP(f), linewidth=0, zorder=2)
            a.set_xlim(-WU, WU); a.set_ylim(-WV, WV); a.set_aspect('equal')
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_color(AXIS); sp.set_linewidth(.6)
            a.plot([-WU + 2, -WU + 12], [-WV + 2, -WV + 2], color=MUTED, lw=1.5, solid_capstyle='butt')
            a.text(-WU + 7, -WV + 3.2, '10 µm', ha='center', va='bottom', fontsize=6.5, color=MUTED)
            if i == 0:
                med = 'medial →' if s_ == 'L' else '← medial'
                a.set_title(f'{LABEL[ds]} · {"left" if s_ == "L" else "right"} APL\n{len(g):,} synapses, '
                            f'{g[f"w_{name}"].notna().mean():.0%} weighted · {med}', fontsize=8, color=INK, loc='left')
            if j == 0:
                a.set_ylabel(f'{name}', fontsize=13, color=INK, rotation=0, ha='right', va='center', labelpad=8)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax[i, :], shrink=.85, aspect=14, pad=.01)
        cb.set_label(f'{name} response (max {VMAX[name]:.2f}, {TOP[name]})', fontsize=8, color=MUTED)
        cb.ax.tick_params(labelsize=7, colors=MUTED); cb.outline.set_visible(False)
    h = [Line2D([], [], marker='o', ls='', color=NODATA, markersize=5, label='glomerulus not in the response table, multiglomerular or untyped PN'),
         Line2D([], [], marker='o', ls='', color='#2a78d6', markersize=5, label='weighted by the PN\'s glomerulus response (color and size; stronger drawn on top; negatives shown as 0)')]
    fig.legend(handles=h, loc='lower left', ncol=2, frameon=False, fontsize=8, labelcolor=MUTED, bbox_to_anchor=(0.0, -0.035))
    fig.suptitle(title + TAG, fontsize=12, color=INK, x=0.01, ha='left')
    fig.savefig(OUT + fname, dpi=170, bbox_inches='tight'); plt.close(fig)


view = 'calyx, posterior view (fly\'s right on the right, dorsal up, anterior–posterior collapsed); each dot one synapse'
draw('PN_to_APL', 'odor_map_PN_to_APL_posterior_view.png', f'PN → APL synapses colored by the PN glomerulus\'s ORN response · {view}')
draw('APL_to_PN', 'odor_map_APL_to_PN_posterior_view.png', f'APL → PN synapses colored by the PN glomerulus\'s ORN response · {view}')

# ---------------- numbers for the reply ----------------
summ = {'window_half_width_um': [float(WU), float(WV)], 'calyx_share': S.groupby('dataset').mb_region.apply(lambda s: round((s == 'CA').mean(), 4)).to_dict()}
summ['weighted_share'] = S.groupby(['dataset', 'direction']).w_PA.apply(lambda s: round(s.notna().mean(), 3)).unstack().to_dict()
top = {}
for _, name in ODORS:
    for d in ['PN_to_APL', 'APL_to_PN']:
        g = S[(S.direction == d) & S[f'w_{name}'].notna()]
        contrib = g.groupby(['dataset', 'glomerulus'])[f'w_{name}'].sum().clip(lower=0)
        share = contrib / contrib.groupby('dataset').transform('sum')
        top[f'{name} {d}'] = {ds: share[ds].nlargest(4).round(2).to_dict() for ds in DS}
summ['top_glomeruli_share_of_weight'] = top
json.dump(summ, open(OUT + 'odor_maps_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
