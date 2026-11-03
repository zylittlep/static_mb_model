"""Why the calyx looks different across connectomes in the posterior-view maps.

For each calyx (APL synapses, postsynaptic site, µm, native axes mapped to fly's right / dorsal / anterior as in the
maps; left calyces mirrored so lateral is always to the right):
  - posterior-view width/height (SD along right vs dorsal), as drawn in the maps
  - 3D principal axes of the calyx cloud (rotation invariant): face-on aspect PC1/PC2, size = geometric mean SD,
    and the tilt of the thinnest axis (the calyx's own front-back axis) away from the anterior-posterior viewing axis
  - ROI clipping: APL synapses with Kenyon cell or ALPN partners that lie outside every MB ROI but within 8 µm of the
    calyx cloud (each dataset's own ROI labels: FlyWire / BANC neuropil meshes, hemibrain roi, male CNS primary ROI)
Figure: posterior view (as in the maps), side view and face-on view of every calyx; grey = in the calyx ROI,
orange = calyx-partner synapses outside it."""
import sys; sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import json
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from geom import to_rda, DS, LABEL

S = ''   # run from the working directory that holds out/ and dl/, like run.py
O, D = S + 'out/', S + 'dl/'
OUT = '/mnt/project-files/apl_synapses/odor_weighted/'
INK, MUTED, IN_ROI, OFF_ROI = '#0b0b0b', '#52514e', '#9a9994', '#eb6834'
P0 = 720575940000000000
APL = {'flywire': {'720575940624547622': 'L', '720575940613583001': 'R'}, 'hemibrain': {425790257: 'R'},
       'malecns': {10977: 'L', 10540: 'R'}, 'banc': {'720575941734526507': 'L', '720575941482622627': 'R'}}


def raw_table(ds):
    """All APL synapses of a dataset with partner class (KC / ALPN / other), assigned ROI and RDA position."""
    if ds == 'flywire':
        r = pd.read_parquet(O + 'fafbP_apl_raw.parquet')
        pre = (r.pre_root_id_720575940.astype('int64') + P0).astype(str); post = (r.post_root_id_720575940.astype('int64') + P0).astype(str)
        c = pd.read_csv(D + 'classification.csv.gz', dtype={'root_id': str}); cls = pd.Series(c['class'].values, index=c.root_id)
        xyz, roi = (r.post_x.values.astype(float), r.post_y.values.astype(float), r.post_z.values.astype(float)), r.neuropil.fillna('NA')
        kc, pn = 'Kenyon_Cell', 'ALPN'
    elif ds == 'hemibrain':
        r = pd.read_parquet(O + 'hemibrain_apl_raw.parquet')
        pre, post = r.body_pre.astype('int64'), r.body_post.astype('int64')
        t = pd.read_parquet(O + 'hemibrain_body_types.parquet'); t = pd.Series(t['type'].values, index=t.body.astype('int64'))
        t = t[~t.index.duplicated()].fillna('')
        cls = pd.Series(np.where(t.str.startswith('KC'), 'KC', np.where(t.str.contains(r'_[a-z0-9]*PN', regex=True), 'PN', 'other')), index=t.index)
        xyz, roi = (r.x_post.values * 8., r.y_post.values * 8., r.z_post.values * 8.), r.roi_post.fillna('NA')
        kc, pn = 'KC', 'PN'
    elif ds == 'malecns':
        r = pd.read_parquet(O + 'malecns_apl_raw.parquet')
        pre, post = r.body_pre.astype('int64'), r.body_post.astype('int64')
        a = pd.read_feather(D + 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['bodyId', 'class'])
        cls = pd.Series(a['class'].values, index=a.bodyId.astype('int64'))
        xyz, roi = (r.x_post.values * 8., r.y_post.values * 8., r.z_post.values * 8.), r.primary_post.fillna('NA')
        kc, pn = 'Kenyon_Cell', 'ALPN'
    else:
        r = pd.read_parquet(O + 'banc_v2_apl_raw.parquet')
        pre, post = r.pre_root_id.astype(str), r.post_root_id.astype(str)
        b = pd.read_feather(D + 'banc_888_meta.feather', columns=['banc_888_id', 'cell_class'])
        cls = pd.Series(b.cell_class.values, index=b.banc_888_id.astype(str))
        xyz, roi = (r.post_x.values.astype(float), r.post_y.values.astype(float), r.post_z.values.astype(float)), r.neuropil.fillna('NA')
        kc, pn = 'kenyon_cell', 'antennal_lobe_projection_neuron'
    ids = APL[ds]
    pre, post = pd.Series(np.asarray(pre)), pd.Series(np.asarray(post))
    side = pre.map(ids).where(pre.isin(list(ids)), post.map(ids))
    partner = pd.Series(np.where(pre.isin(list(ids)), post, pre))
    pc = partner.map(cls).fillna('none').replace({kc: 'KC', pn: 'PN'})
    T = pd.DataFrame(to_rda(ds, *xyz), columns=['R', 'D', 'A'])
    T['side'], T['pclass'], T['roi'] = side.values, pc.values, np.asarray(roi).astype(str)
    T['in_mb'] = T.roi.str.contains(r"^(?:MB_|CA\(|PED\(|aL\(|a'L\(|bL\(|b'L\(|gL\()", regex=True)
    T['in_ca'] = T.roi.str.contains(r'^(?:MB_CA|CA\()', regex=True)
    return T[T.side.notna()]


rows, views = [], []
for ds in DS:
    T = raw_table(ds)
    lob = T[T.roi.str.contains(r"^(?:MB_VL|MB_ML|aL\(|a'L\(|bL\(|b'L\(|gL\()", regex=True)].A.mean()
    if lob < T[T.in_ca].A.mean(): T['A'] = -T.A                      # anterior = towards the lobes
    for s_ in ['L', 'R']:
        t = T[T.side == s_]
        if t.empty: continue
        X = t[['R', 'D', 'A']].values.copy()
        if s_ == 'L': X[:, 0] = -X[:, 0]                               # mirror left: lateral always +R
        ca = t.in_ca.values
        d, _ = cKDTree(X[ca]).query(X[~t.in_mb.values], k=1)
        off = np.zeros(len(t), bool); off[np.flatnonzero(~t.in_mb.values)[d < 8]] = True
        off &= t.pclass.isin(['KC', 'PN']).values
        c0 = X[ca].mean(0)
        C, N = X[ca] - c0, X[off] - c0
        F = np.vstack([C, N])                                         # calyx incl. the clipped part
        def pca(Y):
            w, V = np.linalg.eigh(np.cov(Y.T)); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
            V[:, 0] *= np.sign(V[0, 0]); V[:, 1] *= np.sign(V[1, 1]); V[:, 2] = np.cross(V[:, 0], V[:, 1])
            return np.sqrt(w), V
        sd, V = pca(C); sdF, VF = pca(F)
        thin = V[:, 2] * np.sign(V[2, 2])
        pn_in = int(((t.pclass == 'PN') & t.in_ca).sum()); pn_off = int(((t.pclass == 'PN').values & off).sum())
        kc_in = int(((t.pclass == 'KC') & t.in_ca).sum()); kc_off = int(((t.pclass == 'KC').values & off).sum())
        rows.append({'calyx': f'{LABEL[ds]} {s_}', 'posterior_view_width_um_SD': C[:, 0].std(), 'posterior_view_height_um_SD': C[:, 1].std(),
                     'posterior_view_aspect': C[:, 0].std() / C[:, 1].std(),
                     'tilt_of_calyx_axis_from_AP_deg': np.degrees(np.arccos(abs(thin[2]))),
                     'tilt_pitch_deg': np.degrees(np.arctan2(-thin[1], thin[2])), 'tilt_yaw_deg': np.degrees(np.arctan2(thin[0], thin[2])),
                     'faceon_aspect_PC1_PC2': sd[0] / sd[1], 'PC_SD_um': [round(v, 2) for v in sd], 'size_geomean_SD_um': float(np.prod(sd) ** (1 / 3)),
                     'faceon_aspect_incl_clipped': sdF[0] / sdF[1], 'size_geomean_incl_clipped': float(np.prod(sdF) ** (1 / 3)),
                     'posterior_view_aspect_incl_clipped': F[:, 0].std() / F[:, 1].std(),
                     'PN_APL_in_CA_ROI': pn_in, 'PN_APL_clipped': pn_off, 'pct_PN_APL_clipped': 100 * pn_off / (pn_in + pn_off),
                     'KC_APL_in_CA_ROI': kc_in, 'KC_APL_clipped': kc_off, 'pct_KC_APL_clipped': 100 * kc_off / (kc_in + kc_off),
                     'clipped_labels': pd.Series(t.roi.values[off]).value_counts().head(3).to_dict()})
        views.append((f'{LABEL[ds]} · {"left" if s_ == "L" else "right"}', C, N, V, rows[-1]))
R = pd.DataFrame(rows).set_index('calyx')
R.round(3).to_csv(OUT + 'calyx_shape_by_dataset.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(R.drop(columns=['PC_SD_um', 'clipped_labels']).round(2).T.to_string())
print(R.clipped_labels.to_string())

# ---------------- figure ----------------
rng = np.random.default_rng(0)
n = len(views)
fig, ax = plt.subplots(3, n, figsize=(2.15 * n + .9, 7.7), squeeze=False)
LIM = 42
for j, (lab, C, N, V, r) in enumerate(views):
    ci = rng.choice(len(C), min(len(C), 9000), replace=False)
    ni = rng.choice(len(N), min(len(N), 4000), replace=False) if len(N) else np.array([], int)
    for i in range(3):
        a = ax[i, j]
        if i == 0:   proj = lambda Y: (Y[:, 0], Y[:, 1])                  # posterior view as in the maps (lateral right, dorsal up)
        elif i == 1: proj = lambda Y: (-Y[:, 2], Y[:, 1])                 # side view from lateral: anterior left, dorsal up
        else:        proj = lambda Y: (Y @ V[:, 0], Y @ V[:, 1])          # face-on: looking down the calyx's own thin axis
        a.scatter(*proj(C[ci]), s=.35, color=IN_ROI, lw=0, rasterized=True)
        if len(ni): a.scatter(*proj(N[ni]), s=.5, color=OFF_ROI, lw=0, rasterized=True)
        a.set_xlim(-LIM, LIM); a.set_ylim(-LIM * .78, LIM * .78); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color('#e1e0d9')
        if i == 0:
            a.set_title(f'{lab}\nwidth/height {r["posterior_view_aspect"]:.2f}', fontsize=8, color=INK, loc='left')
        if i == 1:
            th = np.radians(r['tilt_pitch_deg'])                          # draw the calyx axis and the viewing axis
            a.annotate('', xy=(30, 0), xytext=(-30, 0), arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8, ls=(0, (3, 2))))
            a.plot([-30 * np.cos(th), 30 * np.cos(th)], [-30 * np.sin(th), 30 * np.sin(th)], color=INK, lw=1)
            a.set_xlabel(f'calyx axis tilted {r["tilt_of_calyx_axis_from_AP_deg"]:.0f}° from the line of sight', fontsize=7, color=INK, labelpad=2)
        if i == 2:
            a.set_xlabel(f'width/height {r["faceon_aspect_PC1_PC2"]:.2f} · size {r["size_geomean_SD_um"]:.1f} µm', fontsize=7, color=INK, labelpad=2)
        if j == 0:
            a.set_ylabel(['posterior view\n(as in the maps)', 'side view\n(anterior left)', 'face-on view\n(down the calyx axis)'][i],
                         fontsize=8.5, color=INK, rotation=0, ha='right', va='center', labelpad=6)
        if i == 0 and j == 0:
            a.plot([-LIM + 3, -LIM + 23], [-LIM * .78 + 3] * 2, color=INK, lw=1.2); a.text(-LIM + 13, -LIM * .78 + 4.5, '20 µm', fontsize=6, ha='center')
h = [Line2D([], [], marker='o', ls='', color=IN_ROI, markersize=4, label="APL synapses inside the dataset's calyx ROI (what the maps show)"),
     Line2D([], [], marker='o', ls='', color=OFF_ROI, markersize=4, label='APL↔KC / APL↔PN synapses within 8 µm of the calyx but outside every MB ROI'),
     Line2D([], [], color=INK, lw=1, label='calyx axis (thinnest principal axis)'),
     Line2D([], [], color=MUTED, lw=.8, ls=(0, (3, 2)), label='posterior-view line of sight'),
     Line2D([], [], ls='', label='size = geometric mean of the SDs along the 3 principal axes')]
fig.legend(handles=h, loc='lower left', ncol=2, frameon=False, fontsize=7.5, labelcolor=MUTED, bbox_to_anchor=(.06, -.005))
fig.suptitle('Calyx shape per connectome: posterior view as drawn in the maps, side view, and face-on view (left calyces mirrored; lateral to the right)',
             fontsize=10, color=INK, x=.01, ha='left')
fig.tight_layout(rect=(0, .05, 1, .97)); fig.savefig(OUT + 'calyx_shape_by_dataset.png', dpi=170); plt.close(fig)
