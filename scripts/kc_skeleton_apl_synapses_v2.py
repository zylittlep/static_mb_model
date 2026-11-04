"""KC skeletons with their KC<->APL synapses on the NEW claws (Yang, 2026-11-03: "example skeletons of KCs where APL-KC
synapses are outside of claws, especially at larger branches, e.g. the branch before it bifurcates into claws").
Version of kc_skeleton_apl_synapses.py (2026-10-28, old 2 um claws; its text below still describes that version) with:
- claws = claw_v2m2 (8 um along the KC skeleton, split by PN, >= 2 PSDs; with_outside_ROI_synapses/claws_v2m2/); a calyx
  KC<->APL synapse is on a claw when its claw_id is set there (<= 8 um along the skeleton from its claw's own-PN PSDs);
- each PN->KC PSD belongs to its own claw from the claw-site table (argument 3), not to the nearest claw centroid;
- every right-side KC with a skeleton is classified (population shares in summary.json), and the 10 KCs shown are those with the
  most calyx KC<->APL synapses on the dendritic trunk (upstream of >= 2 claws), at most 2 per KC type.
Usage: python3 kc_skeleton_apl_synapses_v2.py <male CNS SWC dir> <raw PN->KC calyx parquet> <malecns_claw_sites.parquet>
Outputs: apl_synapses/kc_skeletons/claws_v2m2/.

Old description:

Male CNS v1.0, right mushroom body (APL_R, body 10540), KCs of the right side. Ten KCs picked at random (seed 1028) among KCs
with >= 4 claws whose number of calyx KC<->APL synapses lies between the 40th and 90th percentile of their type: 3 KCg-m,
1 each of KCa'b'-ap1, KCa'b'-ap2, KCa'b'-m, 2 KCab-s, 1 KCab-c, 1 KCab-m (ids in pick.csv / the output table).
Skeletons: public male CNS SWC files (gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/<id>.swc, coarse,
8 nm voxels); passed as a directory argument. Synapses: with_outside_ROI_synapses/malecns_APL_MB_synapses.csv.gz (every KC<->APL
synapse of the official table plus those outside the official ROIs; the column mb_region_source tells which). The KC's side of
each synapse (PSD for APL->KC, T-bar for KC->APL) is placed at its nearest skeleton node.
Claws: the derived claws (malecns_KC_claws_derived.csv.gz); a calyx synapse is "on a claw" when claw_id is set (KC-side site
within 2 µm of a claw, the rule used everywhere in this project). PN->KC synapses of each KC (the claw sites) come from the raw
calyx PN->KC table (argument 2); each is assigned to the KC's nearest claw centroid.
Where off-claw calyx synapses sit on the KC: the skeleton is rooted at its most anterior node (a lobe tip), so "downstream" means
away from the axon. For the synapse's node v, n_claws(v) = number of claws with a PN->KC site at or downstream of v:
  claw stalk          n_claws = 1   (on the branch that leads to one claw)
  dendritic trunk     n_claws >= 2  (on the shared dendrite, upstream of two or more claws)
  branch without claw n_claws = 0   (a side branch with no claw beyond it, including the soma neurite)
Also the distance along the KC skeleton from each calyx synapse to the nearest PN->KC site.
Outputs (apl_synapses/kc_skeletons/): KC_skeletons_APL_synapses_overview.png/.pdf (whole KCs, two views), KC_skeletons_APL_synapses_calyx.png/.pdf
(calyx close-ups), KC_skeletons_APL_synapses_3D.html (rotatable), KC_APL_synapses_by_location.csv (per KC), KC_APL_calyx_synapses_on_skeleton.csv
(one row per calyx synapse), README.md."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra

SWC_DIR, PNCA, SITES = sys.argv[1], sys.argv[2], sys.argv[3]
B = '/mnt/project-files/apl_synapses/'
OUT = B + 'kc_skeletons/claws_v2m2/'
CV = B + 'with_outside_ROI_synapses/claws_v2m2/'
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
CAT = {'on claw': '#1f9e6e', 'claw stalk': '#a87a0c', 'dendritic trunk': '#8a5cc2', 'branch without claw': '#d1477f',
       'peduncle': '#2b84b0', 'lobes': MUTED}
plt.rcParams.update({'font.family': 'Inter', 'font.size': 8, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 8.5, 'axes.titleweight': 'semibold',
                     'axes.titlelocation': 'left', 'legend.frameon': False})

syn = pd.read_csv(CV + 'malecns_APL_MB_synapses.csv.gz', low_memory=False)
claws = pd.read_csv(CV + 'malecns_KC_claws_derived.csv.gz')
ST = pd.read_parquet(SITES, columns=['kc_id', 'x', 'y', 'z', 'is_PN', 'partner_id', 'claw_v2m2', 'calyx'])
ST = ST[ST.is_PN & (ST.calyx == 'R')]
have = {int(os.path.splitext(f)[0]) for f in os.listdir(SWC_DIR) if f.endswith('.swc')}
pick = sorted(set(claws[claws.calyx == 'R'].kc_id.astype(int)) & have)        # every right-side KC with claws and a skeleton
aplca = syn[(syn.apl_instance == 'APL_R') & (syn.mb_region == 'CA')]
syn = syn[(syn.apl_instance == 'APL_R') & syn.partner_id.isin(pick)].copy()
pnall = pd.read_parquet(PNCA, columns=['x_post', 'y_post', 'z_post', 'body_post', 'body_pre', 'primary_post', 'roi_assigned'])
# the right calyx as corrected in with_outside_ROI_synapses/: every PN->KC synapse assigned to CA(R) plus every calyx synapse of APL_R
# (KC side for KC partners, partner side otherwise), official ROI or added. CAL_OFF marks those inside the official CA(R) ROI.
cr = pnall[pnall.roi_assigned == 'CA(R)']
CAL = np.r_[cr[['x_post', 'y_post', 'z_post']].values * 8 / 1e3, aplca[['partner_side_x_nm', 'partner_side_y_nm', 'partner_side_z_nm']].values / 1e3]
CAL_OFF = np.r_[(cr.primary_post == 'CA(R)').values, (aplca.mb_region_source == 'official ROI').values]
CAL_TREE = cKDTree(CAL)
IN_CAL_UM = 3.0     # a skeleton node belongs to the calyx when a calyx synapse (any cell) lies within 3 µm
del pnall
TYPE = syn.groupby('partner_id').partner_type.first()
TORD = ['KCg-m', 'KCg-d', "KCa'b'-ap1", "KCa'b'-ap2", "KCa'b'-m", 'KCab-s', 'KCab-c', 'KCab-m', 'KCab-p']
pick = [k for k in pick if k in TYPE.index]
ORDER = sorted(pick, key=lambda k: (TORD.index(TYPE[k]) if TYPE[k] in TORD else 99, k))


def load_swc(path):
    s = pd.read_csv(path, comment='#', sep=r'\s+', header=None, names=['id', 't', 'x', 'y', 'z', 'r', 'p'])
    xyz = s[['x', 'y', 'z']].values * 8 / 1e3                       # µm
    pos = pd.Series(np.arange(len(s)), index=s.id.values)
    ch = np.where(s.p.values != -1)[0]; par = pos.reindex(s.p.values[ch]).values.astype(int)
    w = np.linalg.norm(xyz[ch] - xyz[par], axis=1)
    G = coo_matrix((w, (ch, par)), shape=(len(s), len(s))).tocsr()
    return xyz, G + G.T, ch, par


def category(r):
    if r.mb_region == 'PED': return 'peduncle'
    if r.mb_region != 'CA': return 'lobes'
    if pd.notna(r.claw_id): return 'on claw'
    return {0: 'branch without claw', 1: 'claw stalk'}.get(min(int(r.n_claws_below), 2), 'dendritic trunk')


rows_syn, rows_kc, SK = [], [], {}
for kid in ORDER:
    xyz, G, ch, par = load_swc(os.path.join(SWC_DIR, f'{kid}.swc'))
    tree = cKDTree(xyz)
    s = syn[syn.partner_id == kid].copy()
    P = s[['partner_side_x_nm', 'partner_side_y_nm', 'partner_side_z_nm']].values / 1e3
    s['snap_um'], s['node'] = tree.query(P)
    # claws: each PN->KC PSD of the KC with its claw from the claw-site table (-1 = not on a claw)
    c = claws[claws.kc_id == kid].reset_index(drop=True)
    st = ST[ST.kc_id == kid]
    q = st[['x', 'y', 'z']].values / 1e3
    cl = st.claw_v2m2.values
    pnode = tree.query(q)[1]
    # root at the most anterior node (lowest z), count claws at or downstream of each node
    root = int(np.argmin(xyz[:, 2]))
    order, pred = breadth_first_order(G, root, directed=False, return_predecessors=True)
    below = np.zeros((len(xyz), max(int(cl.max()) + 1 if len(cl) else 1, 1)), bool)
    below[pnode[cl >= 0], cl[cl >= 0]] = True
    for v in order[::-1]:
        p = pred[v]
        if p >= 0: below[p] |= below[v]
    nclaw = below.sum(1)
    s['n_claws_below'] = nclaw[s.node.values]
    # distance along the KC to the nearest PN->KC site, for calyx synapses
    ca = (s.mb_region == 'CA').values
    if ca.any() and len(pnode):
        D = dijkstra(G, directed=False, indices=np.unique(pnode), min_only=True)
        s['dist_along_KC_to_PN_site_um'] = np.where(ca, D[s.node.values], np.nan)
    else:
        s['dist_along_KC_to_PN_site_um'] = np.nan
    s['location'] = [category(r) for r in s.itertuples()]
    s['direction_short'] = np.where(s.direction.str.startswith('APL_output'), 'APL→KC', 'KC→APL')
    s['added_outside_official_ROI'] = s.mb_region_source != 'official ROI'
    incal = CAL_TREE.query(xyz, distance_upper_bound=IN_CAL_UM)[0] <= IN_CAL_UM
    # keep only pieces of that calyx part that carry this KC's own calyx synapses (drops bits of the soma neurite passing by)
    from scipy.sparse.csgraph import connected_components
    e = incal[ch] & incal[par]
    ncomp, lab = connected_components(coo_matrix((np.ones(e.sum()), (ch[e], par[e])), shape=(len(xyz), len(xyz))), directed=False)
    own = np.unique(lab[np.r_[pnode, s.node.values[(s.mb_region == 'CA').values]]])
    incal &= np.isin(lab, own)
    SK[kid] = dict(xyz=xyz, ch=ch, par=par, s=s, pn=q, pcl=cl, claws=c, nclaw=nclaw, incal=incal,
                   cal_len=float(np.linalg.norm(xyz[ch] - xyz[par], axis=1)[incal[ch] & incal[par]].sum()))
    rows_syn.append(s[s.mb_region == 'CA'].assign(kc_type=TYPE[kid])[[
        'partner_id', 'kc_type', 'syn_id', 'direction_short', 'location', 'claw_id', 'claw_dist_nm', 'n_claws_below',
        'dist_along_KC_to_PN_site_um', 'snap_um', 'added_outside_official_ROI', 'partner_side_x_nm', 'partner_side_y_nm', 'partner_side_z_nm']])
    r = dict(kc_id=kid, kc_type=TYPE[kid], claws=len(c), PN_to_KC_synapses=len(q), skeleton_length_um=round(float(np.linalg.norm(xyz[ch] - xyz[par], axis=1).sum()), 1),
             calyx_skeleton_length_um=round(SK[kid]['cal_len'], 1))
    for d_ in ['KC→APL', 'APL→KC']:
        for loc in CAT:
            r[f'{d_} {loc}'] = int(((s.direction_short == d_) & (s.location == loc)).sum())
    cas = s[s.mb_region == 'CA']
    r['calyx synapses on claws (%)'] = round(100 * (cas.location == 'on claw').mean(), 1) if len(cas) else np.nan
    r['all synapses, official ROIs only'] = int((~s.added_outside_official_ROI).sum()); r['all synapses, with outside-ROI'] = len(s)
    r['median snap to skeleton (µm)'] = round(float(s.snap_um.median()), 2)
    rows_kc.append(r)
    r['calyx trunk synapses'] = int((cas.location == 'dendritic trunk').sum())
    if len(rows_kc) % 200 == 0: print(len(rows_kc), 'KCs', flush=True)
KC = pd.DataFrame(rows_kc); CS = pd.concat(rows_syn, ignore_index=True)
KC.to_csv(OUT + 'all_right_KCs_APL_synapses_by_location.csv', index=False)
CS.round(3).to_csv(OUT + 'all_right_KCs_APL_calyx_synapses_on_skeleton.csv', index=False)
POP = {'KCs': int(len(KC)), 'calyx KC<->APL synapses': int(len(CS)),
       'share by location (calyx)': CS.location.value_counts(normalize=True).round(3).to_dict(),
       'share by location, KC->APL': CS[CS.direction_short == 'KC→APL'].location.value_counts(normalize=True).round(3).to_dict(),
       'share by location, APL->KC': CS[CS.direction_short == 'APL→KC'].location.value_counts(normalize=True).round(3).to_dict(),
       'off-claw share by location, per KC type': (CS[CS.location != 'on claw'].groupby('kc_type').location.value_counts(normalize=True)
                                                   .round(3).unstack(fill_value=0).to_dict('index')),
       'off-claw: distance along the KC to the nearest PN->KC site (um), median / 90%':
           [round(float(CS[CS.location != 'on claw'].dist_along_KC_to_PN_site_um.median()), 1),
            round(float(CS[CS.location != 'on claw'].dist_along_KC_to_PN_site_um.quantile(.9)), 1)]}
# the 10 KCs shown: most calyx synapses on the dendritic trunk, at most 2 per type
ranked = KC.sort_values(['calyx trunk synapses', 'kc_id'], ascending=[False, True])
chosen, per = [], {}
for r in ranked.itertuples():
    if per.get(r.kc_type, 0) >= 2: continue
    chosen.append(r.kc_id); per[r.kc_type] = per.get(r.kc_type, 0) + 1
    if len(chosen) == 10: break
ORDER = sorted(chosen, key=lambda k: (TORD.index(TYPE[k]) if TYPE[k] in TORD else 99, k))
KC = KC[KC.kc_id.isin(ORDER)].reset_index(drop=True); CS = CS[CS.partner_id.isin(ORDER)]
KC.to_csv(OUT + 'KC_APL_synapses_by_location.csv', index=False)
CS.round(3).to_csv(OUT + 'KC_APL_calyx_synapses_on_skeleton.csv', index=False)

# ---------------------------------------------------------------- views (male CNS: x grows medially on the right side, y ventrally, z posteriorly)
POST = lambda P: (-P[:, 0], -P[:, 1])            # posterior view: fly right to the right... as Report 10
SIDE = lambda P: (-P[:, 2], -P[:, 1])            # lateral view from the fly's right: anterior to the right
MARK = {'KC→APL': dict(marker='o'), 'APL→KC': dict(marker='^')}


from scipy import ndimage
CAL_FILL, CAL_ADDED, CAL_EDGE = '#efeee9', '#f9d3bd', '#b5b4ae'   # inside the official ROI / only in the corrected calyx / outline


def _area(P, view, xe, ye):
    X, Y = view(P)
    m = np.histogram2d(X, Y, [xe, ye])[0] > 0
    r = 4; yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, structure=np.hypot(xx, yy) <= r))
    lab, n = ndimage.label(m)
    if n > 1: m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndimage.gaussian_filter(m.astype(float), 1.5).T


AREA = {}


def calyx_area(ax, view):
    """Projected area of the corrected right calyx (pale orange where only the added synapses reach, pale gray inside the official CA(R) ROI)."""
    key = id(view)
    if key not in AREA:
        X, Y = view(CAL); b = 0.5
        xe = np.arange(X.min() - 6, X.max() + 6, b); ye = np.arange(Y.min() - 6, Y.max() + 6, b)
        AREA[key] = ((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, _area(CAL, view, xe, ye), _area(CAL[CAL_OFF], view, xe, ye))
    xc, yc, a_all, a_off = AREA[key]
    ax.contourf(xc, yc, a_all, levels=[0.5, 9], colors=[CAL_ADDED], zorder=0)
    ax.contourf(xc, yc, a_off, levels=[0.5, 9], colors=[CAL_FILL], zorder=0.1)
    ax.contour(xc, yc, a_all, levels=[0.5], colors=[CAL_EDGE], linewidths=0.6, zorder=0.2)


def calyx_limits(view, pad=4):
    X, Y = view(CAL)
    return (np.quantile(X, 0.0005) - pad, np.quantile(X, 0.9995) + pad), (np.quantile(Y, 0.0005) - pad, np.quantile(Y, 0.9995) + pad)


def draw(ax, kid, view, zoom=False):
    """zoom=True: calyx part of the KC only (nodes within IN_CAL_UM of the calyx), on the calyx area, calyx synapses only."""
    d = SK[kid]; xyz, s = d['xyz'], d['s']
    X, Y = view(xyz)
    calyx_area(ax, view)
    keep = (d['incal'][d['ch']] & d['incal'][d['par']]) if zoom else np.ones(len(d['ch']), bool)
    segs = np.stack([np.c_[X[d['ch']], Y[d['ch']]], np.c_[X[d['par']], Y[d['par']]]], 1)[keep]
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors='#9b9a94', linewidths=0.7, zorder=1))
    if zoom:
        px, py = view(d['pn']); ax.scatter(px, py, s=3, color='#c9c8c2', lw=0, zorder=2)
        c = d['claws']
        if len(c):
            cx, cy = view(c[['centroid_x_nm', 'centroid_y_nm', 'centroid_z_nm']].values / 1e3)
            ax.scatter(cx, cy, s=260, facecolor='none', edgecolor='#b5b4ae', lw=0.8, ls='--', zorder=2)
    show = s if not zoom else s[s.mb_region == 'CA']
    Q = show[['partner_side_x_nm', 'partner_side_y_nm', 'partner_side_z_nm']].values / 1e3
    sx, sy = view(Q) if len(Q) else ([], [])
    for i, r in enumerate(show.itertuples()):
        ax.scatter(sx[i], sy[i], s=(26 if zoom else 11), color=CAT[r.location], edgecolor=(INK if r.added_outside_official_ROI else 'white'),
                   lw=(0.9 if r.added_outside_official_ROI else 0.3), zorder=4, **MARK[r.direction_short])
    if zoom:
        xl, yl = calyx_limits(view); ax.set_xlim(*xl); ax.set_ylim(*yl)
    else:
        (cx0, cx1), (cy0, cy1) = calyx_limits(view, pad=2)     # whole KC and the whole calyx area
        ax.set_xlim(min(X.min(), cx0) - 5, max(X.max(), cx1) + 5); ax.set_ylim(min(Y.min(), cy0) - 5, max(Y.max(), cy1) + 5)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6.5)


def legend_handles(zoom):
    h = [Line2D([], [], marker='s', ls='', color=c, ms=7, label=l) for l, c in CAT.items() if not (zoom and l in ('peduncle', 'lobes'))]
    h += [Line2D([], [], marker='o', ls='', mfc='white', mec=INK, ms=6, label='KC → APL (KC presynaptic)'),
          Line2D([], [], marker='^', ls='', mfc='white', mec=INK, ms=6, label='APL → KC (APL presynaptic)'),
          Line2D([], [], marker='o', ls='', mfc=GRID, mec=INK, mew=1.2, ms=6, label='black outline: outside the official ROIs (added)')]
    if zoom:
        h += [Line2D([], [], marker='o', ls='', color='#c9c8c2', ms=3, label='PN → KC synapse (claw input)'),
              Line2D([], [], marker='o', ls='', mfc='none', mec='#b5b4ae', ms=12, label='claw centroid')]
    from matplotlib.patches import Patch
    h += [Patch(facecolor=CAL_FILL, edgecolor=CAL_EDGE, lw=0.6, label='calyx, inside the official CA(R) ROI'),
          Patch(facecolor=CAL_ADDED, edgecolor=CAL_EDGE, lw=0.6, label='calyx, only in the corrected calyx (outside the official ROI)')]
    return h


def title(kid):
    r = KC[KC.kc_id == kid].iloc[0]
    return f'{TYPE[kid]}  {kid}  ({r.claws} claws; calyx on claws {r["calyx synapses on claws (%)"]:.0f}%)'


# overview: 10 KCs x 2 views
fig, axs = plt.subplots(5, 4, figsize=(15, 19.5))
for i, kid in enumerate(ORDER):
    a1, a2 = axs[i // 2, 2 * (i % 2)], axs[i // 2, 2 * (i % 2) + 1]
    draw(a1, kid, POST); draw(a2, kid, SIDE)
    a1.set_title(title(kid)); a2.set_title('same KC, side view', color=MUTED, fontweight='normal')
    a1.set_xlabel('fly right → (µm)'); a1.set_ylabel('dorsal ↑ (µm)'); a2.set_xlabel('anterior → (µm)')
fig.legend(handles=legend_handles(False), loc='upper center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, 1.0))
fig.suptitle('Ten male CNS KCs (right mushroom body) with the most calyx KC↔APL synapses on the dendritic trunk, new claws (8 µm along the KC, split by PN)', y=1.02, fontsize=11, fontweight='semibold', x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.975))
for ext in ('png', 'pdf'): fig.savefig(OUT + f'KC_skeletons_APL_synapses_overview.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)

# calyx only: the calyx part of each KC on the corrected calyx, posterior and side view, same limits in every panel
(pxl, pyl), (sxl, syl) = calyx_limits(POST), calyx_limits(SIDE)
wr = [pxl[1] - pxl[0], sxl[1] - sxl[0]] * 2
fig, axs = plt.subplots(5, 4, figsize=(15, 19), gridspec_kw=dict(width_ratios=wr))
for i, kid in enumerate(ORDER):
    a1, a2 = axs[i // 2, 2 * (i % 2)], axs[i // 2, 2 * (i % 2) + 1]
    draw(a1, kid, POST, zoom=True); draw(a2, kid, SIDE, zoom=True)
    a1.set_title(title(kid).replace('  (', '\n(')); a2.set_title('same KC, side view\n(seen from the fly\'s right)', color=MUTED, fontweight='normal')
    a1.set_xlabel('fly right → (µm)'); a1.set_ylabel('dorsal ↑ (µm)'); a2.set_xlabel('anterior → (µm)')
fig.legend(handles=legend_handles(True), loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.0))
fig.suptitle('Calyx part of ten male CNS KCs (right calyx) with their calyx KC↔APL synapses; new claws (8 µm along the KC, split by PN)', y=1.025, fontsize=11, fontweight='semibold', x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.945))
for ext in ('png', 'pdf'): fig.savefig(OUT + f'KC_skeletons_APL_synapses_calyx.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)

# 3D viewer
import plotly.graph_objects as go
traces, vis = [], []
# the corrected right calyx as a faint cloud (random subsample), always shown
rng = np.random.default_rng(0)
smp = rng.choice(len(CAL), 12000, replace=False)       # one uniform sample, so the two parts keep their true proportions
for lab, m, col in [('calyx, inside the official CA(R) ROI', CAL_OFF, '#bdbcb6'),
                    ('calyx, only in the corrected calyx (outside the official ROI)', ~CAL_OFF, '#f08a5d')]:
    P = CAL[smp[m[smp]]]
    traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers', marker=dict(size=1.3, color=col, opacity=0.35),
                               name=lab, hoverinfo='skip', visible=True))
NCLOUD = len(traces)
for i, kid in enumerate(ORDER):
    d = SK[kid]; xyz = d['xyz']; idx = []
    lx, ly, lz = [], [], []
    for a, b in zip(d['ch'], d['par']):
        if not (d['incal'][a] and d['incal'][b]): continue
        lx += [xyz[a, 0], xyz[b, 0], None]; ly += [xyz[a, 1], xyz[b, 1], None]; lz += [xyz[a, 2], xyz[b, 2], None]
    traces.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='#8f8e88', width=3), name=f'{TYPE[kid]} {kid}, calyx part',
                               hoverinfo='skip', visible=(i == 0))); idx.append(len(traces) - 1)
    traces.append(go.Scatter3d(x=d['pn'][:, 0], y=d['pn'][:, 1], z=d['pn'][:, 2], mode='markers', marker=dict(size=1.6, color='#c9c8c2'),
                               name='PN → KC synapses (claws)', hoverinfo='skip', visible=(i == 0))); idx.append(len(traces) - 1)
    s = d['s'][d['s'].mb_region == 'CA']
    for loc, col in CAT.items():
        for dr, sym in [('KC→APL', 'circle'), ('APL→KC', 'diamond')]:
            q = s[(s.location == loc) & (s.direction_short == dr)]
            if not len(q): continue
            P = q[['partner_side_x_nm', 'partner_side_y_nm', 'partner_side_z_nm']].values / 1e3
            txt = [f'{dr}, {loc}<br>region {r.mb_region}' + (f', claw distance {r.claw_dist_nm / 1e3:.1f} µm' if pd.notna(r.claw_dist_nm) else '')
                   + ('<br>outside the official ROIs (added)' if r.added_outside_official_ROI else '') for r in q.itertuples()]
            traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers', name=f'{loc}, {dr}', text=txt, hoverinfo='text',
                                       marker=dict(size=4.5, color=col, symbol=sym, line=dict(color='white', width=0.5)), visible=(i == 0)))
            idx.append(len(traces) - 1)
    vis.append(idx)
buttons = []
for i, kid in enumerate(ORDER):
    v = [True] * NCLOUD + [False] * (len(traces) - NCLOUD)
    for j in vis[i]: v[j] = True
    buttons.append(dict(label=f'{TYPE[kid]} {kid}', method='update', args=[{'visible': v}, {'title.text': f'{title(kid)}: drag to rotate, scroll to zoom'}]))
fig3 = go.Figure(traces)
fig3.update_layout(title=dict(text=f'{title(ORDER[0])}: drag to rotate, scroll to zoom', x=0.17, xanchor='left'), template='plotly_white', height=760,
                   updatemenus=[dict(buttons=buttons, direction='down', x=0.0, y=1.0, xanchor='left', yanchor='bottom', showactive=True)],
                   scene=dict(aspectmode='data', xaxis_title='x (µm, toward midline)', yaxis_title='y (µm, ventral)', zaxis_title='z (µm, posterior)',
                              camera=dict(eye=dict(x=0.2, y=-0.4, z=2.0), up=dict(x=0, y=-1, z=0))),
                   legend=dict(itemsizing='constant', font=dict(size=11)), margin=dict(l=10, r=0, t=80, b=40),
                   annotations=[dict(text='Calyx part of the KC only. Hover a synapse for its location and whether it lies outside the official ROIs (added).', x=0, y=0, xref='paper', yref='paper', xanchor='left', yanchor='top', showarrow=False, font=dict(size=11, color='#6b6a65'))])
fig3.write_html(OUT + 'KC_skeletons_APL_synapses_3D.html', include_plotlyjs=True, full_html=True)

# summary numbers for the README
cs = CS
summ = {'all right KCs': POP, 'KCs shown': len(KC), 'calyx KC<->APL synapses': len(cs),
        'share by location (calyx)': cs.location.value_counts(normalize=True).round(3).to_dict(),
        'share by location, KC->APL': cs[cs.direction_short == 'KC→APL'].location.value_counts(normalize=True).round(3).to_dict(),
        'share by location, APL->KC': cs[cs.direction_short == 'APL→KC'].location.value_counts(normalize=True).round(3).to_dict(),
        'off-claw: distance along the KC to the nearest PN->KC site (µm), median / 90%': [round(float(cs[cs.location != 'on claw'].dist_along_KC_to_PN_site_um.median()), 1),
                                                                                       round(float(cs[cs.location != 'on claw'].dist_along_KC_to_PN_site_um.quantile(.9)), 1)],
        'all KC<->APL synapses by region': pd.concat([SK[k]['s'] for k in ORDER]).location.value_counts().to_dict()}
json.dump(summ, open(OUT + 'summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False))
