"""Is the male CNS calyx ROI misaligned? (Yang, 2026-11-03: about a third of left KC<->APL calyx synapses fall outside CA(L).)

Calyx synapses = the corrected calyx of with_outside_ROI_synapses/: ALPN->KC synapses (postsynaptic site) assigned to CA(L)/CA(R),
and APL<->KC calyx synapses of APL_L/APL_R (postsynaptic site, the point the release labels), inside the official ROI or added.
ROI = segment CA(L)/CA(R) of gs://flyem-male-cns/rois/fullbrain-roi-v4 at 256 nm (it reproduces the release's synapse labels
exactly, calyx_roi_check/roi_volume_vs_release_labels.json; v4.1 and v5 agree).
Tests per side:
  coverage            share of calyx synapses inside the ROI
  shift               the translation of the ROI (grid +-14 µm, 1 µm, then 0.25 µm around the best) that covers most synapses:
                      a misaligned ROI would reach ~100% at some shift without losing synapses elsewhere
  rotation+shift      the same with small rotations of the ROI about the calyx centre (+-8 degrees around each axis, 2 degree steps)
  empty ROI           share of ROI volume farther than 3 µm from every calyx synapse (a shifted ROI would leave an empty band)
  where the misses are: distance of each outside synapse to the ROI surface, and their position along the calyx's own axes
Outputs: calyx_roi_check/calyx_roi_check.json, calyx_roi_check.png, calyx_roi_shift_scan.csv.
Usage: python3 scripts/calyx_roi_check.py <raw PN->KC calyx parquet>   (run from a folder with dl/ for meta_util.types)"""
import sys, json, itertools
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd
import tensorstore as ts
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from roilookup import names
from meta_util import types, kc_ids
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/mnt/project-files/apl_synapses/'; OUT = B + 'calyx_roi_check/'
VOL = 'gs://flyem-male-cns/rois/fullbrain-roi-v4/'; RES = 256.0
typ, _ = types('malecns'); KC = set(int(i) for i in kc_ids(typ))
pn = pd.read_parquet(sys.argv[1], columns=['x_post', 'y_post', 'z_post', 'body_post', 'primary_post', 'roi_assigned', 'outside_roi'])
pn = pn[pn.body_post.isin(KC)]
apl = pd.read_csv(B + 'with_outside_ROI_synapses/malecns_APL_MB_synapses.csv.gz', low_memory=False,
                  usecols=['apl_instance', 'partner_class', 'partner_type', 'mb_region', 'mb_region_source', 'post_x_nm', 'post_y_nm', 'post_z_nm', 'roi_official', 'direction'])
apl = apl[(apl.mb_region == 'CA') & apl.partner_type.astype(str).str.startswith('KC')]
nm = names(VOL); ID = {v: k for k, v in nm.items()}
store = ts.open({'driver': 'neuroglancer_precomputed', 'kvstore': VOL, 'scale_index': 0}).result()


def roi_mask(seg, P, pad_um=20):
    lo = np.floor((P.min(0) - pad_um * 1e3) / RES).astype(int); hi = np.ceil((P.max(0) + pad_um * 1e3) / RES).astype(int)
    a = store[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 0].read().result()
    return (a == ID[seg]), lo


def inside(mask, lo, P):
    v = np.floor(P / RES).astype(int) - lo
    ok = np.all((v >= 0) & (v < mask.shape), 1)
    r = np.zeros(len(P), bool); r[ok] = mask[v[ok, 0], v[ok, 1], v[ok, 2]]
    return r


res, scans, FIG = {}, [], {}
for side in ['L', 'R']:
    seg = f'CA({side})'
    q = pn[pn.roi_assigned == seg]
    Ppn = q[['x_post', 'y_post', 'z_post']].values * 8.0
    a = apl[apl.apl_instance == f'APL_{side}']
    Pap = a[['post_x_nm', 'post_y_nm', 'post_z_nm']].values.astype(float)
    P = np.r_[Ppn, Pap]
    mask, lo = roi_mask(seg, P)
    inn_pn, inn_ap = inside(mask, lo, Ppn), inside(mask, lo, Pap)
    # check: the volume reproduces the release label
    chk_pn = float((inn_pn == ~q.outside_roi.values).mean()); chk_ap = float((inn_ap == (a.mb_region_source == 'official ROI').values).mean())
    rng = np.random.default_rng(1)
    S = P[rng.choice(len(P), 30000, replace=False)]
    # ---- translation scan (shift the ROI by s == shift the points by -s)
    best = (-1, None)
    for s in itertools.product(range(-14, 15), repeat=3):
        f = inside(mask, lo, S - np.array(s) * 1e3).mean()
        scans.append(dict(side=side, dx_um=s[0], dy_um=s[1], dz_um=s[2], share_inside=round(float(f), 4)))
        if f > best[0]: best = (f, np.array(s, float))
    for s in itertools.product(np.arange(-1, 1.01, 0.25), repeat=3):
        t = best[1] + np.array(s)
        f = inside(mask, lo, S - t * 1e3).mean()
        if f > best[0]: best = (f, t)
    sh = best[1]
    cov_shift_pn = inside(mask, lo, Ppn - sh * 1e3).mean(); cov_shift_ap = inside(mask, lo, Pap - sh * 1e3).mean()
    # ---- rotation + translation (rotate the points the opposite way about the calyx centre)
    c0 = P.mean(0); bestr = (best[0], np.zeros(3), sh)
    for ang in itertools.product(range(-8, 9, 2), repeat=3):
        if not any(ang): continue
        R = Rotation.from_euler('xyz', ang, degrees=True)
        Sr = R.apply(S - c0) + c0
        for s in itertools.product(range(-6, 7, 2), repeat=3):
            t = sh + np.array(s)
            f = inside(mask, lo, Sr - t * 1e3).mean()
            if f > bestr[0]: bestr = (f, np.array(ang, float), t)
    # ---- empty ROI: ROI voxels farther than 3 µm from every calyx synapse, before and after the best shift
    tree = cKDTree(P)
    vox = np.argwhere(mask); centres = (vox + lo + 0.5) * RES
    if len(centres) > 300000: centres = centres[rng.choice(len(centres), 300000, replace=False)]
    d0 = tree.query(centres, distance_upper_bound=3000)[0]; d1 = tree.query(centres - sh * 1e3, distance_upper_bound=3000)[0]
    # ---- where the misses are: distance to the ROI (outside points), calyx axes
    dist_out = ndimage.distance_transform_edt(~mask) * RES / 1e3
    out_all = ~np.r_[inn_pn, inn_ap]
    v = np.floor(P / RES).astype(int) - lo
    dmiss = dist_out[v[out_all, 0], v[out_all, 1], v[out_all, 2]]
    evals, evecs = np.linalg.eigh(np.cov((P - c0).T)); thin = evecs[:, 0]          # thinnest axis = calyx's own front-back axis
    if thin[2] < 0: thin = -thin                                                  # point it posteriorly (+z)
    proj = (P - c0) @ thin
    res[side] = {
        'ALPN->KC calyx synapses: n, share inside the ROI': [len(Ppn), round(float(inn_pn.mean()), 4)],
        'KC<->APL calyx synapses: n, share inside the ROI': [len(Pap), round(float(inn_ap.mean()), 4)],
        'check: ROI volume lookup reproduces the release label (ALPN->KC, KC<->APL)': [round(chk_pn, 4), round(chk_ap, 4)],
        'best translation of the ROI (µm, x y z)': [round(float(x), 2) for x in sh],
        'share inside after the best translation (ALPN->KC, KC<->APL)': [round(float(cov_shift_pn), 4), round(float(cov_shift_ap), 4)],
        'best rotation (deg about x y z) + translation (µm), share inside (sample)': [list(map(float, bestr[1])), [round(float(x), 2) for x in bestr[2]], round(float(bestr[0]), 4)],
        'ROI volume (µm3), share farther than 3 µm from every calyx synapse: as is, after the best translation':
            [round(float(mask.sum() * (RES / 1e3) ** 3)), round(float(np.isinf(d0).mean()), 4), round(float(np.isinf(d1).mean()), 4)],
        'outside synapses: distance to the ROI (µm) median, 90%, max': [round(float(np.median(dmiss)), 2), round(float(np.quantile(dmiss, .9)), 2), round(float(dmiss.max()), 2)],
        'calyx front-back axis (unit vector, x y z; + = posterior)': [round(float(x), 3) for x in thin],
        'position along that axis (µm from centre): inside median, outside median, outside share in the back third':
            [round(float(np.median(proj[~out_all])) / 1e3, 1), round(float(np.median(proj[out_all])) / 1e3, 1),
             round(float((proj[out_all] > np.quantile(proj, 2 / 3)).mean()), 3)]}
    FIG[side] = dict(mask=mask, lo=lo, P=P, out=out_all, sh=sh)
    print(side, json.dumps(res[side], indent=1, ensure_ascii=False), flush=True)
pd.DataFrame(scans).to_csv(OUT + 'calyx_roi_shift_scan.csv', index=False)
json.dump(res, open(OUT + 'calyx_roi_check.json', 'w'), indent=1, ensure_ascii=False)
np.savez_compressed(OUT + 'calyx_roi_masks_256nm.npz', **{f'{s}_mask': FIG[s]['mask'] for s in FIG}, **{f'{s}_lo_voxel256': FIG[s]['lo'] for s in FIG})

# ---------------------------------------------------------------- figure: ROI silhouette vs synapses, three views per side
VIEWS = [('posterior view', (0, 1), (-1, -1), 'fly right → (µm)', 'dorsal ↑ (µm)', 2),
         ('side view', (2, 1), (-1, -1), 'anterior → (µm)', 'dorsal ↑ (µm)', 0),
         ('top view', (0, 2), (-1, -1), 'fly right → (µm)', 'anterior ↑ (µm)', 1)]
IN_C, OUT_C, ROI_F, ROI_E = '#9a9994', '#eb6834', '#dfe9f5', '#2a78d6'
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, side in enumerate(['L', 'R']):
    d = FIG[side]; m, lo, P, o = d['mask'], d['lo'], d['P'], d['out']
    rng = np.random.default_rng(2); k = rng.choice(len(P), min(25000, len(P)), replace=False)
    for j, (name, (a0, a1), (s0, s1), xl, yl, ax_proj) in enumerate(VIEWS):
        ax = axs[i, j]
        sil = m.any(axis=ax_proj)                             # projection of the ROI along the viewing axis
        keep = [ax_ for ax_ in range(3) if ax_ != ax_proj]
        gx = (np.arange(sil.shape[0]) + lo[keep[0]] + 0.5) * RES / 1e3; gy = (np.arange(sil.shape[1]) + lo[keep[1]] + 0.5) * RES / 1e3
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        if keep != [a0, a1]: sil, X, Y = sil.T, Y.T, X.T
        ax.contourf(s0 * X, s1 * Y, sil.astype(float), levels=[0.5, 1.5], colors=[ROI_F], zorder=0)
        ax.contour(s0 * X, s1 * Y, sil.astype(float), levels=[0.5], colors=[ROI_E], linewidths=1.2, zorder=3)
        Q = P[k] / 1e3; oo = o[k]
        ax.scatter(s0 * Q[~oo, a0], s1 * Q[~oo, a1], s=0.4, color=IN_C, lw=0, alpha=0.5, zorder=1, rasterized=True)
        ax.scatter(s0 * Q[oo, a0], s1 * Q[oo, a1], s=0.6, color=OUT_C, lw=0, alpha=0.7, zorder=2, rasterized=True)
        ax.set_aspect('equal'); ax.set_xlabel(xl); ax.set_ylabel(yl)
        r = res[side]
        ax.set_title(f'CA({side}), {name}' + (f'\n{100 * (1 - r["ALPN->KC calyx synapses: n, share inside the ROI"][1]):.1f}% of ALPN→KC and '
                                              f'{100 * (1 - r["KC<->APL calyx synapses: n, share inside the ROI"][1]):.1f}% of KC↔APL outside' if j == 0 else ''),
                     loc='left', fontsize=9)
h = [Line2D([], [], marker='o', ls='', color=IN_C, ms=4, label='calyx synapse inside the official ROI'),
     Line2D([], [], marker='o', ls='', color=OUT_C, ms=4, label='calyx synapse outside it (added back)'),
     Line2D([], [], color=ROI_E, lw=1.5, label='official calyx ROI (fullbrain-roi-v4), projected')]
fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT + 'calyx_roi_check.png', dpi=170, facecolor='white')
print('done')
