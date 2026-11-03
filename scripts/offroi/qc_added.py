"""Where the outside-ROI version adds synapses, and how many (offroi/assign.py output + the rebuilt tables).
Writes to apl_synapses/with_outside_ROI_synapses/:
  where_synapses_were_added.png  APL synapses per dataset: posterior view of both MBs and a side view of each MB;
                                 grey = inside the official MB ROIs, colored = added (by the MB region they join),
                                 small dark dots = outside every MB ROI and not added (more than 3 µm from the MB anchors, or in
                                 the accessory calyces)
  added_synapses_summary.csv     official-ROI vs added counts: APL synapses in the delivered tables (per APL side, MB region,
                                 direction, partner group), ALPN -> KC synapses in the calyx, KC -> MBON synapses in the lobes
Run from the working folder after build_variant.sh (needs out/outside_roi/final_<ds>_apl_mb.parquet)."""
import sys, re
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from meta_util import types

A = 'out/outside_roi/'
OUT = '/mnt/project-files/apl_synapses/with_outside_ROI_synapses/'
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
PFX = 720575940000000000
FLYEM_RX = r"^(CA|PED|aL|a'L|bL|b'L|gL)\(([LR])\)$"
FW_RX = r'^MB_(CA|PED|VL|ML)_([LR])$'
COARSE = {'CA': 'calyx', 'PED': 'peduncle', 'aL': 'vertical lobe', "a'L": 'vertical lobe', 'VL': 'vertical lobe',
          'bL': 'medial lobe', "b'L": 'medial lobe', 'gL': 'medial lobe', 'ML': 'medial lobe'}
COL = {'calyx': '#2a78d6', 'peduncle': '#9467bd', 'vertical lobe': '#1baf7a', 'medial lobe': '#eb6834'}
INK, MUTED, OFFICIAL, NOTADDED = '#0b0b0b', '#52514e', '#cfcec8', '#3d3c39'
# posterior view (fly's right on the right, dorsal up) and side view (anterior-posterior, dorsal up), native axes in µm
VIEW = {'flywire': lambda x, y, z: (x, -y), 'hemibrain': lambda x, y, z: (-x, -z),
        'malecns': lambda x, y, z: (-x, -y), 'banc': lambda x, y, z: (-x, z)}
SIDE = {'flywire': lambda x, y, z: (z, -y), 'hemibrain': lambda x, y, z: (y, -z),
        'malecns': lambda x, y, z: (z, -y), 'banc': lambda x, y, z: (y, z)}
APL_SIDE = {'flywire': {'720575940624547622': 'L', '720575940613583001': 'R'}, 'hemibrain': {'425790257': 'R'},
            'malecns': {'10977': 'L', '10540': 'R'}, 'banc': {'720575941734526507': 'L', '720575941482622627': 'R'}}


def raw_apl(ds):
    """All raw APL synapses (postsynaptic site in µm), APL side, MB label from assign.py and its source."""
    if ds == 'flywire':
        r = pd.read_parquet('out/fafbP_apl_raw.parquet')
        pre = (r.pre_root_id_720575940.astype('int64') + PFX).astype(str); post = (r.post_root_id_720575940.astype('int64') + PFX).astype(str)
        X = r[['post_x', 'post_y', 'post_z']].values.astype(float) / 1e3
    elif ds == 'banc':
        r = pd.read_parquet('out/banc_v2_apl_raw.parquet')
        pre, post = r.pre_root_id.astype(str), r.post_root_id.astype(str)
        X = r[['post_x', 'post_y', 'post_z']].values.astype(float) / 1e3
    else:
        r = pd.read_parquet(f'out/{ds}_apl_raw.parquet')
        pre, post = r.body_pre.astype('int64').astype(str), r.body_post.astype('int64').astype(str)
        X = r[['x_post', 'y_post', 'z_post']].values.astype(float) * 8 / 1e3
    g = pd.read_parquet(A + f'{ds}_apl_assigned.parquet')
    sd = APL_SIDE[ds]
    t = pd.DataFrame({'apl_side': pre.map(sd).fillna(post.map(sd)).values, 'src': g.mb_region_source.values, 'lab': g.roi_assigned.values})
    m = t.lab.astype(str).str.extract(FLYEM_RX if ds in ('hemibrain', 'malecns') else FW_RX)
    t['region'] = m[0].map(COARSE).values; t['mb_side'] = m[1].values
    t[['x', 'y', 'z']] = X
    keep = (pre.values != post.values) & t.apl_side.notna().values
    return t[keep].reset_index(drop=True)


# ---------------- figure ----------------
fig, ax = plt.subplots(4, 3, figsize=(16.5, 19), gridspec_kw={'width_ratios': [1.25, 1, 1]})
rng = np.random.default_rng(0)
for i, ds in enumerate(DS):
    t = raw_apl(ds)
    off = t[t.src == 'official ROI']; add = t[t.src.str.startswith('added')]; nadd = t[t.src == '']
    for j, (view, sel) in enumerate([('posterior', None), ('side', 'L'), ('side', 'R')]):
        a_ = ax[i, j]
        f = VIEW[ds] if view == 'posterior' else SIDE[ds]
        parts = []
        for k, (d, kind) in enumerate([(off, 'official'), (add, 'added'), (nadd, 'not added')]):
            if sel is not None:
                d = d[(d.mb_side == sel) if kind != 'not added' else (d.apl_side == sel)]
            u, v = f(d.x.values, d.y.values, d.z.values)
            parts.append((d, np.asarray(u), np.asarray(v), kind))
        po = parts[0]
        if len(po[0]) < 5000:
            a_.text(.03, .5, f'{LABEL[ds]} · {"left" if sel == "L" else "right"} MB\nonly the lobe tips are in this volume', transform=a_.transAxes,
                    ha='left', va='center', color=MUTED, fontsize=10)
            a_.set_axis_off(); continue
        # side view: posterior (the calyx) on the right
        flip = 1.0
        if view == 'side':
            ca = po[0].region.values == 'calyx'
            if ca.any() and (~ca).any() and po[1][ca].mean() < po[1][~ca].mean():
                flip = -1.0
        lo_u, hi_u = np.percentile(po[1] * flip, [0.05, 99.95]); lo_v, hi_v = np.percentile(po[2], [0.05, 99.95])
        pad = 8
        for d, u, v, kind in parts:
            u = u * flip
            if kind == 'official':
                n = len(u); s = rng.choice(n, min(n, 120000), replace=False)
                a_.scatter(u[s], v[s], s=.25, color=OFFICIAL, alpha=.5, linewidth=0, rasterized=True)
            elif kind == 'added':
                a_.scatter(u, v, s=1.6, c=d.region.map(COL).values, alpha=.75, linewidth=0, rasterized=True)
            else:
                w = (u > lo_u - pad) & (u < hi_u + pad) & (v > lo_v - pad) & (v < hi_v + pad)
                a_.scatter(u[w], v[w], s=1.2, color=NOTADDED, alpha=.8, linewidth=0, rasterized=True)
        a_.set_xlim(lo_u - pad, hi_u + pad); a_.set_ylim(lo_v - pad, hi_v + pad); a_.set_aspect('equal')
        a_.set_xticks([]); a_.set_yticks([])
        for s in a_.spines.values(): s.set_color('#e1e0d9')
        x0, y0 = lo_u - pad + 4, lo_v - pad + 4
        a_.plot([x0, x0 + 20], [y0, y0], color=INK, lw=1.2, solid_capstyle='butt')
        a_.text(x0 + 10, y0 + 1.5, '20 µm', ha='center', va='bottom', fontsize=7, color=INK)
        if view == 'posterior':
            nA, nO = len(add), len(off)
            title = f'{LABEL[ds]} · posterior view, both MBs\n{nO:,} APL synapses in the official ROIs, {nA:,} added (+{100 * nA / nO:.0f}%)'
        else:
            dd = parts[1][0]
            title = (f'{LABEL[ds]} · {"left" if sel == "L" else "right"} MB, side view (anterior left, posterior right)\n'
                     f'{len(dd):,} added: ' + ', '.join(f'{r} {n:,}' for r, n in dd.region.value_counts().items()))
        a_.set_title(title, fontsize=9, color=INK, loc='left')
h = [Line2D([], [], marker='o', ls='', color=OFFICIAL, markersize=6, label='inside the official MB ROIs')] + \
    [Line2D([], [], marker='o', ls='', color=c, markersize=6, label=f'added, joins the {r}') for r, c in COL.items()] + \
    [Line2D([], [], marker='o', ls='', color=NOTADDED, markersize=4, label='outside every MB ROI, not added (> 3 µm away, or accessory calyx)')]
fig.legend(handles=h, loc='upper left', ncol=3, frameon=False, fontsize=9, labelcolor=MUTED, bbox_to_anchor=(0.01, 0.958))
fig.suptitle('Where the outside-ROI version adds APL synapses (all APL synapses, postsynaptic site; FlyWire without the per-pair threshold)\n'
             'Rule: a synapse outside every official MB ROI joins the MB when it lies within 3 µm of the calyx anchor (ALPN→KC synapses) or the '
             'peduncle/lobe anchor (APL↔KC synapses),\nboth extended through outside KC synapses in ≤3 µm steps. Accessory calyces (γd and αβp KC '
             'dendrites) stay out, as in the official tables.', fontsize=11.5, color=INK, x=0.01, ha='left', y=.995)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(OUT + 'where_synapses_were_added.png', dpi=110); plt.close(fig)
print('figure done', flush=True)

# ---------------- counts ----------------
ALPN = {'ALPN', 'antennal_lobe_projection_neuron'}
rows = []
for ds in DS:
    typ, mbons = types(ds)
    kcs = set(typ.index[typ.astype(str).str.startswith('KC')])
    a = pd.read_parquet(A + f'final_{ds}_apl_mb.parquet', columns=['apl_side', 'apl_instance', 'apl_id', 'direction', 'mb_region', 'mb_region_source',
                                                                  'partner_id', 'partner_class'])
    sd = a.apl_side.where(a.apl_side.notna(), a.apl_instance.str[-1]).astype(str).str[0].str.upper()
    grp = np.where(a.partner_id.astype(str).isin(kcs), 'KC', np.where(a.partner_class.isin(ALPN), 'ALPN',
                   np.where(a.partner_id.astype(str).isin(APL_SIDE[ds]), 'APL', 'other')))
    g = pd.DataFrame({'dataset': ds, 'synapses': 'APL <-> partner (delivered APL table)', 'direction': a.direction.str.split(' ').str[0].values,
                      'side': sd.values, 'mb_region': a.mb_region.values, 'partner_group': grp,
                      'added': (a.mb_region_source != 'official ROI').values})
    rows.append(g)
    p = pd.read_parquet(A + f'{ds}_pnca_raw.parquet')
    post = (p.post_root_id_720575940.astype('int64') + PFX).astype(str) if ds == 'flywire' else \
        (p.post_root_id.astype(str) if ds == 'banc' else p.body_post.astype('Int64').astype(str))
    kc = post.isin(kcs).values
    side = p.roi_assigned.astype(str).str.extract(r'([LR])\)?$')[0]
    rows.append(pd.DataFrame({'dataset': ds, 'synapses': 'ALPN -> KC (calyx)', 'direction': 'ALPN_to_KC', 'side': side.values[kc],
                              'mb_region': 'CA', 'partner_group': 'KC', 'added': p.outside_roi.values[kc]}))
    lab = {'flywire': 'neuropil', 'hemibrain': 'roi_post', 'malecns': 'roi', 'banc': 'neuropil'}[ds]
    for f, added in [('out/' + f'{ds}_kcmbon_lobes.parquet', False), (A + f'{ds}_kcmbon_added.parquet', True)]:
        k = pd.read_parquet(f)
        if ds == 'flywire':
            pre = (k.pre_root_id_720575940.astype('int64') + PFX).astype(str); post = (k.post_root_id_720575940.astype('int64') + PFX).astype(str)
            reg, side = k.neuropil.str[3:5], k.neuropil.str[-1]
        elif ds == 'banc':
            pre, post = k.pre_root_id.astype(str), k.post_root_id.astype(str)
            reg, side = k.neuropil.str[3:5], k.side.astype(str).str[0].str.upper()
        else:
            pre, post = k.body_pre.astype('int64').astype(str), k.body_post.astype('int64').astype(str)
            m = k[lab].astype(str).str.extract(r"^([a-z]'?L)\(([LR])\)$"); reg, side = m[0], m[1]
        s = pre.isin(kcs).values & post.isin(mbons).values
        rows.append(pd.DataFrame({'dataset': ds, 'synapses': 'KC -> MBON (lobes)', 'direction': 'KC_to_MBON', 'side': side.values[s],
                                  'mb_region': reg.values[s], 'partner_group': 'MBON', 'added': added}))
R = pd.concat(rows, ignore_index=True)
key = ['dataset', 'synapses', 'direction', 'side', 'mb_region', 'partner_group']
S = R.groupby(key + ['added']).size().unstack('added', fill_value=0).rename(columns={False: 'n_official_ROI', True: 'n_added'})
for c in ['n_official_ROI', 'n_added']:
    if c not in S: S[c] = 0
S = S[['n_official_ROI', 'n_added']].reset_index()
S['pct_added'] = (100 * S.n_added / S.n_official_ROI.where(S.n_official_ROI > 0)).round(1)
order = {d: i for i, d in enumerate(DS)}
S = S.sort_values(['dataset', 'synapses', 'direction', 'side', 'mb_region', 'partner_group'], key=lambda c: c.map(order) if c.name == 'dataset' else c)
S.to_csv(OUT + 'added_synapses_summary.csv', index=False)
T = S.groupby(['dataset', 'synapses'])[['n_official_ROI', 'n_added']].sum()
T['pct_added'] = (100 * T.n_added / T.n_official_ROI).round(1)
print(T.to_string())
K = S[S.partner_group == 'KC'].groupby(['dataset', 'synapses', 'mb_region'])[['n_official_ROI', 'n_added']].sum()
K['pct'] = (100 * K.n_added / K.n_official_ROI).round(1)
print(K.to_string())
C = S[S.synapses.str.startswith('APL')].assign(region=lambda d: d.mb_region.where(d.mb_region.isin(['CA', 'PED']), 'lobes'))
C = C.groupby(['dataset', 'region'])[['n_official_ROI', 'n_added']].sum()
C['pct'] = (100 * C.n_added / C.n_official_ROI).round(1)
print(C.to_string())
