"""Per-glomerulus maps of KC<->APL synapses on KC claws, all 58 antennal-lobe glomeruli (51 olfactory + 7 VP).

Claws are the derived claws of the main tables: single-linkage clusters (2 µm) of each KC's postsynaptic sites from
antennal-lobe PNs in the calyx. A claw belongs to the glomerulus of its main PN (the PN that makes most of the claw's
PN synapses). A KC<->APL synapse is on a claw when its KC-side site is within 2 µm of that claw's PN-synapse sites
(nearest claw of the same KC) -- the rule behind the claw_id column of the main tables.
One panel per glomerulus x APL: KC<->APL synapses on claws of that glomerulus are colored (blue: the claw's main PN is
uniglomerular; orange: its type names two glomeruli, VP1d+VP4 / VP1m+VP5 / VP3+VP1l, shown in both rows); every other
KC<->APL synapse on a claw (other glomeruli, multiglomerular or untyped main PN) is grey.
FlyWire uses every Princeton synapse between proofread neurons (no per-pair threshold), like the other maps; the other
datasets use the delivered tables. Posterior view and panel windows as in the PN<->APL maps (glom_maps.py)."""
import sys, re, json
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'odor')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from prep import malecns, hemibrain, flywire, banc
from core import cluster_by_owner, assign_nearest
from odor_meta import meta_types, apl_rows
from variant import OUT as VOUT, FINAL, TAG

OUT = VOUT + 'odor_weighted/'
CLAW_D = 2000; ASSIGN_D = 2000      # nm, as in run.py
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
VIEW = {'flywire': lambda x, y, z: (x, -y), 'hemibrain': lambda x, y, z: (-x, -z),
        'malecns': lambda x, y, z: (-x, -y), 'banc': lambda x, y, z: (-x, z)}
INK, MUTED = '#0b0b0b', '#52514e'
REST, UNI, BI, FRAME = '#d3d2cc', '#2a78d6', '#eb6834', '#e1e0d9'
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'
FN = {'flywire': flywire, 'hemibrain': hemibrain, 'malecns': malecns, 'banc': banc}


def gloms_of(t, ds):
    """Glomeruli named by a PN type: 'DM1_lPN' -> [DM1]; 'VP1d+VP4_l2PN1' -> [VP1d, VP4]; 'VP2+_adPN' -> [VP2]."""
    t = str(t)
    if t.startswith('M_') or '_' not in t: return []
    names = [n for n in t.split('_')[0].split('+') if re.fullmatch(GPAT, n)]
    return [HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in names]


rows, claw_rows, checks = [], [], {}
for ds in DS:
    typ, pns, proof, fw_side = meta_types(ds)
    kcs = set(typ.index[typ.astype(str).str.startswith('KC')])
    d = FN[ds]()
    # ---- claws, exactly as run.py builds them (same PN table, same clustering) ----
    p = d['pnca'].copy()
    p['post_type'] = p.post_id.map(typ).astype(str)
    kc = p[p.post_type.str.startswith('KC')].reset_index(drop=True)
    kcx = kc[['post_x', 'post_y', 'post_z']].values
    kc['claw_id'] = cluster_by_owner(kc.post_id.values, kcx, CLAW_D, 'claw')
    cl = pd.read_parquet(FINAL + f'final_{ds}_claws.parquet').set_index('claw_id')
    n_here = kc.groupby('claw_id').size()
    checks[ds] = {'claws': int(len(n_here)), 'claws_in_table': int(len(cl)),
                  'same_sizes': bool((n_here.reindex(cl.index) == cl.n_PN_synapses).all())}
    # ---- KC<->APL synapses in the calyx, KC-side site ----
    if ds == 'flywire':
        a = apl_rows(ds, proof)
        a = a[(a.region == 'CA') & a.partner_id.isin(kcs)].reset_index(drop=True)
        a['apl_side'] = a.apl_id.map(fw_side).astype(str).str[0].str.upper()
        ks = np.where(a.apl_is_pre.values[:, None], a[['post_x', 'post_y', 'post_z']].values, a[['pre_x', 'pre_y', 'pre_z']].values)
    else:
        f = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet')
        f = f[(f.mb_region == 'CA') & f.partner_id.astype(str).isin(kcs)].reset_index(drop=True)
        sd = f.apl_side.where(f.apl_side.notna(), f.apl_instance.str[-1])
        a = pd.DataFrame({'apl_id': f.apl_id.astype(str), 'apl_is_pre': f.apl_is_pre.values, 'partner_id': f.partner_id.astype(str),
                          'apl_side': sd.astype(str).str[0].str.upper().values})
        for c in ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z']: a[c] = f[c].astype(float).values
        ks = f[['partner_side_x', 'partner_side_y', 'partner_side_z']].values.astype(float)
        old_claw = f.claw_id.values
    lab, dist = assign_nearest(a.partner_id.values, ks, kc.post_id.values, kcx, kc.claw_id.values, np.inf)
    a['claw_id'] = np.where(dist <= ASSIGN_D, lab, None)
    a['claw_dist_nm'] = np.round(dist)
    if ds != 'flywire':      # must reproduce the delivered claw_id column
        checks[ds]['claw_id_matches_table'] = bool((pd.Series(a.claw_id).fillna('-').values == pd.Series(old_claw).fillna('-').values).all())
    a['direction'] = np.where(a.apl_is_pre, 'APL_to_KC', 'KC_to_APL')
    a['kc_type'] = a.partner_id.map(typ)
    a['claw_main_PN_id'] = a.claw_id.map(cl.main_PN_id)
    a['claw_main_PN_type'] = a.claw_id.map(cl.main_PN_type)
    u, v = VIEW[ds](a.post_x.values / 1e3, a.post_y.values / 1e3, a.post_z.values / 1e3)
    a['view_right_um'], a['view_dorsal_um'] = np.round(u, 3), np.round(v, 3)
    a.insert(0, 'dataset', ds)
    rows.append(a)
    # claws per glomerulus and calyx: side of a claw = side of the APL calyx synapses nearest its centroid
    tr = cKDTree(a[['post_x', 'post_y', 'post_z']].values)
    _, j = tr.query(cl[['centroid_x_nm', 'centroid_y_nm', 'centroid_z_nm']].values)
    claw_rows.append(pd.DataFrame({'dataset': ds, 'claw_id': cl.index, 'apl_side': a.apl_side.values[j], 'main_PN_type': cl.main_PN_type.values}))
    print(ds, checks[ds], 'KC<->APL calyx synapses', len(a), 'on a claw', round(a.claw_id.notna().mean(), 3), flush=True)

A = pd.concat(rows, ignore_index=True)
A = A.rename(columns={'partner_id': 'kc_id', 'post_x': 'x_nm', 'post_y': 'y_nm', 'post_z': 'z_nm'})
A['glomeruli'] = ['+'.join(gloms_of(t, ds)) if isinstance(t, str) else '' for t, ds in zip(A.claw_main_PN_type, A.dataset)]
on = A[A.claw_id.notna()].copy()
on[['dataset', 'apl_id', 'apl_side', 'direction', 'kc_id', 'kc_type', 'claw_id', 'claw_dist_nm', 'claw_main_PN_id',
    'claw_main_PN_type', 'glomeruli', 'x_nm', 'y_nm', 'z_nm', 'view_right_um', 'view_dorsal_um']].to_csv(
    OUT + 'KC_APL_synapses_on_claws_by_glomerulus.csv', index=False)

on['gl_list'] = on.glomeruli.str.split('+').map(lambda l: [g for g in l if g])
on['n_gl'] = on.gl_list.str.len()
E = on[on.n_gl > 0].explode('gl_list').rename(columns={'gl_list': 'glom'})
GLOMS = sorted(E.glom.unique())
print('glomeruli with claws:', len(GLOMS))

# summary per glomerulus and calyx
CL = pd.concat(claw_rows, ignore_index=True)
CL['gl_list'] = [gloms_of(t, ds) for t, ds in zip(CL.main_PN_type, CL.dataset)]
CE = CL[CL.gl_list.str.len() > 0].explode('gl_list').rename(columns={'gl_list': 'glomerulus'})
n_claws = CE.groupby(['dataset', 'apl_side', 'glomerulus']).size().rename('n_claws')
g = E.groupby(['dataset', 'apl_side', 'glom', 'direction']).size().unstack('direction', fill_value=0)
g.index = g.index.set_names('glomerulus', level='glom')
wc = E.groupby(['dataset', 'apl_side', 'glom']).claw_id.nunique().rename('n_claws_with_APL_synapse')
wc.index = wc.index.set_names('glomerulus', level='glom')
T = pd.concat([n_claws, wc, g], axis=1).fillna(0).astype(int).reset_index()
for c in ['KC_to_APL', 'APL_to_KC']:
    T[f'{c}_per_claw'] = (T[c] / T.n_claws.where(T.n_claws > 0)).round(2)
T.to_csv(OUT + 'KC_APL_on_claws_per_glomerulus_summary.csv', index=False)

# ---------------- figures: windows exactly as the PN<->APL glomerulus maps ----------------
S = pd.read_csv(OUT + 'PN_APL_synapses_GH146_weighted.csv', usecols=['dataset', 'apl_side', 'mb_region', 'view_right_um', 'view_dorsal_um'])
C = S[S.mb_region == 'CA']
box = C.groupby(['dataset', 'apl_side'])[['view_right_um', 'view_dorsal_um']].quantile([.002, .998]).unstack()
mid_u = (box[('view_right_um', .002)] + box[('view_right_um', .998)]) / 2
mid_v = (box[('view_dorsal_um', .002)] + box[('view_dorsal_um', .998)]) / 2
WU = np.ceil(((box[('view_right_um', .998)] - box[('view_right_um', .002)]) / 2).max() + 3)
WV = np.ceil(((box[('view_dorsal_um', .998)] - box[('view_dorsal_um', .002)]) / 2).max() + 3)
for t_ in (on, E):
    key = list(zip(t_.dataset, t_.apl_side))
    t_['du'] = t_.view_right_um.values - mid_u.loc[key].values
    t_['dv'] = t_.view_dorsal_um.values - mid_v.loc[key].values
PANELS = [(ds, s_) for ds in DS for s_ in ['L', 'R'] if (ds, s_) in mid_u.index]
NB = 2
ROWS = int(np.ceil(len(GLOMS) / NB))


def draw(direction, fname, arrow):
    D = on[on.direction == direction]; G = E[E.direction == direction]
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
                rest = t[~t.index.isin(h.index)]
                a.scatter(rest.du, rest.dv, s=.8, color=REST, linewidth=0, zorder=1, rasterized=True)
                a.scatter(h.du, h.dv, s=4, c=np.where(h.n_gl > 1, BI, UNI), linewidth=0, zorder=2, rasterized=True)
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
    fig.suptitle(f'{arrow} synapses on KC claws, by the glomerulus of the claw\'s main PN · calyx, posterior view '
                 f'(fly\'s right on the right, dorsal up){TAG}\n'
                 f'Claw = cluster of a KC\'s PN-synapse sites (2 µm); synapse on a claw = KC-side site within 2 µm of it. '
                 f'Blue = claws whose main PN is uniglomerular; orange = main PN innervates that glomerulus and one other '
                 f'(shown in both rows); grey = {arrow} synapses on every other claw. Number = colored synapses.',
                 fontsize=10, color=INK, x=0.005, ha='left', y=1 - .12 / H, va='top')
    fig.savefig(OUT + fname, dpi=130, bbox_inches='tight'); plt.close(fig)


draw('KC_to_APL', 'claw_glomerulus_map_KC_to_APL_posterior_view.png', 'KC → APL')
draw('APL_to_KC', 'claw_glomerulus_map_APL_to_KC_posterior_view.png', 'APL → KC')
summ = {'checks': checks, 'glomeruli': GLOMS,
        'share_of_KC_APL_calyx_synapses_on_a_claw': A.groupby(['dataset', 'direction']).claw_id.apply(lambda s: round(s.notna().mean(), 3)).unstack().to_dict('index'),
        'share_of_on_claw_synapses_with_a_glomerulus': on.groupby('dataset').n_gl.apply(lambda s: round((s > 0).mean(), 3)).to_dict()}
json.dump(summ, open(OUT + 'claw_glomerulus_maps_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != 'glomeruli'}, indent=1))
print(T.groupby('dataset')[['n_claws', 'KC_to_APL', 'APL_to_KC']].sum())
