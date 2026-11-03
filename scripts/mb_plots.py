"""Figures 1-3 again, counting synapses anywhere in the mushroom body (calyx, peduncle, lobes) instead of the calyx only.
1. per PN: PN->APL (x) vs APL->PN (y)
2. per KC: PN->KC (x) vs KC->APL (row 1) / APL->KC (row 2)
3. per KC: KC->APL (x) vs APL->KC (y)
MB = the ROIs of the delivered APL tables: hemibrain / male CNS CA, PED, aL, a'L, bL, b'L, gL; FlyWire / BANC MB_CA, MB_PED, MB_VL, MB_ML.
In the outside-ROI version (variant.py) the MB also gets the synapses outside these ROIs that offroi/assign.py adds.
Counts are T-bar->PSD pairs, summed over both APLs. FlyWire uses every Princeton synapse between proofread neurons."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import sys; sys.path.insert(0, 'scripts')
from variant import OUT, FINAL, TAG, OUTSIDE, ADDED, fw_apl_raw

D = 'dl/'; O = 'out/'
DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
INK, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#ecebe7', '#c9c8c2'
BLUE = '#2a78d6'
CLASSES = ['γ', 'α′β′', 'αβ', 'no subtype']
CCOL = {'γ': '#2a78d6', 'α′β′': '#eb6834', 'αβ': '#1baf7a', 'no subtype': '#9a9993'}
FW_APL = {'720575940613583001', '720575940624547622'}
JIT = 0.3
rng = np.random.default_rng(0)


def kc_class(t):
    t = str(t)
    if t.startswith('KCg'): return 'γ'
    if t.startswith("KCa'b'") or t.startswith('KCapbp'): return 'α′β′'
    if t.startswith('KCab'): return 'αβ'
    if t.startswith('KC'): return 'no subtype'
    return None


# ---------------- metadata: types and PN lists ----------------
def meta_types(ds):
    if ds == 'flywire':
        c = pd.read_csv(D + 'classification.csv.gz', dtype={'root_id': str})
        t = pd.read_csv(D + 'consolidated_cell_types.csv.gz', dtype={'root_id': str}).drop_duplicates('root_id').set_index('root_id')['primary_type']
        return c.root_id.map(t).set_axis(c.root_id), set(c.root_id[c['class'] == 'ALPN']), set(c.root_id)
    if ds == 'hemibrain':
        h = pd.read_feather(D + 'hemibrain_121_meta.feather')
        typ = pd.Series(h.cell_type.values, index=h.hemibrain_121_id.astype(str))
        off = pd.read_parquet(O + 'hemibrain_body_types.parquet')
        off = pd.Series(off['type'].values, index=off.body.astype('int64').astype(str))
        off = off[~off.index.duplicated()]
        typ = off.combine_first(typ[~typ.index.duplicated()])
        return typ, set(h.hemibrain_121_id[h.cell_class == 'antennal_lobe_projection_neuron'].astype(str)), None
    if ds == 'malecns':
        m = pd.read_feather(D + 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
        ids = m.bodyId.astype('int64').astype(str)
        return pd.Series(m.type.values, index=ids), set(ids[m['class'] == 'ALPN']), None
    b = pd.read_feather(D + 'banc_888_meta.feather', columns=['banc_888_id', 'cell_type', 'cell_class'])
    ids = b.banc_888_id.astype(str)
    typ = pd.Series(b.cell_type.values, index=ids)
    cx = pd.read_csv(D + 'neurons.csv.gz', dtype={'Root ID': str}).set_index('Root ID')['Primary Cell Type']
    typ = typ.where(typ.notna(), typ.index.to_series().map(cx))
    return typ, set(ids[b.cell_class == 'antennal_lobe_projection_neuron']), None


# ---------------- PN -> X pairs anywhere in the MB ----------------
def pn_mb_pairs(ds, proof):
    """One row per (pre PN, post) with n = synapse count in the MB."""
    if ds == 'flywire':
        t = pd.read_parquet(O + 'flywire_pnmb_pairs.parquet')
        if OUTSIDE:
            t = pd.concat([t, pd.read_parquet(ADDED + 'flywire_pnmb_added_pairs.parquet').rename(columns={'n': 'syn_count'})])
        t = t[(t.pre_id != t.post_id) & t.post_id.isin(proof)]
        return t.groupby(['pre_id', 'post_id']).syn_count.sum().rename('n').reset_index()
    if ds == 'hemibrain':
        ca = pd.read_parquet(O + 'hemibrain_pnca_raw.parquet', columns=['body_pre', 'body_post'])
        ca = pd.DataFrame({'pre_id': ca.body_pre.astype('int64').astype(str), 'post_id': ca.body_post.astype('int64').astype(str)})
        nc = pd.read_parquet(O + 'hemibrain_pn_nonCA_mb_pairs.parquet')
        nc = pd.DataFrame({'pre_id': nc.pre_body.astype('int64').astype(str), 'post_id': nc.post_body.astype('int64').astype(str)})
        t = pd.concat([ca, nc])
    elif ds == 'malecns':
        t = pd.read_parquet(O + 'malecns_pnmb_raw.parquet')
        t = pd.DataFrame({'pre_id': t.pre_id.astype(str), 'post_id': t.post_id.astype(str)})
    else:
        t = pd.read_parquet(O + 'banc_pnmb_raw.parquet')
        t = pd.DataFrame({'pre_id': t.pre_root_id.astype(str), 'post_id': t.post_root_id.astype(str)})
    t = t.groupby(['pre_id', 'post_id']).size().rename('n').reset_index()
    if OUTSIDE:
        t = pd.concat([t, pd.read_parquet(ADDED + f'{ds}_pnmb_added_pairs.parquet')]).groupby(['pre_id', 'post_id']).n.sum().reset_index()
    return t


# ---------------- APL rows anywhere in the MB ----------------
def apl_mb(ds, proof):
    if ds == 'flywire':
        r = fw_apl_raw(['pre_root_id_720575940', 'post_root_id_720575940', 'neuropil'])
        P = 720575940000000000
        r['pre_id'] = (r.pre_root_id_720575940.astype('int64') + P).astype(str)
        r['post_id'] = (r.post_root_id_720575940.astype('int64') + P).astype(str)
        r = r[r.neuropil.fillna('').str.startswith('MB_') & (r.pre_id != r.post_id) & r.pre_id.isin(proof) & r.post_id.isin(proof)]
        r['apl_is_pre'] = r.pre_id.isin(FW_APL)
        r['partner_id'] = np.where(r.apl_is_pre, r.post_id, r.pre_id)
        return r[['apl_is_pre', 'partner_id']]
    a = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet', columns=['apl_is_pre', 'partner_id'])
    a['partner_id'] = a.partner_id.astype(str)
    return a


PNrows, KCadd = [], {}
for ds in DS:
    typ, pns, proof = meta_types(ds)
    kc_ids = set(typ.index[typ.astype(str).str.startswith('KC')])
    pp = pn_mb_pairs(ds, proof)
    pp = pp[pp.pre_id.isin(pns)]
    pk = pp[pp.post_id.isin(kc_ids)]
    a = apl_mb(ds, proof)
    ap = a[a.partner_id.isin(pns)]
    t = pd.concat({'PN_to_KC_MB': pk.groupby('pre_id').n.sum(),
                   'PN_to_APL_MB': ap[~ap.apl_is_pre].groupby('partner_id').size(),
                   'APL_to_PN_MB': ap[ap.apl_is_pre].groupby('partner_id').size()}, axis=1).fillna(0).astype(int)
    t.index.name = 'pn_id'; t = t.reset_index()
    t.insert(0, 'dataset', ds); t['pn_type'] = t.pn_id.map(typ)
    PNrows.append(t)
    KCadd[ds] = pk.groupby('post_id').n.sum()
    print(ds, 'PNs', len(t), t[['PN_to_KC_MB', 'PN_to_APL_MB', 'APL_to_PN_MB']].sum().to_dict(), flush=True)
PN = pd.concat(PNrows, ignore_index=True)
PN.to_csv(OUT + 'PN_MB_weights_APL_vs_KC.csv', index=False)

# per-KC: add whole-MB PN input to the existing per-KC table (KC->APL / APL->KC whole-MB columns are already there)
KC = pd.read_csv(OUT + 'KC_weights_APL_PN_per_KC.csv', dtype={'kc_id': str})
KC = KC.drop(columns=[c for c in ['PN_to_KC_MB'] if c in KC.columns])
parts = []
for ds in DS:
    k = KC[KC.dataset == ds].set_index('kc_id')
    add = KCadd[ds]
    new = add.index.difference(k.index)
    if len(new):
        typ, _, _ = meta_types(ds)
        extra = pd.DataFrame(index=new); extra.index.name = 'kc_id'
        extra['dataset'] = ds; extra['kc_type'] = extra.index.map(typ); extra['kc_class'] = extra.kc_type.map(kc_class)
        k = pd.concat([k, extra])
    k['PN_to_KC_MB'] = add.reindex(k.index).fillna(0).astype(int)
    parts.append(k.reset_index())
    print(ds, 'KCs', len(k), 'new', len(new), 'PN->KC calyx vs MB', int(k.PN_to_KC_calyx.fillna(0).sum()), int(k.PN_to_KC_MB.sum()), flush=True)
KC = pd.concat(parts, ignore_index=True)
num = [c for c in KC.columns if c.endswith('_calyx') or c.endswith('_MB')]
KC[num] = KC[num].fillna(0).astype(int)
KC = KC[['dataset', 'kc_id', 'kc_type', 'kc_class', 'PN_to_KC_calyx', 'KC_to_APL_calyx', 'APL_to_KC_calyx',
         'PN_to_KC_MB', 'KC_to_APL_MB', 'APL_to_KC_MB']]
KC.to_csv(OUT + 'KC_weights_APL_PN_per_KC.csv', index=False)


# ---------------- drawing (same style as the calyx figures) ----------------
def rho(x, y):
    return spearmanr(x, y)[0]


def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)


def diag(ax, m):
    ax.plot([0, m], [0, m], ls='--', lw=1, color=AXIS, zorder=0)
    ax.text(m * .98, m * .98, 'y = x', fontsize=7.5, color=MUTED, ha='right', va='top')


def kc_panel(a_, t, xc, yc, ident=False, q=99.9):
    t = t.sample(frac=1, random_state=0)
    x, y = t[xc].values, t[yc].values
    xm, ym = np.percentile(x, q) * 1.08 + 1, np.percentile(y, q) * 1.08 + 1
    if ident:
        xm = ym = max(xm, ym)
    inside = (x <= xm) & (y <= ym)
    a_.scatter(x[inside] + rng.uniform(-JIT, JIT, inside.sum()), y[inside] + rng.uniform(-JIT, JIT, inside.sum()),
               s=6, c=t.kc_class.map(CCOL).values[inside], alpha=.4, linewidth=0)
    a_.set_xlim(-1, xm); a_.set_ylim(-1, ym)
    if ident:
        diag(a_, xm)
    if (~inside).sum():
        a_.text(.01, .99, f'{(~inside).sum()} KC(s) beyond axes, not drawn', transform=a_.transAxes,
                fontsize=7.5, color=MUTED, va='top')
    style(a_)
    return rho(x, y)


def legend(fig):
    present = set(KC.kc_class.dropna())
    h = [Line2D([], [], marker='o', ls='', color=CCOL[c], markersize=7,
                label=f'{c} KCs' if c != 'no subtype' else 'KC, no subtype') for c in CLASSES if c in present]
    fig.legend(handles=h, loc='upper right', ncol=len(h), frameon=False, fontsize=9, labelcolor=MUTED)


# 1 · per PN
fig, ax = plt.subplots(1, 4, figsize=(17, 4.7), constrained_layout=True)
for j, ds in enumerate(DS):
    t = PN[(PN.dataset == ds) & ((PN.PN_to_APL_MB > 0) | (PN.APL_to_PN_MB > 0) | (PN.PN_to_KC_MB > 0))]; a_ = ax[j]
    a_.scatter(t.PN_to_APL_MB, t.APL_to_PN_MB, s=22, color=BLUE, alpha=.7, edgecolor='white', linewidth=.6)
    diag(a_, max(t.PN_to_APL_MB.max(), t.APL_to_PN_MB.max()) * 1.05)
    lab = pd.concat([t.nlargest(2, 'APL_to_PN_MB'), t.nlargest(2, 'PN_to_APL_MB')]).drop_duplicates('pn_id')
    for _, rr in lab.iterrows():
        a_.annotate(str(rr.pn_type).replace('_', ' '), (rr.PN_to_APL_MB, rr.APL_to_PN_MB), fontsize=7.5, color=MUTED,
                    xytext=(4, 2), textcoords='offset points')
    a_.set_title(f'{LABEL[ds]}   n={len(t)} PNs, Spearman ρ={rho(t.PN_to_APL_MB, t.APL_to_PN_MB):.2f}', fontsize=10, color=INK, loc='left')
    a_.set_xlabel('PN → APL synapses (APL input)', color=MUTED, fontsize=9)
    if j == 0: a_.set_ylabel('APL → PN synapses (APL output)', color=MUTED, fontsize=9)
    style(a_)
fig.suptitle('1 · Per PN, whole mushroom body: PN → APL vs APL → PN' + TAG, fontsize=13, color=INK, x=0.01, ha='left')
fig.savefig(OUT + 'scatter1_PN_APL_reciprocal_MB.png', dpi=160); plt.close(fig)

# 2 · per KC vs PN input
fig, ax = plt.subplots(2, 4, figsize=(17, 8.4), constrained_layout=True)
for j, ds in enumerate(DS):
    t = KC[KC.dataset == ds]
    for i, (yc, ylab) in enumerate([('KC_to_APL_MB', 'KC → APL synapses (whole MB)'), ('APL_to_KC_MB', 'APL → KC synapses (whole MB)')]):
        r_ = kc_panel(ax[i, j], t, 'PN_to_KC_MB', yc)
        ax[i, j].set_title(f'{LABEL[ds]}   n={len(t)} KCs, Spearman ρ={r_:.2f}', fontsize=10, color=INK, loc='left')
        ax[i, j].set_xlabel('PN → KC synapses onto this KC (whole MB)', color=MUTED, fontsize=9)
        if j == 0: ax[i, j].set_ylabel(ylab, color=MUTED, fontsize=9)
fig.suptitle('2 · Per KC, whole mushroom body: PN input vs KC → APL and APL → KC   (points jittered ±0.3 synapse)' + TAG, fontsize=13, color=INK, x=0.01, ha='left')
legend(fig)
fig.savefig(OUT + 'scatter2_KC_PNinput_vs_APL_MB.png', dpi=160); plt.close(fig)

# 3 · per KC reciprocal
fig, ax = plt.subplots(1, 4, figsize=(17, 4.9), constrained_layout=True)
for j, ds in enumerate(DS):
    t = KC[KC.dataset == ds]
    t = t[(t.KC_to_APL_MB > 0) | (t.APL_to_KC_MB > 0)]
    r_ = kc_panel(ax[j], t, 'KC_to_APL_MB', 'APL_to_KC_MB', ident=True)
    ax[j].set_title(f'{LABEL[ds]}   n={len(t)} KCs, Spearman ρ={r_:.2f}', fontsize=10, color=INK, loc='left')
    ax[j].set_xlabel('KC → APL synapses (whole MB)', color=MUTED, fontsize=9)
    if j == 0: ax[j].set_ylabel('APL → KC synapses (whole MB)', color=MUTED, fontsize=9)
fig.suptitle('3 · Per KC, whole mushroom body: KC → APL vs APL → KC   (points jittered ±0.3 synapse)' + TAG, fontsize=13, color=INK, x=0.01, ha='left')
legend(fig)
fig.savefig(OUT + 'scatter3_KC_APL_reciprocal_MB.png', dpi=160); plt.close(fig)

# ---------------- numbers for the reply ----------------
PNc = pd.read_csv(OUT + 'PN_calyx_weights_APL_vs_KC.csv', dtype={'pn_id': str})
for ds in DS:
    t = PN[PN.dataset == ds]; c = PNc[PNc.dataset == ds]
    print(ds, 'PN: MB vs calyx sums PN->APL', int(t.PN_to_APL_MB.sum()), int(c.PN_to_APL.sum()), 'APL->PN', int(t.APL_to_PN_MB.sum()), int(c.APL_to_PN.sum()),
          'rho MB', round(rho(t.PN_to_APL_MB, t.APL_to_PN_MB), 2), 'rho calyx', round(rho(c.PN_to_APL, c.APL_to_PN), 2))
    k = KC[KC.dataset == ds]; r3 = k[(k.KC_to_APL_MB > 0) | (k.APL_to_KC_MB > 0)]
    print(ds, 'KC fig2 rho (KC->APL, APL->KC):', round(rho(k.PN_to_KC_MB, k.KC_to_APL_MB), 2), round(rho(k.PN_to_KC_MB, k.APL_to_KC_MB), 2),
          '| fig3 rho', round(rho(r3.KC_to_APL_MB, r3.APL_to_KC_MB), 2),
          '| calyx share of KC->APL, APL->KC:', round(k.KC_to_APL_calyx.sum() / k.KC_to_APL_MB.sum(), 2), round(k.APL_to_KC_calyx.sum() / k.APL_to_KC_MB.sum(), 2))
    print('   medians by class (PN->KC, KC->APL, APL->KC, whole MB):', k.groupby('kc_class')[['PN_to_KC_MB', 'KC_to_APL_MB', 'APL_to_KC_MB']].median().to_dict('index'))
    print('   per-class rho fig3:', {cl: round(rho(g.KC_to_APL_MB, g.APL_to_KC_MB), 2) for cl, g in r3.groupby('kc_class') if len(g) > 20})
