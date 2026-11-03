"""Three scatter figures requested 2026-10-10:
1. per PN, calyx: PN->APL (x) vs APL->PN (y)
2. per KC, calyx: PN->KC (x) vs KC->APL / APL->KC (y)
3. per KC: KC->APL (x) vs APL->KC (y), whole MB and calyx only
All counts are raw synapse (T-bar->PSD connection) counts, summed over both APLs.
FlyWire uses all Princeton synapses between proofread neurons (no >=5 per-pair threshold),
so per-KC weights are not zeroed by the Codex threshold."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from prep import malecns, hemibrain, flywire, banc
from variant import OUT, FINAL, TAG, fw_apl_raw

DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
INK, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#ecebe7', '#c9c8c2'
BLUE = '#2a78d6'
CLASSES = ['γ', 'α′β′', 'αβ', 'no subtype']
CCOL = {'γ': '#2a78d6', 'α′β′': '#eb6834', 'αβ': '#1baf7a', 'no subtype': '#9a9993'}
FW_APL = {'720575940613583001', '720575940624547622'}


def kc_class(t):
    t = str(t)
    if t.startswith('KCg'): return 'γ'
    if t.startswith("KCa'b'") or t.startswith('KCapbp'): return 'α′β′'
    if t.startswith('KCab'): return 'αβ'
    if t.startswith('KC'): return 'no subtype'
    return None


def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)


def identity(ax, x, y):
    m = max(np.max(x), np.max(y)) * 1.05
    ax.plot([0, m], [0, m], ls='--', lw=1, color=AXIS, zorder=0)
    ax.text(m, m, 'y = x', fontsize=7.5, color=MUTED, ha='right', va='bottom')


def rho(x, y):
    return spearmanr(x, y)[0]


# ---------------- per-KC table ----------------
rows = []
for ds in DS:
    d = {'malecns': malecns, 'hemibrain': hemibrain, 'flywire': flywire, 'banc': banc}[ds]()
    meta = d['meta'].drop_duplicates('id').set_index('id')
    cls = meta['type'].map(kc_class)
    kc_ids = set(cls.dropna().index)
    proof = set(meta.index)
    pn = d['pnca']
    if ds == 'flywire':
        pn = pn[(pn.pre_id != pn.post_id) & pn.post_id.isin(proof)]
    pnkc = pn[pn.post_id.isin(kc_ids)].groupby('post_id').size()
    if ds == 'flywire':
        r = fw_apl_raw(['pre_root_id_720575940', 'post_root_id_720575940', 'neuropil'])
        P = 720575940000000000
        r['pre_id'] = (r.pre_root_id_720575940.astype('int64') + P).astype(str)
        r['post_id'] = (r.post_root_id_720575940.astype('int64') + P).astype(str)
        npil = r.neuropil.fillna('')
        r = r[npil.str.startswith('MB_') & (r.pre_id != r.post_id) & r.pre_id.isin(proof) & r.post_id.isin(proof)]
        r['apl_is_pre'] = r.pre_id.isin(FW_APL)
        r['partner_id'] = np.where(r.apl_is_pre, r.post_id, r.pre_id)
        r['mb_region'] = np.where(r.neuropil.str.startswith('MB_CA'), 'CA', 'other')
        a = r[['apl_is_pre', 'partner_id', 'mb_region']]
    else:
        a = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet', columns=['apl_is_pre', 'partner_id', 'mb_region'])
    k = a[a.partner_id.isin(kc_ids)]
    ca = k.mb_region == 'CA'
    t = pd.concat({
        'PN_to_KC_calyx': pnkc,
        'KC_to_APL_calyx': k[ca & ~k.apl_is_pre].groupby('partner_id').size(),
        'APL_to_KC_calyx': k[ca & k.apl_is_pre].groupby('partner_id').size(),
        'KC_to_APL_MB': k[~k.apl_is_pre].groupby('partner_id').size(),
        'APL_to_KC_MB': k[k.apl_is_pre].groupby('partner_id').size(),
    }, axis=1).fillna(0).astype(int)
    t.index.name = 'kc_id'; t = t.reset_index()
    t.insert(0, 'dataset', ds)
    t.insert(2, 'kc_type', t.kc_id.map(meta['type']))
    t.insert(3, 'kc_class', t.kc_id.map(cls))
    rows.append(t)
    print(ds, 'KCs', len(t), t.kc_class.value_counts().to_dict(), t.iloc[:, 4:].sum().to_dict(), flush=True)
KC = pd.concat(rows, ignore_index=True)
KC.to_csv(OUT + 'KC_weights_APL_PN_per_KC.csv', index=False)
PN = pd.read_csv(OUT + 'PN_calyx_weights_APL_vs_KC.csv')

# ---------------- figure 1: per PN ----------------
fig, ax = plt.subplots(1, 4, figsize=(17, 4.7), constrained_layout=True)
for j, ds in enumerate(DS):
    t = PN[PN.dataset == ds]; a_ = ax[j]
    a_.scatter(t.PN_to_APL, t.APL_to_PN, s=22, color=BLUE, alpha=.7, edgecolor='white', linewidth=.6)
    identity(a_, t.PN_to_APL, t.APL_to_PN)
    lab = pd.concat([t.nlargest(2, 'APL_to_PN'), t.nlargest(2, 'PN_to_APL')]).drop_duplicates('pn_id')
    for _, rr in lab.iterrows():
        a_.annotate(str(rr.pn_type).replace('_', ' '), (rr.PN_to_APL, rr.APL_to_PN), fontsize=7.5, color=MUTED,
                    xytext=(4, 2), textcoords='offset points')
    a_.set_title(f'{LABEL[ds]}   n={len(t)} PNs, Spearman ρ={rho(t.PN_to_APL, t.APL_to_PN):.2f}', fontsize=10, color=INK, loc='left')
    a_.set_xlabel('PN → APL synapses (APL input)', color=MUTED, fontsize=9)
    if j == 0: a_.set_ylabel('APL → PN synapses (APL output)', color=MUTED, fontsize=9)
    style(a_)
fig.suptitle('1 · Per PN, calyx only: PN → APL vs APL → PN' + TAG, fontsize=13, color=INK, x=0.01, ha='left')
fig.savefig(OUT + 'scatter1_PN_APL_reciprocal_calyx.png', dpi=160); plt.close(fig)


JIT = 0.3
rng = np.random.default_rng(0)


def kc_panel(a_, t, xc, yc, ident=False, q=99.9):
    """Scatter one KC panel. Axes stop at the 99.9th percentile; KCs beyond are counted, not drawn.
    Integer counts are jittered by +-JIT synapse so overlapping KCs stay visible."""
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
        a_.plot([0, xm], [0, xm], ls='--', lw=1, color=AXIS, zorder=0)
        a_.text(xm * .98, xm * .98, 'y = x', fontsize=7.5, color=MUTED, ha='right', va='top')
    if (~inside).sum():
        a_.text(.01, .99, f'{(~inside).sum()} KC(s) beyond axes, not drawn', transform=a_.transAxes,
                fontsize=7.5, color=MUTED, va='top')
    style(a_)
    return rho(x, y)


def legend(fig, present):
    h = [Line2D([], [], marker='o', ls='', color=CCOL[c], markersize=7, label=f'{c} KCs' if c != 'no subtype' else 'KC, no subtype')
         for c in CLASSES if c in present]
    fig.legend(handles=h, loc='upper right', ncol=len(h), frameon=False, fontsize=9, labelcolor=MUTED)


present = set(KC.kc_class.dropna())
# ---------------- figure 2: per KC, calyx, vs PN input ----------------
fig, ax = plt.subplots(2, 4, figsize=(17, 8.4), constrained_layout=True)
for j, ds in enumerate(DS):
    t = KC[KC.dataset == ds]
    for i, (yc, ylab) in enumerate([('KC_to_APL_calyx', 'KC → APL synapses (calyx)'), ('APL_to_KC_calyx', 'APL → KC synapses (calyx)')]):
        r_ = kc_panel(ax[i, j], t, 'PN_to_KC_calyx', yc)
        ax[i, j].set_title(f'{LABEL[ds]}   n={len(t)} KCs, ρ={r_:.2f}', fontsize=10, color=INK, loc='left')
        ax[i, j].set_xlabel('PN → KC synapses onto this KC (calyx)', color=MUTED, fontsize=9)
        if j == 0: ax[i, j].set_ylabel(ylab, color=MUTED, fontsize=9)
fig.suptitle('2 · Per KC, calyx only: PN input vs KC → APL and APL → KC   (points jittered ±0.3 synapse)' + TAG, fontsize=13, color=INK, x=0.01, ha='left')
legend(fig, present)
fig.savefig(OUT + 'scatter2_KC_PNinput_vs_APL_calyx.png', dpi=160); plt.close(fig)

# ---------------- figure 3: per KC, KC->APL vs APL->KC ----------------
fig, ax = plt.subplots(2, 4, figsize=(17, 8.4), constrained_layout=True)
for j, ds in enumerate(DS):
    t = KC[KC.dataset == ds]
    for i, (sfx, where) in enumerate([('MB', 'whole mushroom body'), ('calyx', 'calyx only')]):
        tt = t[(t[f'KC_to_APL_{sfx}'] > 0) | (t[f'APL_to_KC_{sfx}'] > 0)]
        r_ = kc_panel(ax[i, j], tt, f'KC_to_APL_{sfx}', f'APL_to_KC_{sfx}', ident=True)
        ax[i, j].set_title(f'{LABEL[ds]}, {where}   n={len(tt)} KCs, ρ={r_:.2f}', fontsize=10, color=INK, loc='left')
        ax[i, j].set_xlabel(f'KC → APL synapses ({where})', color=MUTED, fontsize=9)
        if j == 0: ax[i, j].set_ylabel('APL → KC synapses', color=MUTED, fontsize=9)
fig.suptitle('3 · Per KC: KC → APL vs APL → KC   (points jittered ±0.3 synapse)' + TAG, fontsize=13, color=INK, x=0.01, ha='left')
legend(fig, present)
fig.savefig(OUT + 'scatter3_KC_APL_reciprocal.png', dpi=160); plt.close(fig)

# ---------------- numbers for the reply ----------------
for ds in DS:
    t = PN[PN.dataset == ds]
    print(ds, 'PN rho', round(rho(t.PN_to_APL, t.APL_to_PN), 2), 'median APL->PN / PN->APL',
          round(float((t.APL_to_PN.sum() / max(t.PN_to_APL.sum(), 1))), 2))
    t = KC[KC.dataset == ds]
    out = {}
    for c in CLASSES:
        u = t[t.kc_class == c]
        if len(u) < 20: continue
        out[c] = dict(n=len(u),
                      rho_pn_kc2apl=round(rho(u.PN_to_KC_calyx, u.KC_to_APL_calyx), 2),
                      rho_pn_apl2kc=round(rho(u.PN_to_KC_calyx, u.APL_to_KC_calyx), 2),
                      rho_recip_MB=round(rho(u.KC_to_APL_MB, u.APL_to_KC_MB), 2),
                      med_KC2APL_MB=float(u.KC_to_APL_MB.median()), med_APL2KC_MB=float(u.APL_to_KC_MB.median()),
                      med_PN2KC=float(u.PN_to_KC_calyx.median()))
    print(ds, out)
