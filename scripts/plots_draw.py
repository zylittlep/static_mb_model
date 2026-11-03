"""Draw the three requested scatter figures from the per-PN / per-KC CSVs written by scatter.py and plots3.py.
1. per PN, calyx: PN->APL (x) vs APL->PN (y)
2. per KC, calyx: PN->KC (x) vs KC->APL (row 1) / APL->KC (row 2)
3. per KC, calyx: KC->APL (x) vs APL->KC (y)"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import sys; sys.path.insert(0, 'scripts')
from variant import OUT, TAG

DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
INK, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#ecebe7', '#c9c8c2'
BLUE = '#2a78d6'
CLASSES = ['γ', 'α′β′', 'αβ', 'no subtype']
CCOL = {'γ': '#2a78d6', 'α′β′': '#eb6834', 'αβ': '#1baf7a', 'no subtype': '#9a9993'}
JIT = 0.3
rng = np.random.default_rng(0)
PN = pd.read_csv(OUT + 'PN_calyx_weights_APL_vs_KC.csv', dtype={'pn_id': str})
KC = pd.read_csv(OUT + 'KC_weights_APL_PN_per_KC.csv', dtype={'kc_id': str})


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
    """Axes stop at the 99.9th percentile; KCs beyond are counted, not drawn.
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
    t = PN[PN.dataset == ds]; a_ = ax[j]
    a_.scatter(t.PN_to_APL, t.APL_to_PN, s=22, color=BLUE, alpha=.7, edgecolor='white', linewidth=.6)
    diag(a_, max(t.PN_to_APL.max(), t.APL_to_PN.max()) * 1.05)
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

# 2 · per KC vs PN input
fig, ax = plt.subplots(2, 4, figsize=(17, 8.4), constrained_layout=True)
for j, ds in enumerate(DS):
    t = KC[KC.dataset == ds]
    for i, (yc, ylab) in enumerate([('KC_to_APL_calyx', 'KC → APL synapses (calyx)'), ('APL_to_KC_calyx', 'APL → KC synapses (calyx)')]):
        r_ = kc_panel(ax[i, j], t, 'PN_to_KC_calyx', yc)
        ax[i, j].set_title(f'{LABEL[ds]}   n={len(t)} KCs, Spearman ρ={r_:.2f}', fontsize=10, color=INK, loc='left')
        ax[i, j].set_xlabel('PN → KC synapses onto this KC (calyx)', color=MUTED, fontsize=9)
        if j == 0: ax[i, j].set_ylabel(ylab, color=MUTED, fontsize=9)
fig.suptitle('2 · Per KC, calyx only: PN input vs KC → APL and APL → KC   (points jittered ±0.3 synapse)' + TAG, fontsize=13, color=INK, x=0.01, ha='left')
legend(fig)
fig.savefig(OUT + 'scatter2_KC_PNinput_vs_APL_calyx.png', dpi=160); plt.close(fig)

# 3 · per KC reciprocal, calyx only
fig, ax = plt.subplots(1, 4, figsize=(17, 4.9), constrained_layout=True)
for j, ds in enumerate(DS):
    t = KC[KC.dataset == ds]
    t = t[(t.KC_to_APL_calyx > 0) | (t.APL_to_KC_calyx > 0)]
    r_ = kc_panel(ax[j], t, 'KC_to_APL_calyx', 'APL_to_KC_calyx', ident=True)
    ax[j].set_title(f'{LABEL[ds]}   n={len(t)} KCs, Spearman ρ={r_:.2f}', fontsize=10, color=INK, loc='left')
    ax[j].set_xlabel('KC → APL synapses (calyx)', color=MUTED, fontsize=9)
    if j == 0: ax[j].set_ylabel('APL → KC synapses (calyx)', color=MUTED, fontsize=9)
fig.suptitle('3 · Per KC, calyx only: KC → APL vs APL → KC   (points jittered ±0.3 synapse)' + TAG, fontsize=13, color=INK, x=0.01, ha='left')
legend(fig)
fig.savefig(OUT + 'scatter3_KC_APL_reciprocal_calyx.png', dpi=160); plt.close(fig)

for ds in DS:
    t = KC[KC.dataset == ds]; c = t[(t.KC_to_APL_calyx > 0) | (t.APL_to_KC_calyx > 0)]
    print(ds, 'fig2 rho', round(rho(t.PN_to_KC_calyx, t.KC_to_APL_calyx), 2), round(rho(t.PN_to_KC_calyx, t.APL_to_KC_calyx), 2),
          'fig3 rho', round(rho(c.KC_to_APL_calyx, c.APL_to_KC_calyx), 2),
          'per class fig3', {k: round(rho(g.KC_to_APL_calyx, g.APL_to_KC_calyx), 2) for k, g in c.groupby('kc_class') if len(g) > 20},
          'calyx medians by class (PN->KC, KC->APL, APL->KC)', t.groupby('kc_class')[['PN_to_KC_calyx', 'KC_to_APL_calyx', 'APL_to_KC_calyx']].median().to_dict('index'))
