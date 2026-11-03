"""Per KC, lobes only: KC->MBON (x) vs KC->APL or APL->KC (y), one panel per lobe (or compartment) and dataset.
Reads KC_lobe_compartment_weights_MBON_APL.csv written by lobe_tables.py."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sys; sys.path.insert(0, 'scripts')
from variant import OUT, TAG

DS = ['flywire', 'hemibrain', 'malecns', 'banc']
LABEL = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
INK, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#ecebe7', '#c9c8c2'
CCOL = {'γ': '#2a78d6', 'α′β′': '#eb6834', 'αβ': '#1baf7a'}
LOBES = [('aL', 'α'), ('bL', 'β'), ("a'L", 'α′'), ("b'L", 'β′'), ('gL', 'γ')]
LOBE_CLASS = {'aL': 'αβ', 'bL': 'αβ', "a'L": 'α′β′', "b'L": 'α′β′', 'gL': 'γ'}
COMPS = [(f'{k}{i}', f'{v}{i}', lob) for lob, k, v, n in [('aL', 'a', 'α', 3), ('bL', 'b', 'β', 2), ("a'L", "a'", 'α′', 3),
                                                           ("b'L", "b'", 'β′', 2), ('gL', 'g', 'γ', 5)] for i in range(1, n + 1)]
JIT = 0.3
rng = np.random.default_rng(0)
W = pd.read_csv(OUT + 'KC_lobe_compartment_weights_MBON_APL.csv', dtype={'kc_id': str})


def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)


def panel(ax, t, yc, color, title, q=99.5):
    if len(t) < 3:
        ax.text(.5, .5, 'no KCs', transform=ax.transAxes, ha='center', va='center', color=MUTED, fontsize=9)
        ax.set_title(title, fontsize=9, color=INK, loc='left'); style(ax); return
    t = t.sample(frac=1, random_state=0)
    x, y = t.KC_to_MBON.values, t[yc].values
    xm, ym = np.percentile(x, q) * 1.08 + 1, np.percentile(y, q) * 1.08 + 1
    inside = (x <= xm) & (y <= ym)
    ax.scatter(x[inside] + rng.uniform(-JIT, JIT, inside.sum()), y[inside] + rng.uniform(-JIT, JIT, inside.sum()),
               s=5, color=color, alpha=.35, linewidth=0)
    ax.set_xlim(-1, xm); ax.set_ylim(-1, ym)
    r = spearmanr(x, y)[0] if np.ptp(x) and np.ptp(y) else np.nan
    ax.set_title(f'{title}   n={len(t)}, ρ={r:.2f}', fontsize=9, color=INK, loc='left')
    if (~inside).sum():
        ax.text(.99, .02, f'{(~inside).sum()} beyond axes', transform=ax.transAxes, fontsize=7, color=MUTED, ha='right')
    style(ax)


def figure(level, rows, yc, ylab, fname, suptitle):
    fig, ax = plt.subplots(len(rows), 4, figsize=(16, 2.75 * len(rows) + .6), constrained_layout=True, squeeze=False)
    for i, (reg, name, lobe) in enumerate(rows):
        for j, ds in enumerate(DS):
            t = W[(W.dataset == ds) & (W.level == level) & (W.region == reg) & (W.kc_class == LOBE_CLASS[lobe])]
            panel(ax[i, j], t, yc, CCOL[LOBE_CLASS[lobe]], f'{LABEL[ds]} · {name}')
            if i == len(rows) - 1: ax[i, j].set_xlabel('KC → MBON synapses', color=MUTED, fontsize=9)
            if j == 0: ax[i, j].set_ylabel(f'{ylab}\n({name})', color=MUTED, fontsize=9)
    fig.suptitle(suptitle + TAG, fontsize=13, color=INK, x=0.01, ha='left')
    fig.savefig(OUT + fname, dpi=130); plt.close(fig)


lobe_rows = [(k, v + ' lobe', k) for k, v in LOBES]
comp_rows = [(c, n, lob) for c, n, lob in COMPS]
figure('lobe', lobe_rows, 'KC_to_APL', 'KC → APL synapses', 'scatter4_KC_MBON_vs_KC_to_APL_by_lobe.png',
       'Per KC, lobes only: KC → MBON vs KC → APL, by lobe   (each dot one KC; jittered ±0.3 synapse)')
figure('lobe', lobe_rows, 'APL_to_KC', 'APL → KC synapses', 'scatter5_KC_MBON_vs_APL_to_KC_by_lobe.png',
       'Per KC, lobes only: KC → MBON vs APL → KC, by lobe   (each dot one KC; jittered ±0.3 synapse)')
figure('compartment', comp_rows, 'KC_to_APL', 'KC → APL synapses', 'scatter6_KC_MBON_vs_KC_to_APL_by_compartment.png',
       'Per KC, by lobe compartment: KC → MBON vs KC → APL   (each dot one KC; jittered ±0.3 synapse)')
figure('compartment', comp_rows, 'APL_to_KC', 'APL → KC synapses', 'scatter7_KC_MBON_vs_APL_to_KC_by_compartment.png',
       'Per KC, by lobe compartment: KC → MBON vs APL → KC   (each dot one KC; jittered ±0.3 synapse)')

# numbers for the reply
for level, rows in [('lobe', lobe_rows), ('compartment', comp_rows)]:
    out = []
    for reg, name, lobe in rows:
        for ds in DS:
            t = W[(W.dataset == ds) & (W.level == level) & (W.region == reg) & (W.kc_class == LOBE_CLASS[lobe])]
            if len(t) < 3: continue
            out.append(dict(region=name, ds=ds, n=len(t), rho_KC2APL=round(spearmanr(t.KC_to_MBON, t.KC_to_APL)[0], 2),
                            rho_APL2KC=round(spearmanr(t.KC_to_MBON, t.APL_to_KC)[0], 2),
                            med_MBON=t.KC_to_MBON.median(), med_KC2APL=t.KC_to_APL.median(), med_APL2KC=t.APL_to_KC.median()))
    o = pd.DataFrame(out)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    print(o.to_string())
