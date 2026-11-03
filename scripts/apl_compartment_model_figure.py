"""Figure for the compartment (sub-arbor) APL model: odor_weighted/APL_compartment_model.png.

APL_VARIANT=outside_roi draws the version with synapses outside the official ROIs (with_outside_ROI_synapses/).
Reads the outputs of apl_compartment_model.py and, as reference, apl_local_model.py (local APL without compartment walls).
Panels: rank-order Spearman (blocked vs intact similarity) by version, r of the model's change in similarity with the
TNT - TNT3c change by version, and the within-class site-shuffle null for the isopotential compartment version.
"""
import os, sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts/reports')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import report_lib as L

L.style()
OUTSIDE = os.environ.get('APL_VARIANT', '') == 'outside_roi'
OW = L.B + ('with_outside_ROI_synapses/' if OUTSIDE else '') + 'odor_weighted/'
SC = json.load(open(OW + 'APL_compartment_model_summary.json')); SL = json.load(open(OW + 'APL_local_model_summary.json'))
NUL = pd.read_csv(OW + 'APL_compartment_model_nulls.csv')
RAW_CORR, CEIL = 0.92, 0.84
MBS = ['hemibrain R', 'malecns L', 'malecns R']; MBN = {'hemibrain R': 'hemibrain R', 'malecns L': 'male CNS L', 'malecns R': 'male CNS R'}
MK = {'hemibrain R': 'o', 'malecns L': 's', 'malecns R': '^'}
COMP = '#a8820e'
GCOL = {'ref': L.GRAY, 'loc': L.PURPLE, 'cmp': COMP, 'bo': L.GREEN, 'kb': L.INK}
GNAME = {'ref': 'global APL', 'loc': 'local APL, no walls\n(Report 11)', 'cmp': 'APL confined to\nsub-arbors, on KCs',
         'bo': 'boutons\nonly', 'kb': 'KCs +\nboutons'}       # both confined to sub-arbors; named in full in the caption
LOC = ['200', '100', '50', '25', '10', '5', '2']


def lam_lab(key):
    t = key.split('compartments ')[-1].replace('local ', '')
    return {'inf': 'λ = ∞', 'inf, per length': 'λ = ∞, ÷ length'}.get(t, f'λ = {t}')


def versions():
    """(label, group, source, key): source 'c' = compartment run, 'l' = local run."""
    v = [('uniform (U)', 'ref', 'c', 'U'), ('per-KC counts (C)', 'ref', 'c', 'C')]
    v += [(f'λ = {l} µm', 'loc', 'l', f'local {l} µm') for l in LOC]
    v += [(lam_lab(k), 'cmp', 'c', k) for k in ['compartments inf', 'compartments inf, per length'] +
          [f'compartments {l} µm' for l in ['200', '100', '50', '25', '10']]]
    v += [(lam_lab(f'compartments {l}'), 'bo', 'c', f'boutons, compartments {l}') for l in ['inf', '50 µm']]
    v += [(lam_lab(f'compartments {l}'), 'kb', 'c', f'KC + boutons, compartments {l}') for l in ['inf', '50 µm']]
    return v


def per_mb(rn):
    return {'c': pd.read_csv(OW + f'APL_compartment_model_per_MB_{rn}.csv').set_index(['version', 'mushroom_body']),
            'l': pd.read_csv(OW + f'APL_local_model_per_MB_{rn}.csv').set_index(['version', 'mushroom_body'])}


def mean_rank(rn, src, key):
    return (SC if src == 'c' else SL)[rn]['model: Spearman blocked vs intact (mean over MBs) by version'][key]


def mean_r(rn, src, key):
    return SC[rn]['r over odor pairs, model change (MB-averaged) vs data change'][key] if src == 'c' else \
        SL[rn]['r over odor pairs, model change vs data change'][key]


def draw(ax, rn, kind):
    V = versions(); P = per_mb(rn); x = []; pos = 0; prev = None; vals = []
    for lab, g, src, key in V:
        if prev is not None and g != prev: pos += 1.1
        x.append(pos); prev = g; pos += 1
    for xi, (lab, g, src, key) in zip(x, V):
        col = GCOL[g]
        for k in MBS:
            val = P[src].loc[(key, k), 'spearman_blocked_vs_intact' if kind == 'rank' else 'r_dS_with_data']
            ax.scatter(xi, val, marker=MK[k], s=16, facecolor='white', edgecolor=col, lw=0.9, zorder=3); vals.append(val)
        m = mean_rank(rn, src, key) if kind == 'rank' else mean_r(rn, src, key)
        ax.plot([xi - 0.32, xi + 0.32], [m, m], color=col, lw=2.4, solid_capstyle='butt', zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([v[0] for v in V], rotation=90, fontsize=7.2)
    grp = {}
    for xi, v in zip(x, V): grp.setdefault(v[1], []).append(xi)
    for g, xs in grp.items():
        ax.text(np.mean(xs), -0.5, GNAME[g], transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.3, color=GCOL[g],
                fontweight='semibold')
    ax.set_xlim(min(x) - 0.7, max(x) + 0.7)
    if kind == 'rank':
        dv = SC[rn]['data: Spearman TNT vs TNT3c similarity over odor pairs']
        ax.axhspan(dv, RAW_CORR, color=L.BLOCKED, alpha=0.08, lw=0)
        ax.axhline(dv, color=L.BLOCKED, lw=1, ls='--'); ax.axhline(RAW_CORR, color=L.BLOCKED, lw=1, ls=':')
        ax.text(min(x) - 0.6, dv, f'data, all recordings ({dv:.2f})', ha='left', va='bottom', fontsize=7, color=L.BLOCKED)
        ax.text(min(x) - 0.6, RAW_CORR, 'data, corrected for noise (≈ 0.92)', ha='left', va='bottom', fontsize=7, color=L.BLOCKED)
        ax.axhline(1, color=L.GRID, lw=0.8)
        ax.set_ylim(min(0.85, min(vals) - 0.01), 1.003); ax.set_ylabel('Spearman ρ over odor pairs:\nsimilarity blocked vs intact')
    else:
        ax.axhline(CEIL, color=L.BLOCKED, lw=1, ls=':'); ax.axhline(0, color=L.GRID, lw=0.8)
        ax.text(max(x) + 0.6, CEIL, 'ceiling set by the data\'s noise (≈ 0.84)', ha='right', va='bottom', fontsize=7, color=L.BLOCKED)
        ax.set_ylim(min(0, min(vals) - 0.04), 0.9); ax.set_ylabel('r over odor pairs: model change\nvs TNT − TNT3c change in similarity')
    ax.grid(axis='y', color=L.GRID, lw=0.5); ax.set_axisbelow(True)


def fig():
    f = plt.figure(figsize=(13.5, 12.5))
    gs = f.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=1.0, wspace=0.17)
    for j, rn in enumerate(['ORN', 'GH146']):
        a = f.add_subplot(gs[0, j]); draw(a, rn, 'rank')
        a.set_title(f'{"ab"[j]}  Does blocking APL reorder odor pairs? ({rn} input)')
        a = f.add_subplot(gs[1, j]); draw(a, rn, 'r')
        a.set_title(f'{"cd"[j]}  Does the model change match the data\'s change? ({rn} input)')
        a = f.add_subplot(gs[2, j])
        nk = [k for k in SC[rn] if k.startswith('null')][0]; best = NUL[NUL.responses == rn].version.iloc[0]
        nr = NUL[NUL.responses == rn].r_dS_with_data.values; real, nm, (q5, q95), p = SC[rn][nk]['r real, null mean, null 5-95%, p']
        a.hist(nr, bins=20, color=COMP, alpha=0.35, edgecolor=COMP, lw=0.6, label=f'{len(nr)} shuffles of KC sites on APL\n(within KC class)')
        a.axvline(real, color=COMP, lw=2, label=f'real wiring: r = {real:.2f} (p = {p:.2f})')
        ru = SC[rn]['r over odor pairs, model change (MB-averaged) vs data change']['U']
        a.axvline(ru, color=L.GRAY, lw=1.5, ls='--', label=f'uniform APL: r = {ru:.2f}')
        a.set_xlabel(f'r with the data change, APL confined to sub-arbors ({lam_lab(best)})'); a.set_ylabel('number of shuffles')
        a.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); a.legend(fontsize=7.2, loc='upper left')
        a.set_title(f'{"ef"[j]}  Does it matter which KC sits in which sub-arbor? ({rn})')
    hs = [plt.Line2D([], [], marker=MK[k], ls='', mfc='white', mec=L.MUTED, label=MBN[k]) for k in MBS]
    hs.append(plt.Line2D([], [], color=L.MUTED, lw=2.4, label='mean of the three\n(rank: of per-MB values;\nr: of the MB-averaged change)'))
    f.legend(handles=hs, loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    return f


if __name__ == '__main__':
    F = fig(); F.savefig(OW + 'APL_compartment_model.png', dpi=200, bbox_inches='tight', facecolor='white')
    print('wrote', OW + 'APL_compartment_model.png')
