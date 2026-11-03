"""Which odor pairs APL inhibition on PN boutons changes, in each mushroom body (Yang asked to see the data, 2026-10-21,
for the "APL on PN boutons" bullet of odor_weighted/APL_block_model_notes.md). Official ROIs only.

Model: scripts/apl_block_model.py (20% of KCs respond with APL blocked, 10% with it intact), read from
odor_weighted/APL_block_model_odor_pairs.csv. Similarity = cosine of two odors' KC population responses;
dcos = cos(blocked) - cos(intact).
Extra dcos = dcos with APL on PN boutons weighted per glomerulus (P) - dcos with every glomerulus alike (P0). The blocked
responses are the same in both, so it equals cos_intact(P0) - cos_intact(P). M - M0 is the same with the boutons at their
synapse share (10-11% of APL's calyx output) and the rest uniform on the KCs.
Bouton weight of a glomerulus (panels g, h): APL->PN synapses per PN->KC synapse of its uniglomerular PNs, divided by the
connectome's overall ratio (all uniglomerular PNs), mean of hemibrain and male CNS (APL_wiring_by_glomerulus.csv).
Connectomes: hemibrain and male CNS only (Yang, 2026-10-21); the FlyWire/BANC version is in
odor_weighted/archive_2026-10-21_four_connectomes/.
Glomerular check: cosine of two odors' glomerular response vectors with each glomerulus divided by (1 + c x its weight),
vs by (1 + c), for c = 1 and 3.
Outputs: odor_weighted/APL_bouton_pairs.png, APL_bouton_pairs.csv (the six named pairs per mushroom body),
APL_bouton_pairs.json."""
import os, json
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Rectangle

B = '/mnt/project-files/apl_synapses/'
OUT = B + 'odor_weighted/'
ORN_TABLE = '/mnt/project-files/uploads/hearth/5d477ed3-345f-4332-8b48-1ad3e2e0a0c1'   # ORN_ovlp16_resp_flyave_merge_dropna.csv
MAIN = '20% -> 10%'
S = 'cos'
MBS = ['hemibrain R', 'malecns L', 'malecns R']     # hemibrain and male CNS only (Yang, 2026-10-21)
DS = ['hemibrain', 'malecns']
NAMED = [('pa', 'eb'), ('pa', 'ep'), ('eb', '1-6ol'), ('eb', '1-8ol'), ('ep', '1-6ol'), ('ep', '1-8ol')]
FOCUS = ['pa', 'eb', 'ep', '1-6ol', '1-8ol']
INK, MUTED, GRID, NAN_C = '#0b0b0b', '#52514e', '#e1e0d9', '#d6d6d4'
GREEN, BLUE = '#1f9e6e', '#2a78d6'          # as P and U in APL_block_model.png
SEQ = LinearSegmentedColormap.from_list('seq', ['#f7f7f5', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = matplotlib.colormaps['RdBu_r']

orn = pd.read_csv(ORN_TABLE, index_col=0)
gh = pd.read_csv(OUT + 'GH146_mean_by_glomerulus.csv', index_col=0).drop(columns='n_flies')
col = {o: f'megamat: {o} @ -3' for o in orn.columns} | {'paa': 'glomeruli_diagnostics: paa @ -5'}
RESP = {'ORN': orn, 'GH146': pd.DataFrame({o: gh[c] for o, c in col.items() if c in gh.columns})}
ODORS = list(orn.columns)

PR = pd.read_csv(OUT + 'APL_block_model_odor_pairs.csv')
PR = PR[PR.coding_levels == MAIN].copy()
assert sorted(PR.mushroom_body.unique()) == sorted(MBS), PR.mushroom_body.unique()
PR['x_P'] = PR[f'{S}_intact_P0'] - PR[f'{S}_intact_P']
PR['x_M'] = PR[f'{S}_intact_M0'] - PR[f'{S}_intact_M']
PR['P0_minus_U'] = PR[f'{S}_intact_U'] - PR[f'{S}_intact_P0']       # dcos(P0) - dcos(U)

GW = pd.read_csv(OUT + 'APL_wiring_by_glomerulus.csv')
GW = GW[GW.PN_to_KC > 0]
tot = GW.groupby('dataset')[['APL_to_PN', 'PN_to_KC']].sum()
GW['rel'] = (GW.APL_to_PN / GW.PN_to_KC) / GW.dataset.map(tot.APL_to_PN / tot.PN_to_KC)
REL = GW.pivot_table(index='glomerulus', columns='dataset', values='rel').reindex(columns=DS).mean(1)


def pair(rn, a, b):
    P = PR[PR.responses == rn]
    q = P[((P.odor_a == a) & (P.odor_b == b)) | ((P.odor_a == b) & (P.odor_b == a))]
    return q.set_index('mushroom_body').reindex(MBS)


def cosv(x, y):
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


# ---------------- numbers ----------------
summary, rows = {'coding_levels': MAIN, 'similarity': 'cosine of KC population responses'}, []
tri = np.triu_indices(len(MBS), 1)
for rn, R in RESP.items():
    P = PR[(PR.responses == rn) & ~PR.odor_dropped]
    X = P.pivot_table(index=['odor_a', 'odor_b'], columns='mushroom_body', values='x_P', sort=False)[MBS]
    XM = P.pivot_table(index=['odor_a', 'odor_b'], columns='mushroom_body', values='x_M', sort=False)[MBS]
    m = X.mean(1); order = list(m.abs().sort_values(ascending=False).index)
    Q = PR[PR.responses == rn]
    s = {'pairs': int(len(X)), 'odors left out (< 0.5% of KCs respond)': [o for o in R.columns if Q[(Q.odor_a == o) | (Q.odor_b == o)].odor_dropped.all()],
         'mean |extra dcos| over pairs and MBs, all inhibition on boutons (P - P0)': round(float(X.abs().values.mean()), 3),
         'same, boutons at their synapse share (M - M0)': round(float(XM.abs().values.mean()), 3),
         'r between MBs of the P - P0 maps (median)': round(float(np.median(X.corr().values[tri])), 2),
         'mean |dcos(P0) - dcos(U)| over pairs and MBs': round(float(P.P0_minus_U.abs().mean()), 4),
         'largest |extra dcos| (mean over MBs)': {f'{a}-{b}': round(float(m[(a, b)]), 3) for a, b in order[:8]},
         'named pairs': {}}
    for a, b in NAMED:
        q = pair(rn, a, b)
        k = (a, b) if (a, b) in m.index else (b, a)
        s['named pairs'][f'{a}-{b}'] = {
            'extra dcos P - P0: mean, min, max over MBs': [round(float(q.x_P.mean()), 3), round(float(q.x_P.min()), 3), round(float(q.x_P.max()), 3)],
            'MBs with the same sign as the mean': int((np.sign(q.x_P) == np.sign(q.x_P.mean())).sum()),
            'rank by |mean| among pairs': order.index(k) + 1,
            'extra dcos M - M0: mean, min, max': [round(float(q.x_M.mean()), 3), round(float(q.x_M.min()), 3), round(float(q.x_M.max()), 3)],
            'cos blocked, intact P0, intact P, intact U (mean over MBs)': [round(float(q[f'{S}_{c}'].mean()), 3) for c in ['blocked', 'intact_P0', 'intact_P', 'intact_U']]}
        for mb in MBS:
            rows.append({'responses': rn, 'pair': f'{a}-{b}', 'mushroom_body': mb, **{f'{S}_{c}': q.loc[mb, f'{S}_{c}'] for c in
                         ['blocked', 'intact_U', 'intact_P0', 'intact_P', 'intact_M0', 'intact_M']},
                         'extra_dcos_P_minus_P0': q.loc[mb, 'x_P'], 'extra_dcos_M_minus_M0': q.loc[mb, 'x_M']})
    Rg = R.clip(lower=0); w = REL.reindex(Rg.index)
    s['glomerular check: cosine of response vectors, glomeruli alike -> per bouton weight'] = {
        f'c = {c}': {f'{a}-{b}': [round(cosv((Rg[a] / (1 + c)).values, (Rg[b] / (1 + c)).values), 2),
                                  round(cosv((Rg[a] / (1 + c * w)).values, (Rg[b] / (1 + c * w)).values), 2)] for a, b in NAMED} for c in (1, 3)}
    summary[rn] = s
pd.DataFrame(rows).round(5).to_csv(OUT + 'APL_bouton_pairs.csv', index=False)
json.dump(summary, open(OUT + 'APL_bouton_pairs.json', 'w'), indent=1, default=str)

# ---------------- figure ----------------
xs = PR[PR.responses.isin(RESP)]
lim_dots = np.ceil(max(abs(pd.concat([pair(rn, a, b)[c] for rn in RESP for a, b in NAMED for c in ('x_P', 'x_M')])).max(), 0.05) * 20 + 0.5) / 20
cols_r = [f'{S}_blocked', f'{S}_intact_P', f'{S}_intact_P0']
vals = pd.concat([pair(rn, a, b)[cols_r].mean() for rn in RESP for a, b in NAMED])
lim_arr = (np.floor(vals.min() * 10) / 10 - 0.05, min(1.0, np.ceil(vals.max() * 10) / 10 + 0.05))
VM = 0.2
NORM = Normalize(-VM, VM)
VMAX_R = float(max(RESP[rn].clip(lower=0)[FOCUS].values.max() for rn in RESP))

fig = plt.figure(figsize=(18.5, 17))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.86], hspace=0.5, top=0.895, bottom=0.05, left=0.05, right=0.965)


def heat(ax, rn, letter):
    s = summary[rn]
    P = PR[(PR.responses == rn) & ~PR.odor_dropped]
    m = P.groupby(['odor_a', 'odor_b']).x_P.mean()
    n = len(ODORS)
    img = np.ones((n, n, 4))
    for i in range(n):
        for j in range(i): img[i, j] = to_rgba(NAN_C)
    for (a, b), v in m.items():
        i, j = sorted((ODORS.index(a), ODORS.index(b)), reverse=True)
        img[i, j] = DIV(NORM(v))
    ax.imshow(img, interpolation='nearest')
    for a, b in NAMED:
        i, j = sorted((ODORS.index(a), ODORS.index(b)), reverse=True)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.8, zorder=3))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ODORS, rotation=90, fontsize=8, color=INK); ax.set_yticklabels(ODORS, fontsize=8, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    miss = [o for o in ODORS if o not in RESP[rn].columns] + s['odors left out (< 0.5% of KCs respond)']
    ax.set_title(f'{letter}  {rn} inputs: extra Δcos for every odor pair, mean of {len(MBS)} MBs\n'
                 f'mean |extra Δcos| {s["mean |extra dcos| over pairs and MBs, all inhibition on boutons (P - P0)"]:.3f}; '
                 f'same pattern in every MB (median r {s["r between MBs of the P - P0 maps (median)"]:.2f})\n'
                 f'boxes: the six pairs in the next panels; grey: {", ".join(miss)} (' +
                 ('not in the table' if rn == 'GH146' else '< 0.5% of KCs respond') + ')', fontsize=9.3, color=INK, loc='left')


def dots(ax, rn, letter):
    for k, (a, b) in enumerate(NAMED):
        q = pair(rn, a, b)
        for c, dy, filled in [('x_P', -0.2, True), ('x_M', 0.2, False)]:
            y = k + dy + np.linspace(-0.1, 0.1, len(MBS))
            ax.scatter(q[c], y, s=30, facecolor=GREEN if filled else 'white', edgecolor=GREEN, lw=1.2, zorder=3)
            mu = q[c].mean()
            ax.plot([mu, mu], [k + dy - 0.17, k + dy + 0.17], color=INK, lw=2.2, zorder=4, solid_capstyle='butt')
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
    ax.axhline(1.5, color=GRID, lw=1.2, zorder=0)
    ax.set_ylim(len(NAMED) - 0.45, -0.55)
    ax.set_yticks(range(len(NAMED))); ax.set_yticklabels([f'{a} – {b}' for a, b in NAMED], fontsize=9.5, color=INK)
    ax.set_xlim(-lim_dots, lim_dots)
    ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=8.5, colors=MUTED); ax.tick_params(axis='y', length=0)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.set_xlabel('extra Δcos: per glomerulus − all glomeruli alike', fontsize=9, color=MUTED)
    ax.annotate('← less alike when APL is blocked', xy=(0, 0), xycoords='axes fraction', xytext=(0, -41), textcoords='offset points',
                ha='left', va='top', fontsize=8.5, color=MUTED)
    ax.annotate('more alike when APL is blocked →', xy=(1, 0), xycoords='axes fraction', xytext=(0, -41), textcoords='offset points',
                ha='right', va='top', fontsize=8.5, color=MUTED)
    ax.set_title(f'{letter}  The six pairs in each mushroom body (one dot per MB)\n'
                 'filled: all of APL\'s inhibition on boutons (upper bound)\n'
                 f'open: boutons at their synapse share (10–11%), the rest uniform on KCs;  bar: mean of {len(MBS)} MBs',
                 fontsize=9.3, color=INK, loc='left')


def arrows(ax, rn, letter):
    for k, (a, b) in enumerate(NAMED):
        q = pair(rn, a, b)[cols_r].mean()
        xb = q[f'{S}_blocked']
        for c, dy, colr in [(f'{S}_intact_P', -0.17, GREEN), (f'{S}_intact_P0', 0.17, BLUE)]:
            x0 = q[c]
            if abs(xb - x0) > 0.006:
                ax.annotate('', xy=(xb, k + dy), xytext=(x0, k + dy), zorder=3,
                            arrowprops=dict(arrowstyle='-|>', color=colr, lw=2, mutation_scale=12, shrinkA=0, shrinkB=0))
            ax.scatter([x0], [k + dy], s=30, color=colr, zorder=4, lw=0)
        ax.plot([xb, xb], [k - 0.36, k + 0.36], color=INK, lw=1.8, zorder=5, solid_capstyle='butt')
    ax.axhline(1.5, color=GRID, lw=1.2, zorder=0)
    ax.set_ylim(len(NAMED) - 0.45, -0.55)
    ax.set_yticks(range(len(NAMED))); ax.set_yticklabels([f'{a} – {b}' for a, b in NAMED], fontsize=9.5, color=INK)
    ax.set_xlim(*lim_arr)
    ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=8.5, colors=MUTED); ax.tick_params(axis='y', length=0)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.set_xlabel(f'cosine similarity of the two odors\' KC responses (mean of {len(MBS)} MBs)', fontsize=9, color=MUTED)
    ax.set_title(f'{letter}  Similarity of the six pairs: dot = APL intact, arrow tip = blocked\n'
                 'green: bouton inhibition per glomerulus; blue: all glomeruli alike\n'
                 '(the same as a uniform APL on the KCs); black bar: APL blocked', fontsize=9.3, color=INK, loc='left')


HEAT_AXES = []
for row, rn, L in [(0, 'ORN', 'abc'), (1, 'GH146', 'def')]:
    sub = gs[row].subgridspec(1, 3, width_ratios=[0.86, 1.05, 1.05], wspace=0.34)
    HEAT_AXES.append(fig.add_subplot(sub[0])); heat(HEAT_AXES[-1], rn, L[0])
    dots(fig.add_subplot(sub[1]), rn, L[1])
    arrows(fig.add_subplot(sub[2]), rn, L[2])

sub = gs[2].subgridspec(1, 2, width_ratios=[36, 29], wspace=0.12)
axes_r = []
for i, (rn, letter) in enumerate([('ORN', 'g'), ('GH146', 'h')]):
    R = RESP[rn].clip(lower=0)[FOCUS]
    w = REL.reindex(R.index)
    order = w.sort_values().index
    g2 = sub[i].subgridspec(2, 1, height_ratios=[1, 1.35], hspace=0.06)
    axb = fig.add_subplot(g2[0]); axh = fig.add_subplot(g2[1])
    x = np.arange(len(order))
    axb.bar(x, w[order], color=MUTED, width=0.72, zorder=2)
    axb.axhline(1, color=INK, lw=0.8, ls=(0, (3, 3)), zorder=3)
    axb.set_xlim(-0.5, len(order) - 0.5); axb.set_xticks([])
    axb.set_ylabel('bouton weight', fontsize=8.5, color=MUTED); axb.tick_params(axis='y', labelsize=8, colors=MUTED)
    for sp in ['top', 'right', 'bottom']: axb.spines[sp].set_visible(False)
    im = axh.imshow(R.loc[order].T.values, aspect='auto', cmap=SEQ, vmin=0, vmax=VMAX_R, interpolation='nearest')
    axh.set_yticks(range(len(FOCUS))); axh.set_yticklabels(FOCUS, fontsize=9, color=INK)
    axh.set_xticks(x); axh.set_xticklabels(order, rotation=90, fontsize=8, color=INK)
    axh.tick_params(length=0)
    for sp in axh.spines.values(): sp.set_visible(False)
    axb.set_title(f'{letter}  {rn} responses of the five odors per glomerulus (negative set to 0), glomeruli ordered by bouton weight:\n'
                  'APL→PN synapses per PN→KC synapse of the glomerulus\'s PNs ÷ the connectome\'s overall ratio; mean of hemibrain and male CNS',
                  fontsize=9.3, color=INK, loc='left')
    axes_r.append((axb, axh))
fig.canvas.draw()
p = axes_r[-1][1].get_position()
cax = fig.add_axes([p.x1 + 0.006, p.y0, 0.007, p.height])
cb = fig.colorbar(im, cax=cax); cb.set_label('response', fontsize=8.5, color=MUTED)
cb.ax.tick_params(labelsize=8, colors=MUTED); cb.outline.set_visible(False)
rend = fig.canvas.get_renderer()
tb = axes_r[0][1].get_tightbbox(rend).transformed(fig.transFigure.inverted())
gc = summary['GH146']['glomerular check: cosine of response vectors, glomeruli alike -> per bouton weight']['c = 1']
go = summary['ORN']['glomerular check: cosine of response vectors, glomeruli alike -> per bouton weight']['c = 1']
fig.text(0.05, tb.y0 - 0.012,
         'Why these pairs (g, h).  Bouton inhibition holds down glomeruli with a high weight (right) more than those with a low weight (left), so with APL intact '
         'each odor is represented mostly by its low-weight glomeruli.\n'
         'For eb, ep, 1-6ol and 1-8ol these are the same glomeruli (DM5, VM2 and VM7d, all below the overall ratio), so with APL intact they look more alike than with even inhibition, and blocking APL '
         'makes them less alike.\n'
         'For pa they are mostly DM3, while much of what pa shares with eb and ep is in DM2 (high weight), so with APL intact pa looks less like eb and ep, and blocking '
         'APL brings them together.\n'
         'The glomerular responses alone show the same: dividing each glomerulus by 1 + its weight changes the GH146 cosine of eb and 1-6ol from '
         f'{gc["eb-1-6ol"][0]:.2f} to {gc["eb-1-6ol"][1]:.2f} and of pa and eb from {gc["pa-eb"][0]:.2f} to {gc["pa-eb"][1]:.2f} '
         f'(ORN table: {go["eb-1-6ol"][0]:.2f} to {go["eb-1-6ol"][1]:.2f} and {go["pa-eb"][0]:.2f} to {go["pa-eb"][1]:.2f}).',
         fontsize=9.5, color=INK, ha='left', va='top')

so, sg = summary['ORN'], summary['GH146']
fig.suptitle('Which odor pairs does APL inhibition on PN boutons change?  KC model on each connectome\'s PN→KC claws (hemibrain R, male CNS L and R; official ROIs); 20% of KCs respond with APL blocked, '
             '10% with it intact\n'
             'cos = cosine similarity of two odors\' KC population responses; Δcos = cos(blocked) − cos(intact). Extra Δcos = Δcos with bouton inhibition weighted per glomerulus by its '
             'APL→PN synapses − Δcos with every glomerulus alike.\n'
             'Positive: blocking APL makes the pair more similar than it would with the inhibition spread evenly. Every glomerulus alike gives the same Δcos as a uniform APL on the KCs '
             f'(mean |difference| {so["mean |dcos(P0) - dcos(U)| over pairs and MBs"]:.3f} with ORN, {sg["mean |dcos(P0) - dcos(U)| over pairs and MBs"]:.3f} with GH146 inputs).',
             fontsize=10.5, color=INK, x=0.01, ha='left', y=0.985)
for ax0 in HEAT_AXES:
    p0 = ax0.get_position()
    cax = fig.add_axes([p0.x1 + 0.006, p0.y0, 0.006, p0.height])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=DIV), cax=cax)
    cb.set_label('extra Δcos', fontsize=8.5, color=MUTED); cb.ax.tick_params(labelsize=8, colors=MUTED); cb.outline.set_visible(False)
fig.savefig(OUT + 'APL_bouton_pairs.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print(json.dumps(summary, indent=1, default=str)[:6000])
