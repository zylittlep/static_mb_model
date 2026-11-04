"""Claws per hemibrain KC by three segmentations (Yang asked 2026-11-04): the old 2 µm claws, the new rule chosen by Yang
(8 µm geodesic clusters of all calyx sites, split by PN partner, a PN needs >= 2 PSDs; claw_v2m2), and Prat's segmentation
(the uploaded KC_Claw_Counts_Hemibrain csv: n_claws and n_claws_corrected per KC; counts only, no claw positions).

Usage: python3 compare_three.py <claws_per_KC.csv from compare_claws.py> <Prat csv> <out dir>"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

CPK, PRAT, OUT = sys.argv[1:4]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#c9c8c2'
C = {'old': '#eda100', 'new': '#1baf7a', 'Prat': '#6250d6'}
SHORT = {'old': 'old', 'new': 'new', 'Prat': "Prat's"}
LAB = {'old': 'old (2 µm)', 'new': 'new (8 µm, split by PN, ≥2 PSDs)', 'Prat': "Prat's (corrected)"}
NEWCOL = 'new v2, all sites, split by PN (PN claw >= 2 PSDs)'

k = pd.read_csv(CPK)
k = k[k.mb == 'hemibrain R'].set_index('kc_id')
p = pd.read_csv(PRAT).set_index('bodyId')
H = pd.DataFrame({'old': k['old (2 µm, PN PSDs)'], 'new': k[NEWCOL], 'type': k.kc_type}).join(
    p[['KC_subtype', 'n_claws', 'n_claws_corrected']], how='inner').rename(columns={'n_claws_corrected': 'Prat', 'n_claws': 'Prat_before'})
missing = p.loc[~p.index.isin(k.index)]


def stats(a, b):
    d = H[a] - H[b]
    return {'pair': f'{a} vs {b}', 'r': round(float(np.corrcoef(H[a], H[b])[0, 1]), 3), 'mean difference': round(float(d.mean()), 2),
            'same count': round(float((d == 0).mean()), 3), 'within 1': round(float((d.abs() <= 1).mean()), 3),
            'within 2': round(float((d.abs() <= 2).mean()), 3)}


AG = pd.DataFrame([stats('old', 'Prat'), stats('new', 'Prat'), stats('old', 'Prat_before'), stats('new', 'Prat_before'), stats('new', 'old')])
AG.to_csv(os.path.join(OUT, 'three_way_agreement.csv'), index=False)
order = ["KCg-m", "KCg-d", "KCg-t", "KCab-c", "KCab-s", "KCab-m", "KCab-p", "KCa'b'-ap1", "KCa'b'-ap2", "KCa'b'-m"]
BY = H.groupby('type')[['old', 'new', 'Prat', 'Prat_before']].agg(['mean', 'size']).reindex(order).dropna(how='all')
BY.columns = [f'{a} {b}' for a, b in BY.columns]
BY = BY[['old size', 'old mean', 'new mean', 'Prat mean', 'Prat_before mean']].rename(columns={'old size': 'KCs'})
BY.round(2).to_csv(os.path.join(OUT, 'three_way_by_subtype.csv'))

fig, axs = plt.subplots(2, 3, figsize=(15, 9.6))


def heat(ax, a, b):
    x, y = H[b].astype(int).values, H[a].astype(int).values
    m = int(max(x.max(), y.max())) + 1
    M = np.zeros((m, m)); np.add.at(M, (y, x), 1)
    im = ax.imshow(np.where(M > 0, M, np.nan), origin='lower', cmap='Blues', aspect='equal', interpolation='nearest')
    ax.plot([0, m - 1], [0, m - 1], color=MUTED, lw=0.8, ls='--')
    s = stats(a, b)
    ax.set_title(f'{LAB[a]} vs {LAB[b]}\nr = {s["r"]:.2f}, mean diff {s["mean difference"]:+.2f}, same {s["same count"]:.0%}, '
                 f'within 1 {s["within 1"]:.0%}', fontsize=9.5, loc='left', color=INK)
    ax.set_xlabel(f'claws per KC, {LAB[b]}', fontsize=9); ax.set_ylabel(f'claws per KC, {LAB[a]}', fontsize=9)
    ax.set_xlim(-0.5, 22.5); ax.set_ylim(-0.5, 22.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('KCs', fontsize=8); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)


heat(axs[0, 0], 'old', 'Prat'); heat(axs[0, 1], 'new', 'Prat'); heat(axs[0, 2], 'new', 'old')

ax = axs[1, 0]
for m in ['old', 'new', 'Prat']:
    v = H[m].value_counts().sort_index(); v = v / v.sum()
    ax.plot(v.index, v.values, marker='o', ms=4, lw=2, color=C[m], label=SHORT[m])
ax.set_xlabel('claws per KC', fontsize=9); ax.set_ylabel('share of KCs', fontsize=9); ax.set_xlim(-0.5, 22.5)
ax.legend(frameon=False, fontsize=8.5, loc='upper right'); ax.set_title('Distribution of claws per KC\n' + ', '.join(f'{SHORT[m]} mean {H[m].mean():.1f}' for m in ['old', 'new', 'Prat']), fontsize=9.5, loc='left')

ax = axs[1, 1]
yy = np.arange(len(BY))
for i, m in enumerate(['old', 'new', 'Prat']):
    ax.scatter(BY[f'{m} mean'], yy + (i - 1) * 0.22, s=36, color=C[m], label=LAB[m], zorder=3)
for j in yy: ax.axhline(j, color=GRID, lw=0.5, zorder=1)
ax.set_yticks(yy, [f'{t} (n = {int(n)})' for t, n in zip(BY.index, BY.KCs)], fontsize=8); ax.invert_yaxis()
ax.set_xlabel('mean claws per KC', fontsize=9); ax.set_xlim(0, 12)
ax.set_title('Mean claws per KC by hemibrain KC subtype\n(colors as in the left panel)', fontsize=9.5, loc='left')

ax = axs[1, 2]
bins = np.arange(-8.5, 12.5, 1)
for m in ['old', 'new']:
    d = H[m] - H['Prat']
    ax.hist(d.clip(-8, 12), bins=bins, histtype='step', lw=2, color=C[m], density=True, label=f'{SHORT[m]} − Prat\'s')
ax.axvline(0, color=MUTED, lw=0.8, ls='--')
ax.set_xlabel('difference in claws per KC (clipped at −8 and +12)', fontsize=9); ax.set_ylabel('share of KCs', fontsize=9)
ax.legend(frameon=False, fontsize=8, loc='upper right'); ax.set_title('Difference from Prat\'s count, per KC', fontsize=9.5, loc='left')

for a in axs.ravel():
    for sp in ['top', 'right']: a.spines[sp].set_visible(False)
    a.tick_params(labelsize=8)
fig.suptitle(f'Claws per KC in hemibrain R by three segmentations ({len(H):,} KCs in all three). Prat\'s counts are the corrected ones '
             f'(n_claws_corrected).\n{len(missing)} of Prat\'s KCs have no corrected-calyx PN input in our tables and are left out.',
             fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, 'three_way_claw_comparison.png'), dpi=130); plt.close(fig)

summ = {'KCs': len(H), 'Prat KCs left out': len(missing), 'left out by subtype': missing.KC_subtype.value_counts().to_dict(),
        'left out with 0 claws in Prat': int((missing.n_claws_corrected == 0).sum()),
        'means': {m: round(float(H[m].mean()), 2) for m in ['old', 'new', 'Prat', 'Prat_before']},
        'agreement': AG.to_dict('records')}
json.dump(summ, open(os.path.join(OUT, 'three_way_summary.json'), 'w'), indent=1)
print(json.dumps(summ, indent=1)); print(BY.round(2))
