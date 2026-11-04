"""Compare the geodesic claws with the old 2 um claws and with Prat's hemibrain claw counts; draw the 40 KCs where the old
and new methods differ most (2D pages + a 3D viewer) and a summary figure.

Usage: python3 compare_claws.py <out dir with <ds>_claw_sites.parquet> <hemibrain swc dir> <malecns swc dir> <Prat's claw-count csv (Yang's upload)>
Writes into the out dir: claws_per_KC.csv, claws_new_<variant>.csv, claw_compare_summary.json, claw_compare_summary.png,
claw_disparity_KCs_page<k>.png + claw_disparity_KCs.pdf, claw_disparity_KCs_3D.html, disparity_KCs.csv."""
import sys, os, json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree

OUT, SWC_HB, SWC_MC, YANG = sys.argv[1:5]
MAIN = os.environ.get('CLAW_MAIN', 'pn')            # main new version: 'all' or 'pn' (2026-11-03 rule), 'v2' (Yang's rule of 2026-11-04)
REC = os.environ.get('CLAW_REC', '1') == '1'           # split recursively (2026-11-03 rule)
V2M = os.environ.get('CLAW_V2M', '1')                  # v2: minimum PSDs for a PN to make its own claw
if MAIN == 'v2':
    NEWCOL, SPLITCOL, BASE = f'claw_v2m{V2M}', f'split_v2m{V2M}', 'all'
else:
    NEWCOL = ('clawrec_' if REC else 'claw_') + MAIN; SPLITCOL = ('splitrec_' if REC else 'split_') + MAIN; BASE = MAIN
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
MK = ['o', 's', '^', 'D']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#c9c8c2'


def glom(t, ds):
    t = str(t)
    if t.startswith('M_') or '_' not in t: return '?'
    g = [n for n in t.split('_')[0].split('+') if re.fullmatch(GPAT, n)]
    g = [HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in g]
    return '+'.join(g) if g else '?'


def kc_class(t):
    t = str(t)
    if t.startswith('KCg') or t.startswith('KCy'): return 'γ'
    if t.startswith("KCa'b'"): return "α′β′"
    if t.startswith('KCab'): return 'αβ'
    return 'other'


S = {}
for ds in ['hemibrain', 'malecns']:
    s = pd.read_parquet(os.path.join(OUT, f'{ds}_claw_sites.parquet')); s['dataset'] = ds
    s['mb'] = ('hemibrain ' if ds == 'hemibrain' else 'male CNS ') + s.calyx
    s['glom'] = [glom(t, ds) if p else '' for t, p in zip(s.pn_type, s.is_PN)]
    S[ds] = s
S = pd.concat(S.values(), ignore_index=True)
S['kc_class'] = S.kc_type.map(kc_class)

# ---------------------------------------------------------------- claw tables
def claw_table(s, col, splitcol=None):
    c = s[s[col] >= 0]
    g = c.groupby(['mb', 'kc_id', col])
    t = g.agg(kc_type=('kc_type', 'first'), n_sites=('kind', 'size'), n_PN_PSDs=('is_PN', 'sum'),
              x=('x', 'mean'), y=('y', 'mean'), z=('z', 'mean')).reset_index().rename(columns={col: 'claw'})
    pnp = c[c.is_PN].groupby(['mb', 'kc_id', col]).partner_id.agg(lambda v: v.value_counts().index[0]).rename('main_PN_id')
    pt = c[c.is_PN].groupby(['mb', 'kc_id', col]).apply(
        lambda d: pd.Series({'main_PN_type': d.pn_type[d.partner_id == d.partner_id.value_counts().index[0]].iloc[0],
                             'main_PN_share': d.partner_id.value_counts().iloc[0] / len(d),
                             'second_PN_PSDs': int(d.partner_id.value_counts().iloc[1]) if d.partner_id.nunique() > 1 else 0}),
        include_groups=False)
    t = t.join(pnp, on=['mb', 'kc_id', 'claw']).join(pt, on=['mb', 'kc_id', 'claw'])
    t['glomerulus'] = [glom(x, 'hemibrain' if m.startswith('hemi') else 'malecns') for x, m in zip(t.main_PN_type, t.mb)]
    if splitcol: t = t.join(c.groupby(['mb', 'kc_id', col])[splitcol].any().rename('from_split'), on=['mb', 'kc_id', 'claw'])
    return t


VERS = {'old (2 µm, PN PSDs)': ('claw_old', None), 'new, all sites': ('claw_all', 'split_all'),
        'new, all sites, recursive split': ('clawrec_all', 'splitrec_all'), 'new, PN PSDs': ('claw_pn', 'split_pn'),
        'new, PN PSDs, recursive split': ('clawrec_pn', 'splitrec_pn')}
for _m in ['1', '2', '3', '5']:
    if f'claw_v2m{_m}' in S.columns:
        VERS[f'new v2, all sites, split by PN (PN claw >= {_m} PSD{"s" if _m != "1" else ""})'] = (f'claw_v2m{_m}', f'split_v2m{_m}')
T = {k: claw_table(S, c, sc) for k, (c, sc) in VERS.items()}
pre = S[S.is_PN].groupby(['mb', 'kc_id']).agg(kc_type=('kc_type', 'first'), n_PN_PSDs=('is_PN', 'size'))
K = pre.copy()
for k, (c, _) in VERS.items():
    K[k] = T[k].groupby(['mb', 'kc_id']).size().reindex(K.index).fillna(0).astype(int)
for v in ['all', 'pn']:     # clusters with PN input before any split
    K[f'new {v} sites, before split'] = S[(S[f'cl0_{v}'] >= 0) & S.is_PN].groupby(['mb', 'kc_id'])[f'cl0_{v}'].nunique().reindex(K.index).fillna(0).astype(int)
K['old, claws with >=3 PN PSDs'] = T['old (2 µm, PN PSDs)'].query('n_PN_PSDs >= 3').groupby(['mb', 'kc_id']).size().reindex(K.index).fillna(0).astype(int)
K = K.reset_index()
y = pd.read_csv(YANG)
K = K.merge(y[['bodyId', 'KC_type', 'n_claws', 'n_claws_corrected']].rename(columns={'bodyId': 'kc_id', 'KC_type': 'Prat_KC_type',
            'n_claws': 'Prat_n_claws', 'n_claws_corrected': 'Prat_n_claws_corrected'}), on='kc_id', how='left')
K.loc[~K.mb.str.startswith('hemi'), ['Prat_n_claws', 'Prat_n_claws_corrected']] = np.nan
K['kc_class'] = K.kc_type.map(kc_class)
mainname = [k for k, (c, _) in VERS.items() if c == NEWCOL][0]
K['new_main'] = K[mainname]; K['old'] = K['old (2 µm, PN PSDs)']

# agreement with Prat (hemibrain)
H = K[K.mb.str.startswith('hemi') & K.Prat_n_claws.notna()]
agree = {}
for k in [c for c in K.columns if c.startswith('old') or c.startswith('new')]:
    if k in ('new_main',): continue
    for ref in ['Prat_n_claws', 'Prat_n_claws_corrected']:
        d = H[k] - H[ref]
        agree[f'{k} vs {ref}'] = {'r': round(float(np.corrcoef(H[k], H[ref])[0, 1]), 3), 'mean_diff': round(float(d.mean()), 2),
                                  'exact': round(float((d == 0).mean()), 3), 'within_1': round(float((d.abs() <= 1).mean()), 3)}
# missing-from-us and missing-from-Prat
yk = set(y.bodyId); hk = set(H.kc_id)

# ---------------------------------------------------------------- old vs new agreement per KC (on PN PSDs)
def ari(a, b):
    from math import comb
    ct = pd.crosstab(a, b).values
    sc = sum(comb(int(n), 2) for n in ct.ravel()); sa = sum(comb(int(n), 2) for n in ct.sum(1)); sb = sum(comb(int(n), 2) for n in ct.sum(0))
    tot = comb(int(ct.sum()), 2)
    if tot == 0: return 1.0
    exp = sa * sb / tot; mx = (sa + sb) / 2
    return 1.0 if mx == exp else (sc - exp) / (mx - exp)


P = S[S.is_PN]
K = K.join(P.groupby(['mb', 'kc_id']).apply(lambda d: ari(d.claw_old.values, d[NEWCOL].values), include_groups=False).rename('ARI_old_new'),
           on=['mb', 'kc_id'])
K['delta'] = K.new_main - K.old
K['score'] = K.delta.abs() + (1 - K.ARI_old_new)
K.to_csv(os.path.join(OUT, 'claws_per_KC.csv'), index=False)
for k in VERS:
    if k.startswith('new'):
        T[k].to_csv(os.path.join(OUT, 'claws_' + k.replace(', ', '_').replace(' ', '_') + '.csv'), index=False)

# ---------------------------------------------------------------- pick 40 KCs
rng = np.random.default_rng(1103)
quota = {'hemibrain R': 20, 'male CNS L': 10, 'male CNS R': 10}
pick = []
for mb, q in quota.items():
    k = K[(K.mb == mb) & K.kc_class.isin(['γ', 'αβ', "α′β′"])].sort_values('score', ascending=False)
    per = {c: q // 3 + (1 if i < q % 3 else 0) for i, c in enumerate(['γ', 'αβ', "α′β′"])}
    for c, n in per.items(): pick.append(k[k.kc_class == c].head(n))
pick = pd.concat(pick).sort_values(['mb', 'kc_class', 'score'], ascending=[True, True, False])
pick.to_csv(os.path.join(OUT, 'disparity_KCs.csv'), index=False)


# ---------------------------------------------------------------- skeleton helpers
def swc(ds, kid):
    p = os.path.join(SWC_HB if ds == 'hemibrain' else SWC_MC, f'{kid}.swc')
    s = pd.read_csv(p, comment='#', sep=r'\s+', header=None, names=['id', 't', 'x', 'y', 'z', 'r', 'p'])
    xyz = s[['x', 'y', 'z']].values * 8.
    pos = pd.Series(np.arange(len(s)), index=s.id.values)
    ch = np.where(s.p.values != -1)[0]; par = pos.reindex(s.p.values[ch]).values.astype(int)
    return xyz, ch, par


def calyx_part(xyz, ch, par, sites):
    near = np.isfinite(cKDTree(sites).query(xyz, distance_upper_bound=3000)[0])
    keep = near[ch] & near[par]
    return ch[keep], par[keep]


def view_basis(pts):
    c = pts.mean(0); u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[:2]


def style(i):
    return COL[i % 8], MK[(i // 8) % 4]


# ---------------------------------------------------------------- 2D pages
def glom_styles(s):
    """One color + marker per glomerulus of the KC's PN inputs (most PSDs first), the same in every panel."""
    g = s.glom[s.is_PN].replace('', '?').value_counts()
    return {k: style(i) for i, k in enumerate(g.index)}


def claw_glom(s, col):
    """main glomerulus of each claw (most PN PSDs)"""
    c = s[(s[col] >= 0) & s.is_PN]
    return c.groupby(col).glom.agg(lambda v: v.replace('', '?').value_counts().index[0])


def hull(ax, q, color):
    from scipy.spatial import ConvexHull
    if len(q) < 3: return
    try:
        h = ConvexHull(q)
    except Exception:
        return
    v = np.r_[h.vertices, h.vertices[:1]]
    ax.plot(q[v, 0], q[v, 1], color=color, lw=0.7, alpha=0.8, zorder=2)


def draw_kc(ax_old, ax_new, row):
    ds = 'hemibrain' if row.mb.startswith('hemi') else 'malecns'
    s = S[(S.mb == row.mb) & (S.kc_id == row.kc_id)]
    pts = s[['x', 'y', 'z']].values
    xyz, ch, par = swc(ds, row.kc_id); a, b = calyx_part(xyz, ch, par, pts)
    c, B = view_basis(pts)
    proj = lambda X: (X - c) @ B.T / 1e3
    sk = proj(xyz)
    GS = glom_styles(s)
    q = proj(pts)
    isPN = s.is_PN.values
    pg = s.glom.replace('', '?').values
    for ax, col, title in [(ax_old, 'claw_old', 'old: 2 µm, PN PSDs'), (ax_new, NEWCOL, 'new: 8 µm along skeleton, split by PN' if MAIN == 'v2' else 'new: 8 µm along skeleton')]:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(np.stack([sk[a], sk[b]], 1), colors=GRID, linewidths=0.8, zorder=1))
        lab = s[col].values
        cg = claw_glom(s, col)
        # other (non-PN) sites: colored by their claw's glomerulus, grey when not on a claw (old claws hold PN PSDs only)
        oth = ~isPN
        on = oth & (lab >= 0)
        ax.scatter(q[oth & ~on, 0], q[oth & ~on, 1], s=3, c='#cfcec8', zorder=2, lw=0)
        if on.any():
            ax.scatter(q[on, 0], q[on, 1], s=4, c=[GS.get(cg.get(l, '?'), ('#bdbcb6', 'o'))[0] for l in lab[on]], alpha=0.55, zorder=3, lw=0)
        # PN PSDs: color + marker = the presynaptic PN's glomerulus (identical in both panels); hollow = not on a claw
        for g, (colr, mk) in GS.items():
            m = isPN & (pg == g)
            ax.scatter(q[m & (lab >= 0), 0], q[m & (lab >= 0), 1], s=13, c=colr, zorder=4, lw=0.3, edgecolors='white', marker=mk)
            ax.scatter(q[m & (lab < 0), 0], q[m & (lab < 0), 1], s=13, facecolors='none', zorder=4, lw=0.8, edgecolors=colr, marker=mk)
        # claw outlines and labels
        for k in np.unique(lab[lab >= 0]):
            mm = lab == k
            colr = GS.get(cg.get(k, '?'), ('#898781', 'o'))[0]
            hull(ax, q[mm], colr)
            if (mm & isPN).sum() >= 3:
                cx, cy = q[mm].mean(0)
                sp = col != 'claw_old' and s[SPLITCOL][mm].any()
                ax.text(cx, cy, cg.get(k, '?') + ('*' if sp else ''), fontsize=5.5, color=INK, ha='center', va='center', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.75))
        n = len(np.unique(lab[lab >= 0]))
        head = (f'{row.mb} · {row.kc_id} · {row.kc_type} · {int(row.n_PN_PSDs)} PN PSDs'
                + ('' if pd.isna(row.Prat_n_claws) else f' · Prat {int(row.Prat_n_claws)} (corrected {int(row.Prat_n_claws_corrected)})'))
        ax.set_title((head if col == 'claw_old' else '') + f'\n{title}: {n} claws', fontsize=7.5, color=INK, loc='left', pad=3)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_visible(False)
    lim_x = np.percentile(q[:, 0], [0, 100]) + [-2, 2]; lim_y = np.percentile(q[:, 1], [0, 100]) + [-2, 2]
    for ax in (ax_old, ax_new):
        ax.set_xlim(*lim_x); ax.set_ylim(*lim_y)
        x0 = lim_x[0] + 0.5; y0 = lim_y[0] + 0.5
        ax.plot([x0, x0 + 5], [y0, y0], color=INK2, lw=1.2); ax.text(x0 + 2.5, y0 + 0.4, '5 µm', fontsize=5, ha='center', color=INK2)


PER = 8
pages = [pick.iloc[i:i + PER] for i in range(0, len(pick), PER)]
pdf = PdfPages(os.path.join(OUT, 'claw_disparity_KCs.pdf'))
for pi, pg in enumerate(pages):
    fig, axs = plt.subplots(PER // 2, 4, figsize=(15, 17))
    for i, (_, row) in enumerate(pg.iterrows()):
        r, cc = i // 2, (i % 2) * 2
        draw_kc(axs[r, cc], axs[r, cc + 1], row)
    for j in range(len(pg), PER):
        r, cc = j // 2, (j % 2) * 2; axs[r, cc].axis('off'); axs[r, cc + 1].axis('off')
    fig.suptitle(f'KCs where the old and new claw methods differ most (page {pi + 1} of {len(pages)})\n'
                 + (('Each KC: left = old claws (2 µm straight-line single linkage on PN→KC PSDs); right = new claws (Yang\'s rule of 2026-11-04: all calyx synapse\n'
                     'sites clustered by single linkage at 8 µm along the KC\'s skeleton; each PN with ≥' + V2M + ' PSD(s) in a cluster gets its own claw; other sites join the claw\n'
                     'of the nearest PN→KC PSD if it is ≤8 µm away along the skeleton).\n') if MAIN == 'v2' else
                    ('Each KC: left = old claws (2 µm straight-line single linkage on PN→KC PSDs); right = new claws (single linkage at 8 µm along the KC\'s\n'
                     'skeleton on ' + ('all its calyx synapse sites' if MAIN == 'all' else 'its PN→KC PSDs') + '; clusters with >40 ' + ('sites' if MAIN == 'all' else 'PN PSDs') + ' and a second PN with ≥5 PSDs split by PN, repeatedly).\n'))
                 + 'Big symbols = PN→KC PSDs; color + shape = glomerulus of the presynaptic PN (the same in both panels; hollow = PSD not on a claw).\n'
                 + ('Small dots = the KC\'s other calyx synapse sites, colored by the glomerulus of the claw they join (grey = not on a claw). Thin outline = one claw.\n' if MAIN != 'pn' else
                    'Small grey dots = the KC\'s other calyx synapse sites (not used to make claws). Thin outline = one claw.\n')
                 + 'Label = glomerulus of the claw\'s main PN (claws with ≥3 PN PSDs); * = claw ' + ('sharing its 8 µm cluster with another PN\'s claw' if MAIN == 'v2' else 'made by a split') + '. Grey lines = KC skeleton near its calyx sites.\n'
                 'View = the plane in which the KC\'s calyx sites spread most (same for both panels).', fontsize=9.5, color=INK, x=0.01, ha='left', y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if MAIN == 'v2' else 0.93), h_pad=2.5)
    fig.savefig(os.path.join(OUT, f'claw_disparity_KCs_page{pi + 1}.png'), dpi=110); pdf.savefig(fig); plt.close(fig)
pdf.close()

# ---------------------------------------------------------------- summary figure
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
def heat(ax, x, yv, xl, yl, title):
    m = int(max(x.max(), yv.max())) + 1
    H2 = np.zeros((m, m)); np.add.at(H2, (yv.astype(int), x.astype(int)), 1)
    ax.imshow(np.where(H2 > 0, H2, np.nan), origin='lower', cmap='Blues', aspect='equal')
    ax.plot([0, m - 1], [0, m - 1], color=INK2, lw=0.8, ls='--')
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    r = np.corrcoef(x, yv)[0, 1]; d = (yv - x)
    ax.set_title(f'{title}\nr = {r:.2f}, mean diff {d.mean():+.2f}, same {np.mean(d == 0):.0%}, within 1 {np.mean(np.abs(d) <= 1):.0%}',
                 fontsize=9, loc='left')
heat(axs[0, 0], H.Prat_n_claws_corrected, H.old, "Prat's corrected claws per KC", 'old claws per KC', 'Hemibrain R: old vs Prat (corrected)')
heat(axs[0, 1], H.Prat_n_claws_corrected, H.new_main, "Prat's corrected claws per KC", 'new claws per KC', 'Hemibrain R: new vs Prat (corrected)')
heat(axs[0, 2], H.Prat_n_claws, H[f'new {BASE} sites, before split'], "Prat's claws per KC (before correction)", 'new clusters per KC, before split',
     'Hemibrain R: new before split vs Prat (before)')
ax = axs[1, 0]
for i, mb in enumerate(['hemibrain R', 'male CNS L', 'male CNS R']):
    d = K[K.mb == mb].delta.value_counts().sort_index()
    ax.plot(d.index, d.values / d.sum(), marker='o', ms=4, lw=1.5, color=COL[i], label=mb)
ax.set_xlabel('new − old claws per KC', fontsize=9); ax.set_ylabel('share of KCs', fontsize=9); ax.legend(frameon=False, fontsize=8)
ax.set_title('Change in claws per KC, by mushroom body', fontsize=9, loc='left')
ax = axs[1, 1]
for i, (k, lab) in enumerate([('old (2 µm, PN PSDs)', 'old'), (mainname, 'new')]):
    v = T[k].n_PN_PSDs
    ax.hist(v.clip(upper=80), bins=np.arange(0, 82, 2), histtype='step', lw=1.6, color=COL[i], label=f'{lab} (median {v.median():.0f})')
ax.set_xlabel('PN→KC PSDs per claw (80 = 80 or more)', fontsize=9); ax.set_ylabel('claws', fontsize=9); ax.legend(frameon=False, fontsize=8)
ax.set_title('Claw size, all three mushroom bodies', fontsize=9, loc='left')
ax = axs[1, 2]
tt = T[mainname]; cls = ['γ', 'αβ', "α′β′"]
x = np.arange(3); w = 0.38
for i, (col, lab) in enumerate([('old', 'old'), ('new_main', 'new')]):
    v = [H[H.kc_class == c][col].mean() for c in cls]
    ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=COL[i], label=lab)
yv = [H[H.kc_class == c].Prat_n_claws_corrected.mean() for c in cls]
ax.scatter(x + 0.5 * w, yv, color=INK, marker='_', s=300, zorder=5, label="Prat, corrected")
ax.set_xticks(x, cls); ax.set_ylabel('mean claws per KC', fontsize=9); ax.legend(frameon=False, fontsize=8)
ax.set_title('Hemibrain R: mean claws per KC by class', fontsize=9, loc='left')
for a in axs.ravel():
    for sp_ in ['top', 'right']: a.spines[sp_].set_visible(False)
    a.tick_params(labelsize=8)
fig.suptitle(f'Geodesic claws ({mainname}) vs the old 2 µm claws and Prat\'s hemibrain counts', fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(OUT, 'claw_compare_summary.png'), dpi=130); plt.close(fig)

# ---------------------------------------------------------------- numbers
_c = S[S[f'cl0_{BASE}'] >= 0]
_n = _c.groupby(['mb', 'kc_id', f'cl0_{BASE}']).size()
_p = _c[_c.is_PN].groupby(['mb', 'kc_id', f'cl0_{BASE}', 'partner_id']).size().sort_values(ascending=False)
_second = _p.groupby(level=[0, 1, 2]).apply(lambda v: v.iloc[1] if len(v) > 1 else 0)
BIGSHARE = ((_n.reindex(_second.index) > 40) & (_second >= 5)).mean()
spl = T[mainname]
summ = {'main_new_version': mainname, 'agreement_with_Prat_hemibrain': agree,
        'KCs': K.groupby('mb').size().to_dict(),
        'hemibrain KCs in Prat but not here': len(yk - hk), 'hemibrain KCs here but not in Prat': len(set(K[K.mb.str.startswith('hemi')].kc_id) - yk),
        'claws per KC mean (old, new) by MB': {mb: [round(float(g.old.mean()), 2), round(float(g.new_main.mean()), 2)] for mb, g in K.groupby('mb')},
        'claws from a split, share': round(float(spl.from_split.mean()), 3),
        'clusters before split (with PN input): share with >40 sites and a second PN >= 5 PSDs': round(float(BIGSHARE), 3),
        'purity: PSD-weighted share of a claw\'s PN PSDs from its main PN (old, new)': [
            round(float((lambda p: (p.groupby(['mb', 'kc_id', c]).partner_id.agg(lambda v: v.value_counts().iloc[0]).sum() / len(p)))(P[P[c] >= 0])), 3)
            for c in ['claw_old', NEWCOL]],
        'median ARI old vs new': round(float(K.ARI_old_new.median()), 3),
        'clusters before split (with PN input): share with PSDs from 2 or more PNs': round(float((_second > 0).mean()), 3),
        'share of calyx sites on a claw (new main version) by MB': {mb: round(float((g[NEWCOL] >= 0).mean()), 3) for mb, g in S.groupby('mb')}}
json.dump(summ, open(os.path.join(OUT, 'claw_compare_summary.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False)[:3000])

# ---------------------------------------------------------------- 3D viewer of the same KCs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
SYM = ['circle', 'square', 'diamond', 'cross']
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]], horizontal_spacing=0.01,
                    subplot_titles=('old claws (2 µm, PN→KC PSDs)', 'new claws (8 µm along the KC skeleton' + (', split by PN)' if MAIN == 'v2' else ')')))
labels, ntr = [], []
for ki, (_, row) in enumerate(pick.iterrows()):
    ds = 'hemibrain' if row.mb.startswith('hemi') else 'malecns'
    s = S[(S.mb == row.mb) & (S.kc_id == row.kc_id)]
    pts = s[['x', 'y', 'z']].values; c0 = pts.mean(0)
    xyz, ch, par = swc(ds, row.kc_id); a, b = calyx_part(xyz, ch, par, pts)
    P = np.round((xyz - c0) / 1e3, 2); Q = np.round((pts - c0) / 1e3, 2)
    seg = np.full((len(a) * 3, 3), np.nan); seg[0::3] = P[a]; seg[1::3] = P[b]
    n0 = len(fig.data)
    for colno, col in [(1, 'claw_old'), (2, NEWCOL)]:
        fig.add_trace(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode='lines', line=dict(color='#c9c8c2', width=2),
                                   hoverinfo='skip', showlegend=False, visible=ki == 0), row=1, col=colno)
        lab = s[col].values; cl = pd.Series(lab[lab >= 0]).value_counts(); order = {k: i for i, k in enumerate(cl.index)}
        GS = glom_styles(s); cg = claw_glom(s, col); MKS = {'o': 'circle', 's': 'square', '^': 'diamond', 'D': 'cross'}
        pg = s.glom.replace('', '?').values
        own = [GS.get(g, ('#bdbcb6', 'o')) for g in pg]
        clw = [GS.get(cg.get(l, '?'), ('#bdbcb6', 'o')) if l >= 0 else ('#d6d5cf', 'o') for l in lab]
        colr = np.array([o[0] if p else c_[0] for o, c_, p in zip(own, clw, s.is_PN.values)])
        sym = np.array([MKS[o[1]] if p else 'circle' for o, p in zip(own, s.is_PN.values)])
        size = np.where(s.is_PN.values, 5.5, 2.6)
        gl = s.groupby(lab).apply(lambda d: d.glom[d.is_PN].value_counts().index[0] if d.is_PN.any() else '', include_groups=False)
        txt = [('not on a claw' if l < 0 else f'claw {order[l] + 1} · {gl.get(l, "")}' + (' (shares its 8 µm cluster)' if col != 'claw_old' and sp else ''))
               + (' · PN→KC PSD from ' + str(t) if p else ' · other KC synapse site')
               for l, p, t, sp in zip(lab, s.is_PN.values, s.pn_type.values, s[SPLITCOL].values)]
        fig.add_trace(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode='markers', text=txt, hoverinfo='text',
                                   marker=dict(color=colr, size=size, symbol=sym, line=dict(width=0)), showlegend=False, visible=ki == 0),
                      row=1, col=colno)
    ntr.append((n0, len(fig.data)))
    yy = '' if pd.isna(row.Prat_n_claws) else f', Prat {int(row.Prat_n_claws)} / corrected {int(row.Prat_n_claws_corrected)}'
    labels.append(f'{ki + 1}. {row.mb} {row.kc_id} {row.kc_type}: old {int(row.old)}, new {int(row.new_main)}{yy}')
nT = len(fig.data); buttons = []
for ki, (lo, hi) in enumerate(ntr):
    vis = [lo <= i < hi for i in range(nT)]
    buttons.append(dict(label=labels[ki], method='update', args=[{'visible': vis}]))
sc = dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='white')
fig.update_layout(scene=sc, scene2=sc, height=780, margin=dict(l=0, r=0, t=120, b=30), paper_bgcolor='white',
                  updatemenus=[dict(buttons=buttons, direction='down', x=0.01, y=1.09, xanchor='left', yanchor='bottom', showactive=True)],
                  annotations=list(fig.layout.annotations) + [dict(text='Big = PN→KC PSD, color + shape = its PN’s glomerulus (same in both panels); small = other KC synapse site, colored by its claw’s glomerulus; '
                                                                    'grey = not on a claw. Coordinates in µm, centered on the KC. '
                                                                    'Drag to rotate (both panels turn together); hover a point for its claw.',
                                                                    x=0.01, y=-0.02, xref='paper', yref='paper', showarrow=False, xanchor='left',
                                                                    font=dict(size=11, color=INK2))])
sync = """var gd=document.getElementsByClassName('plotly-graph-div')[0];var busy=false;
gd.on('plotly_relayout',function(e){if(busy)return;var k=Object.keys(e);var s=null,t=null;
if(k.indexOf('scene.camera')>=0){s='scene';t='scene2';}else if(k.indexOf('scene2.camera')>=0){s='scene2';t='scene';}
if(!s)return;busy=true;var u={};u[t+'.camera']=e[s+'.camera'];Plotly.relayout(gd,u).then(function(){busy=false;});});"""
html = fig.to_html(include_plotlyjs=True, full_html=True, post_script=sync)
html = html.replace('<head>', '<head><meta name="viewport" content="width=device-width, initial-scale=1"><title>Claws: old vs new, 40 KCs</title>', 1)
open(os.path.join(OUT, 'claw_disparity_KCs_3D.html'), 'w').write(html)
print('3D viewer written', len(html) / 1e6, 'MB')
