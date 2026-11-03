"""Geodesic distance along APL between its synapses on microglomeruli, glomerulus by glomerulus (Yang, 2026-10-23).
Hemibrain and male CNS only (Yang). Goal: a basis for estimating how far inhibition spreads between glomeruli (and within
one) in a non-spiking APL.

Microglomerulus = one derived PN bouton (2 µm single-linkage cluster of a PN's T-bars in the calyx) with the KC claws whose
main bouton it is (the bouton giving the claw most of its PN synapses). Its glomerulus = the bouton's PN's glomerulus
(uniglomerular PN types only; hemibrain names as in FlyWire: VC3l -> VC3, VC3m -> VC5, VC5 -> VM6).
APL synapses on a microglomerulus: APL<->PN synapses with a bouton_id (partner-side site within 2 µm of the bouton) and
APL<->KC synapses with a claw_id (KC-side site within 2 µm of the claw), from the main tables, calyx only.
Each synapse is placed at the skeleton node nearest to its APL-side site (APL output: the T-bar; APL input: the PSD).
Skeletons: hemibrain v1.2 (425790257) and male CNS v1.0 (10540 APL_R, 10977 APL_L) SWC files, 8 nm units. Skeleton pieces
that are not connected are joined by the shortest bridges between pieces (minimum spanning tree over the pieces).

Weighted mean geodesic distance between glomeruli A and B = the mean of d(i, j) over pairs of synapses, i on A's
microglomeruli and j on B's, each pair weighted by w_i x w_j. On a tree this is exact as
  D(A, B) = sum over skeleton edges e of length(e) x [a_e (1 - b_e) + b_e (1 - a_e)],
with a_e, b_e the shares of A's and B's weight on the far side of e.
  all    w = 1 for every APL synapse on the microglomerulus (both directions): symmetric.
  in->out  i = APL inputs at A (KC->APL, PN->APL), j = APL outputs at B (APL->KC, APL->PN); the pair weight is the product
           of input and output synapse counts (Yang's idea of multiplying the two directions, made directional).
Diagonal = the same glomerulus (self term), all pairs including those on the same microglomerulus.
Passive spread (illustration only, APL's length constant is not known): C_lambda(A -> B) = weighted mean of exp(-d / lambda)
over in->out pairs, computed exactly on the tree with two passes; 1 means every input sits on its outputs.
Euclidean check: the same weighted mean with straight-line distances.
Calyx sub-arbors: skeleton nodes within 5 µm of a calyx APL synapse of the outside-ROI table (which covers the whole calyx in
both versions) are calyx nodes; the connected pieces of the skeleton restricted to them are APL's calyx sub-arbors, which
connect to each other only through the peduncle or lobes. Pieces holding >= 2% of the synapses on microglomeruli are named
S1, S2, ... by size; the rest are pooled as 'small pieces'. Distances between and within sub-arbors use the same formula.
Env: APL_VARIANT=outside_roi uses with_outside_ROI_synapses/ (synapses outside the official ROIs added).
Outputs (<base>/apl_geodesic/): APL_geodesic_glomerulus_pairs.csv, APL_geodesic_microglomeruli.csv, APL_geodesic_summary.json,
APL_geodesic_heatmaps.png, APL_geodesic_checks.png, APL_geodesic_passive_spread.png, APL_geodesic_subarbors.png,
APL_geodesic_glomerulus_by_subarbor.csv."""
import os, re, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, breadth_first_order
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr

OUTSIDE = os.environ.get('APL_VARIANT', '') == 'outside_roi'
ROOT = '/mnt/project-files/apl_synapses/'
BASE = ROOT + ('with_outside_ROI_synapses/' if OUTSIDE else '')
OUT = BASE + 'apl_geodesic/'
SK = ROOT + 'data/skeletons/'
APLS = {'hemibrain R': ('hemibrain', 'APL_R', 'hemibrain_425790257_APL_R.swc'),
        'male CNS L': ('malecns', 'APL_L', 'malecns_10977_APL_L.swc'),
        'male CNS R': ('malecns', 'APL_R', 'malecns_10540_APL_R.swc')}
MIN_SYN = 20                    # a glomerulus is shown when >= 20 APL synapses sit on its microglomeruli in every APL
LAMBDAS = [10, 25, 50, 100]     # µm, passive spread illustration
HB_RENAME = {'VC3l': 'VC3', 'VC3m': 'VC5', 'VC5': 'VM6'}
GPAT = r'(?:D|DA\d[lm]?|DC\d|DL\d[dv]?|DM\d|DP1[lm]|V|VA\d[dlmv]?|VC\d[lm]?|VL\d[ap]?|VM\d[dv]?|VP\d[dlm]?)'
INK, MUTED, GRID, NAN_C = '#0b0b0b', '#52514e', '#e1e0d9', '#d6d6d4'
SEQ = LinearSegmentedColormap.from_list('seq', ['#f7f7f5', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']); SEQ.set_bad(NAN_C)
DIV = LinearSegmentedColormap.from_list('div', ['#0d366b', '#3987e5', '#a9cdf5', '#efeee9', '#f6c0a6', '#eb6834', '#8f3412']); DIV.set_bad(NAN_C)
TAG = ' (incl. synapses outside the official ROIs)' if OUTSIDE else ' (official ROIs)'


def uni(t, ds):
    t = str(t)
    if t.startswith('M_') or '_' not in t: return ''
    g = [n for n in t.split('_')[0].split('+') if re.fullmatch(GPAT, n)]
    g = [HB_RENAME.get(n, n) if ds == 'hemibrain' else n for n in g]
    return g[0] if len(g) == 1 else ''


# ---------------------------------------------------------------- skeleton
def load_tree(path):
    """SWC (8 nm units) -> node positions (nm), parent of each node in one connected tree, edge length to the parent (nm),
    BFS order from the root, and the bridges added between unconnected pieces."""
    s = pd.read_csv(path, comment='#', sep=r'\s+', header=None, names=['id', 't', 'x', 'y', 'z', 'r', 'p'])
    xyz = s[['x', 'y', 'z']].values.astype(float) * 8
    n = len(s); pos = pd.Series(np.arange(n), index=s.id.values)
    child = np.where(s.p.values != -1)[0]; par = pos.reindex(s.p.values[child]).values.astype(int)
    ei, ej = list(child), list(par)
    A = coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    bridges = []
    if ncomp > 1:   # shortest link between every two pieces, then a minimum spanning tree over the pieces
        comps = [np.where(lab == c)[0] for c in range(ncomp)]
        trees = [cKDTree(xyz[c]) for c in comps]
        Dm = np.full((ncomp, ncomp), np.inf); link = {}
        for a in range(ncomp):
            for b in range(a + 1, ncomp):
                small, big = (a, b) if len(comps[a]) <= len(comps[b]) else (b, a)
                d, k = trees[big].query(xyz[comps[small]])
                m = int(np.argmin(d)); Dm[a, b] = Dm[b, a] = d[m] + 1e-6
                link[(a, b)] = (comps[small][m], comps[big][k[m]])
        T = minimum_spanning_tree(np.where(np.isfinite(Dm), Dm, 0)).tocoo()
        for a, b in zip(T.row, T.col):
            i, j = link[(min(a, b), max(a, b))]; ei.append(i); ej.append(j)
            bridges.append(float(np.linalg.norm(xyz[i] - xyz[j])))
    A = coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n)).tocsr()
    root = int(np.argmax(np.bincount(lab)) if False else child[0] if False else 0)
    order, pred = breadth_first_order(A, root, directed=False, return_predecessors=True)
    assert len(order) == n, 'tree not connected'
    parent = pred.copy(); parent[root] = -1
    L = np.zeros(n); nz = parent >= 0
    L[nz] = np.linalg.norm(xyz[nz] - xyz[parent[nz]], axis=1)
    return dict(xyz=xyz, parent=parent, L=L, order=order, root=root, n_pieces=int(ncomp), bridges=bridges,
                length_mm=float(L.sum() / 1e6))


def subtree_sums(T, X):
    """Sum of the rows of X over each node's subtree (children before parents)."""
    S = X.astype(float).copy(); par = T['parent']
    for v in T['order'][::-1]:
        p = par[v]
        if p >= 0: S[p] += S[v]
    return S


def mean_dist(T, Wa, Wb):
    """Weighted mean geodesic distance (µm) between the weight columns of Wa and of Wb (nodes x glomeruli)."""
    Sa, Sb = subtree_sums(T, Wa), subtree_sums(T, Wb)
    Fa, Fb = Sa / Sa[T['root']], Sb / Sb[T['root']]
    l = T['L'][:, None] / 1e3
    return (l * Fa).T @ (1 - Fb) + (l * (1 - Fa)).T @ Fb


def passive(T, Win, Wout, lam_um):
    """Weighted mean of exp(-d / lambda) over (input at A, output at B) pairs, exact on the tree: two passes."""
    par, order = T['parent'], T['order']
    k = np.exp(-T['L'] / (lam_um * 1e3))
    up = Win.astype(float).copy()
    for v in order[::-1]:
        p = par[v]
        if p >= 0: up[p] += up[v] * k[v]
    phi = up.copy()
    for v in order:
        p = par[v]
        if p >= 0: phi[v] = up[v] + (phi[p] - up[v] * k[v]) * k[v]
    C = phi.T @ Wout
    return C / np.outer(Win.sum(0), Wout.sum(0))


def subarbors(T, ds, inst, min_share=0.02, node_of=None):
    """Connected pieces of the skeleton restricted to calyx nodes (within 5 µm of a calyx APL synapse, outside-ROI table)."""
    t = pd.read_csv(ROOT + f'with_outside_ROI_synapses/{ds}_APL_MB_synapses.csv.gz', low_memory=False,
                    usecols=['apl_instance', 'mb_region', 'direction', 'pre_x_nm', 'pre_y_nm', 'pre_z_nm', 'post_x_nm', 'post_y_nm', 'post_z_nm'])
    t = t[(t.apl_instance == inst) & (t.mb_region == 'CA')]
    out = t.direction.str.startswith('APL_output').values[:, None]
    P = np.where(out, t[['pre_x_nm', 'pre_y_nm', 'pre_z_nm']].values, t[['post_x_nm', 'post_y_nm', 'post_z_nm']].values)
    ca = cKDTree(P).query(T['xyz'], distance_upper_bound=5000)[0] < 5000
    n = len(ca); ch = np.where(T['parent'] >= 0)[0]; k = ca[ch] & ca[T['parent'][ch]]
    A = coo_matrix((np.ones(k.sum()), (ch[k], T['parent'][ch][k])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    lab = np.where(ca, lab, -1)
    return lab


# ---------------------------------------------------------------- synapses on microglomeruli
def synapses(ds, inst):
    cols = ['direction', 'apl_instance', 'mb_region', 'partner_type', 'bouton_id', 'claw_id', 'claw_main_bouton_id',
            'pre_x_nm', 'pre_y_nm', 'pre_z_nm', 'post_x_nm', 'post_y_nm', 'post_z_nm']
    t = pd.read_csv(BASE + f'{ds}_APL_MB_synapses.csv.gz', usecols=cols, low_memory=False)
    t = t[(t.apl_instance == inst) & (t.mb_region == 'CA')]
    b = pd.read_csv(BASE + f'{ds}_PN_boutons_derived.csv.gz', usecols=['bouton_id', 'pn_id', 'pn_type'])
    bg = pd.Series([uni(x, ds) for x in b.pn_type], index=b.bouton_id)
    out = t.direction.str.startswith('APL_output').values
    mg = np.where(t.bouton_id.notna(), t.bouton_id, t.claw_main_bouton_id)       # PN synapse: its bouton; KC: its claw's main bouton
    on = t.bouton_id.notna() | t.claw_id.notna()
    d = pd.DataFrame({'mg': mg, 'partner': np.where(t.bouton_id.notna(), 'PN', 'KC'), 'out': out,
                      'x': np.where(out, t.pre_x_nm, t.post_x_nm), 'y': np.where(out, t.pre_y_nm, t.post_y_nm),
                      'z': np.where(out, t.pre_z_nm, t.post_z_nm)})[on.values]
    d['glom'] = bg.reindex(d.mg.values).fillna('').values
    n_ca = {'APL<->PN in calyx': int(t.partner_type.notna().sum()), 'calyx APL synapses': len(t)}
    return d, n_ca, len(t)


def heat(ax, M, names, vmin, vmax, cmap, title):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=5.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=5.6)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(title, loc='left', fontsize=9, color=INK)
    return im


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({'font.family': 'Inter', 'font.size': 8, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False})
    R, summary, rows_mg = {}, {'version': 'official ROIs' if not OUTSIDE else 'with synapses outside the official ROIs', 'apl': {}}, []
    for key, (ds, inst, f) in APLS.items():
        T = load_tree(SK + f)
        d, n_ca, n_all = synapses(ds, inst)
        node_d, node = cKDTree(T['xyz']).query(d[['x', 'y', 'z']].values)
        d['node'], d['snap_nm'] = node, node_d
        lab = subarbors(T, ds, inst)
        pc = lab[d.node.values]
        vc = pd.Series(pc[(d.glom != '').values]).value_counts(normalize=True)
        big = [p for p in vc.index if p >= 0 and vc[p] >= 0.02]
        name = {p: f'S{i + 1}' for i, p in enumerate(big)}
        d['subarbor'] = [name.get(p, 'small pieces' if p >= 0 else 'not calyx') for p in pc]
        gl = d[d.glom != '']
        R[key] = dict(T=T, d=d, gl=gl, lab=lab, name=name)
        mgs = gl.groupby('mg').agg(glomerulus=('glom', 'first'), n_in=('out', lambda s: int((~s).sum())), n_out=('out', 'sum'),
                                   n_PN=('partner', lambda s: int((s == 'PN').sum())), n_KC=('partner', lambda s: int((s == 'KC').sum())))
        rows_mg.append(mgs.reset_index().assign(apl=key))
        summary['apl'][key] = {
            'skeleton: nodes, total length (mm), pieces joined, bridges (µm)': [len(T['L']), round(T['length_mm'], 1), T['n_pieces'],
                                                                              [round(b / 1e3, 2) for b in sorted(T['bridges'], reverse=True)[:8]]],
            'calyx APL synapses': n_all,
            'on a microglomerulus (bouton or claw)': len(d), 'of those, uniglomerular PN glomerulus': len(gl),
            'by partner and direction': {f'{p} {"APL->" if o else "->APL"}': int(((gl.partner == p) & (gl.out == o)).sum()) for p in ['PN', 'KC'] for o in [False, True]},
            'microglomeruli with an APL synapse': int(gl.mg.nunique()),
            'calyx sub-arbors holding >= 2% of these synapses, and their shares': {n_: round(float((gl.subarbor == n_).mean()), 3) for n_ in list(name.values()) + ['small pieces', 'not calyx']},
            'distance from synapse to nearest skeleton node (µm): median, 95th percentile': [round(float(np.median(d.snap_nm)) / 1e3, 2), round(float(np.quantile(d.snap_nm, 0.95)) / 1e3, 2)]}
        print(key, summary['apl'][key], flush=True)
    MG = pd.concat(rows_mg, ignore_index=True)
    cnt = pd.DataFrame({k: v['gl'].glom.value_counts() for k, v in R.items()}).fillna(0)
    G = sorted(cnt.index[(cnt >= MIN_SYN).all(1)])
    summary['glomeruli shown (>= %d APL synapses on their microglomeruli in every APL)' % MIN_SYN] = len(G)
    summary['glomeruli left out'] = {g: cnt.loc[g].astype(int).to_dict() for g in cnt.index if g not in G}
    gi = {g: i for i, g in enumerate(G)}
    MAT, rows = {}, []
    for key, r in R.items():
        T, gl = r['T'], r['gl']; gl = gl[gl.glom.isin(G)]
        n = len(T['L']); M = len(G)
        def W(mask):
            X = np.zeros((n, M)); np.add.at(X, (gl.node.values[mask], gl.glom.map(gi).values[mask]), 1.0); return X
        Wall, Win, Wout = W(np.ones(len(gl), bool)), W(~gl.out.values), W(gl.out.values)
        Dall, Dio = mean_dist(T, Wall, Wall), mean_dist(T, Win, Wout)
        # Euclidean, same weights (all synapses)
        P = gl[['x', 'y', 'z']].values / 1e3; gid = gl.glom.map(gi).values
        E = np.zeros((M, M)); order = np.argsort(gid); Ps, gs = P[order], gid[order]; starts = np.searchsorted(gs, np.arange(M))
        for a in range(M):
            Xa = Ps[gs == a]
            dd = np.sqrt(((Xa[:, None, :] - Ps[None]) ** 2).sum(-1)).sum(0)
            E[a] = np.add.reduceat(dd, starts) / (len(Xa) * np.bincount(gs, minlength=M))
        C = {lam: passive(T, Win, Wout, lam) for lam in LAMBDAS}
        MAT[key] = dict(all=Dall, io=Dio, euc=E, C=C, n_in=Win.sum(0), n_out=Wout.sum(0))
        for a, A in enumerate(G):
            for b, Bg in enumerate(G):
                rows.append(dict(apl=key, glomerulus_A=A, glomerulus_B=Bg, n_syn_A=int(Wall[:, a].sum()), n_syn_B=int(Wall[:, b].sum()),
                                 n_APL_inputs_A=int(Win[:, a].sum()), n_APL_outputs_B=int(Wout[:, b].sum()),
                                 geodesic_um_all_synapses=Dall[a, b], geodesic_um_inputs_A_to_outputs_B=Dio[a, b],
                                 geodesic_um_in_out_symmetric=(Dio[a, b] + Dio[b, a]) / 2, euclidean_um_all_synapses=E[a, b],
                                 **{f'passive_spread_inputs_A_to_outputs_B_lambda_{l}um': C[l][a, b] for l in LAMBDAS}))
    PAIRS = pd.DataFrame(rows)
    # within-microglomerulus vs between microglomeruli of one glomerulus (exact, from the pair formula per microglomerulus pair is
    # costly; use the self term split: D_self = within share x D_within + rest) -> computed directly for microglomeruli below
    keys = list(R)
    mean_all = np.mean([MAT[k]['all'] for k in keys], 0)
    Z = linkage(squareform((mean_all + mean_all.T) / 2 - np.diag(np.diag(mean_all)), checks=False), 'average', optimal_ordering=True)
    o = leaves_list(Z); names = [G[i] for i in o]
    iu = np.triu_indices(len(G), 1)
    def off(Mx): return Mx[iu]
    summary['weighted mean geodesic distance (µm), all synapses: median over glomerulus pairs (off-diagonal), median of diagonal'] = {
        k: [round(float(np.median(off(MAT[k]['all']))), 1), round(float(np.median(np.diag(MAT[k]['all']))), 1)] for k in keys}
    summary['same, inputs at A -> outputs at B'] = {
        k: [round(float(np.median(np.r_[MAT[k]['io'][iu], MAT[k]['io'].T[iu]])), 1), round(float(np.median(np.diag(MAT[k]['io']))), 1)] for k in keys}
    summary['spearman between APLs over glomerulus pairs (all synapses, off-diagonal)'] = {
        f'{a} vs {b}': round(float(spearmanr(off(MAT[a]['all']), off(MAT[b]['all']))[0]), 3) for i, a in enumerate(keys) for b in keys[i + 1:]}
    summary['geodesic / euclidean, median over glomerulus pairs; pearson r between them'] = {
        k: [round(float(np.median(off(MAT[k]['all']) / off(MAT[k]['euc']))), 2), round(float(pearsonr(off(MAT[k]['all']), off(MAT[k]['euc']))[0]), 3)] for k in keys}
    summary['in->out asymmetry: median |D(A->B) - D(B->A)| (µm), median D'] = {
        k: [round(float(np.median(np.abs(MAT[k]['io'] - MAT[k]['io'].T)[iu])), 2), round(float(np.median(MAT[k]['io'][iu])), 1)] for k in keys}
    summary['passive spread (illustration): self / mean cross-glomerulus, median over glomeruli'] = {
        k: {f'lambda {l} µm': round(float(np.median(np.diag(MAT[k]['C'][l]) / ((MAT[k]['C'][l].sum(1) - np.diag(MAT[k]['C'][l])) / (len(G) - 1)))), 2) for l in LAMBDAS} for k in keys}

    # within one microglomerulus vs between microglomeruli of the same glomerulus (exact pairwise, via each microglomerulus's
    # weight vector on the tree; microglomeruli with >= 2 synapses)
    within = {}
    for key, r in R.items():
        T, gl = r['T'], r['gl']; gl = gl[gl.glom.isin(G)]
        vc = gl.mg.value_counts(); big = vc.index[vc >= 2]
        mgi = {m: i for i, m in enumerate(big)}; sub = gl[gl.mg.isin(big)]
        X = np.zeros((len(T['L']), len(big)), np.float32)
        np.add.at(X, (sub.node.values, sub.mg.map(mgi).values), 1.0)
        S = subtree_sums(T, X); F = S / S[T['root']]; l = T['L'] / 1e3
        dw = (l[:, None] * 2 * F * (1 - F)).sum(0)            # mean distance between two synapses of the same microglomerulus
        within[key] = float(np.average(dw, weights=vc[big].values ** 2))
        MG.loc[(MG.apl == key) & MG.mg.isin(big), 'geodesic_um_within_microglomerulus'] = pd.Series(dw, index=big).reindex(
            MG.loc[(MG.apl == key) & MG.mg.isin(big), 'mg']).values
    summary['mean geodesic distance between two APL synapses on the same microglomerulus (µm, pair-weighted)'] = {k: round(v, 2) for k, v in within.items()}
    PAIRS.round(4).to_csv(OUT + 'APL_geodesic_glomerulus_pairs.csv', index=False)
    MG.rename(columns={'mg': 'microglomerulus_bouton_id'}).round(3).to_csv(OUT + 'APL_geodesic_microglomeruli.csv', index=False)
    json.dump(summary, open(OUT + 'APL_geodesic_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1)[:6000])

    # ---------------------------------------------------------------- calyx sub-arbors
    SUBS, rows_s = {}, []
    for key, r in R.items():
        T, gl = r['T'], r['gl'][r['gl'].glom.isin(G)]
        ns = [x for x in sorted(gl.subarbor.unique(), key=lambda x: (not x[1:].isdigit(), int(x[1:]) if x[1:].isdigit() else 0, x))]
        si = {x: i for i, x in enumerate(ns)}
        X = np.zeros((len(T['L']), len(ns))); np.add.at(X, (gl.node.values, gl.subarbor.map(si).values), 1.0)
        Ds = mean_dist(T, X, X); w = X.sum(0) / X.sum()
        share = pd.crosstab(gl.glom, gl.subarbor, normalize='index').reindex(index=G, columns=ns).fillna(0)
        SUBS[key] = dict(names=ns, D=Ds, w=w, share=share)
        named = [i for i, x in enumerate(ns) if x[1:].isdigit()]
        Dn, wn = Ds[np.ix_(named, named)], w[named] / w[named].sum()
        same = float(np.sum(np.diag(Dn) * wn ** 2) / np.sum(wn ** 2)); P2 = np.outer(wn, wn); np.fill_diagonal(P2, 0)
        diff = float(np.sum(Dn * P2) / P2.sum()) if P2.sum() else float('nan')
        top = share[[ns[i] for i in named]].max(1)
        summary['apl'][key]['sub-arbors: mean geodesic distance (µm) between two synapses in the same / different named sub-arbors'] = [round(same, 1), round(diff, 1)]
        summary['apl'][key]['sub-arbors: chance that two synapses on microglomeruli share a named sub-arbor'] = round(float(np.sum(wn ** 2)), 3)
        summary['apl'][key]["sub-arbors: share of a glomerulus's synapses in its main sub-arbor, median over glomeruli (largest sub-arbor's overall share)"] = [
            round(float(top.median()), 3), round(float(wn.max()), 3)]
        for g in G:
            rows_s.append(dict(apl=key, glomerulus=g, **{f'share_in_{x}': share.loc[g, x] for x in ns}))
    pd.DataFrame(rows_s).round(4).to_csv(OUT + 'APL_geodesic_glomerulus_by_subarbor.csv', index=False)
    json.dump(summary, open(OUT + 'APL_geodesic_summary.json', 'w'), indent=1)

    # ---------------------------------------------------------------- figures
    ix = np.ix_(o, o)
    vmax = float(np.quantile(np.concatenate([MAT[k][m][ix].ravel() for k in keys for m in ('all', 'io')]), 0.99))
    vmin = float(min(np.min(MAT[k][m]) for k in keys for m in ('all', 'io')))
    fig, axs = plt.subplots(2, 3, figsize=(19, 13.6))
    for j, k in enumerate(keys):
        im = heat(axs[0, j], MAT[k]['all'][ix], names, vmin, vmax, SEQ, f'{k}: all APL synapses on the microglomeruli\n'
                  f'pair weight = 1 per synapse pair (w_i = w_j = 1)')
        heat(axs[1, j], MAT[k]['io'][ix], names, vmin, vmax, SEQ, f'{k}: APL inputs at row glomerulus → APL outputs at column\n'
             f'pair weight = input synapses × output synapses')
        axs[1, j].set_ylabel('glomerulus A (APL input: KC→APL, PN→APL)', fontsize=8); axs[1, j].set_xlabel('glomerulus B (APL output: APL→KC, APL→PN)', fontsize=8)
    for a in axs.ravel():
        pass
    cax = fig.add_axes([0.93, 0.25, 0.008, 0.5]); cb = fig.colorbar(im, cax=cax)
    cb.set_label('weighted mean geodesic distance along the APL skeleton (µm)', fontsize=9); cb.outline.set_visible(False)
    fig.suptitle(f'Geodesic distance along APL between its synapses on the microglomeruli of two glomeruli{TAG}\n'
                 f'{len(G)} glomeruli with ≥ {MIN_SYN} APL synapses on their microglomeruli in each APL; same order and color scale in every '
                 'panel (average-linkage clustering of the mean of the top row); diagonal = the same glomerulus', x=0.06, ha='left', fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.91, top=0.9, bottom=0.07, wspace=0.28, hspace=0.32)
    fig.savefig(OUT + 'APL_geodesic_heatmaps.png', dpi=170); plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6))
    a = axs[0]
    for k, c in zip(keys, ['#eb6834', '#1f9e6e', '#2a78d6']):
        a.scatter(off(MAT[k]['euc']), off(MAT[k]['all']), s=3, alpha=0.35, color=c, lw=0, label=k)
    lim = [0, max(a.get_xlim()[1], a.get_ylim()[1])]; a.plot(lim, lim, color=MUTED, lw=0.8, ls='--')
    a.set_xlabel('weighted mean straight-line distance (µm)'); a.set_ylabel('weighted mean geodesic distance along APL (µm)')
    a.set_title('Glomerulus pairs: geodesic vs straight-line\n(all synapses; dashed: equal)', loc='left', fontsize=9); a.legend(fontsize=7, markerscale=3, frameon=False)
    for a, (k1, k2) in zip(axs[1:3], [('male CNS L', 'male CNS R'), ('hemibrain R', 'male CNS R')]):
        a.scatter(off(MAT[k1]['all']), off(MAT[k2]['all']), s=3, alpha=0.4, color='#52514e', lw=0)
        a.set_xlabel(f'{k1}: geodesic distance (µm)'); a.set_ylabel(f'{k2}: geodesic distance (µm)')
        a.set_title(f'Same glomerulus pairs in two APLs\nSpearman {spearmanr(off(MAT[k1]["all"]), off(MAT[k2]["all"]))[0]:.2f} '
                    f'over {len(iu[0])} pairs', loc='left', fontsize=9)
    a = axs[3]
    for k, c in zip(keys, ['#eb6834', '#1f9e6e', '#2a78d6']):
        dg = np.diag(MAT[k]['all']); cr = (MAT[k]['all'].sum(1) - dg) / (len(G) - 1)
        a.scatter(cr, dg, s=12, color=c, lw=0, label=k)
    lim = [0, max(a.get_xlim()[1], a.get_ylim()[1])]; a.plot(lim, lim, color=MUTED, lw=0.8, ls='--')
    a.set_xlabel('mean distance to the other glomeruli (µm)'); a.set_ylabel('distance within the glomerulus (µm)')
    a.set_title('Per glomerulus: within (diagonal) vs to others\n(one point per glomerulus and APL)', loc='left', fontsize=9); a.legend(fontsize=7, frameon=False)
    fig.suptitle(f'Checks{TAG}', x=0.04, ha='left', fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(OUT + 'APL_geodesic_checks.png', dpi=170, bbox_inches='tight'); plt.close(fig)

    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(len(keys), len(LAMBDAS), figsize=(5.0 * len(LAMBDAS), 5.2 * len(keys)))
    for row, k in enumerate(keys):
        for col, l in enumerate(LAMBDAS):
            a = axs[row, col]; Cm = MAT[k]['C'][l][ix]; offm = Cm[~np.eye(len(G), dtype=bool)].mean()
            im = a.imshow(Cm / offm, cmap=DIV, norm=LogNorm(0.25, 4), interpolation='nearest')
            a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=90, fontsize=4.8); a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=4.8)
            a.tick_params(length=0); [sp.set_visible(False) for sp in a.spines.values()]
            dg = np.diag(Cm) / offm
            a.set_title(f'{k}, λ = {l} µm\nmean cross-glomerulus coupling {offm:.3f}; diagonal median ×{np.median(dg):.1f}', loc='left', fontsize=8.5)
            if col == 0: a.set_ylabel('glomerulus A: APL inputs (KC→APL, PN→APL)', fontsize=7.5)
            if row == len(keys) - 1: a.set_xlabel('glomerulus B: APL outputs (APL→KC, APL→PN)', fontsize=7.5)
    cax = fig.add_axes([0.93, 0.3, 0.008, 0.4]); cb = fig.colorbar(im, cax=cax)
    cb.set_label('coupling A → B ÷ the panel\'s mean cross-glomerulus coupling (log scale)', fontsize=9); cb.outline.set_visible(False)
    fig.suptitle('Illustration only: passive spread if APL signals fell off as exp(−distance / λ) along the skeleton; APL\'s length constant λ is not known' + TAG +
                 '\ncoupling A → B = mean of exp(−d / λ) over pairs of an APL input synapse at A\'s microglomeruli and an APL output synapse at B\'s '
                 '(pair weight = product of synapse counts); 1 would mean every input sits on every output', x=0.03, ha='left', fontsize=10.5)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.93, bottom=0.05, wspace=0.3, hspace=0.3)
    fig.savefig(OUT + 'APL_geodesic_passive_spread.png', dpi=150); plt.close(fig)

    VIEW = {'hemibrain': lambda P: (-P[:, 0], -P[:, 2]), 'malecns': lambda P: (-P[:, 0], -P[:, 1])}
    PAL = ['#2a78d6', '#eb6834', '#1f9e6e', '#8a5cc2', '#c8a21c', '#d35d9b', '#4bb3c7', '#7a5c3e', '#9aa832', '#3d4fb8']
    fig, axs = plt.subplots(3, 3, figsize=(19, 17.5), gridspec_kw=dict(width_ratios=[1.15, 0.8, 1.6]))
    for row, key in enumerate(keys):
        ds = APLS[key][0]; T, r = R[key]['T'], R[key]; S = SUBS[key]
        a = axs[row, 0]; lab = r['lab']; ca = lab >= 0
        inv = {v: k_ for k_, v in r['name'].items()}
        X, Y = VIEW[ds](T['xyz'][ca] / 1e3)
        col = np.array([GRID] * ca.sum(), dtype=object)
        for i, nm in enumerate([x for x in S['names'] if x[1:].isdigit()]):
            col[lab[ca] == inv[nm]] = PAL[i % len(PAL)]
        a.scatter(X, Y, s=0.6, c=list(col), lw=0, rasterized=True)
        for i, nm in enumerate([x for x in S['names'] if x[1:].isdigit()]):
            a.scatter([], [], s=25, color=PAL[i % len(PAL)], label=f'{nm} ({100 * S["w"][S["names"].index(nm)]:.0f}%)')
        a.scatter([], [], s=25, color=GRID, label='small pieces')
        a.set_aspect('equal'); a.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0), title='sub-arbor (share of\nsynapses on micro-\nglomeruli)', title_fontsize=7)
        a.set_xlabel('fly right → (µm, posterior view)'); a.set_ylabel('dorsal ↑ (µm)')
        a.set_title(f'{key}: APL skeleton nodes in the calyx,\ncolored by connected sub-arbor', loc='left', fontsize=9)
        a = axs[row, 1]; nm = S['names']
        im = a.imshow(S['D'], cmap=SEQ, vmin=0, vmax=float(np.max([SUBS[k]['D'].max() for k in keys])))
        for i in range(len(nm)):
            for j in range(len(nm)):
                a.text(j, i, f'{S["D"][i, j]:.0f}', ha='center', va='center', fontsize=6.5, color='white' if S['D'][i, j] > 0.6 * a.get_images()[0].get_clim()[1] else INK)
        a.set_xticks(range(len(nm))); a.set_xticklabels(nm, rotation=90, fontsize=7); a.set_yticks(range(len(nm))); a.set_yticklabels(nm, fontsize=7)
        a.tick_params(length=0); [sp.set_visible(False) for sp in a.spines.values()]
        a.set_title('Mean geodesic distance (µm) between\nsynapses in two sub-arbors', loc='left', fontsize=9)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.set_label('weighted mean geodesic distance (µm)', fontsize=7.5); cb.outline.set_visible(False)
        a = axs[row, 2]; sh = S['share'].loc[names]
        im2 = a.imshow(sh.values.T, cmap=SEQ, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        a.set_yticks(range(len(nm))); a.set_yticklabels(nm, fontsize=7); a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=90, fontsize=5.6)
        a.tick_params(length=0); [sp.set_visible(False) for sp in a.spines.values()]
        a.set_title("Share of each glomerulus's APL synapses (on its microglomeruli) in each sub-arbor", loc='left', fontsize=9)
        cb = fig.colorbar(im2, ax=a, fraction=0.03, pad=0.01); cb.set_label("share of the glomerulus's synapses", fontsize=7.5); cb.outline.set_visible(False)
    fig.suptitle('APL\'s calyx arbor is split into sub-arbors that connect only through the peduncle or lobes' + TAG +
                 '\nCalyx nodes = skeleton nodes within 5 µm of a calyx APL synapse (outside-ROI table, whole calyx); sub-arbor = connected piece of them; '
                 'glomeruli in the order of the distance heatmaps', x=0.03, ha='left', fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.05, wspace=0.45, hspace=0.3)
    fig.savefig(OUT + 'APL_geodesic_subarbors.png', dpi=150); plt.close(fig)
    print('done')
