"""Chance overlap of two unrelated odors: simulation and data (Yang, 2026-10-21).
Claim (APL_block_model_notes.md): for two unrelated odors, the cosine similarity of their 0/1 responses is about
sqrt(p_a p_b), p = fraction of units responding.
Why: for 0/1 vectors, cosine = n_ab / sqrt(n_a n_b) (n_a, n_b responders, n_ab responders to both). If odor b's n_b responders
are placed at random, independently of odor a's, the overlap is hypergeometric: E[n_ab] = n_a n_b / N, so
E[cosine] = sqrt(n_a n_b) / N = sqrt(p_a p_b) exactly, with SD sqrt((1 - p_a)(1 - p_b) / (N - 1)).
a. Random 0/1 codes: N = 2000 units (about one mushroom body or one recording); each unit responds to odor a with probability
   p_a and to odor b with probability p_b, independently; 2000 pairs per (p_a, p_b).
b. Connectome model (official ROIs, the mushroom bodies of apl_block_model.py: hemibrain R, male CNS L/R since 2026-10-21;
   KC input as in that model): unrelated odors =
   glomerular patterns drawn independently, (i) as in the model: a random ORN-table odor with its responses shuffled across the
   36 table glomeruli (other glomeruli give 0); (ii) every glomerulus of the wiring responding, each value drawn from all
   ORN-table responses. 300 odors per mushroom body and variant; an odor's KC code at density p = the top p of KC input.
   Cosine averaged over pairs of different odors; prediction from each KC's response probability pi_k over the 300 odors:
   E[n_ab] = sum_k pi_k(p_a) pi_k(p_b).
c. Data (official TNT / TNT3c files): 0/1 responses (> 1) per recording; similarity and sqrt(p_a p_b) per odor pair, averaged
   over recordings; unrelated baseline = each odor's column shuffled across the units of its recording (50 shuffles); and a
   shuffle that also keeps each unit's number of responses (trades between odor columns, 1500 per sample, 20 samples).
d. Data: change TNT - TNT3c in 0/1 similarity vs change in sqrt(p_a p_b).
Outputs: odor_weighted/chance_overlap_simulation.png, chance_overlap_simulation.csv (a, b), chance_overlap_data_pairs.csv (c, d),
chance_overlap_simulation.json (summary)."""
import sys, json
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from orn_load import load
import apl_block_model as abm

OUT = abm.B + 'odor_weighted/'
UP = '/mnt/project-files/uploads/hearth/'
D = {'TNT3c': load(UP + '76debf61-257e-4375-b4d8-a02b48c4f62a'), 'TNT': load(UP + '2e7fb552-f4ec-49d7-b06f-14ee778f6433')}
O = list(next(iter(D['TNT'].values())).columns); IU = np.triu_indices(len(O), 1)
PS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
PP = [(a, b) for i, a in enumerate(PS) for b in PS[i:]]
N, NREP, NODOR, NSHUF, NFM = 2000, 2000, 300, 50, 20
rs = np.random.default_rng(11)


def cos01(B):
    """cosine between the columns of a 0/1 matrix (units x odors)."""
    B = B.astype(np.float32); n = B.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (B.T @ B) / np.sqrt(np.outer(n, n))


# ---------------- a. random 0/1 codes ----------------
rows = []
for pa, pb in PP:
    A = rs.random((NREP, N)) < pa; Bv = rs.random((NREP, N)) < pb
    c = (A & Bv).sum(1) / np.sqrt(A.sum(1) * Bv.sum(1))
    rows.append(dict(part='a random codes', variant=f'N = {N}', p_a=pa, p_b=pb, sqrt_pa_pb=np.sqrt(pa * pb), cosine_mean=c.mean(),
                     cosine_5=np.quantile(c, 0.05), cosine_95=np.quantile(c, 0.95), cosine_sd=c.std(),
                     sd_formula=np.sqrt((1 - pa) * (1 - pb) / (N - 1))))

# ---------------- b. connectome model, unrelated odors ----------------
R = abm.responses()['ORN']; Rv = np.clip(R.values, 0, None); POOL = Rv.ravel()
MB, _ = abm.mushroom_bodies()
KEY_B = f'b connectome model, mean over {len(MB)} mushroom bodies'
het = {}
for k, mb in MB.items():
    W = mb['W']; gl_t = [g for g in W.columns if g in R.index]; idx = [list(R.index).index(g) for g in gl_t]
    for var in ['ORN-table glomeruli (as in the model)', 'every glomerulus responding']:
        if var.startswith('ORN'):
            G = np.array([rs.permutation(Rv[:, rs.integers(Rv.shape[1])]) for _ in range(NODOR)]).T     # 36 glomeruli x odors
            H = W[gl_t].values @ G[idx]
        else:
            H = W.values @ rs.choice(POOL, (W.shape[1], NODOR))
        Bp = {}
        for p in PS:
            thr = np.quantile(H, 1 - p, axis=0); Bp[p] = H > thr[None]
        for pa, pb in PP:
            Ba, Bb = Bp[pa].astype(np.float32), Bp[pb].astype(np.float32)
            na, nb = Ba.sum(0), Bb.sum(0); off = ~np.eye(NODOR, dtype=bool)
            C = (Ba.T @ Bb) / np.sqrt(np.outer(na, nb)); X = np.sqrt(np.outer(na, nb)) / len(H)
            pia, pib = Ba.mean(1), Bb.mean(1)
            rows.append(dict(part='b connectome model', variant=var, mushroom_body=k, p_a=pa, p_b=pb, sqrt_pa_pb=X[off].mean(),
                             cosine_mean=C[off].mean(), cosine_5=np.quantile(C[off], 0.05), cosine_95=np.quantile(C[off], 0.95),
                             cosine_sd=C[off].std(), prediction_from_KC_response_probability=(pia @ pib) / np.sqrt(na.mean() * nb.mean())))
        pi = Bp[0.1].mean(1); neff = (W.values.sum(1) ** 2) / np.maximum((W.values ** 2).sum(1), 1e-12)
        cover = W[gl_t].values.sum(1) if var.startswith('ORN') else W.values.sum(1)
        het[(k, var)] = dict(never_respond_at_10pct=float((pi == 0).mean()), sd_response_probability_at_10pct=float(pi.std()),
                             spearman_pi_vs_effective_inputs=float(pd.Series(pi).corr(pd.Series(neff), method='spearman')),
                             spearman_pi_vs_covered_input=float(pd.Series(pi).corr(pd.Series(cover), method='spearman')))
    print(k, flush=True)
S = pd.DataFrame(rows)

# ---------------- c, d. data ----------------
def curveball(B, ntr):
    """Random 0/1 matrix with the same number of responses per unit and per odor: trades between two odor columns."""
    B = B.copy(); m = B.shape[1]
    for _ in range(ntr):
        a, b = rs.choice(m, 2, replace=False); idx = np.flatnonzero(B[:, a] ^ B[:, b])
        na = int(B[idx, a].sum()); perm = rs.permutation(idx)
        B[idx, a] = False; B[idx, b] = False; B[perm[:na], a] = True; B[perm[na:], b] = True
    return B


Dd = {}
for c, d in D.items():
    s01, ch, sh, fm = [], [], [], []
    for v in d.values():
        B = v.values > 1; p = B.mean(0)
        s01.append(cos01(B)[IU]); ch.append(np.sqrt(np.outer(p, p))[IU])
        sh.append(np.mean([cos01(np.array([rs.permutation(B[:, j]) for j in range(B.shape[1])]).T)[IU] for _ in range(NSHUF)], 0))
        fm.append(np.mean([cos01(curveball(B, 1500))[IU] for _ in range(NFM)], 0))
    Dd[c] = dict(s01=np.nanmean(s01, 0), chance=np.mean(ch, 0), shuffled=np.mean(sh, 0), unit_counts_kept=np.mean(fm, 0))
P = pd.DataFrame({'odor_a': [O[i] for i in IU[0]], 'odor_b': [O[j] for j in IU[1]],
                  **{f'{x}_{c}': Dd[c][x] for c in D for x in ['s01', 'chance', 'shuffled', 'unit_counts_kept']}})
P['d_s01'] = P.s01_TNT - P.s01_TNT3c; P['d_chance'] = P.chance_TNT - P.chance_TNT3c

if __name__ == '__main__':
    a = S[S.part == 'a random codes']
    b = S[S.part == 'b connectome model'].groupby(['variant', 'p_a', 'p_b'], sort=False).mean(numeric_only=True).reset_index()
    summary = {
        'a random codes (N = 2000)': {
            'max |mean cosine - sqrt(pa pb)|': round(float((a.cosine_mean - a.sqrt_pa_pb).abs().max()), 4),
            'simulated SD / formula SD, range': [round(float((a.cosine_sd / a.sd_formula).min()), 3), round(float((a.cosine_sd / a.sd_formula).max()), 3)]},
        KEY_B: {
            var: {'cosine / sqrt(pa pb), range over the grid': [round(float((g.cosine_mean / g.sqrt_pa_pb).min()), 2), round(float((g.cosine_mean / g.sqrt_pa_pb).max()), 2)],
                  'cosine / sqrt(pa pb) at pa = pb = 0.1': round(float(g.query('p_a == 0.1 and p_b == 0.1').eval('cosine_mean / sqrt_pa_pb').iloc[0]), 2),
                  'prediction from KC response probability / simulated, range': [round(float((g.prediction_from_KC_response_probability / g.cosine_mean).min()), 3),
                                                                                 round(float((g.prediction_from_KC_response_probability / g.cosine_mean).max()), 3)],
                  'KCs never responding at 10% (fraction), mean over MBs': round(float(np.mean([h['never_respond_at_10pct'] for (k, v), h in het.items() if v == var])), 3),
                  'spearman, KC response probability vs effective number of glomerular inputs, mean': round(float(np.mean([h['spearman_pi_vs_effective_inputs'] for (k, v), h in het.items() if v == var])), 2),
                  'spearman, KC response probability vs share of claw input with a response, mean': round(float(np.mean([h['spearman_pi_vs_covered_input'] for (k, v), h in het.items() if v == var])), 2)}
            for var, g in b.groupby('variant', sort=False)},
        'c data, mean over the 120 odor pairs': {c: {'0/1 similarity': round(float(Dd[c]['s01'].mean()), 3), 'sqrt(pa pb)': round(float(Dd[c]['chance'].mean()), 3),
                                                     'units shuffled (unrelated baseline)': round(float(Dd[c]['shuffled'].mean()), 3),
                                                     "shuffled keeping each unit's number of responses": round(float(Dd[c]['unit_counts_kept'].mean()), 3),
                                                     'max |shuffled - sqrt(pa pb)|': round(float(np.abs(Dd[c]['shuffled'] - Dd[c]['chance']).max()), 4),
                                                     'pairs above sqrt(pa pb)': int((Dd[c]['s01'] > Dd[c]['chance']).sum())} for c in D},
        'd data change TNT - TNT3c': {'r over pairs, change in 0/1 similarity vs change in sqrt(pa pb)': round(float(np.corrcoef(P.d_s01, P.d_chance)[0, 1]), 3),
                                      'mean change in 0/1 similarity': round(float(P.d_s01.mean()), 3), 'mean change in sqrt(pa pb)': round(float(P.d_chance.mean()), 3),
                                      'least-squares slope': round(float(np.polyfit(P.d_chance, P.d_s01, 1)[0]), 2)}}
    # ---------------- figure ----------------
    INK, MUTED, GRID, SURF = abm.INK, abm.MUTED, abm.GRID, '#ffffff'
    CC = {'TNT3c': '#2a78d6', 'TNT': '#c8553d'}
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 10.4))
    fig.suptitle('Two unrelated odors: cosine similarity of their 0/1 responses vs √(p_a·p_b)', fontsize=12, color=INK, x=0.04, ha='left', y=0.99)
    TK = [0.02, 0.05, 0.1, 0.2, 0.5]

    def loglog(a, lim):
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(*lim); a.set_ylim(*lim)
        a.set_xticks(TK); a.set_yticks(TK); a.set_xticklabels([f'{t:g}' for t in TK]); a.set_yticklabels([f'{t:g}' for t in TK])
        a.minorticks_off(); a.plot(lim, lim, color=MUTED, lw=1, zorder=1); a.grid(color=GRID, lw=0.6, zorder=0)

    def head(a, letter, title, lines):
        """panel letter, title and a muted note above the axes"""
        a.annotate('\n'.join(lines), xy=(0, 1), xycoords='axes fraction', xytext=(0, 7), textcoords='offset points', va='bottom', ha='left',
                   color=MUTED, fontsize=8, linespacing=1.45, annotation_clip=False, in_layout=False)
        t = a.set_title(title, loc='left', color=INK, fontsize=10, pad=7 + 8 * 1.2 * 1.45 * len(lines) + 7)
        a.annotate(letter, xy=(0, 1), xycoords=t, xytext=(-30, 0), textcoords='offset points', fontsize=13, fontweight='bold', color=INK,
                   va='top', annotation_clip=False)

    lim = (0.015, 0.62)
    A = ax[0, 0]; loglog(A, lim)
    A.vlines(a.sqrt_pa_pb, a.cosine_5, a.cosine_95, color=MUTED, lw=1.2, zorder=2)
    A.scatter(a.sqrt_pa_pb, a.cosine_mean, s=30, color=INK, edgecolor=SURF, lw=0.8, zorder=3)
    A.set_xlabel('√(p_a·p_b)'); A.set_ylabel('cosine similarity of the 0/1 responses')
    head(A, 'a', 'Random 0/1 codes', [f'N = {N:,} units; each responds to odor a with chance p_a',
                                       'and to odor b with chance p_b, independently',
                                       f'dots: mean of {NREP:,} pairs; bars: 5–95% (clipped at the axis)',
                                       f'every mean is within {summary["a random codes (N = 2000)"]["max |mean cosine - sqrt(pa pb)|"]:.3f} of √(p_a·p_b), the line'])
    B_ = ax[0, 1]; loglog(B_, lim)
    sb = summary[KEY_B]
    v1, v2 = sb['ORN-table glomeruli (as in the model)'], sb['every glomerulus responding']
    for var in ['ORN-table glomeruli (as in the model)', 'every glomerulus responding']:
        g = b[b.variant == var]; filled = var.startswith('ORN')
        B_.scatter(g.sqrt_pa_pb, g.cosine_mean, s=30, color=INK if filled else SURF, edgecolor=SURF if filled else INK, lw=0.8 if filled else 0.9,
                   zorder=3, label='random odors on the ORN-table glomeruli (as in the model)' if filled else 'random odors on every glomerulus')
    B_.set_xlabel('√(p_a·p_b)'); B_.set_ylabel('cosine similarity of the 0/1 KC responses')
    head(B_, 'b', f'Connectome KC model, {len(MB)} mushroom bodies (hemibrain, male CNS)',
         ['unrelated odors: glomerular patterns drawn independently',
          f'KC code = top p of KC input; mean over pairs and the {len(MB)} MBs',
          f'at p_a = p_b = 0.1: ×{v1["cosine / sqrt(pa pb) at pa = pb = 0.1"]:.2f} (ORN table), ×{v2["cosine / sqrt(pa pb) at pa = pb = 0.1"]:.2f} (every glomerulus)',
          'above the line because KCs differ in how easily they respond'])
    B_.legend(loc='upper left', frameon=False, fontsize=7.8, handletextpad=0.3, borderaxespad=0.4)
    C_ = ax[1, 0]
    m = 0.7
    C_.plot([0, m], [0, m], color=MUTED, lw=1, zorder=1); C_.grid(color=GRID, lw=0.6, zorder=0)
    for c in ['TNT3c', 'TNT']:
        C_.scatter(P[f'chance_{c}'], P[f'shuffled_{c}'], s=12, color='#9a9994', edgecolor='none', zorder=2,
                   label='units shuffled' if c == 'TNT' else None)
    for c, lab in [('TNT3c', 'TNT3c (APL intact)'), ('TNT', 'TNT (APL blocked)')]:
        C_.scatter(P[f'chance_{c}'], P[f's01_{c}'], s=26, color=CC[c], edgecolor=SURF, lw=0.7, zorder=3, label=lab)
    C_.set_xlim(0, 0.3); C_.set_ylim(0, m)
    C_.set_xlabel('√(p_a·p_b), p = fraction of units > 1'); C_.set_ylabel('cosine similarity of the 0/1 responses')
    sc = summary['c data, mean over the 120 odor pairs']
    head(C_, 'c', "Yang's recordings: real odor pairs",
         ['120 odor pairs; per recording, then averaged over recordings',
          f'shuffled units sit on the line, √(p_a·p_b) (within {max(sc[c]["max |shuffled - sqrt(pa pb)|"] for c in sc):.3f})',
          f'real pairs: {sc["TNT3c"]["0/1 similarity"]:.3f} vs {sc["TNT3c"]["sqrt(pa pb)"]:.3f} (TNT3c), '
          f'{sc["TNT"]["0/1 similarity"]:.3f} vs {sc["TNT"]["sqrt(pa pb)"]:.3f} (TNT)'])
    C_.legend(loc='lower right', frameon=False, fontsize=7.8, handletextpad=0.3, borderaxespad=0.3)
    D_ = ax[1, 1]
    sd = summary['d data change TNT - TNT3c']
    hi = 0.42
    D_.plot([0, hi], [0, hi], color=MUTED, lw=1, zorder=1); D_.grid(color=GRID, lw=0.6, zorder=0)
    D_.text(0.165, 0.165, 'slope 1: unrelated odors', color=MUTED, fontsize=8, rotation=45, rotation_mode='anchor', ha='center', va='bottom',
            transform_rotates_text=True)
    D_.scatter(P.d_chance, P.d_s01, s=26, color=INK, edgecolor=SURF, lw=0.7, zorder=3)
    D_.set_xlim(0, 0.2); D_.set_ylim(-0.05, hi); D_.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    D_.set_xlabel('change in √(p_a·p_b)'); D_.set_ylabel('change in 0/1 similarity')
    head(D_, 'd', 'TNT − TNT3c, per odor pair',
         [f'r = {sd["r over pairs, change in 0/1 similarity vs change in sqrt(pa pb)"]:.2f} over the 120 pairs; least-squares slope {sd["least-squares slope"]:.1f}',
          f'mean change {sd["mean change in 0/1 similarity"]:.3f}, vs {sd["mean change in sqrt(pa pb)"]:.3f} if the odors were unrelated',
          'line: the change that chance overlap alone would give'])
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=3.0, w_pad=2.6)
    fig.savefig(OUT + 'chance_overlap_simulation.png', dpi=160, facecolor='white')
    S.round(5).to_csv(OUT + 'chance_overlap_simulation.csv', index=False)
    P.round(5).to_csv(OUT + 'chance_overlap_data_pairs.csv', index=False)
    json.dump(summary, open(OUT + 'chance_overlap_simulation.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))
