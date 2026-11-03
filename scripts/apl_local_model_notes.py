"""Write odor_weighted/APL_local_model_notes.md (plain-words summary of apl_local_model.py + apl_local_boutons.py) from their
outputs. APL_VARIANT=outside_roi writes the version in with_outside_ROI_synapses/."""
import os, json
OUTSIDE = os.environ.get('APL_VARIANT', '') == 'outside_roi'
OW = '/mnt/project-files/apl_synapses/' + ('with_outside_ROI_synapses/' if OUTSIDE else '') + 'odor_weighted/'
S1 = json.load(open(OW + 'APL_local_model_summary.json')); S2 = json.load(open(OW + 'APL_local_boutons_summary.json'))
RNS = ['ORN', 'GH146']; LAMS = ['200', '100', '50', '25', '10', '5', '2']; LB = ['inf', '50', '25', '10', '5']
f3 = lambda x: f'{x:.3f}'


def rk1(rn, v): return S1[rn]['model: Spearman blocked vs intact (mean over MBs) by version'][v]
def r1(rn, v): return S1[rn]['r over odor pairs, model change vs data change'][v]
def rk2(rn, v): return S2[rn]['Spearman blocked vs intact, mean over MBs'][v]
def r2(rn, v): return S2[rn]['r with the data change, model change averaged over MBs (as tnt_compare.py)'][v]
def ds1(rn, v): return S1[rn]['mean change in similarity: data, model by version'][1][v]


nk = {rn: next(k for k in S1[rn] if k.startswith('null')) for rn in RNS}
best = {rn: nk[rn].split('version ')[-1] for rn in RNS}
loc = {rn: [f'local {l} µm' for l in LAMS if f'local {l} µm' in S1[rn]['r over odor pairs, model change vs data change']] for rn in RNS}
L = []
L.append('# Local, non-spiking APL in the KC model' + (' (version with synapses outside the official ROIs)' if OUTSIDE else ' (official ROIs)'))
L.append('')
L.append('Yang asked (2026-10-23): does a model in which APL is non-spiking, so its inhibition stays local, and uses its real connectivity, '
         'reshape the KC representation (especially the Spearman rank order of odor-pair similarity between inhibition on and off) more like '
         'the data than a uniform APL or a simple connectivity APL? Hemibrain R, male CNS L and R. Step-by-step explanation with figures: '
         '`reports/11_local_APL_model.html` / `.pdf`. Figure: `APL_local_model.png`.')
L.append('')
L.append('## Model in short')
L.append('- Same pinned KC model as `tnt_compare.py`: each odor\'s input scaled so its fraction of KCs responding with APL blocked equals '
         'TNT; one APL gain per version sets the mean intact fraction to TNT3c\'s (6.83%). y = [x − 1 − a M y]+, solved per odor.')
L.append('- U: M = 1/N. C: M_kj = u_k v_j / N (calyx APL→KC and KC→APL counts). local λ: M_kj = c Σ exp(−d(s,t)/λ) over APL→KC sites s on KC k '
         'and KC→APL sites t of KC j, d = distance along the APL skeleton; λ = ∞ gives C exactly. λ = 200 … 2 µm.')
L.append('- Boutons: each bouton\'s output divided by (1 + β Q), Q = local APL signal at its APL→PN synapses per claw synapse. "boutons only" = '
         'all inhibition there; "KC + boutons" = both routes at the boutons\' real share of APL\'s calyx output (hemibrain 0.097, male CNS 0.112).')
L.append('- Null: KCs swap their APL synapse sites within class (100 shuffles), for the local λ with the best pattern r.')
L.append('- Solver: exact active-set method (falls back to a damped iteration); it matches the damped iteration (gains to 4 decimals, '
         'responses within ~1e-6). The tree sums were checked against brute-force shortest paths (relative error ~1e-16).')
L.append('')
L.append('## Answer')
for rn in RNS:
    d = S1[rn]['data: Spearman TNT vs TNT3c similarity over odor pairs']
    kc = [rk1(rn, v) for v in loc[rn]]
    bo = [rk2(rn, f'boutons {l} µm' if l != 'inf' else 'boutons inf') for l in LB]
    kb = [rk2(rn, f'KC + boutons {l} µm' if l != 'inf' else 'KC + boutons inf') for l in LB]
    nb = S1[rn][nk[rn]]['r real, null mean, null 5-95%, p']; nr = S1[rn][nk[rn]]['rank Spearman real, null mean, null 5-95%, p (lower = more reordering)']
    L.append(f'**{rn} input.** Rank order, blocked vs intact (data {f3(d)}, ≈ 0.92 noise-corrected): U {f3(rk1(rn, "U"))}, C {f3(rk1(rn, "C"))}, '
             f'local on KCs {f3(min(kc))}–{f3(max(kc))}, boutons only {f3(min(bo))}–{f3(max(bo))}, KC + boutons {f3(min(kb))}–{f3(max(kb))}. '
             f'Pattern r with the data\'s change (ceiling ≈ 0.84): U {f3(r1(rn, "U"))}, C {f3(r1(rn, "C"))}, best local on KCs {f3(r1(rn, best[rn]))} '
             f'({best[rn].replace("local ", "λ = ")}), best KC + boutons {f3(max(r2(rn, k) for k in S2[rn]["r with the data change, model change averaged over MBs (as tnt_compare.py)"] if k.startswith("KC")))}. '
             f'Shuffled KC sites ({best[rn].replace("local ", "λ = ")}): r {f3(nb[0])} vs null {f3(nb[1])} (5–95% {nb[2][0]}–{nb[2][1]}), p = {nb[3]}; '
             f'rank {f3(nr[0])} vs null {f3(nr[1])}, p = {nr[3]}. Mean change in similarity: data {f3(S1[rn]["mean change in similarity: data, model by version"][0])}, '
             f'U {f3(ds1(rn, "U"))}, best local {f3(ds1(rn, best[rn]))}.')
    L.append('')
rt = {rn: [ds1(rn, v) / S1[rn]['mean change in similarity: data, model by version'][0] for v in ['U'] + loc[rn]] for rn in RNS}
pv = {rn: S1[rn][nk[rn]]['r real, null mean, null 5-95%, p'][3] for rn in RNS}
if all(p <= 0.05 for p in pv.values()):
    shuf = 'and the real placement of KCs on its arbor beats within-class shuffles for both inputs'
elif not any(p <= 0.05 for p in pv.values()):
    shuf = 'but the real placement of KCs on its arbor does not beat within-class shuffles (so the gain is locality itself)'
else:
    shuf = ('and the real placement of KCs on its arbor beats within-class shuffles for '
            + ', '.join(f'{rn} (p = {pv[rn]})' for rn in RNS if pv[rn] <= 0.05) + ' but not for '
            + ', '.join(f'{rn} (p = {pv[rn]})' for rn in RNS if pv[rn] > 0.05))
L.append('In words: a local, non-spiking APL predicts **which** pairs change most better than a uniform or count-weighted APL, '
         + shuf + '. It does **not** reorder the pairs much more: the rank order stays near '
         'the uniform model\'s. Only inhibition placed entirely on the PN boutons comes near the data\'s reordering, and at the boutons\' real '
         'share of APL output it does not. The model\'s mean change in similarity (U and local on KCs) is '
         + ', '.join(f'{min(rt[rn]):.0%}–{max(rt[rn]):.0%} of the data\'s ({rn})' for rn in RNS) + '.')
L.append('')
L.append('## All versions')
for rn in RNS:
    L.append(f'**{rn}** | rank (mean of MBs) | pattern r')
    L.append('---|---|---')
    for v in ['U', 'C'] + loc[rn]:
        L.append(f'{v} | {f3(rk1(rn, v))} | {f3(r1(rn, v))}')
    for l in LB:
        for p in ['boutons', 'KC + boutons']:
            k = f'{p} inf' if l == 'inf' else f'{p} {l} µm'
            L.append(f'{k} | {f3(rk2(rn, k))} | {f3(r2(rn, k))}')
    L.append('')
L.append('## Caveats')
L.append('- λ = 2 µm is near the skeleton\'s resolution (synapses snap 0.4–0.8 µm to nodes).')
L.append('- The λ for the shuffle test was picked first (best r), so p does not correct for trying several λ.')
L.append('- Synapse counts stand in for strength; passive steady-state decay only; a strongly coupled threshold-linear network can have '
         'several steady states, and the solver finds the one reached from the blocked responses.')
L.append('')
L.append('Files: `APL_local_model_summary.json`, `_fits.csv`, `_per_MB_{ORN,GH146}.csv`, `_odor_pairs.csv`, `_nulls.csv`; '
         '`APL_local_boutons_summary.json`, `_per_MB_{ORN,GH146}.csv`. Scripts: `scripts/apl_local_model.py`, `apl_local_boutons.py`, '
         '`apl_local_model_figure.py`, this file `apl_local_model_notes.py`.')
open(OW + 'APL_local_model_notes.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
