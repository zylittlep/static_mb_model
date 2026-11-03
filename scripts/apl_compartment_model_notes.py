"""Write odor_weighted/APL_compartment_model_notes.md (plain-words summary of apl_compartment_model.py) from its outputs, with the
local model of apl_local_model.py as reference. APL_VARIANT=outside_roi writes the version in with_outside_ROI_synapses/."""
import os, json
OUTSIDE = os.environ.get('APL_VARIANT', '') == 'outside_roi'
OW = '/mnt/project-files/apl_synapses/' + ('with_outside_ROI_synapses/' if OUTSIDE else '') + 'odor_weighted/'
SC = json.load(open(OW + 'APL_compartment_model_summary.json')); SL = json.load(open(OW + 'APL_local_model_summary.json'))
RNS = ['ORN', 'GH146']; CL = ['200', '100', '50', '25', '10']
RK = 'model: Spearman blocked vs intact (mean over MBs) by version'; RC = 'r over odor pairs, model change (MB-averaged) vs data change'
f3 = lambda x: f'{x:.3f}'
ISO = ['compartments inf', 'compartments inf, per length']
KCV = ISO + [f'compartments {l} µm' for l in CL]
L = ['# APL confined to its calyx sub-arbors' + (' (version with synapses outside the official ROIs)' if OUTSIDE else ' (official ROIs)'), '',
     'Yang asked (2026-10-24): a model in which signals inside APL stay within the calyx sub-arbors of Report 10 (no crossing between them); '
     'does it model the reshaping (rank order of odor-pair similarity, inhibition on vs off) better? Hemibrain R, male CNS L and R. '
     'Step-by-step report with figures: `reports/12_APL_sub_arbor_compartments.html` / `.pdf`. Figure: `APL_compartment_model.png`.', '',
     '## Model in short',
     '- Same pinned KC model as Reports 07 and 11: y = [x − 1 − a M y]+, TNT fractions per odor with APL blocked, one APL gain per version for TNT3c\'s 6.83%.',
     '- Compartments = connected pieces of APL\'s calyx skeleton (nodes within 5 µm of a calyx APL synapse); every piece is its own compartment.',
     '- Isopotential: M_kj ∝ Σ_c O_ck I_cj (O, I: APL→KC and KC→APL synapses of the KC in compartment c). "÷ length": each term divided by the '
     'compartment\'s cable length. λ inside: Report 11\'s local kernel with every edge leaving a compartment cut (checked: equals the isopotential '
     'formula at λ = ∞ exactly). Bouton route as in Report 11 with the walled signal.',
     '- Null: KCs swap APL synapse sites within class, 100x, for the better isopotential version.', '', '## Answer']
for rn in RNS:
    s = SC[rn]; nk = next(k for k in s if k.startswith('null')); nb = s[nk]['r real, null mean, null 5-95%, p']
    bl = max([k for k in SL[rn]['r over odor pairs, model change vs data change'] if k.startswith('local')], key=lambda k: SL[rn]['r over odor pairs, model change vs data change'][k])
    L.append(f'**{rn} input.** Rank order (data {f3(s["data: Spearman TNT vs TNT3c similarity over odor pairs"])}, ≈ 0.92 noise-corrected): '
             f'U {f3(s[RK]["U"])}, C {f3(s[RK]["C"])}, sub-arbors on KCs {f3(min(s[RK][v] for v in KCV))}–{f3(max(s[RK][v] for v in KCV))}, '
             f'boutons only in sub-arbors {f3(s[RK]["boutons, compartments inf"])} (∞) / {f3(s[RK]["boutons, compartments 50 µm"])} (50 µm), '
             f'KC + boutons {f3(s[RK]["KC + boutons, compartments inf"])}. Pattern r (ceiling ≈ 0.84): U {f3(s[RC]["U"])}, C {f3(s[RC]["C"])}, '
             f'isopotential {f3(s[RC]["compartments inf"])}, ÷ length {f3(s[RC]["compartments inf, per length"])}, λ = 10 µm inside '
             f'{f3(s[RC]["compartments 10 µm"])}; without walls (Report 11) best {f3(SL[rn]["r over odor pairs, model change vs data change"][bl])} '
             f'({bl.replace("local ", "λ = ")}). Shuffle ({nk.split("version ")[-1]}): r {f3(nb[0])} vs null {f3(nb[1])}, p = {nb[3]}. '
             f'Mean change in similarity: data {f3(s["mean change in similarity: data, model by version"][0])}, U '
             f'{f3(s["mean change in similarity: data, model by version"][1]["U"])}, sub-arbors '
             f'{f3(min(s["mean change in similarity: data, model by version"][1][v] for v in KCV))}–{f3(max(s["mean change in similarity: data, model by version"][1][v] for v in KCV))}.')
    L.append('')
pv = {rn: SC[rn][next(k for k in SC[rn] if k.startswith('null'))]['r real, null mean, null 5-95%, p'][3] for rn in RNS}
sig = [rn for rn in RNS if pv[rn] <= 0.05]
L.append('In words: confining APL\'s signal to its calyx sub-arbors does **not** make inhibition reorder odor pairs more; the rank order stays '
         'near the uniform model\'s (about 0.98). Isopotential sub-arbors predict which pairs change no better than one APL. Dividing each '
         'sub-arbor\'s potential by its cable length raises the pattern r above U and C'
         + (f', and the real placement beats within-class shuffles for {" and ".join(sig)} but not for {" and ".join(rn for rn in RNS if rn not in sig)}'
            if sig and len(sig) < len(RNS) else (', and the real placement beats within-class shuffles for both inputs' if sig else
                                                 ', but the real placement does not beat within-class shuffles'))
         + '. No compartment version beats the local model without walls at short λ, where the walls do not matter.')
L.append('')
L.append('## All versions')
for rn in RNS:
    L += [f'**{rn}** | rank (mean of MBs) | pattern r | mean ΔS', '---|---|---|---']
    for v in SC[rn][RK]:
        L.append(f'{v} | {f3(SC[rn][RK][v])} | {f3(SC[rn][RC][v])} | {f3(SC[rn]["mean change in similarity: data, model by version"][1][v])}')
    L.append('')
L += ['## Checks', '```', json.dumps(SC['checks'], indent=1, ensure_ascii=False), '```', '',
      'Files: `APL_compartment_model_summary.json`, `_fits.csv`, `_per_MB_{ORN,GH146}.csv`, `_odor_pairs.csv`, `_nulls.csv`, `_compartments.csv`. '
      'Scripts: `scripts/apl_compartment_model.py`, `apl_compartment_model_figure.py`, this file `apl_compartment_model_notes.py`.']
open(OW + 'APL_compartment_model_notes.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:30]))
