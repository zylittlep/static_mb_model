"""Make with_outside_ROI_synapses/claws_v2m2/ a complete input folder for the downstream scripts (models, maps, reports),
hemibrain and male CNS only. Run after export_claws_v2.py.

- <ds>_APL_MB_synapses.csv.gz: the corrected-calyx table with the claw columns of the KC rows in the calyx replaced by the
  new claws (claw_id, nearest_claw_id = claw_id, claw_dist_nm along the skeleton, claw_main_PN_id/type, claw_main_bouton_id,
  claw_bouton_source). Every other column and row is unchanged.
- Unchanged copies: <ds>_PN_boutons_derived.csv.gz, KC_weights_APL_PN_per_KC.csv, PN_calyx_weights_APL_vs_KC.csv,
  odor_weighted/GH146_mean_by_glomerulus.csv.
- odor_weighted/KC_APL_synapses_on_claws_by_glomerulus.csv: copy of the new on-claw table (where the scripts look for it).
Usage: python3 export_claws_v2_tables.py"""
import os, shutil, json
import numpy as np, pandas as pd

BO = '/mnt/project-files/apl_synapses/with_outside_ROI_synapses/'
NEW = BO + 'claws_v2m2/'
os.makedirs(NEW + 'odor_weighted/', exist_ok=True)
A = pd.read_csv(NEW + 'KC_APL_calyx_synapses_claws_v2m2.csv.gz', dtype={'kc_id': str, 'apl_id': str}, low_memory=False)
CLAWCOLS = ['claw_id', 'nearest_claw_id', 'claw_dist_nm', 'claw_main_PN_id', 'claw_main_PN_type', 'claw_main_bouton_id', 'claw_bouton_source']
check = {}
for ds in ['hemibrain', 'malecns']:
    t = pd.read_csv(BO + f'{ds}_APL_MB_synapses.csv.gz', low_memory=False, dtype={'partner_id': str, 'apl_id': str})
    a = A[A.dataset == ds].reset_index(drop=True)
    # same rows, same order as export_claws_v2.py took them from the final table: calyx rows with a KC partner
    ix = np.flatnonzero((t.mb_region == 'CA').values & t.partner_id.isin(set(a.kc_id)).values)
    ok = len(ix) == len(a) and (t.partner_id.values[ix] == a.kc_id.values).all() and \
        np.allclose(t.post_x_nm.values[ix], a.x_nm.values) and np.allclose(t.post_z_nm.values[ix], a.z_nm.values)
    assert ok, (ds, len(ix), len(a))
    old_on = t.claw_id.notna().sum()
    kcrow = np.zeros(len(t), bool); kcrow[ix] = True
    # KC rows outside the calyx have no claw anyway; the KC rows in the calyx get the new claw columns
    for c in CLAWCOLS: t[c] = t[c].astype(object); t.loc[kcrow, c] = None
    t.loc[ix, 'claw_id'] = a.claw_id.values
    t.loc[ix, 'nearest_claw_id'] = a.claw_id.values
    t.loc[ix, 'claw_dist_nm'] = a.claw_dist_nm.values
    t.loc[ix, 'claw_main_PN_id'] = a.claw_main_PN_id.values
    t.loc[ix, 'claw_main_PN_type'] = a.claw_main_PN_type.values
    t.loc[ix, 'claw_main_bouton_id'] = a.claw_main_bouton_id.values
    on = t.claw_id.notna().values
    t.loc[on, 'claw_bouton_source'] = 'claws: 8 um geodesic, split by PN (>=2 PSDs)'
    t.loc[t.bouton_id.notna().values, 'claw_bouton_source'] = 'derived (spatial clustering)'
    t.to_csv(NEW + f'{ds}_APL_MB_synapses.csv.gz', index=False)
    shutil.copy(BO + f'{ds}_PN_boutons_derived.csv.gz', NEW)
    check[ds] = {'rows': int(len(t)), 'KC calyx rows': int(len(ix)), 'on a claw, old': int(old_on), 'on a claw, new': int(on.sum())}
    print(ds, check[ds], flush=True)
for f in ['KC_weights_APL_PN_per_KC.csv', 'PN_calyx_weights_APL_vs_KC.csv']:
    shutil.copy(BO + f, NEW)
shutil.copy(BO + 'odor_weighted/GH146_mean_by_glomerulus.csv', NEW + 'odor_weighted/')
shutil.copy(NEW + 'KC_APL_synapses_on_claws_by_glomerulus.csv', NEW + 'odor_weighted/')
json.dump(check, open(NEW + 'export_tables_check.json', 'w'), indent=1)
