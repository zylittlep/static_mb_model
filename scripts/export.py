"""Write the delivered CSVs for one dataset from out/final_<ds>_*.parquet (same columns as the first delivery,
plus partner_type_harmonized where the dataset has it)."""
import sys, pandas as pd
sys.path.insert(0, 'scripts')
from variant import OUT as O, FINAL
FIRST = ['dataset', 'syn_id', 'direction', 'apl_id', 'apl_instance', 'apl_side', 'pre_id', 'post_id', 'partner_id', 'partner_type',
         'partner_type_harmonized', 'partner_class', 'partner_superclass', 'partner_side',
         'pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z', 'partner_side_x', 'partner_side_y', 'partner_side_z',
         'roi_official', 'mb_region', 'mb_region_source', 'outside_roi_dist_nm', 'mb_compartment', 'mb_compartment_source', 'mb_compartment_official', 'mb_compartment_derived',
         'compartment_vote_share', 'nearest_DAN_tbar_nm',
         'claw_id', 'nearest_claw_id', 'claw_dist_nm', 'claw_main_PN_id', 'claw_main_PN_type', 'claw_main_bouton_id',
         'bouton_id', 'nearest_bouton_id', 'bouton_dist_nm', 'claw_bouton_source', 'conf_pre', 'conf_post', 'syn_size', 'pair_n',
         'ctr_x', 'ctr_y', 'ctr_z', 'roi_pre']
for ds in sys.argv[1:]:
    a = pd.read_parquet(FINAL + f'final_{ds}_apl_mb.parquet')
    cols = [c for c in FIRST if c in a.columns]
    a = a[cols].rename(columns={c: c + '_nm' for c in cols if c[-2:] in ('_x', '_y', '_z')})
    a.to_csv(O + f'{ds}_APL_MB_synapses.csv.gz', index=False)
    pd.read_parquet(FINAL + f'final_{ds}_claws.parquet').to_csv(O + f'{ds}_KC_claws_derived.csv.gz', index=False)
    pd.read_parquet(FINAL + f'final_{ds}_boutons.parquet').to_csv(O + f'{ds}_PN_boutons_derived.csv.gz', index=False)
    print(ds, len(a), 'cols', len(cols))
