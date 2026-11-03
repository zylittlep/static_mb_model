import sys, json
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from prep import malecns, hemibrain, flywire, banc, flyem_lobe, fw_lobe, MB_PREFIX
from core import cluster_by_owner, assign_nearest, derive_compartment
from variant import OUTSIDE, FINAL

CLAW_D = 2000      # nm, single-linkage distance for KC claw clustering
BOUTON_D = 2000    # nm, single-linkage distance for PN bouton clustering
ASSIGN_D = 2000    # nm, max distance from an APL synapse to a claw / bouton point
APL_TYPES = {'APL'}

ds = sys.argv[1]
d = {'malecns': malecns, 'hemibrain': hemibrain, 'flywire': flywire, 'banc': banc}[ds]()
meta = d['meta'].drop_duplicates('id').set_index('id')
apl_ids = set(meta.index[meta['type'].astype(str).isin(APL_TYPES)])
a = d['apl'].copy()
a = a[a.pre_id.isin(apl_ids) | a.post_id.isin(apl_ids)]
a = a[a.pre_id != a.post_id]

# ---- MB filter --------------------------------------------------------------
# outside-ROI version: the MB label assigned by offroi/assign.py (official label, or the MB region an outside synapse joined)
ROI = 'roi_assigned' if OUTSIDE else 'roi_official'
if ds in ('malecns', 'hemibrain'):
    inmb = a[ROI].astype(str).str.startswith(MB_PREFIX)
    a['mb_region'] = a[ROI].str.extract(r"^([^()]+)\(")[0]
else:
    inmb = a[ROI].astype(str).str.startswith('MB_')
    a['mb_region'] = a[ROI].str.extract(r"^MB_([A-Z]+)")[0]
a = a[inmb].reset_index(drop=True)

a['direction'] = np.where(a.pre_id.isin(apl_ids), 'APL_output (APL presynaptic)', 'APL_input (APL postsynaptic)')
a['apl_id'] = np.where(a.pre_id.isin(apl_ids), a.pre_id, a.post_id)
a['partner_id'] = np.where(a.pre_id.isin(apl_ids), a.post_id, a.pre_id)
a['apl_is_pre'] = a.pre_id.isin(apl_ids)
for c in ['type', 'type_harmonized', 'class', 'superclass', 'instance', 'side']:
    if c in meta.columns:
        a['partner_' + c] = a.partner_id.map(meta[c])
a['apl_instance'] = a.apl_id.map(meta['instance'])
a['apl_side'] = a.apl_id.map(meta['side'])
P = a[['pre_x', 'pre_y', 'pre_z']].values; Q = a[['post_x', 'post_y', 'post_z']].values

# ---- MB compartment ---------------------------------------------------------
famdist = pd.read_parquet('out/malecns_dan_comp.parquet')
famdist['family'] = famdist['type'].str.extract(r'^(PAM\d\d|PPL\d\d\d)')[0]
famdist = famdist[famdist.comp.str.match(r"^(a|a'|b|b'|g)\d$")]
famdist = pd.crosstab(famdist.family, famdist.comp, normalize='index')
dan = d['dan'].dropna(subset=['lobe', 'family'])
if ds in ('malecns', 'hemibrain'):
    q_region = a[ROI].map(flyem_lobe).values
else:
    q_region = a[ROI].map(fw_lobe).values
q_region = np.array([r if r else '' for r in q_region], dtype=object)
comp, share, nd = derive_compartment(Q, q_region, dan[['x', 'y', 'z']].values, dan.lobe.values,
                                     dan.family.values, famdist)
a['mb_compartment_derived'] = comp
a['compartment_vote_share'] = np.round(share, 3)
a['nearest_DAN_tbar_nm'] = np.round(nd)
if ds == 'malecns':
    from roilookup import lookup
    oc = lookup('gs://flyem-male-cns/rois/malecns-subcompartments-v3/', Q, 0, 256)
    oc = pd.Series(oc).str.replace(r'\((L|R)\)', '', regex=True)
    oc[~oc.str.match(r"^(a|a'|b|b'|g)\d$")] = None
    a['mb_compartment_official'] = oc.values
    lobe = a.mb_region.isin(['aL', "a'L", 'bL', "b'L", 'gL'])
    v = a[lobe & a.mb_compartment_official.notna() & a.mb_compartment_derived.notna()]
    acc = (v.mb_compartment_official == v.mb_compartment_derived).mean()
    json.dump({'n': int(len(v)), 'agreement': float(acc),
               'per_comp': (v.mb_compartment_official == v.mb_compartment_derived).groupby(v.mb_compartment_official).mean().round(3).to_dict()},
              open(FINAL + 'malecns_compartment_validation.json', 'w'), indent=1)
    print('compartment derived vs official agreement', acc, len(v))
# finest available label
if ds == 'malecns':
    a['mb_compartment'] = a.mb_compartment_official.fillna(a.mb_region.where(a.mb_region.isin(['CA', 'PED'])))
    a['mb_compartment_source'] = np.where(a.mb_compartment_official.notna(), 'official ROI (malecns-subcompartments-v3)',
                                          np.where(a.mb_region.isin(['CA', 'PED']), 'official ROI', None))
    if OUTSIDE:   # added lobe synapses that the subcompartment volume does not cover: DAN-vote compartment
        f = (a.mb_region_source != 'official ROI') & a.mb_compartment.isna() & a.mb_compartment_derived.notna()
        a.loc[f, 'mb_compartment'] = a.mb_compartment_derived[f]; a.loc[f, 'mb_compartment_source'] = 'derived (DAN kNN)'
else:
    a['mb_compartment'] = a.mb_compartment_derived.where(a.mb_region.isin(['aL', "a'L", 'bL', "b'L", 'gL', 'VL', 'ML']),
                                                         a.mb_region.where(a.mb_region.isin(['CA', 'PED'])))
    a['mb_compartment_source'] = np.where(a.mb_region.isin(['CA', 'PED']), 'official ROI',
                                          np.where(a.mb_compartment.notna(), 'derived (DAN kNN)', None))
if OUTSIDE:   # added calyx/peduncle rows: the compartment is the region, so it takes the region's source
    f = (a.mb_region_source != 'official ROI') & a.mb_region.isin(['CA', 'PED'])
    a.loc[f, 'mb_compartment_source'] = a.mb_region_source[f]

# ---- PN boutons & KC claws --------------------------------------------------
p = d['pnca'].copy()
p['post_type'] = p.post_id.map(meta['type']).astype(str)
tb = p.drop_duplicates(['pre_id', 'pre_x', 'pre_y', 'pre_z']).reset_index(drop=True)
tbx = tb[['pre_x', 'pre_y', 'pre_z']].values
tb['bouton_id'] = cluster_by_owner(tb.pre_id.values, tbx, BOUTON_D, 'bouton')
kc = p[p.post_type.str.startswith('KC')].reset_index(drop=True)
kcx = kc[['post_x', 'post_y', 'post_z']].values
kc['claw_id'] = cluster_by_owner(kc.post_id.values, kcx, CLAW_D, 'claw')
kc = kc.merge(tb[['pre_id', 'pre_x', 'pre_y', 'pre_z', 'bouton_id']], on=['pre_id', 'pre_x', 'pre_y', 'pre_z'], how='left')

claws = kc.groupby('claw_id').agg(kc_id=('post_id', 'first'), kc_type=('post_type', 'first'),
                                  n_PN_synapses=('pre_id', 'size'),
                                  main_PN_id=('pre_id', lambda s: s.value_counts().index[0]),
                                  main_bouton_id=('bouton_id', lambda s: s.value_counts().index[0] if s.notna().any() else None),
                                  centroid_x_nm=('post_x', 'mean'), centroid_y_nm=('post_y', 'mean'), centroid_z_nm=('post_z', 'mean')).reset_index()
claws['main_PN_type'] = claws.main_PN_id.map(meta['type'])
tb2 = tb.merge(p.groupby(['pre_id', 'pre_x', 'pre_y', 'pre_z']).size().rename('n_post').reset_index(), on=['pre_id', 'pre_x', 'pre_y', 'pre_z'])
boutons = tb2.groupby('bouton_id').agg(pn_id=('pre_id', 'first'), n_tbars=('pre_id', 'size'), n_postsynaptic_partners=('n_post', 'sum'),
                                       centroid_x_nm=('pre_x', 'mean'), centroid_y_nm=('pre_y', 'mean'), centroid_z_nm=('pre_z', 'mean')).reset_index()
boutons['pn_type'] = boutons.pn_id.map(meta['type'])

# APL <-> KC in calyx
a['partner_side_x'] = np.where(a.apl_is_pre, a.post_x, a.pre_x)
a['partner_side_y'] = np.where(a.apl_is_pre, a.post_y, a.pre_y)
a['partner_side_z'] = np.where(a.apl_is_pre, a.post_z, a.pre_z)
R = a[['partner_side_x', 'partner_side_y', 'partner_side_z']].values
a['claw_id'] = None; a['claw_dist_nm'] = np.nan; a['bouton_id'] = None; a['bouton_dist_nm'] = np.nan
ca = (a.mb_region == 'CA').values
ik = np.flatnonzero(ca & a.partner_id.isin(set(kc.post_id)).values)
lab, dist = assign_nearest(a.partner_id.values[ik], R[ik], kc.post_id.values, kcx, kc.claw_id.values, np.inf)
a['nearest_claw_id'] = None
a.loc[ik, 'nearest_claw_id'] = lab; a.loc[ik, 'claw_dist_nm'] = np.round(dist)
a.loc[ik, 'claw_id'] = np.where(dist <= ASSIGN_D, lab, None)
ip = np.flatnonzero(ca & a.partner_id.isin(set(tb.pre_id)).values)
lab, dist = assign_nearest(a.partner_id.values[ip], R[ip], tb.pre_id.values, tbx, tb.bouton_id.values, np.inf)
a['nearest_bouton_id'] = None
a.loc[ip, 'nearest_bouton_id'] = lab; a.loc[ip, 'bouton_dist_nm'] = np.round(dist)
a.loc[ip, 'bouton_id'] = np.where(dist <= ASSIGN_D, lab, None)
a['claw_bouton_source'] = np.where(a.claw_id.notna() | a.bouton_id.notna(), 'derived (spatial clustering)', None)
cmap = claws.set_index('claw_id')
a['claw_main_PN_id'] = a.claw_id.map(cmap.main_PN_id); a['claw_main_PN_type'] = a.claw_id.map(cmap.main_PN_type)
a['claw_main_bouton_id'] = a.claw_id.map(cmap.main_bouton_id)

a.insert(0, 'dataset', ds)
a.to_parquet(FINAL + f'final_{ds}_apl_mb.parquet')
claws.insert(0, 'dataset', ds); boutons.insert(0, 'dataset', ds)
claws.to_parquet(FINAL + f'final_{ds}_claws.parquet'); boutons.to_parquet(FINAL + f'final_{ds}_boutons.parquet')
cpk = claws.groupby('kc_id').size()
summ = {'dataset': ds, 'apl_ids': sorted(apl_ids), 'n_mb_synapse_rows': int(len(a)),
        'by_direction': a.direction.value_counts().to_dict(), 'by_region': a.mb_region.value_counts().to_dict(),
        'by_compartment': a.mb_compartment.value_counts().to_dict(),
        **({'outside_roi_rows': int((a.mb_region_source != 'official ROI').sum())} if OUTSIDE else {}),
        'KC_rows_in_CA': int(len(ik)), 'KC_rows_on_claw': int(a.claw_id.notna().sum()),
        'PN_rows_in_CA': int(len(ip)), 'PN_rows_on_bouton': int(a.bouton_id.notna().sum()),
        'claws_per_KC_median': float(cpk.median()), 'claws_per_KC_mean': float(cpk.mean()),
        'boutons_per_PN_median': float(boutons.groupby('pn_id').size().median()), 'n_boutons': int(len(boutons)), 'n_claws': int(len(claws))}
json.dump(summ, open(FINAL + f'final_{ds}_summary.json', 'w'), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
