"""Synapses outside every official mushroom body ROI that belong to the mushroom body (the with_outside_ROI_synapses version).

Anchors (postsynaptic sites, nm; both MBs of a dataset together):
  calyx            ALPN -> KC synapses inside the official calyx ROI, extended through ALPN -> KC synapses outside every MB ROI
                   that join them in steps of <= 3 µm (KC claws the ROI leaves out).
  peduncle / lobes APL <-> KC synapses inside the official PED and lobe ROIs.
Then APL <-> KC synapses outside every MB ROI join whichever anchor reaches them first, in steps of <= 3 µm, and take the region
of the point they joined (KC dendrites and axons exist only in the mushroom body, so these chains stay inside it).
No chain runs through synapses with gamma-d or alpha/beta-p KCs, whose dendrites form the accessory calyces next to the calyx and
peduncle: the accessory calyces stay out, as in the official tables.
A synapse outside every MB ROI joins the MB when its postsynaptic site is within 3 µm of an anchor point, with the region and side
of the nearest anchor point (CA, PED, or the lobe in the dataset's own terms: aL / a'L / bL / b'L / gL in hemibrain and male CNS,
VL / ML in FlyWire and BANC). Applied to the APL tables, the ALPN -> any tables (claws, boutons, PN weights) and the KC -> MBON
tables (lobe plots). ALPN synapses that join the calyx are kept when they are ALPN -> KC, or when their T-bar belongs to a calyx
bouton (joins, in steps of <= 2 µm along T-bars of the same PN, a T-bar that contacts a KC in the calyx); PN synapses near the
calyx that are not on such boutons (e.g. onto lateral horn or SLP neurons) stay out. Writes to out/outside_roi/:
  <ds>_apl_assigned.parquet     row-aligned with the raw APL table: roi_assigned, mb_region_source, outside_roi_dist_nm
  <ds>_pnca_raw.parquet         official calyx PN table + added calyx rows (same columns, plus outside_roi / roi_assigned)
  <ds>_pnmb_added_pairs.parquet added ALPN -> any synapses in any MB region, per pair (pre_id, post_id, n)
  <ds>_kcmbon_added.parquet     added KC -> MBON synapses in the lobes, same columns as the official lobe table
  assign_<ds>.json              counts
Run from the working folder (with dl/, out/, scripts/): python3 offroi/assign.py <dataset>."""
import sys, json, re
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from meta_util import types, kc_ids

STEP = 3000.   # nm
BOUTON_D = 2000.   # nm, as the bouton clustering in run.py
A = 'out/outside_roi/'
PFX = 720575940000000000
ds = sys.argv[1]
FLYEM = ds in ('hemibrain', 'malecns')
LOBES = ['aL', "a'L", 'bL', "b'L", 'gL'] if FLYEM else ['VL', 'ML']
RX = re.compile(r"^(CA|PED|aL|a'L|bL|b'L|gL)\(([LR])\)$") if FLYEM else re.compile(r'^MB_(CA|PED|VL|ML)_([LR])$')
SRC_CA = 'added: outside the official ROIs, within 3 µm of the calyx'
SRC_LB = 'added: outside the official ROIs, within 3 µm of the peduncle/lobes'
ACA_KC = r'^KC(g-d|ab-p)'   # gamma-d / alpha-beta-p KCs (accessory-calyx dendrites)


def parse(labels):
    """official label -> (region or '', side or '')."""
    s = pd.Series(np.asarray(labels, dtype=object)).fillna('').astype(str)
    m = s.str.extract(RX)
    return m[0].fillna('').values, m[1].fillna('').values


def synth(region, side):
    return f'{region}({side})' if FLYEM else f'MB_{region}_{side}'


def banc_label(t):
    return (t.neuropil.fillna('').astype(str) + '_' + t.side.fillna('').astype(str).str[0].str.upper()).values


def xyz(t, cols, scale=1.):
    return t[cols].values.astype(float) * scale


typ, _ = types(ds)
KC = kc_ids(typ)

# ---------------- raw tables: ids as str, official label, postsynaptic site in nm ----------------
if ds == 'flywire':
    apl = pd.read_parquet('out/fafbP_apl_raw.parquet')
    apl_pre = (apl.pre_root_id_720575940.astype('int64') + PFX).astype(str).values; apl_post = (apl.post_root_id_720575940.astype('int64') + PFX).astype(str).values
    apl_lab = apl.neuropil.fillna('').values; apl_xyz = xyz(apl, ['post_x', 'post_y', 'post_z'])
    APL_IDS = {'720575940613583001', '720575940624547622'}
    pn = pd.read_parquet(A + 'fafbP_pn_all_raw.parquet')
    pn_pre = (pn.pre_root_id_720575940.astype('int64') + PFX).astype(str).values; pn_post = (pn.post_root_id_720575940.astype('int64') + PFX).astype(str).values
    pn_lab = pn.neuropil.fillna('').values; pn_xyz = xyz(pn, ['post_x', 'post_y', 'post_z']); pn_tb = xyz(pn, ['pre_x', 'pre_y', 'pre_z'])
    pnca_off = pd.read_parquet('out/fafbP_pnca_raw.parquet')
    km = pd.read_parquet(A + 'flywire_kcmbon_all.parquet'); km_lab = km.neuropil.fillna('').values; km_xyz = xyz(km, ['post_x', 'post_y', 'post_z'])
elif ds == 'hemibrain':
    apl = pd.read_parquet('out/hemibrain_apl_raw.parquet')
    apl_pre = apl.body_pre.astype('int64').astype(str).values; apl_post = apl.body_post.astype('int64').astype(str).values
    apl_lab = apl.roi_post.fillna('').values; apl_xyz = xyz(apl, ['x_post', 'y_post', 'z_post'], 8.)
    APL_IDS = {'425790257'}
    pn = pd.read_parquet(A + 'hemibrain_pn_all_raw.parquet')
    pn_pre = pn.body_pre.astype('int64').astype(str).values; pn_post = pn.body_post.astype('Int64').astype(str).values
    pn_lab = pn.roi_pre.fillna('').values          # the official calyx PN table selects ALPN T-bars by their own ROI
    pn_xyz = xyz(pn, ['x_post', 'y_post', 'z_post'], 8.); pn_tb = xyz(pn, ['x_pre', 'y_pre', 'z_pre'], 8.)
    pnca_off = pd.read_parquet('out/hemibrain_pnca_raw.parquet')
    km = pd.read_parquet(A + 'hemibrain_kcmbon_all.parquet'); km_lab = km.roi_post.fillna('').values; km_xyz = xyz(km, ['x_post', 'y_post', 'z_post'], 8.)
elif ds == 'malecns':
    apl = pd.read_parquet('out/malecns_apl_raw.parquet')
    apl_pre = apl.body_pre.astype('int64').astype(str).values; apl_post = apl.body_post.astype('int64').astype(str).values
    apl_lab = apl.primary_post.fillna('').values; apl_xyz = xyz(apl, ['x_post', 'y_post', 'z_post'], 8.)
    APL_IDS = {'10540', '10977'}
    pn = pd.read_parquet('out/malecns_pn_all_raw.parquet')
    pn_pre = pn.body_pre.astype('int64').astype(str).values; pn_post = pn.body_post.astype('int64').astype(str).values
    pn_lab = pn.primary_post.fillna('').values; pn_xyz = xyz(pn, ['x_post', 'y_post', 'z_post'], 8.); pn_tb = xyz(pn, ['x_pre', 'y_pre', 'z_pre'], 8.)
    pnca_off = pd.read_parquet('out/malecns_pnca_raw.parquet')
    km = pd.read_parquet(A + 'malecns_kcmbon_all.parquet'); km_lab = km.roi.fillna('').values; km_xyz = xyz(km, ['x_post', 'y_post', 'z_post'], 8.)
else:
    apl = pd.read_parquet('out/banc_v2_apl_raw.parquet')
    apl_pre = apl.pre_root_id.astype(str).values; apl_post = apl.post_root_id.astype(str).values
    apl_lab = banc_label(apl); apl_xyz = xyz(apl, ['post_x', 'post_y', 'post_z'])
    APL_IDS = {'720575941482622627', '720575941734526507'}
    pn = pd.read_parquet(A + 'banc_pn_all_raw.parquet')
    pn_pre = pn.pre_root_id.astype(str).values; pn_post = pn.post_root_id.astype(str).values
    pn_lab = banc_label(pn); pn_xyz = xyz(pn, ['post_x', 'post_y', 'post_z']); pn_tb = xyz(pn, ['pre_x', 'pre_y', 'pre_z'])
    pnca_off = pd.read_parquet('out/banc_pnca_raw.parquet')
    km = pd.read_parquet(A + 'banc_kcmbon_all.parquet'); km_lab = banc_label(km); km_xyz = xyz(km, ['post_x', 'post_y', 'post_z'])

apl_reg, apl_side = parse(apl_lab); pn_reg, pn_side = parse(pn_lab); km_reg, km_side = parse(km_lab)
apl_is_pre = np.isin(apl_pre, list(APL_IDS))
apl_partner = np.where(apl_is_pre, apl_post, apl_pre)
apl_kc = np.isin(apl_partner, list(KC)) & (apl_pre != apl_post)
pn_kc = np.isin(pn_post, list(KC))
aca = lambda ids: pd.Series(ids).map(typ).astype(str).str.match(ACA_KC).values
apl_aca = aca(apl_partner); pn_aca = aca(pn_post)


# ---------------- anchors ----------------
def grow(ref_xyz, ref_code, cand_xyz, step=STEP):
    """Candidates that join the reference points through steps <= step; each takes the code of the point it joined."""
    code = np.full(len(cand_xyz), -1); added = np.zeros(len(cand_xyz), bool)
    fx, fc, it = ref_xyz, ref_code, 0
    while len(fx):
        rem = np.flatnonzero(~added)
        if not len(rem): break
        d, j = cKDTree(fx).query(cand_xyz[rem], k=1, distance_upper_bound=step)
        ok = np.isfinite(d); new = rem[ok]
        if not len(new): break
        added[new] = True; code[new] = fc[j[ok]]
        fx, fc, it = cand_xyz[new], code[new], it + 1
    return added, code, it


CODES = [('CA', 'L'), ('CA', 'R')] + [(r, s) for r in ['PED'] + LOBES for s in 'LR']
CI = {c: i for i, c in enumerate(CODES)}
enc = lambda reg, side: np.array([CI.get((r, s), -1) for r, s in zip(reg, side)])

ref = pn_kc & (pn_reg == 'CA'); cand = pn_kc & (pn_reg == '') & ~pn_aca
ca_add, ca_code, ca_it = grow(pn_xyz[ref], enc(pn_reg[ref], pn_side[ref]), pn_xyz[cand])
claw_X = np.vstack([pn_xyz[ref], pn_xyz[cand][ca_add]]); claw_C = np.r_[enc(pn_reg[ref], pn_side[ref]), ca_code[ca_add]]
n_claw_in, n_claw_cand = int(ref.sum()), int(cand.sum())

# outside APL <-> KC synapses join whichever anchor reaches them first: calyx claws, or official PED / lobe APL <-> KC synapses
ref = apl_kc & np.isin(apl_reg, ['PED'] + LOBES); cand = apl_kc & (apl_reg == '') & ~apl_aca
lb0_X, lb0_C = apl_xyz[ref], enc(apl_reg[ref], apl_side[ref])
k_add, k_code, k_it = grow(np.vstack([claw_X, lb0_X]), np.r_[claw_C, lb0_C], apl_xyz[cand])
to_ca = k_add & np.isin(k_code, [CI[('CA', 'L')], CI[('CA', 'R')]]); to_lb = k_add & ~to_ca
CA_X = np.vstack([claw_X, apl_xyz[cand][to_ca]]); CA_C = np.r_[claw_C, k_code[to_ca]]
LB_X = np.vstack([lb0_X, apl_xyz[cand][to_lb]]); LB_C = np.r_[lb0_C, k_code[to_lb]]
ca_tree, lb_tree = cKDTree(CA_X), cKDTree(LB_X)
stats = {'dataset': ds, 'calyx_anchor': {'ALPN_KC_in_CA': n_claw_in, 'outside_ALPN_KC_candidates': n_claw_cand,
                                          'outside_ALPN_KC_joined': int(ca_add.sum()), 'steps': ca_it,
                                          'outside_APL_KC_joined': int(to_ca.sum())},
         'lobe_anchor': {'APL_KC_in_PED_lobes': int(ref.sum()), 'outside_APL_KC_joined': int(to_lb.sum())},
         'outside_APL_KC_candidates': int(cand.sum()), 'APL_KC_chain_steps': k_it}


def assign(x, reg):
    """For rows outside every MB ROI (reg == ''): region, side, source and distance of the nearest anchor point within STEP."""
    n = len(x); out_reg = np.full(n, '', dtype=object); out_side = np.full(n, '', dtype=object)
    src = np.full(n, '', dtype=object); dist = np.full(n, np.nan)
    o = np.flatnonzero(reg == '')
    d1, j1 = ca_tree.query(x[o], k=1, distance_upper_bound=STEP)
    d2, j2 = lb_tree.query(x[o], k=1, distance_upper_bound=STEP)
    use_ca = np.isfinite(d1) & (d1 <= d2); use_lb = np.isfinite(d2) & ~use_ca
    for m, D, J, C, s in [(use_ca, d1, j1, CA_C, SRC_CA), (use_lb, d2, j2, LB_C, SRC_LB)]:
        k = o[m]; c = C[J[m]]
        out_reg[k] = [CODES[i][0] for i in c]; out_side[k] = [CODES[i][1] for i in c]
        src[k] = s; dist[k] = D[m]
    return out_reg, out_side, src, dist


# ---------------- APL table ----------------
r_, s_, src, dist = assign(apl_xyz, apl_reg)
inmb = apl_reg != ''
roi_assigned = np.where(inmb | (src == ''), apl_lab, [synth(a, b) if a else '' for a, b in zip(r_, s_)])
mb_src = np.where(inmb, 'official ROI', src)
pd.DataFrame({'roi_assigned': roi_assigned.astype(str), 'mb_region_source': mb_src.astype(str), 'outside_roi_dist_nm': np.round(dist)}).to_parquet(A + f'{ds}_apl_assigned.parquet')
apl_side_of = {'flywire': {'720575940624547622': 'L', '720575940613583001': 'R'}, 'hemibrain': {'425790257': 'R'},
               'malecns': {'10977': 'L', '10540': 'R'}, 'banc': {'720575941734526507': 'L', '720575941482622627': 'R'}}[ds]
own = pd.Series(np.where(apl_is_pre, apl_pre, apl_post)).map(apl_side_of).values
added = src != ''
stats['apl'] = {'rows': int(len(apl)), 'official_MB': int(inmb.sum()), 'added': int(added.sum()),
                'added_by_region': pd.Series(r_[added]).value_counts().to_dict(),
                'added_KC_partner_by_region': pd.Series(r_[added & apl_kc]).value_counts().to_dict(),
                'added_side_differs_from_APL_side': int((added & (s_ != own)).sum()),
                'outside_not_added': int(((~inmb) & ~added).sum())}

# ---------------- ALPN -> any table ----------------
r_, s_, src, dist = assign(pn_xyz, pn_reg)
near_ca = r_ == 'CA'
# keep ALPN -> KC, and other partners only on calyx boutons: T-bar joins a KC-contacting calyx T-bar of the same PN in <= 2 µm steps
X = pn_tb.copy(); X[:, 0] += pd.factorize(pn_pre)[0] * 1e8          # T-bars of different PNs never within reach
ref = pn_kc & ((pn_reg == 'CA') | near_ca); cand = near_ca & ~pn_kc
on_bouton, _, b_it = grow(X[ref], np.zeros(int(ref.sum()), int), X[cand], step=BOUTON_D)
add_ca = near_ca & pn_kc
add_ca[np.flatnonzero(cand)[on_bouton]] = True
addrows = pn[add_ca].copy()
for c in pnca_off.columns:                       # same dtypes as the official calyx table
    if c in addrows: addrows[c] = addrows[c].astype(pnca_off[c].dtype)
addrows = addrows[list(pnca_off.columns)]
lab_col = {'flywire': 'neuropil', 'hemibrain': 'roi_pre', 'malecns': 'primary_post', 'banc': 'neuropil'}[ds]
off = pnca_off.assign(outside_roi=False, roi_assigned=banc_label(pnca_off) if ds == 'banc' else pnca_off[lab_col].astype(str).values)
addrows = addrows.assign(outside_roi=True, roi_assigned=[synth('CA', s) for s in s_[add_ca]])
pd.concat([off, addrows], ignore_index=True).to_parquet(A + f'{ds}_pnca_raw.parquet')
mb_add = add_ca | ((src != '') & (r_ != 'CA'))
pairs = pd.DataFrame({'pre_id': pn_pre[mb_add], 'post_id': pn_post[mb_add], 'region': r_[mb_add]})
pairs.groupby(['pre_id', 'post_id']).size().rename('n').reset_index().to_parquet(A + f'{ds}_pnmb_added_pairs.parquet')
stats['alpn'] = {'rows': int(len(pn)), 'official_calyx_rows': int(len(pnca_off)), 'added_calyx_rows': int(add_ca.sum()),
                 'added_calyx_ALPN_to_KC': int((add_ca & pn_kc).sum()), 'added_calyx_other_partners_on_KC_boutons': int(on_bouton.sum()),
                 'near_calyx_left_out_not_on_KC_bouton': int((~on_bouton).sum()), 'bouton_steps': b_it,
                 'added_MB_rows_by_region': pd.Series(r_[mb_add]).value_counts().to_dict()}

# ---------------- KC -> MBON table (lobes only, like the official lobe table) ----------------
r_, s_, src, dist = assign(km_xyz, km_reg)
add_lb = np.isin(r_, LOBES)
k = km[add_lb].copy()
k['outside_roi_dist_nm'] = np.round(dist[add_lb])
if ds == 'flywire':
    k['neuropil_official'] = k.neuropil; k['neuropil'] = [synth(a, b) for a, b in zip(r_[add_lb], s_[add_lb])]
elif ds == 'hemibrain':
    k['roi_post_official'] = k.roi_post; k['roi_post'] = [synth(a, b) for a, b in zip(r_[add_lb], s_[add_lb])]
elif ds == 'malecns':
    k['roi_official'] = k.roi; k['roi'] = [synth(a, b) for a, b in zip(r_[add_lb], s_[add_lb])]
else:
    k['neuropil_official'] = k.neuropil; k['neuropil'] = ['MB_' + a for a in r_[add_lb]]
k.to_parquet(A + f'{ds}_kcmbon_added.parquet')
stats['kc_mbon'] = {'rows': int(len(km)), 'official_lobe_rows': int(np.isin(km_reg, LOBES).sum()), 'added_lobe_rows': int(add_lb.sum()),
                    'added_by_region': pd.Series(r_[add_lb]).value_counts().to_dict(),
                    'outside_rows_assigned_CA_or_PED': int(np.isin(r_, ['CA', 'PED']).sum())}
json.dump(stats, open(A + f'assign_{ds}.json', 'w'), indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
