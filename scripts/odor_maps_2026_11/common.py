"""Shared loaders for the Report 03 revision (2026-11): new ORN tables, GH146, PN<->APL calyx synapses (corrected calyx,
i.e. with_outside_ROI_synapses) in posterior view and in 3D anatomical axes."""
import numpy as np, pandas as pd, matplotlib
from matplotlib.colors import TwoSlopeNorm

B = '/mnt/project-files/apl_synapses/'
OW = B + 'with_outside_ROI_synapses/odor_weighted/'
NEW = B + 'orn_inputs_2026-11-03/'
OUT = OW + 'new_ORN_2026-11-03/'

DSN = {'flywire': 'FlyWire 783', 'hemibrain': 'Hemibrain v1.2', 'malecns': 'Male CNS v1.0', 'banc': 'BANC 888'}
CALYCES = [('flywire', 'L'), ('flywire', 'R'), ('hemibrain', 'R'), ('malecns', 'L'), ('malecns', 'R')]   # BANC left out (Yang)
MODEL_MBS = [('hemibrain', 'R'), ('malecns', 'L'), ('malecns', 'R')]
INK, MUTED = '#0b0b0b', '#52514e'
PANEL_BG, NODATA = '#d6d6d4', '#8f8e8a'
CMAP = matplotlib.colormaps['RdBu_r']

# native nm -> anatomical µm axes (right, anterior, dorsal). Right and dorsal are the posterior-view axes of the earlier maps
# (odor_maps.py VIEW); anterior is the third axis, signed so that the peduncle lies anterior of the calyx (checked on the
# APL calyx and peduncle synapses of each dataset, 2026-11-04).
AXES = {'flywire': lambda x, y, z: (x, -z, -y), 'hemibrain': lambda x, y, z: (-x, y, -z),
        'malecns': lambda x, y, z: (-x, -z, -y), 'banc': lambda x, y, z: (-x, -y, z)}


def short(c):
    return c.split(': ')[-1]


def response_sets():
    """{name: (glomerulus x odor DataFrame, panel label per odor, title)} in Yang's odor order."""
    old = pd.read_csv(OW + 'ORN_fly_means_by_glomerulus.csv', index_col=0)            # megamat + validation2 in Yang's order
    gh = pd.read_csv(OW + 'GH146_mean_by_glomerulus.csv', index_col=0).drop(columns='n_flies')
    diag_order = [short(c) for c in gh.columns if c.startswith('glomeruli_diagnostics')]
    out = {}
    for name, f, main, title in [('ORN_megamat_diag', 'ORN_megamat_diagnostics_flymean.csv', 'megamat', 'ORN, megamat + diagnostics (dataset 1, 9 flies)'),
                                 ('ORN_valid2_diag', 'ORN_validation2_diagnostics_flymean.csv', 'validation2', 'ORN, validation2 + diagnostics (dataset 4, 6 flies)')]:
        M = pd.read_csv(NEW + f, index_col=0)
        first = [c for c in old.columns if c in M.columns and (c.endswith('@ -3') if main == 'megamat' else True)]
        first = first[:17] if main == 'megamat' else [c for c in old.columns[17:] if c in M.columns]
        rest = [c for c in diag_order if c in M.columns and c not in first] + sorted(c for c in M.columns if c not in first and c not in diag_order)
        assert len(first) + len(rest) == M.shape[1], (name, len(first), len(rest), M.shape)
        out[name] = (M[first + rest], [main] * len(first) + ['diagnostics'] * len(rest), title)
    cols = list(gh.columns)
    names = [short(c) for c in cols]
    dup = {n for n in names if names.count(n) > 1}                   # ms @ -3 was given in both GH146 panels
    names = [n + (' (diag)' if c.startswith('glom') else ' (megamat)') if n in dup else n for n, c in zip(names, cols)]
    out['GH146'] = (gh.set_axis(names, axis=1), ['diagnostics' if c.startswith('glom') else 'megamat' for c in cols],
                    'GH146 PNs, diagnostics + megamat (7 flies)')
    return out


def norms(rs):
    orn = pd.concat([rs['ORN_megamat_diag'][0], rs['ORN_valid2_diag'][0]], axis=1)
    gh = rs['GH146'][0]
    return {'ORN': TwoSlopeNorm(vcenter=0, vmin=np.nanmin(orn.values), vmax=np.nanmax(orn.values)),
            'GH146': TwoSlopeNorm(vcenter=0, vmin=np.nanmin(gh.values), vmax=np.nanmax(gh.values))}


def pn_synapses():
    """PN<->APL calyx synapses of the corrected calyx with posterior-view and 3D anatomical coordinates (µm)."""
    S = pd.read_csv(OW + 'PN_APL_synapses_ORN_per_odor_weighted.csv',
                    usecols=['dataset', 'apl_id', 'apl_side', 'direction', 'pn_id', 'pn_type', 'glomerulus', 'mb_region',
                             'x_nm', 'y_nm', 'z_nm', 'view_right_um', 'view_dorsal_um'], dtype={'apl_id': str, 'pn_id': str})
    S = S[S.mb_region == 'CA'].copy()
    for ds, f in AXES.items():
        m = S.dataset == ds
        r, a, d = f(S.x_nm[m].values / 1e3, S.y_nm[m].values / 1e3, S.z_nm[m].values / 1e3)
        S.loc[m, 'right_um'], S.loc[m, 'anterior_um'], S.loc[m, 'dorsal_um'] = r, a, d
    assert np.allclose(S.right_um, S.view_right_um, atol=1e-3) and np.allclose(S.dorsal_um, S.view_dorsal_um, atol=1e-3)
    return S


CV = B + 'with_outside_ROI_synapses/claws_v2m2/'


def kc_synapses():
    """KC<->APL calyx synapses on the new claws (claw_v2m2; hemibrain and male CNS only), with the glomerulus of the claw's PN
    (empty when its type names two glomeruli or none) and posterior-view / 3D anatomical coordinates (µm) of the
    postsynaptic site, as for the PN synapses."""
    K = pd.read_csv(CV + 'KC_APL_synapses_on_claws_by_glomerulus.csv', dtype={'apl_id': str, 'kc_id': str},
                    usecols=['dataset', 'apl_id', 'apl_side', 'direction', 'kc_id', 'claw_id', 'glomeruli', 'x_nm', 'y_nm', 'z_nm',
                             'view_right_um', 'view_dorsal_um'])
    K['glomerulus'] = K.glomeruli.fillna('').where(~K.glomeruli.fillna('').str.contains(r'\+'), '')
    K['glomerulus'] = K.glomerulus.replace('', np.nan)
    for ds in ['hemibrain', 'malecns']:
        m = K.dataset == ds
        r, a, d = AXES[ds](K.x_nm[m].values / 1e3, K.y_nm[m].values / 1e3, K.z_nm[m].values / 1e3)
        K.loc[m, 'right_um'], K.loc[m, 'anterior_um'], K.loc[m, 'dorsal_um'] = r, a, d
    assert np.allclose(K.right_um, K.view_right_um, atol=1e-3) and np.allclose(K.dorsal_um, K.view_dorsal_um, atol=1e-3)
    return K
