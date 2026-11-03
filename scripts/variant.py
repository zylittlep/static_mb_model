"""Which version of the outputs a script builds.
Default: official ROIs only (the original tables and plots).
APL_VARIANT=outside_roi: also the synapses outside every official MB ROI that lie within 3 µm of the mushroom body
(offroi/assign.py), with intermediate tables in out/outside_roi/ and outputs in apl_synapses/with_outside_ROI_synapses/.
APL_OUT / APL_FINAL override the two folders (used to check that the default build is unchanged)."""
import os
import pandas as pd

OUTSIDE = os.environ.get('APL_VARIANT', '') == 'outside_roi'
ADDED = 'out/outside_roi/'
FINAL = os.environ.get('APL_FINAL') or (ADDED if OUTSIDE else 'out/')
OUT = os.environ.get('APL_OUT') or ('/mnt/project-files/apl_synapses/' + ('with_outside_ROI_synapses/' if OUTSIDE else ''))
TAG = '   [incl. synapses outside the official ROIs]' if OUTSIDE else ''


def fw_apl_raw(columns=None):
    """FlyWire Princeton APL synapses, all neuropils. In the outside-ROI version, `neuropil` of each synapse added to the MB
    is replaced by the MB neuropil it was assigned to (MB_CA_L, MB_ML_R, ...); the dataset's own label is `neuropil_official`."""
    r = pd.read_parquet('out/fafbP_apl_raw.parquet')
    if OUTSIDE:
        a = pd.read_parquet(ADDED + 'flywire_apl_assigned.parquet', columns=['roi_assigned'])
        r['neuropil_official'] = r.neuropil
        r['neuropil'] = a.roi_assigned.values
    return r if columns is None else r[columns]
