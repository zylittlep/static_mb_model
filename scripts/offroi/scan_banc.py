"""BANC 888 v2 enriched synapses: every ALPN-presynaptic synapse and every KC -> MBON synapse, in any neuropil.
Each worker thread keeps its own ParquetFile (footer read once). Run from the working folder (with dl/, out/, scripts/)."""
import sys; sys.path.insert(0, 'scripts')
import threading, pyarrow.fs as fs, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from meta_util import types, kc_ids
b = pd.read_feather('dl/banc_888_meta.feather', columns=['banc_888_id', 'cell_class'])
typ, mbon = types('banc')
PN = pa.array(b.banc_888_id[b.cell_class == 'antennal_lobe_projection_neuron'].astype(str).tolist(), type=pa.large_string())
KC = pa.array(sorted(kc_ids(typ)), type=pa.large_string()); MB = pa.array(sorted(mbon), type=pa.large_string())
path = 'lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_synapses_v2_enriched.parquet'
g = fs.GcsFileSystem(anonymous=True)
meta = pq.ParquetFile(path, filesystem=g).metadata; n = meta.num_row_groups
COLS = ['id', 'size', 'pre_root_id', 'post_root_id', 'X', 'Y', 'Z', 'pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z', 'neuropil', 'side']
tl = threading.local()
def work(i):
    if not hasattr(tl, 'f'):
        tl.f = pq.ParquetFile(path, filesystem=g, metadata=meta)
    t = tl.f.read_row_group(i, columns=COLS)
    pre = t['pre_root_id'].cast(pa.large_string()); post = t['post_root_id'].cast(pa.large_string())
    return t.filter(pc.is_in(pre, PN)), t.filter(pc.and_(pc.is_in(pre, KC), pc.is_in(post, MB)))
pn, km = [], []
with ThreadPoolExecutor(16) as ex:
    for k, (a, m) in enumerate(ex.map(work, range(n))):
        pn.append(a); km.append(m)
        if k % 100 == 0: print(k, n, flush=True)
for nm, l in [('banc_pn_all_raw', pn), ('banc_kcmbon_all', km)]:
    t = pa.concat_tables(l).to_pandas(); t['pre_root_id'] = t.pre_root_id.astype(str); t['post_root_id'] = t.post_root_id.astype(str)
    t.to_parquet(f'out/outside_roi/{nm}.parquet'); print(nm, len(t)); print(t.neuropil.value_counts().head(12).to_dict())
