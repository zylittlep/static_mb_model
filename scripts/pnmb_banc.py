"""BANC: every ALPN -> X synapse in any MB neuropil (same v2 enriched table as the calyx scan).
Each worker thread keeps its own open ParquetFile so the (large) footer is read once per thread, not once per row group."""
import threading, pyarrow.fs as fs, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, pandas as pd
from concurrent.futures import ThreadPoolExecutor
b = pd.read_feather('dl/banc_888_meta.feather')
PN = pa.array(b.banc_888_id[b.cell_class == 'antennal_lobe_projection_neuron'].astype(str).tolist(), type=pa.large_string())
path = 'lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_synapses_v2_enriched.parquet'
g = fs.GcsFileSystem(anonymous=True)
meta = pq.ParquetFile(path, filesystem=g).metadata
n = meta.num_row_groups
tl = threading.local()
def work(i):
    if not hasattr(tl, 'f'):
        tl.f = pq.ParquetFile(path, filesystem=g, metadata=meta)
    t = tl.f.read_row_group(i, columns=['pre_root_id', 'post_root_id', 'neuropil'])
    pre = t['pre_root_id'].cast(pa.large_string()); npil = t['neuropil'].cast(pa.string())
    return t.filter(pc.and_(pc.is_in(pre, PN), pc.starts_with(npil, 'MB_')))
out = []
with ThreadPoolExecutor(24) as ex:
    for k, a in enumerate(ex.map(work, range(n))):
        out.append(a)
        if k % 100 == 0: print(k, n, flush=True)
t = pa.concat_tables(out).to_pandas(); t['pre_root_id'] = t.pre_root_id.astype(str); t['post_root_id'] = t.post_root_id.astype(str)
t.to_parquet('out/banc_pnmb_raw.parquet'); print('rows', len(t)); print(t.neuropil.value_counts())
