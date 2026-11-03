"""BANC v2 enriched synapses: KC -> MBON in MB_VL / MB_ML."""
import sys; sys.path.insert(0, 'scripts')
import threading, pyarrow.fs as fs, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
from concurrent.futures import ThreadPoolExecutor
from meta_util import types, kc_ids
typ, mbon = types('banc')
KC = pa.array(sorted(kc_ids(typ)), type=pa.large_string()); MB = pa.array(sorted(mbon), type=pa.large_string())
path = 'lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_synapses_v2_enriched.parquet'
g = fs.GcsFileSystem(anonymous=True)
meta = pq.ParquetFile(path, filesystem=g).metadata; n = meta.num_row_groups
tl = threading.local()
COLS = ['pre_root_id', 'post_root_id', 'pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z', 'neuropil', 'side']
def work(i):
    if not hasattr(tl, 'f'):
        tl.f = pq.ParquetFile(path, filesystem=g, metadata=meta)
    t = tl.f.read_row_group(i, columns=COLS)
    npil = t['neuropil'].cast(pa.string())
    m = pc.and_(pc.and_(pc.is_in(t['pre_root_id'].cast(pa.large_string()), KC), pc.is_in(t['post_root_id'].cast(pa.large_string()), MB)),
                pc.or_(pc.equal(npil, 'MB_VL'), pc.equal(npil, 'MB_ML')))
    return t.filter(m)
out = []
with ThreadPoolExecutor(24) as ex:
    for k, a in enumerate(ex.map(work, range(n))):
        out.append(a)
        if k % 100 == 0: print(k, n, flush=True)
t = pa.concat_tables(out).to_pandas(); t['pre_root_id'] = t.pre_root_id.astype(str); t['post_root_id'] = t.post_root_id.astype(str)
t.to_parquet('out/banc_kcmbon_lobes.parquet'); print('rows', len(t)); print(t.neuropil.value_counts())
