"""Male CNS v1.0: every KC -> MBON synapse in any ROI (the official-ROI table kept lobe ROIs only). Run from the working folder (with dl/, out/, scripts/)."""
import sys; sys.path.insert(0, 'scripts')
import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd
from meta_util import types, kc_ids
typ, mbon = types('malecns')
KC = pa.array(sorted(int(i) for i in kc_ids(typ)), type=pa.int64()); MB = pa.array(sorted(int(i) for i in mbon), type=pa.int64())
r = ipc.open_file('dl/syn-partners-male-cns-v1.0-minconf-0.5.feather'); out = []
for i in range(r.num_record_batches):
    b = r.get_batch(i)
    f = b.filter(pc.and_(pc.is_in(b['body_pre'], KC), pc.is_in(b['body_post'], MB)))
    if f.num_rows:
        t = f.select(['x_pre', 'y_pre', 'z_pre', 'x_post', 'y_post', 'z_post', 'body_pre', 'body_post']).to_pandas()
        t['roi'] = f['primary_post'].cast(pa.string()).to_pylist(); out.append(t)
    if i % 500 == 0: print(i, r.num_record_batches, flush=True)
df = pd.concat(out, ignore_index=True); df.to_parquet('out/outside_roi/malecns_kcmbon_all.parquet')
print('rows', len(df)); print(df.roi.value_counts().head(20).to_dict())
