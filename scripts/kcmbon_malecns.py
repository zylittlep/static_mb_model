"""Male CNS: KC -> MBON synapses whose postsynaptic point is in a lobe ROI."""
import sys; sys.path.insert(0, 'scripts')
import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd
from meta_util import types, kc_ids
typ, mbon = types('malecns')
KC = pa.array(sorted(int(i) for i in kc_ids(typ)), type=pa.int64()); MB = pa.array(sorted(int(i) for i in mbon), type=pa.int64())
LOBES = pa.array(["aL(R)", "aL(L)", "a'L(R)", "a'L(L)", "bL(R)", "bL(L)", "b'L(R)", "b'L(L)", "gL(R)", "gL(L)"])
r = ipc.open_file('dl/syn-partners-male-cns-v1.0-minconf-0.5.feather'); out = []
for i in range(r.num_record_batches):
    b = r.get_batch(i); roi = b['primary_post'].cast(pa.string())
    f = b.filter(pc.and_(pc.and_(pc.is_in(b['body_pre'], KC), pc.is_in(b['body_post'], MB)), pc.is_in(roi, LOBES)))
    if f.num_rows:
        t = f.select(['x_pre', 'y_pre', 'z_pre', 'x_post', 'y_post', 'z_post', 'body_pre', 'body_post']).to_pandas()
        t['roi'] = f['primary_post'].cast(pa.string()).to_pylist(); out.append(t)
    if i % 500 == 0: print(i, r.num_record_batches, flush=True)
df = pd.concat(out, ignore_index=True); df.to_parquet('out/malecns_kcmbon_lobes.parquet'); print('rows', len(df)); print(df.roi.value_counts())
