"""Hemibrain: KC T-bars -> MBON PSDs, with the MBON PSD in a lobe ROI."""
import sys; sys.path.insert(0, 'scripts')
import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd, numpy as np
from meta_util import types, kc_ids
D = 'dl/'; PTS = D + 'hemibrain-v1.2-synapse-points-with-body-roi-type-status.feather'
typ, mbon = types('hemibrain')
KC = pa.array(sorted(int(i) for i in kc_ids(typ)), type=pa.uint64()); MB = pa.array(sorted(int(i) for i in mbon), type=pa.uint64())
LOBES = pa.array(["aL(R)", "aL(L)", "a'L(R)", "a'L(L)", "bL(R)", "bL(L)", "b'L(R)", "b'L(L)", "gL(R)", "gL(L)"])
def pid(t):
    x = pc.cast(t['x'], pa.uint64()); y = pc.cast(t['y'], pa.uint64()); z = pc.cast(t['z'], pa.uint64())
    return pc.bit_wise_or(pc.bit_wise_or(pc.shift_left(z, 42), pc.shift_left(y, 21)), x)
def frame(f):
    return pd.DataFrame({'pid': pid(f).to_numpy(), 'body': f['body'].to_numpy(), 'x': f['x'].to_numpy(), 'y': f['y'].to_numpy(),
                         'z': f['z'].to_numpy(), 'roi': f['roi'].cast(pa.string()).to_pylist()})
r = ipc.open_file(PTS); kt, mp = [], []
for i in range(r.num_record_batches):
    b = r.get_batch(i); kind = b['kind'].cast(pa.string())
    f = b.filter(pc.and_(pc.is_in(b['body'], KC), pc.equal(kind, 'PreSyn')))
    if f.num_rows: kt.append(frame(f))
    f = b.filter(pc.and_(pc.and_(pc.is_in(b['body'], MB), pc.equal(kind, 'PostSyn')), pc.is_in(b['roi'].cast(pa.string()), LOBES)))
    if f.num_rows: mp.append(frame(f))
    if i % 200 == 0: print('points', i, r.num_record_batches, flush=True)
kt = pd.concat(kt, ignore_index=True).drop_duplicates('pid'); mp = pd.concat(mp, ignore_index=True).drop_duplicates('pid')
print('KC T-bars', len(kt), 'MBON lobe PSDs', len(mp), flush=True)
K = pa.array(kt.pid.values, type=pa.uint64()); M = pa.array(mp.pid.values, type=pa.uint64())
r = ipc.open_file(D + 'hemibrain-v1.2-synapse-partners.feather'); sub = []
for i in range(r.num_record_batches):
    b = r.get_batch(i).select(['pre_id', 'post_id'])
    sub.append(b.filter(pc.and_(pc.is_in(b['post_id'], M), pc.is_in(b['pre_id'], K))))
pairs = pa.Table.from_batches(sub).to_pandas()
pairs = pairs.merge(kt.add_suffix('_pre'), left_on='pre_id', right_on='pid_pre').merge(mp.add_suffix('_post'), left_on='post_id', right_on='pid_post')
pairs.to_parquet('out/hemibrain_kcmbon_lobes.parquet'); print('pairs', len(pairs)); print(pairs.roi_post.value_counts())
