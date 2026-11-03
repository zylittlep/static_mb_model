"""Hemibrain: ALPN T-bars in MB ROIs other than the calyx (peduncle, lobes), and their postsynaptic bodies."""
import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd, numpy as np
D = 'dl/'; PTS = D + 'hemibrain-v1.2-synapse-points-with-body-roi-type-status.feather'
h = pd.read_feather(D + 'hemibrain_121_meta.feather')
PN = pa.array(h.hemibrain_121_id[h.cell_class == 'antennal_lobe_projection_neuron'].astype('uint64'), type=pa.uint64())
NONCA = pa.array(["PED(R)", "aL(R)", "aL(L)", "a'L(R)", "a'L(L)", "bL(R)", "bL(L)", "b'L(R)", "b'L(L)", "gL(R)", "gL(L)"])
def pid(t):
    x = pc.cast(t['x'], pa.uint64()); y = pc.cast(t['y'], pa.uint64()); z = pc.cast(t['z'], pa.uint64())
    return pc.bit_wise_or(pc.bit_wise_or(pc.shift_left(z, 42), pc.shift_left(y, 21)), x)
r = ipc.open_file(PTS); tb = []
for i in range(r.num_record_batches):
    b = r.get_batch(i); roi = b['roi'].cast(pa.string()); kind = b['kind'].cast(pa.string())
    f = b.filter(pc.and_(pc.and_(pc.is_in(b['body'], PN), pc.equal(kind, 'PreSyn')), pc.is_in(roi, NONCA)))
    if f.num_rows:
        tb.append(pd.DataFrame({'pid': pid(f).to_numpy(), 'pre_body': f['body'].to_numpy(), 'roi': f['roi'].cast(pa.string()).to_pylist()}))
tb = pd.concat(tb, ignore_index=True) if tb else pd.DataFrame(columns=['pid', 'pre_body', 'roi'])
print('PN T-bars outside calyx but inside MB:', len(tb), tb.roi.value_counts().to_dict(), flush=True)
pairs = pd.DataFrame(columns=['pre_id', 'post_id'])
if len(tb):
    ids = pa.array(tb.pid.values, type=pa.uint64()); r = ipc.open_file(D + 'hemibrain-v1.2-synapse-partners.feather'); sub = []
    for i in range(r.num_record_batches):
        b = r.get_batch(i).select(['pre_id', 'post_id']); sub.append(b.filter(pc.is_in(b['pre_id'], ids)))
    pairs = pa.Table.from_batches(sub).to_pandas()
    need = pa.array(np.unique(pairs.post_id.values), type=pa.uint64()); r = ipc.open_file(PTS); pts = []
    for i in range(r.num_record_batches):
        b = r.get_batch(i); p = pid(b); f = b.filter(pc.is_in(p, need))
        if f.num_rows: pts.append(pd.DataFrame({'pid': pid(f).to_numpy(), 'post_body': f['body'].to_numpy()}))
    pts = pd.concat(pts).drop_duplicates('pid').set_index('pid')['post_body']
    pairs['post_body'] = pairs.post_id.map(pts).values
    pairs = pairs.merge(tb.rename(columns={'pid': 'pre_id'}), on='pre_id')
pairs.to_parquet('out/hemibrain_pn_nonCA_mb_pairs.parquet'); print('pairs', len(pairs))
