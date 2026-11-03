"""Male CNS: every ALPN -> X synapse whose postsynaptic point lies in any MB ROI (calyx, peduncle, lobes)."""
import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd
m = pd.read_feather('dl/body-annotations-male-cns-v1.0-minconf-0.5.feather')
PN = pa.array(m.bodyId[m['class'] == 'ALPN'].astype('int64').tolist(), type=pa.int64())
MBR = pa.array(["CA(R)", "CA(L)", "PED(R)", "PED(L)", "aL(R)", "aL(L)", "a'L(R)", "a'L(L)", "bL(R)", "bL(L)", "b'L(R)", "b'L(L)", "gL(R)", "gL(L)"])
r = ipc.open_file('dl/syn-partners-male-cns-v1.0-minconf-0.5.feather'); out = []
for i in range(r.num_record_batches):
    b = r.get_batch(i); roi = b['primary_post'].cast(pa.string())
    f = b.filter(pc.and_(pc.is_in(b['body_pre'], PN), pc.is_in(roi, MBR)))
    out.append(pd.DataFrame({'pre_id': f['body_pre'].to_numpy(), 'post_id': f['body_post'].to_numpy(), 'roi': f['primary_post'].cast(pa.string()).to_pylist()}))
    if i % 500 == 0: print(i, r.num_record_batches, flush=True)
df = pd.concat(out, ignore_index=True); df.to_parquet('out/malecns_pnmb_raw.parquet')
print('rows', len(df)); print(df.roi.value_counts())
