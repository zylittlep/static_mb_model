"""One pass over the male CNS synapse-partner table: every synapse whose presynaptic body is an ALPN, any ROI."""
import time
import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd
m = pd.read_feather('dl/body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['bodyId', 'class'])
PN = pa.array(m.bodyId[m['class'] == 'ALPN'].astype('int64').tolist(), type=pa.int64())
r = ipc.open_file('dl/syn-partners-male-cns-v1.0-minconf-0.5.feather'); out = []; t0 = time.time()
print('batches', r.num_record_batches, 'schema', r.schema.names, flush=True)
for i in range(r.num_record_batches):
    b = r.get_batch(i)
    f = b.filter(pc.is_in(b['body_pre'], PN))
    if f.num_rows:
        d = f.to_pandas(); d['primary_post'] = d.primary_post.astype(str); out.append(d)
    if i % 1000 == 0: print(i, round(time.time() - t0), 'rows so far', sum(len(x) for x in out), flush=True)
df = pd.concat(out, ignore_index=True)
df.to_parquet('out/malecns_pn_all_raw.parquet')
print('rows', len(df), 'time', round(time.time() - t0)); print(df.primary_post.value_counts().head(20))
