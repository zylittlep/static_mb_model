# official hemibrain v1.2 type per body, read from the synapse-points release file
import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd
r=ipc.open_file('dl/hemibrain-v1.2-synapse-points-with-body-roi-type-status.feather'); out=[]
for i in range(r.num_record_batches):
    b=r.get_batch(i).select(['body','type'])
    t=pa.Table.from_batches([b]); t=t.set_column(1,'type',pc.cast(t['type'],pa.string()))
    t=t.filter(pc.is_valid(t['type']))
    out.append(t.group_by(['body','type']).aggregate([]))
df=pa.concat_tables(out).to_pandas().drop_duplicates()
df.to_parquet('out/hemibrain_body_types.parquet'); print(len(df), df.body.duplicated().sum())
k=df[df.type.str.startswith('KC')]; print(k.type.value_counts())
