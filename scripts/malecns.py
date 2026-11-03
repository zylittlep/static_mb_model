import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd
r=ipc.open_file('dl/syn-partners-male-cns-v1.0-minconf-0.5.feather')
apl=pa.array([10540,10977],type=pa.int64()); out=[]
for i in range(r.num_record_batches):
    b=r.get_batch(i)
    m=pc.or_(pc.is_in(b['body_pre'],apl),pc.is_in(b['body_post'],apl))
    out.append(b.filter(m).to_pandas())
df=pd.concat(out,ignore_index=True); df['primary_post']=df.primary_post.astype(str)
print(len(df)); print(df.primary_post.value_counts().head(30)); print(df.head())
df.to_parquet('out/malecns_apl_raw.parquet')
