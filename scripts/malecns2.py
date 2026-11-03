import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pandas as pd
m=pd.read_feather('dl/body-annotations-male-cns-v1.0-minconf-0.5.feather')
PN=pa.array(m.bodyId[m['class']=='ALPN'].astype('int64').tolist(),type=pa.int64())
DAN=pa.array(m.bodyId[m['class']=='DAN'].astype('int64').tolist(),type=pa.int64())
r=ipc.open_file('dl/syn-partners-male-cns-v1.0-minconf-0.5.feather')
ca=[];dan=[]
MBR=pa.array(["CA(R)","CA(L)","PED(R)","PED(L)","aL(R)","aL(L)","a'L(R)","a'L(L)","bL(R)","bL(L)","b'L(R)","b'L(L)","gL(R)","gL(L)"])
for i in range(r.num_record_batches):
    b=r.get_batch(i)
    roi=b['primary_post'].cast(pa.string())
    ca.append(b.filter(pc.and_(pc.is_in(b['body_pre'],PN),pc.starts_with(roi,'CA('))).to_pandas())
    dan.append(b.filter(pc.and_(pc.is_in(b['body_pre'],DAN),pc.is_in(roi,MBR))).to_pandas())
for nm,l in [('pnca',ca),('dan',dan)]:
    df=pd.concat(l,ignore_index=True); df['primary_post']=df.primary_post.astype(str); df.to_parquet(f'out/malecns_{nm}_raw.parquet'); print(nm,len(df))
