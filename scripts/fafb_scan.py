import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, pandas as pd
D='dl/'
c=pd.read_csv(D+'classification.csv.gz',dtype={'root_id':str})
APL=['720575940613583001','720575940624547622']
PN=c.root_id[c['class']=='ALPN'].tolist()
DAN=c.root_id[c['class']=='DAN'].tolist()
f=pq.ParquetFile(D+'fafb_783_synapses.parquet')
cols=['id','connector_id','x','y','z','confidence','prepost','pre','post','neuropil','pre_label','post_label','side']
A=pa.array(APL); P=pa.array(PN); Dn=pa.array(DAN)
apl=[];ca=[];dan=[]
for i in range(f.metadata.num_row_groups):
    t=f.read_row_group(i,columns=cols)
    pre=t['pre'].cast(pa.string()); post=t['post'].cast(pa.string()); npil=t['neuropil'].cast(pa.string())
    apl.append(t.filter(pc.or_(pc.is_in(pre,A),pc.is_in(post,A))))
    ca.append(t.filter(pc.and_(pc.is_in(pre,P),pc.starts_with(npil,'MB_CA'))))
    dan.append(t.filter(pc.and_(pc.is_in(pre,Dn),pc.starts_with(npil,'MB_'))))
for n,l in [('apl',apl),('pnca',ca),('dan',dan)]:
    tt=pa.concat_tables(l); pq.write_table(tt,f'out/fafb_{n}_raw.parquet'); print(n,tt.num_rows)
