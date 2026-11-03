import pyarrow.csv as pcsv, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq, pandas as pd, urllib.request, gzip
c=pd.read_csv('dl/classification.csv.gz')
def suf(ids): return pa.array([int(str(i)[9:]) for i in ids],type=pa.int64())
APL=suf([720575940613583001,720575940624547622]); PN=suf(c.root_id[c['class']=='ALPN']); DAN=suf(c.root_id[c['class']=='DAN'])
url='https://storage.googleapis.com/flywire-data/codex/data/fafb/783/fafb_v783_princeton_synapse_table.csv.gz'
src=gzip.GzipFile(fileobj=urllib.request.urlopen(url))
co=pcsv.ConvertOptions(column_types={'pre_root_id_720575940':pa.int64(),'post_root_id_720575940':pa.int64(),'neuropil':pa.string()})
r=pcsv.open_csv(src,read_options=pcsv.ReadOptions(block_size=1<<26),convert_options=co)
apl=[];ca=[];dan=[];n=0
for b in r:
    pre=b['pre_root_id_720575940']; post=b['post_root_id_720575940']; np_=pc.fill_null(b['neuropil'],'')
    apl.append(pa.Table.from_batches([b]).filter(pc.or_(pc.is_in(pre,APL),pc.is_in(post,APL))))
    ca.append(pa.Table.from_batches([b]).filter(pc.and_(pc.is_in(pre,PN),pc.starts_with(np_,'MB_CA'))))
    dan.append(pa.Table.from_batches([b]).filter(pc.and_(pc.is_in(pre,DAN),pc.starts_with(np_,'MB_'))))
    n+=b.num_rows
    if len(apl)%20==0: print(n,flush=True)
for nm,l in [('apl',apl),('pnca',ca),('dan',dan)]:
    t=pa.concat_tables(l); pq.write_table(t,f'out/fafbP_{nm}_raw.parquet'); print(nm,t.num_rows)
print('total',n)
