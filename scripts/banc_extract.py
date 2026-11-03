import pyarrow.fs as fs, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, sys
from concurrent.futures import ThreadPoolExecutor
APL={'720575941482622627','720575941734526507'}
path=sys.argv[1]; out=sys.argv[2]
g=fs.GcsFileSystem(anonymous=True)
f0=pq.ParquetFile(path, filesystem=g); n=f0.metadata.num_row_groups
names=f0.schema_arrow.names
cols=[c for c in ['id','size','pre_root_id','post_root_id','X','Y','Z','pre_x','pre_y','pre_z','post_x','post_y','post_z','neuropil','neuropil_detailed','region','side','label'] if c in names]
apl=pa.array(list(APL),type=pa.large_string())
def work(i):
    f=pq.ParquetFile(path, filesystem=g)
    t=f.read_row_group(i, columns=cols)
    m=pc.or_(pc.is_in(t['pre_root_id'].cast(pa.large_string()),apl), pc.is_in(t['post_root_id'].cast(pa.large_string()),apl))
    return t.filter(m)
res=[]
with ThreadPoolExecutor(12) as ex:
    for k,t in enumerate(ex.map(work, range(n))):
        res.append(t)
        if k%100==0: print(k, n, sum(r.num_rows for r in res), flush=True)
pq.write_table(pa.concat_tables(res), out)
print('done')
