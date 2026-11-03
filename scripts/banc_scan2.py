import pyarrow.fs as fs, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, pandas as pd
from concurrent.futures import ThreadPoolExecutor
b=pd.read_feather('dl/banc_888_meta.feather')
PN=pa.array(b.banc_888_id[b.cell_class=='antennal_lobe_projection_neuron'].astype(str).tolist(),type=pa.large_string())
DAN=pa.array(b.banc_888_id[b.cell_class=='mushroom_body_dopaminergic_neuron'].astype(str).tolist(),type=pa.large_string())
path='lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_synapses_v2_enriched.parquet'
g=fs.GcsFileSystem(anonymous=True)
n=pq.ParquetFile(path,filesystem=g).metadata.num_row_groups
cols=['id','size','pre_root_id','post_root_id','X','Y','Z','pre_x','pre_y','pre_z','post_x','post_y','post_z','neuropil','side']
def work(i):
    t=pq.ParquetFile(path,filesystem=g).read_row_group(i,columns=cols)
    pre=t['pre_root_id'].cast(pa.large_string()); npil=t['neuropil'].cast(pa.string())
    isMB=pc.starts_with(npil,'MB_')
    return (t.filter(pc.and_(pc.is_in(pre,PN),pc.starts_with(npil,'MB_CA'))), t.filter(pc.and_(pc.is_in(pre,DAN),isMB)))
ca=[];dan=[]
with ThreadPoolExecutor(12) as ex:
    for k,(a,d) in enumerate(ex.map(work,range(n))):
        ca.append(a); dan.append(d)
        if k%100==0: print(k,n,flush=True)
for nm,l in [('pnca',ca),('dan',dan)]:
    tt=pa.concat_tables(l); pq.write_table(tt,f'out/banc_{nm}_raw.parquet'); print(nm,tt.num_rows)
