import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq, numpy as np, pandas as pd
D='dl/'; PTS=D+'hemibrain-v1.2-synapse-points-with-body-roi-type-status.feather'
h=pd.read_feather(D+'hemibrain_121_meta.feather')
APL=pa.array([425790257],type=pa.uint64())
PN=pa.array(h.hemibrain_121_id[h.cell_class=='antennal_lobe_projection_neuron'].astype('uint64'),type=pa.uint64())
DAN=pa.array(h.hemibrain_121_id[h.cell_class=='mushroom_body_dopaminergic_neuron'].astype('uint64'),type=pa.uint64())
MB=pa.array(["CA(R)","CA(L)","PED(R)","aL(R)","aL(L)","a'L(R)","a'L(L)","bL(R)","bL(L)","b'L(R)","b'L(L)","gL(R)","gL(L)"])
COLS=['x','y','z','kind','conf','body','roi']
def withpid(b):
    x=pc.cast(b['x'],pa.uint64());y=pc.cast(b['y'],pa.uint64());z=pc.cast(b['z'],pa.uint64())
    pid=pc.bit_wise_or(pc.bit_wise_or(pc.shift_left(z,42),pc.shift_left(y,21)),x)
    t=pa.Table.from_batches([b.select(COLS)])
    t=t.set_column(t.schema.get_field_index('kind'),'kind',pc.cast(t['kind'],pa.string()))
    t=t.set_column(t.schema.get_field_index('roi'),'roi',pc.cast(t['roi'],pa.string()))
    return t.append_column('pid',pid)
r=ipc.open_file(PTS); apl=[];pnca=[];dan=[]
for i in range(r.num_record_batches):
    t=withpid(r.get_batch(i))
    apl.append(t.filter(pc.is_in(t['body'],APL)))
    pnca.append(t.filter(pc.and_(pc.and_(pc.is_in(t['body'],PN),pc.equal(t['kind'],'PreSyn')),pc.starts_with(t['roi'],'CA('))))
    dan.append(t.filter(pc.and_(pc.and_(pc.is_in(t['body'],DAN),pc.equal(t['kind'],'PreSyn')),pc.is_in(t['roi'],MB))))
apl=pa.concat_tables(apl); pnca=pa.concat_tables(pnca); dan=pa.concat_tables(dan)
print('apl pts',apl.num_rows,'pn ca tbars',pnca.num_rows,'dan tbars',dan.num_rows,flush=True)
pq.write_table(dan,'out/hemibrain_dan_raw.parquet'); pq.write_table(pnca,'out/hemibrain_pnca_tbars.parquet')
r=ipc.open_file(D+'hemibrain-v1.2-synapse-partners.feather'); subA=[];subP=[]
aplids=apl['pid'].combine_chunks(); pnids=pnca['pid'].combine_chunks()
for i in range(r.num_record_batches):
    b=r.get_batch(i).select(['pre_id','post_id'])
    subA.append(b.filter(pc.or_(pc.is_in(b['pre_id'],aplids),pc.is_in(b['post_id'],aplids))))
    subP.append(b.filter(pc.is_in(b['pre_id'],pnids)))
subA=pa.Table.from_batches(subA).to_pandas(); subP=pa.Table.from_batches(subP).to_pandas()
print('apl pairs',len(subA),'pn pairs',len(subP),flush=True)
need=pa.array(np.unique(np.concatenate([subA.pre_id.values,subA.post_id.values,subP.pre_id.values,subP.post_id.values])),type=pa.uint64())
r=ipc.open_file(PTS); pts=[]
for i in range(r.num_record_batches):
    t=withpid(r.get_batch(i)); pts.append(t.filter(pc.is_in(t['pid'],need)))
pts=pa.concat_tables(pts).to_pandas().drop_duplicates('pid').set_index('pid')
def join(sub):
    pre=pts.reindex(sub.pre_id.values).add_suffix('_pre').reset_index(drop=True)
    post=pts.reindex(sub.post_id.values).add_suffix('_post').reset_index(drop=True)
    return pd.concat([pre,post],axis=1)
join(subA).to_parquet('out/hemibrain_apl_raw.parquet')
join(subP).to_parquet('out/hemibrain_pnca_raw.parquet')
print('done')
