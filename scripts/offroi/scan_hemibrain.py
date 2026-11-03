"""Hemibrain v1.2: every ALPN T-bar -> partner (any ROI), and every KC T-bar -> MBON PSD (any ROI), with both points.
Same files and point-id scheme as scripts/hemibrain.py and scripts/kcmbon_hemibrain.py. Run from the working folder (with dl/, out/, scripts/)."""
import sys; sys.path.insert(0, 'scripts')
import pyarrow.ipc as ipc, pyarrow as pa, pyarrow.compute as pc, numpy as np, pandas as pd
from meta_util import types, kc_ids
D = 'dl/'; PTS = D + 'hemibrain-v1.2-synapse-points-with-body-roi-type-status.feather'
h = pd.read_feather(D + 'hemibrain_121_meta.feather')
typ, mbon = types('hemibrain')
PN = pa.array(h.hemibrain_121_id[h.cell_class == 'antennal_lobe_projection_neuron'].astype('uint64'), type=pa.uint64())
KC = pa.array(sorted(int(i) for i in kc_ids(typ)), type=pa.uint64()); MB = pa.array(sorted(int(i) for i in mbon), type=pa.uint64())
COLS = ['x', 'y', 'z', 'kind', 'conf', 'body', 'roi']
def pid(t):
    x = pc.cast(t['x'], pa.uint64()); y = pc.cast(t['y'], pa.uint64()); z = pc.cast(t['z'], pa.uint64())
    return pc.bit_wise_or(pc.bit_wise_or(pc.shift_left(z, 42), pc.shift_left(y, 21)), x)
def frame(f):
    d = f.select(COLS).to_pandas(); d['kind'] = d['kind'].astype(str); d['roi'] = d['roi'].astype(str); d['pid'] = pid(f).to_numpy()
    return d
r = ipc.open_file(PTS); pnt, kct, mbp = [], [], []
for i in range(r.num_record_batches):
    b = r.get_batch(i); kind = b['kind'].cast(pa.string())
    pre = pc.equal(kind, 'PreSyn')
    for lst, m in [(pnt, pc.and_(pc.is_in(b['body'], PN), pre)), (kct, pc.and_(pc.is_in(b['body'], KC), pre)),
                   (mbp, pc.and_(pc.is_in(b['body'], MB), pc.equal(kind, 'PostSyn')))]:
        f = b.filter(m)
        if f.num_rows: lst.append(frame(f))
    if i % 200 == 0: print('points', i, r.num_record_batches, flush=True)
pnt, kct, mbp = [pd.concat(l, ignore_index=True).drop_duplicates('pid') for l in (pnt, kct, mbp)]
print('PN T-bars', len(pnt), 'KC T-bars', len(kct), 'MBON PSDs', len(mbp), flush=True)
P_ = pa.array(pnt.pid.values, type=pa.uint64()); K_ = pa.array(kct.pid.values, type=pa.uint64()); M_ = pa.array(mbp.pid.values, type=pa.uint64())
r = ipc.open_file(D + 'hemibrain-v1.2-synapse-partners.feather'); sp, sk = [], []
for i in range(r.num_record_batches):
    b = r.get_batch(i).select(['pre_id', 'post_id'])
    sp.append(b.filter(pc.is_in(b['pre_id'], P_)))
    sk.append(b.filter(pc.and_(pc.is_in(b['pre_id'], K_), pc.is_in(b['post_id'], M_))))
sp = pa.Table.from_batches(sp).to_pandas(); sk = pa.Table.from_batches(sk).to_pandas()
print('PN pairs', len(sp), 'KC->MBON pairs', len(sk), flush=True)
need = pa.array(np.unique(sp.post_id.values), type=pa.uint64()); r = ipc.open_file(PTS); pts = []
for i in range(r.num_record_batches):
    b = r.get_batch(i); f = b.filter(pc.is_in(pid(b), need))
    if f.num_rows: pts.append(frame(f))
pts = pd.concat(pts).drop_duplicates('pid').set_index('pid')
pre = pnt.set_index('pid').reindex(sp.pre_id.values).add_suffix('_pre').reset_index(drop=True)
post = pts.reindex(sp.post_id.values).add_suffix('_post').reset_index(drop=True)
pd.concat([pre, post], axis=1)[['x_pre', 'y_pre', 'z_pre', 'kind_pre', 'conf_pre', 'body_pre', 'roi_pre',
                                'x_post', 'y_post', 'z_post', 'kind_post', 'conf_post', 'body_post', 'roi_post']].to_parquet('out/outside_roi/hemibrain_pn_all_raw.parquet')
k = sk.merge(kct.drop(columns=['kind', 'conf']).add_suffix('_pre'), left_on='pre_id', right_on='pid_pre').merge(
    mbp.drop(columns=['kind', 'conf']).add_suffix('_post'), left_on='post_id', right_on='pid_post')
k.to_parquet('out/outside_roi/hemibrain_kcmbon_all.parquet')
print('done', len(sp), len(k)); print(k.roi_post.value_counts().head(12).to_dict())
