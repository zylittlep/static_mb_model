"""Every synapse site of every KC near the calyx (T-bars where the KC is presynaptic, PSDs where it is postsynaptic),
for the geodesic claw segmentation. Run from the working folder that holds dl/ (release files), out/ and scripts/.

Usage: python3 scripts/claws_geodesic/extract_kc_sites.py hemibrain|malecns
Output: out/claws/<ds>_kc_sites.parquet, one row per KC site: kc_id, kind (Tbar/PSD), x/y/z (nm), roi label,
partner_id (PSD: the presynaptic body; T-bar: -1), n_partners (T-bar: number of PSDs it drives; PSD: 1).
Sites are kept inside a box around the calyx (corrected-calyx PN->KC PSDs +- 15 um); the calyx itself is decided later.
Hemibrain: points + partners release files (a point counts only if it is in a release connection).
Male CNS: syn-partners v1.0, confidence >= 0.5; T-bar ROI = the ROI of its first PSD (the file labels PSDs only)."""
import sys, os; sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.ipc as ipc
from meta_util import types, kc_ids
ds = sys.argv[1]; os.makedirs('out/claws', exist_ok=True)
typ, _ = types(ds); KCS = sorted(int(i) for i in kc_ids(typ))
anc = pd.read_parquet(f'out/outside_roi/{ds}_pnca_raw.parquet')
anc = anc[anc.roi_assigned.astype(str).str.startswith('CA')]
lo = anc[['x_post', 'y_post', 'z_post']].min().values - 15000 / 8; hi = anc[['x_post', 'y_post', 'z_post']].max().values + 15000 / 8
print(ds, 'KCs', len(KCS), 'box (voxels)', lo, hi, flush=True)

def inbox(b, sx):   # sx: suffix '' (hemibrain points) or '_pre' / '_post'
    m = None
    for k, c in enumerate('xyz'):
        v = b[c + sx]; mm = pc.and_(pc.greater_equal(v, float(lo[k])), pc.less_equal(v, float(hi[k])))
        m = mm if m is None else pc.and_(m, mm)
    return m

if ds == 'malecns':
    KC = pa.array(KCS, type=pa.int64())
    r = ipc.open_file('dl/syn-partners-male-cns-v1.0-minconf-0.5.feather'); post, pre = [], []
    for i in range(r.num_record_batches):
        b = r.get_batch(i)
        f = b.filter(pc.and_(pc.is_in(b['body_post'], KC), inbox(b, '_post')))
        if f.num_rows:
            t = f.select(['x_post', 'y_post', 'z_post', 'body_post', 'body_pre', 'conf_post']).to_pandas()
            t['roi'] = f['primary_post'].cast(pa.string()).to_numpy(zero_copy_only=False); post.append(t)
        f = b.filter(pc.and_(pc.is_in(b['body_pre'], KC), inbox(b, '_pre')))
        if f.num_rows:
            t = f.select(['x_pre', 'y_pre', 'z_pre', 'body_pre', 'conf_pre']).to_pandas()
            t['roi'] = f['primary_post'].cast(pa.string()).to_numpy(zero_copy_only=False); pre.append(t)
        if i % 500 == 0: print(i, r.num_record_batches, flush=True)
    post = pd.concat(post, ignore_index=True); pre = pd.concat(pre, ignore_index=True)
    P = pd.DataFrame({'kc_id': post.body_post.values, 'kind': 'PSD', 'x': post.x_post * 8., 'y': post.y_post * 8., 'z': post.z_post * 8.,
                      'roi': post.roi.values, 'conf': post.conf_post.values, 'partner_id': post.body_pre.values, 'n_partners': 1})
    T = pre.groupby(['body_pre', 'x_pre', 'y_pre', 'z_pre'], sort=False).agg(roi=('roi', 'first'), conf=('conf_pre', 'first'),
                                                                               n_partners=('roi', 'size')).reset_index()
    T = pd.DataFrame({'kc_id': T.body_pre.values, 'kind': 'Tbar', 'x': T.x_pre * 8., 'y': T.y_pre * 8., 'z': T.z_pre * 8.,
                      'roi': T.roi.values, 'conf': T.conf.values, 'partner_id': -1, 'n_partners': T.n_partners.values})
else:
    PTS = 'dl/hemibrain-v1.2-synapse-points-with-body-roi-type-status.feather'
    KC = pa.array(KCS, type=pa.uint64())
    def pid(t):
        x = pc.cast(t['x'], pa.uint64()); y = pc.cast(t['y'], pa.uint64()); z = pc.cast(t['z'], pa.uint64())
        return pc.bit_wise_or(pc.bit_wise_or(pc.shift_left(z, 42), pc.shift_left(y, 21)), x)
    r = ipc.open_file(PTS); pts = []
    for i in range(r.num_record_batches):
        b = r.get_batch(i); f = b.filter(pc.and_(pc.is_in(b['body'], KC), inbox(b, '')))
        if f.num_rows:
            d = f.select(['x', 'y', 'z', 'conf', 'body']).to_pandas()
            d['kind'] = f['kind'].cast(pa.string()).to_numpy(zero_copy_only=False); d['roi'] = f['roi'].cast(pa.string()).to_numpy(zero_copy_only=False)
            d['pid'] = pid(f).to_numpy(); pts.append(d)
        if i % 200 == 0: print('points', i, r.num_record_batches, flush=True)
    pts = pd.concat(pts, ignore_index=True).drop_duplicates('pid')
    tb = pts[pts.kind == 'PreSyn']; ps = pts[pts.kind == 'PostSyn']
    print('KC T-bars', len(tb), 'KC PSDs', len(ps), flush=True)
    TB = pa.array(tb.pid.values, type=pa.uint64()); PS = pa.array(ps.pid.values, type=pa.uint64())
    r = ipc.open_file('dl/hemibrain-v1.2-synapse-partners.feather'); a, bb = [], []
    for i in range(r.num_record_batches):
        b = r.get_batch(i).select(['pre_id', 'post_id'])
        a.append(b.filter(pc.is_in(b['pre_id'], TB))); bb.append(b.filter(pc.is_in(b['post_id'], PS)))
    a = pa.Table.from_batches(a).to_pandas(); bb = pa.Table.from_batches(bb).to_pandas().drop_duplicates('post_id')
    need = pa.array(np.unique(bb.pre_id.values), type=pa.uint64()); r = ipc.open_file(PTS); pb = []
    for i in range(r.num_record_batches):
        b = r.get_batch(i); m = pid(b); f = b.filter(pc.is_in(m, need))
        if f.num_rows: pb.append(pd.DataFrame({'pid': pid(f).to_numpy(), 'body': f['body'].to_numpy()}))
    pb = pd.concat(pb).drop_duplicates('pid').set_index('pid')['body']
    ntb = a.groupby('pre_id').size()
    tb = tb[tb.pid.isin(ntb.index)]; ps = ps.merge(bb, left_on='pid', right_on='post_id')
    T = pd.DataFrame({'kc_id': tb.body.values.astype('int64'), 'kind': 'Tbar', 'x': tb.x * 8., 'y': tb.y * 8., 'z': tb.z * 8., 'roi': tb.roi.values,
                      'conf': tb.conf.values, 'partner_id': -1, 'n_partners': ntb.reindex(tb.pid.values).values})
    P = pd.DataFrame({'kc_id': ps.body.values.astype('int64'), 'kind': 'PSD', 'x': ps.x * 8., 'y': ps.y * 8., 'z': ps.z * 8., 'roi': ps.roi.values,
                      'conf': ps.conf.values, 'partner_id': pb.reindex(ps.pre_id.values).fillna(-1).astype('int64').values, 'n_partners': 1})
out = pd.concat([T, P], ignore_index=True); out.to_parquet(f'out/claws/{ds}_kc_sites.parquet')
print('done', ds, len(out), out.kind.value_counts().to_dict(), 'KCs', out.kc_id.nunique())
print(out.roi.value_counts().head(10).to_dict())
