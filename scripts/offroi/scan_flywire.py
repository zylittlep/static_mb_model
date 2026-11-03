"""FlyWire 783 Princeton synapses, one streamed pass: every ALPN-presynaptic synapse and every KC -> MBON synapse,
in any neuropil (the official-ROI tables kept only MB_CA / MB_VL / MB_ML rows). Run from the working folder (with dl/, out/, scripts/)."""
import sys; sys.path.insert(0, 'scripts')
import gzip
import pyarrow.csv as pcsv, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq, pandas as pd
from meta_util import types, kc_ids
c = pd.read_csv('dl/classification.csv.gz', dtype={'root_id': str})
typ, mbon = types('flywire')
def suf(ids): return pa.array(sorted(int(str(i)[9:]) for i in ids), type=pa.int64())
PN = suf(c.root_id[c['class'] == 'ALPN']); KC = suf(kc_ids(typ)); MB = suf(mbon)
F = 'dl/fafb_v783_princeton_synapse_table.csv.gz'  # local copy (offroi/dl_princeton.sh); streaming it dropped mid-transfer
src = gzip.open(F, 'rb')
co = pcsv.ConvertOptions(column_types={'pre_root_id_720575940': pa.int64(), 'post_root_id_720575940': pa.int64(), 'neuropil': pa.string()})
r = pcsv.open_csv(src, read_options=pcsv.ReadOptions(block_size=1 << 26), convert_options=co)
pn, km, n = [], [], 0
for b in r:
    t = pa.Table.from_batches([b]); pre = b['pre_root_id_720575940']; post = b['post_root_id_720575940']
    pn.append(t.filter(pc.is_in(pre, PN)))
    km.append(t.filter(pc.and_(pc.is_in(pre, KC), pc.is_in(post, MB))))
    n += b.num_rows
    if len(pn) % 40 == 0: print(n, flush=True)
for nm, l in [('fafbP_pn_all_raw', pn), ('flywire_kcmbon_all', km)]:
    t = pa.concat_tables(l); pq.write_table(t, f'out/outside_roi/{nm}.parquet'); print(nm, t.num_rows)
print('scanned', n)
