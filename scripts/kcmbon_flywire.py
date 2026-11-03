"""FlyWire 783 Princeton synapses: KC -> MBON in MB_VL / MB_ML (streamed from the Codex bucket)."""
import sys; sys.path.insert(0, 'scripts')
import pyarrow.csv as pcsv, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq, urllib.request, gzip
from meta_util import types, kc_ids
typ, mbon = types('flywire')
def suf(ids): return pa.array(sorted(int(str(i)[9:]) for i in ids), type=pa.int64())
KC = suf(kc_ids(typ)); MB = suf(mbon)
url = 'https://storage.googleapis.com/flywire-data/codex/data/fafb/783/fafb_v783_princeton_synapse_table.csv.gz'
src = gzip.GzipFile(fileobj=urllib.request.urlopen(url))
co = pcsv.ConvertOptions(column_types={'pre_root_id_720575940': pa.int64(), 'post_root_id_720575940': pa.int64(), 'neuropil': pa.string()})
r = pcsv.open_csv(src, read_options=pcsv.ReadOptions(block_size=1 << 26), convert_options=co)
out = []; n = 0
for b in r:
    npil = pc.fill_null(b['neuropil'], '')
    lobe = pc.or_(pc.starts_with(npil, 'MB_VL'), pc.starts_with(npil, 'MB_ML'))
    m = pc.and_(pc.and_(pc.is_in(b['pre_root_id_720575940'], KC), pc.is_in(b['post_root_id_720575940'], MB)), lobe)
    out.append(pa.Table.from_batches([b]).filter(m)); n += b.num_rows
    if len(out) % 40 == 0: print(n, flush=True)
t = pa.concat_tables(out); pq.write_table(t, 'out/flywire_kcmbon_lobes.parquet'); print('rows', t.num_rows, 'scanned', n)
