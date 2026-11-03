"""FlyWire: ALPN -> X synapse counts per pair in every MB neuropil, from Codex connections_princeton_no_threshold
(same Princeton synapses as the calyx table, aggregated per pair and neuropil)."""
import pandas as pd
c = pd.read_csv('dl/classification.csv.gz', dtype={'root_id': str})
pns = set(c.root_id[c['class'] == 'ALPN'])
out = []
for ch in pd.read_csv('dl/connections_princeton_no_threshold.csv.gz', dtype={'pre_root_id': str, 'post_root_id': str},
                      usecols=['pre_root_id', 'post_root_id', 'neuropil', 'syn_count'], chunksize=4_000_000):
    out.append(ch[ch.pre_root_id.isin(pns) & ch.neuropil.fillna('').str.startswith('MB_')])
t = pd.concat(out).rename(columns={'pre_root_id': 'pre_id', 'post_root_id': 'post_id'})
t.to_parquet('out/flywire_pnmb_pairs.parquet'); print(len(t)); print(t.groupby('neuropil').syn_count.sum())
