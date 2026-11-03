"""summary_counts_by_compartment_and_partner_class.csv from the four delivered APL tables: synapse counts per APL and
direction, once by MB compartment and once by partner class (the same two groupbys as the first build).
Writes to the folder of the version being built (variant.py)."""
import sys; sys.path.insert(0, 'scripts')
import pandas as pd
from variant import OUT

COLS = ['dataset', 'apl_id', 'direction', 'mb_compartment', 'n_synapses', 'partner_class']
parts = []
for ds in ['flywire', 'hemibrain', 'malecns', 'banc']:
    a = pd.read_csv(OUT + f'{ds}_APL_MB_synapses.csv.gz', usecols=['dataset', 'apl_id', 'direction', 'mb_compartment', 'partner_class'],
                    dtype={'apl_id': str})
    parts.append(a.groupby(['dataset', 'apl_id', 'direction', 'mb_compartment']).size().rename('n_synapses').reset_index())
    parts.append(a.groupby(['dataset', 'apl_id', 'direction', 'partner_class']).size().rename('n_synapses').reset_index())
S = pd.concat(parts, ignore_index=True)[COLS]
S.to_csv(OUT + 'summary_counts_by_compartment_and_partner_class.csv', index=False)
print('rows', len(S)); print(S.groupby('dataset').n_synapses.sum())
