"""Why do gamma vs alpha/beta KCs separate differently across datasets in figures 2 and 3?
Part 1: KC counts and per-class totals per hemisphere (from the delivered per-KC CSV).
Part 2: derived claws per KC and PN synapses per claw.
Part 3 (FlyWire): PN->KC synapses by neuropil (is calyx input lost to neighbouring neuropils?),
        KC output by lobe (are KCg-m labels consistent with gamma-lobe-only axons?),
        Princeton vs Buhmann synapse tables."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
D = 'dl/'; O = 'out/'; OUT = '/mnt/project-files/apl_synapses/'
KC = pd.read_csv(OUT + 'KC_weights_APL_PN_per_KC.csv', dtype={'kc_id': str})

c = pd.read_csv(D + 'classification.csv.gz', dtype={'root_id': str})
fw_side = c.set_index('root_id')['side']
m = pd.read_feather(D + 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
mc_side = pd.Series(m.somaSide.values, index=m.bodyId.astype('int64').astype(str))
b = pd.read_feather(D + 'banc_888_meta.feather', columns=['banc_888_id', 'side'])
bc_side = pd.Series(b.side.values, index=b.banc_888_id.astype(str))
side = {'flywire': fw_side, 'malecns': mc_side, 'banc': bc_side}
KC['side'] = [('right' if ds == 'hemibrain' else str(side[ds].get(k, '?'))) for ds, k in zip(KC.dataset, KC.kc_id)]
KC['side'] = KC.side.str.lower().str[0].map({'r': 'R', 'l': 'L'}).fillna('?')

print('== Part 1: per class per hemisphere ==')
cols = ['PN_to_KC_calyx', 'APL_to_KC_calyx', 'KC_to_APL_calyx']
g = KC.groupby(['dataset', 'side', 'kc_class'])
t = pd.concat([g.size().rename('n_KC')] + [g[cc].median().rename('med_' + cc) for cc in cols]
              + [(g[cc].sum() / 1000).round(1).rename('sum_k_' + cc) for cc in cols], axis=1)
print(t)
print('\nKC-m only (main gamma):')
gm = KC[KC.kc_type.isin(['KCg-m', 'KCg'])].groupby(['dataset', 'side'])
print(pd.concat([gm.size().rename('n'), gm.PN_to_KC_calyx.median().rename('med_PN'), (gm.PN_to_KC_calyx.sum() / 1000).round(1).rename('sum_k_PN')], axis=1))
print('\nAll KCs, per side: n, total PN->KC calyx (k)')
ga = KC.groupby(['dataset', 'side'])
print(pd.concat([ga.size().rename('n'), (ga.PN_to_KC_calyx.sum() / 1000).round(1).rename('sum_k_PN'),
                 (ga.APL_to_KC_calyx.sum() / 1000).round(1).rename('sum_k_APL2KC'), (ga.KC_to_APL_calyx.sum() / 1000).round(1).rename('sum_k_KC2APL')], axis=1))

print('\n== Part 2: derived claws ==')
for ds in ['flywire', 'hemibrain', 'malecns', 'banc']:
    cl = pd.read_parquet(O + f'final_{ds}_claws.parquet')
    cl['kc_id'] = cl.kc_id.astype(str)
    cl = cl.merge(KC[KC.dataset == ds][['kc_id', 'kc_class', 'kc_type']].rename(columns={'kc_type': 't2'}), on='kc_id', how='inner')
    big = cl[cl.n_PN_synapses >= 5]
    per = pd.concat([cl.groupby('kc_id').size().rename('claws'), big.groupby('kc_id').size().rename('claws5')], axis=1).fillna(0)
    per = per.join(KC[KC.dataset == ds].set_index('kc_id')[['kc_class']])
    print(ds, 'median claws per KC (all / >=5 syn):', per.groupby('kc_class')[['claws', 'claws5']].median().to_dict('index'),
          '| median PN syn per claw (>=5):', big.groupby('kc_class').n_PN_synapses.median().to_dict())

print('\n== Part 3: FlyWire connection tables ==')
kc_fw = KC[KC.dataset == 'flywire'].set_index('kc_id')
kcs = set(kc_fw.index); pns = set(c.root_id[c['class'] == 'ALPN'])
for name in ['connections_princeton_no_threshold', 'connections_no_threshold']:
    pk, ko = [], []
    for ch in pd.read_csv(D + name + '.csv.gz', dtype={'pre_root_id': str, 'post_root_id': str}, chunksize=4_000_000,
                          usecols=['pre_root_id', 'post_root_id', 'neuropil', 'syn_count']):
        pk.append(ch[ch.post_root_id.isin(kcs) & ch.pre_root_id.isin(pns)])
        ko.append(ch[ch.pre_root_id.isin(kcs)])
    pk = pd.concat(pk); ko = pd.concat(ko)
    pk['kc_class'] = pk.post_root_id.map(kc_fw.kc_class); ko['kc_type'] = ko.pre_root_id.map(kc_fw.kc_type)
    pk['in_CA'] = pk.neuropil.str.startswith('MB_CA')
    print(f'\n{name}: PN->KC synapses by KC class, share inside MB_CA and top other neuropils')
    for cl_, u in pk.groupby('kc_class'):
        tot = u.syn_count.sum(); ca = u[u.in_CA].syn_count.sum()
        oth = u[~u.in_CA].groupby('neuropil').syn_count.sum().sort_values(ascending=False).head(4)
        per = u.groupby('post_root_id').syn_count.sum()
        perca = u[u.in_CA].groupby('post_root_id').syn_count.sum()
        print(f'  {cl_}: total {tot}, in MB_CA {ca / tot:.1%}; median per KC all {per.median():.0f}, MB_CA {perca.median():.0f}; outside: {oth.to_dict()}')
    lobe = ko.assign(l=ko.neuropil.str.extract(r'^(MB_VL|MB_ML|MB_CA|MB_PED)')[0].fillna('other'))
    tab = lobe.groupby(['kc_type', 'l']).syn_count.sum().unstack(fill_value=0)
    print(f'{name}: KC output synapses by MB region and KC type (share of row)')
    print((tab.div(tab.sum(1), axis=0) * 100).round(1).assign(total=tab.sum(1)))
    if name.startswith('connections_princeton'):
        # per gamma-m KC: fraction of its MB output in the vertical lobe (gamma axons should have ~0)
        g_ = lobe[lobe.kc_type == 'KCg-m'].groupby(['pre_root_id', 'l']).syn_count.sum().unstack(fill_value=0)
        frac = g_.get('MB_VL', 0) / g_[[x for x in ['MB_VL', 'MB_ML', 'MB_PED'] if x in g_]].sum(1).clip(lower=1)
        print('KCg-m with >20% of lobe output in MB_VL:', int((frac > .2).sum()), 'of', len(frac))
