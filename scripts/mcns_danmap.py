import sys; sys.path.insert(0,'scripts')
from roilookup import lookup
import pandas as pd, numpy as np
m=pd.read_feather('dl/body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId')
d=pd.read_parquet('out/malecns_dan_raw.parquet')
d=d.drop_duplicates(['x_pre','y_pre','z_pre'])
d['comp']=lookup('gs://flyem-male-cns/rois/malecns-subcompartments-v3/',d[['x_pre','y_pre','z_pre']].values*8,0,256)
d['comp']=np.where(d.comp=='', d.primary_post, d.comp)
d['comp']=d.comp.str.replace(r'\((L|R)\)','',regex=True)
d['type']=m.loc[d.body_pre,'type'].values
d.to_parquet('out/malecns_dan_comp.parquet')
ct=pd.crosstab(d.type,d.comp)
frac=ct.div(ct.sum(1),axis=0)
res=pd.DataFrame({'top':frac.idxmax(1),'purity':frac.max(1),'n':ct.sum(1)})
pd.set_option('display.max_rows',200)
print(res.sort_values('top'))
res.to_csv('out/malecns_dan_type_compartment.csv')
