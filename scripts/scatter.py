import sys; sys.path.insert(0,'scripts')
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from prep import malecns, hemibrain, flywire, banc
from prep import XYZ
from variant import OUT, FINAL, TAG, fw_apl_raw
rows=[]
for ds,fn in [('flywire',flywire),('hemibrain',hemibrain),('malecns',malecns),('banc',banc)]:
    d=fn(); meta=d['meta'].drop_duplicates('id').set_index('id')
    p=d['pnca'].copy()
    if ds=='flywire':  # same Codex filter as the APL table
        proof=set(meta.index); p=p[(p.pre_id!=p.post_id)&p.post_id.isin(proof)]
    p=p[p.post_id.map(meta['type']).astype(str).str.startswith('KC')]
    kc=p.groupby('pre_id').size().rename('PN_to_KC')
    a=pd.read_parquet(FINAL+f'final_{ds}_apl_mb.parquet'); a=a[a.mb_region=='CA']
    if ds=='flywire':
        r=fw_apl_raw(); P=720575940000000000
        r['pre_id']=(r.pre_root_id_720575940.astype('int64')+P).astype(str); r['post_id']=(r.post_root_id_720575940.astype('int64')+P).astype(str)
        r=r[r.neuropil.fillna('').str.startswith('MB_CA')&(r.pre_id!=r.post_id)&r.pre_id.isin(proof)&r.post_id.isin(proof)]
        A={'720575940613583001','720575940624547622'}
        r['apl_is_pre']=r.pre_id.isin(A); r['partner_id']=r.post_id.where(r.apl_is_pre,r.pre_id); a=r
    pns=set(d['pnca'].pre_id.astype(str))
    a=a[a.partner_id.astype(str).isin(pns)]
    ain=a[~a.apl_is_pre].groupby('partner_id').size().rename('PN_to_APL')
    aout=a[a.apl_is_pre].groupby('partner_id').size().rename('APL_to_PN')
    t=pd.concat([kc,ain,aout],axis=1).fillna(0).astype(int)
    t.index.name='pn_id'; t=t.reset_index(); t['pn_type']=t.pn_id.map(meta['type']); t.insert(0,'dataset',ds)
    rows.append(t)
T=pd.concat(rows); T.to_csv(OUT+'PN_calyx_weights_APL_vs_KC.csv',index=False)
INK='#0b0b0b'; MUTED='#52514e'; C='#2a78d6'
fig,ax=plt.subplots(2,4,figsize=(17,8.2),constrained_layout=True)
for j,ds in enumerate(['flywire','hemibrain','malecns','banc']):
    t=T[T.dataset==ds]
    for i,(col,lab) in enumerate([('PN_to_APL','PN → APL synapses (APL input)'),('APL_to_PN','APL → PN synapses (APL output)')]):
        a_=ax[i,j]; a_.scatter(t.PN_to_KC,t[col],s=22,color=C,alpha=.7,edgecolor='white',linewidth=.6)
        r,pv=spearmanr(t.PN_to_KC,t[col])
        a_.set_title(f'{ds}  (n={len(t)} PNs, Spearman ρ={r:.2f})',fontsize=10,color=INK,loc='left')
        for _,rr in t.nlargest(3,col).iterrows():
            a_.annotate(str(rr.pn_type).replace('_',' '),(rr.PN_to_KC,rr[col]),fontsize=7.5,color=MUTED,xytext=(4,2),textcoords='offset points')
        a_.set_xlabel('PN → KC synapses in calyx',color=MUTED,fontsize=9)
        if j==0: a_.set_ylabel(lab,color=MUTED,fontsize=9)
        for s in ['top','right']: a_.spines[s].set_visible(False)
        for s in ['left','bottom']: a_.spines[s].set_color('#c9c8c2')
        a_.tick_params(colors=MUTED,labelsize=8); a_.grid(color='#ecebe7',lw=.6); a_.set_axisbelow(True)
fig.suptitle('Per PN, calyx only: APL↔PN synapses vs PN→KC synapses'+TAG,fontsize=13,color=INK,x=0.01,ha='left')
fig.savefig(OUT+'scatter_APL_PN_vs_PN_KC_calyx.png',dpi=160)
for ds in T.dataset.unique():
    t=T[T.dataset==ds]; print(ds,len(t),[round(spearmanr(t.PN_to_KC,t[c])[0],2) for c in ['PN_to_APL','APL_to_PN']], t[['PN_to_KC','PN_to_APL','APL_to_PN']].sum().to_dict())
