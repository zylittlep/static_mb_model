import tensorstore as ts, numpy as np, json, urllib.request
def names(path):
    d=json.load(urllib.request.urlopen('https://storage.googleapis.com/'+path[5:]+'segment_properties/info'))
    return {int(i):n for i,n in zip(d['inline']['ids'],d['inline']['properties'][0]['values'])}
def lookup(path, pts_nm, scale_index, res_nm):
    v=ts.open({'driver':'neuroglancer_precomputed','kvstore':path,'scale_index':scale_index}).result()
    vox=np.floor(np.asarray(pts_nm,float)/res_nm).astype(np.int64)
    lo=vox.min(0); hi=vox.max(0)+1
    shp=np.array([v.domain[i].exclusive_max for i in range(3)])
    lo=np.clip(lo,0,shp); hi=np.clip(hi,0,shp)
    a=v[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2],0].read().result()
    out=np.zeros(len(vox),dtype=np.int64)
    ok=np.all((vox>=lo)&(vox<hi),axis=1)
    rel=vox[ok]-lo
    out[ok]=a[rel[:,0],rel[:,1],rel[:,2]]
    nm=names(path)
    return np.array([nm.get(int(i),'') for i in out],dtype=object)
