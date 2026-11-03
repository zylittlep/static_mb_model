"""Compare the scratch rebuild of the default version (verify_official.sh) with the delivered files and intermediate tables."""
import os, sys, gzip, json, hashlib
import numpy as np, pandas as pd
from PIL import Image

R = 'verify_official/'
DEL = '/mnt/project-files/apl_synapses/'


def md5(b):
    return hashlib.md5(b).hexdigest()


def content(path):
    with open(path, 'rb') as f:
        b = f.read()
    return gzip.decompress(b) if path.endswith('.gz') else b


def same_png(a, b):
    x = np.asarray(Image.open(a).convert('RGBA')).astype(int); y = np.asarray(Image.open(b).convert('RGBA')).astype(int)
    if x.shape != y.shape:
        return f'size {x.shape} vs {y.shape}'
    d = np.abs(x - y).max(-1)
    return f'pixels differing {100 * (d > 0).mean():.3f}%, max diff {d.max()}'


rows = []
# intermediate tables
for ds in ['flywire', 'hemibrain', 'malecns', 'banc']:
    for k in ['apl_mb', 'claws', 'boutons']:
        a = pd.read_parquet(f'out/final_{ds}_{k}.parquet'); b = pd.read_parquet(R + f'final/final_{ds}_{k}.parquet')
        rows.append((f'out/final_{ds}_{k}.parquet', 'same' if a.equals(b) else f'DIFFERENT {a.shape} vs {b.shape}'))
    a = json.load(open(f'out/final_{ds}_summary.json')); b = json.load(open(R + f'final/final_{ds}_summary.json'))
    rows.append((f'out/final_{ds}_summary.json', 'same' if a == b else 'DIFFERENT'))
for f in ['lobe_tables_summary.json', 'malecns_compartment_validation.json']:
    a = json.load(open(DEL + f)); b = json.load(open(R + 'final/' + f))
    rows.append((f, 'same' if a == b else 'DIFFERENT'))
# delivered files
for sub in ['', 'odor_weighted/']:
    for f in sorted(os.listdir(DEL + sub)):
        p = DEL + sub + f
        if os.path.isdir(p) or f == 'README.md':
            continue
        q = R + 'out_dir/' + sub + f
        if not os.path.exists(q):
            rows.append((sub + f, 'not rebuilt by the chain')); continue
        if md5(content(p)) == md5(content(q)):
            rows.append((sub + f, 'same')); continue
        if f.endswith('.png'):
            rows.append((sub + f, 'DIFFERENT png: ' + same_png(p, q)))
        elif f.endswith('.json'):
            rows.append((sub + f, 'same content' if json.load(open(p)) == json.load(open(q)) else 'DIFFERENT json'))
        else:
            a = pd.read_csv(p, low_memory=False); b = pd.read_csv(q, low_memory=False)
            rows.append((sub + f, 'same values' if a.equals(b) else f'DIFFERENT csv {a.shape} vs {b.shape}'))
for sub in ['', 'odor_weighted/']:
    for f in sorted(os.listdir(R + 'out_dir/' + sub)):
        if not os.path.isdir(R + 'out_dir/' + sub + f) and not os.path.exists(DEL + sub + f):
            rows.append((sub + f, 'EXTRA file in rebuild'))
T = pd.DataFrame(rows, columns=['file', 'result'])
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120); pd.set_option('display.max_rows', 200)
print(T.to_string(index=False))
