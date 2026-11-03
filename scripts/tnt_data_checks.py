"""Double-counting checks on Yang's TNT / TNT3c KC responses (2026-10-15).

1. Identical units: rows identical to another row of the same recording, or shared between recordings.
2. Near-duplicate units: each unit's nearest neighbour (Euclidean, 16 odors) in its own recording vs in every other recording of
   the same condition, subsampled to the same number of candidates. A unit is flagged when its own-recording neighbour is
   closer than the 1st percentile of its distances to other recordings. Responsive units = max response > 1.
3. Two hemispheres of one fly (same date and fly number, FL and FR): similarity structure (cosine over the 120 odor pairs) and
   per-odor fraction responding, same-fly vs different-fly pairs of recordings; and the change in similarity (TNT - TNT3c)
   with each fly weighted once instead of each recording.
Outputs: odor_weighted/TNT_data_checks.csv (one row per recording), odor_weighted/TNT_data_checks.json."""
import sys, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
sys.path.insert(0, '/mnt/project-files/apl_synapses/scripts')
from orn_load import load

OUT = '/mnt/project-files/apl_synapses/odor_weighted/'
UP = '/mnt/project-files/uploads/hearth/'
D = {'TNT': load(UP + '2e7fb552-f4ec-49d7-b06f-14ee778f6433'), 'TNT3c': load(UP + '76debf61-257e-4375-b4d8-a02b48c4f62a')}
IU = np.triu_indices(16, 1)
rs = np.random.default_rng(0)
fly = lambda r: r.split('F')[0]


def cos(X):
    n = np.linalg.norm(X, axis=0)
    return (X.T @ X) / np.outer(n, n)


rows, seen = [], {}
for c, d in D.items():
    for a, v in d.items():
        X = v.values; n = len(X)
        _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
        for i, row in enumerate(X): seen.setdefault(row.tobytes(), set()).add((c, a))
        dw, iw = cKDTree(X).query(X, k=2); dw, iw = dw[:, 1], iw[:, 1]
        dac = np.array([cKDTree(d[b].values[rs.choice(len(d[b]), min(len(d[b]), n - 1), replace=False)]).query(X, k=1)[0]
                        for b in d if b != a])
        q01 = np.quantile(dac, 0.01); flag = dw < q01; resp = X.max(1) > 1
        rows.append(dict(condition=c, recording=a, fly=fly(a), units=n, responsive_units=int(resp.sum()),
                         identical_units=int((cnt[inv.ravel()] > 1).sum()),
                         median_nn_distance_same_recording=np.median(dw), median_nn_distance_other_recordings=np.median(dac),
                         frac_units_flagged=flag.mean(), frac_responsive_units_flagged=flag[resp].mean(),
                         flagged_mutual_pairs=int(np.sum(flag & (iw[iw] == np.arange(n))) // 2)))
R = pd.DataFrame(rows)
R.round(4).to_csv(OUT + 'TNT_data_checks.csv', index=False)

S = {c: {r: cos(v.values)[IU] for r, v in d.items()} for c, d in D.items()}
F = {c: {r: (v.values > 1).mean(0) for r, v in d.items()} for c, d in D.items()}
hemi = {}
for c in D:
    names = list(S[c]); same, diff, sameF, diffF = [], [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]; s_ = fly(a) == fly(b)
            (same if s_ else diff).append(np.corrcoef(S[c][a], S[c][b])[0, 1])
            (sameF if s_ else diffF).append(np.corrcoef(F[c][a], F[c][b])[0, 1])
    hemi[c] = {'flies with two recordings': sorted({fly(r) for r in names if sum(fly(x) == fly(r) for x in names) > 1}),
               'similarity structure r, same-fly pairs': [round(x, 3) for x in same],
               'similarity structure r, different-fly pairs: median, min, max': [round(float(np.median(diff)), 3), round(min(diff), 3), round(max(diff), 3)],
               'same-fly pairs: fraction of different-fly pairs below them': [round(float(np.mean(np.array(diff) < x)), 2) for x in same],
               'per-odor fraction responding r, same-fly pairs': [round(x, 3) for x in sameF],
               'per-odor fraction responding r, different-fly pairs: median, min, max': [round(float(np.median(diffF)), 3), round(min(diffF), 3), round(max(diffF), 3)]}


def mean_S(c, by_fly):
    if not by_fly: return np.mean(list(S[c].values()), 0)
    g = {}
    for r, s in S[c].items(): g.setdefault(fly(r), []).append(s)
    return np.mean([np.mean(x, 0) for x in g.values()], 0)


def mean_F(c, by_fly):
    if not by_fly: return np.mean(list(F[c].values()), 0)
    g = {}
    for r, s in F[c].items(): g.setdefault(fly(r), []).append(s)
    return np.mean([np.mean(x, 0) for x in g.values()], 0)


dS_rec, dS_fly = mean_S('TNT', False) - mean_S('TNT3c', False), mean_S('TNT', True) - mean_S('TNT3c', True)
P = pd.read_csv(OUT + 'TNT_vs_model_odor_pairs.csv').query('threshold == 1.0 and responses == "ORN"')
assert np.allclose(P.dS_data.values, dS_rec, atol=1e-4)
sup_rec, sup_fly = mean_F('TNT', False) / mean_F('TNT3c', False), mean_F('TNT', True) / mean_F('TNT3c', True)
summary = {
    'identical units within a recording (all recordings)': int(R.identical_units.sum()),
    'units shared between recordings': int(sum(len(x) > 1 for x in seen.values())),
    'near-duplicate units (own-recording neighbour closer than 1st percentile of other recordings), mean over recordings': {
        c: {'all units': round(float(g.frac_units_flagged.mean()), 4), 'responsive units': round(float(g.frac_responsive_units_flagged.mean()), 4)}
        for c, g in R.groupby('condition')},
    'hemispheres of one fly': hemi,
    'change in similarity, recordings weighted equally vs flies weighted equally': {
        'mean': [round(float(dS_rec.mean()), 4), round(float(dS_fly.mean()), 4)],
        'pattern r over odor pairs': round(float(np.corrcoef(dS_rec, dS_fly)[0, 1]), 4),
        'r with model U (ORN)': [round(float(np.corrcoef(dS_rec, P.dS_model_U)[0, 1]), 3), round(float(np.corrcoef(dS_fly, P.dS_model_U)[0, 1]), 3)],
        'r with model M (ORN)': [round(float(np.corrcoef(dS_rec, P.dS_model_M)[0, 1]), 3), round(float(np.corrcoef(dS_fly, P.dS_model_M)[0, 1]), 3)],
        'per-odor suppression, spearman between weightings': round(float(spearmanr(sup_rec, sup_fly)[0]), 3)}}
json.dump(summary, open(OUT + 'TNT_data_checks.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
