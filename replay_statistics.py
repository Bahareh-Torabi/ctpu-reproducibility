#!/usr/bin/env python3
"""Replay exploratory crossed-bootstrap contrasts from compact document metrics."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
METRICS=['ranking_loss','soft_cardinality_mae','nll','brier']
def stable_int_seed(*parts, bits=63):
    s = '||'.join(map(str, parts)).encode('utf-8')
    h = hashlib.blake2b(s, digest_size=16).digest()
    return int.from_bytes(h, 'little') & ((1 << bits)-1)

def crossed_paired_bootstrap_arrays(pairs, reps=5000, seed=20261205, replicate_chunk=250):
    """Vectorized crossed seed x document percentile bootstrap, matching ctpu/statistics.py."""
    if not pairs:
        raise ValueError('No paired seed runs')
    n_docs = len(next(iter(pairs[0][0].values())))
    if n_docs == 0:
        raise ValueError('No evaluation documents')
    rng = np.random.default_rng(seed)
    S = len(pairs)

    # Precompute the random draws in the exact per-replicate order used by the
    # training implementation: seeds first, followed by a shared document draw.
    sampled_seeds = np.empty((reps, S), dtype=np.int16)
    doc_indices = np.empty((reps, n_docs), dtype=np.int32)
    for r in range(reps):
        sampled_seeds[r] = rng.integers(0, S, size=S)
        doc_indices[r] = rng.integers(0, n_docs, size=n_docs)

    out = {}
    for m in METRICS:
        diff_matrix = np.stack([
            np.asarray(a[m], dtype=np.float64) - np.asarray(b[m], dtype=np.float64)
            for a, b in pairs
        ], axis=0)
        if diff_matrix.shape != (S, n_docs):
            raise ValueError(f'Unexpected paired metric shape for {m}: {diff_matrix.shape}')

        seed_means = np.nanmean(diff_matrix, axis=1)
        observed = float(np.nanmean(seed_means))
        draws = np.full(reps, np.nan, dtype=np.float64)

        # Chunking bounds temporary memory for the EURLEX57K 6,000 x 4,108 runs.
        for start in range(0, reps, replicate_chunk):
            stop = min(reps, start + replicate_chunk)
            ss = sampled_seeds[start:stop]
            di = doc_indices[start:stop]
            accum = np.zeros(stop-start, dtype=np.float64)
            n_valid = np.zeros(stop-start, dtype=np.int16)
            for j in range(S):
                selected = diff_matrix[ss[:, j, None], di]
                finite = np.isfinite(selected)
                counts = finite.sum(axis=1)
                sums = np.where(finite, selected, 0.0).sum(axis=1)
                valid = counts > 0
                accum[valid] += sums[valid] / counts[valid]
                n_valid[valid] += 1
                del selected, finite, counts, sums
            ok = n_valid > 0
            block = np.full(stop-start, np.nan, dtype=np.float64)
            block[ok] = accum[ok] / n_valid[ok]
            draws[start:stop] = block

        finite_draws = draws[np.isfinite(draws)]
        if finite_draws.size == 0:
            raise RuntimeError(f'No finite bootstrap draws for {m}')
        out[m] = {
            'observed_mean_difference': observed,
            'ci_2_5': float(np.percentile(finite_draws, 2.5)),
            'ci_97_5': float(np.percentile(finite_draws, 97.5)),
            'bootstrap_fraction_difference_lt_0': float(np.mean(finite_draws < 0)),
            'bootstrap_fraction_is_not_a_p_value': True,
        }
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'compact_document_metrics')
    parser.add_argument('--output',type=Path,default=Path('replayed_contrasts.csv'))
    args=parser.parse_args()
    records=json.loads((args.root/'MANIFEST.json').read_text())
    cache={}
    for row in records:
        path=args.root/row['relative_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError(f'Compact metric checksum mismatch: {path}')
        with np.load(path) as values:
            cache.setdefault(row['dataset'],{}).setdefault(row['method'],{})[row['seed']]={m:values[m] for m in METRICS}
    rows=[]
    for ds,methods in cache.items():
        for comp in ['pairwise','pairwise_cardinality_penalty','asl','wasl','normmatch','param_pcgrad','score_auxproj']:
            common=sorted(set(methods['ctpu']) & set(methods.get(comp,{})))
            if len(common)<3:continue
            seed=stable_int_seed(20261205,ds) if comp=='pairwise' else stable_int_seed(20261205,ds,comp)
            pairs=[(methods['ctpu'][s],methods[comp][s]) for s in common]
            result=crossed_paired_bootstrap_arrays(pairs,reps=5000,seed=seed)
            for metric,res in result.items():
                rows.append({'dataset':ds,'comparator':comp,'metric':metric,'n_paired_seeds':len(common),
                    'difference_ctpu_minus_comparator':res['observed_mean_difference'],
                    'ci_95_lower':res['ci_2_5'],'ci_95_upper':res['ci_97_5']})
            print(ds,comp,'complete',flush=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    print(args.output)
