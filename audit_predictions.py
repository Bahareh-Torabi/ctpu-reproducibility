#!/usr/bin/env python3
"""Portable audit of frozen CTPU artifacts. Requires NumPy, pandas, matplotlib.
No training, tokenizer, model loading, threshold tuning or test inference.
"""
# Locate the frozen final-test export.

from pathlib import Path
import os, json, csv, zipfile, hashlib, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='Reanalyse frozen CTPU test predictions without retraining.')
parser.add_argument('project_root', type=Path, help='Root containing campaigns/ and PAPER_ARTIFACT_MANIFEST.json')
parser.add_argument('--output', type=Path, default=Path('analysis_results'))
args = parser.parse_args()
PROJECT_ROOT = args.project_root.resolve()
import matplotlib
matplotlib.use('Agg')
def display(value):
    print(value.to_string(index=False) if hasattr(value, 'to_string') else value)

GLOBAL_PAPER = PROJECT_ROOT / 'global_paper'
CAMPAIGNS = PROJECT_ROOT / 'campaigns'
OUTDIR = args.output.resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

# Expected local folder layout:
# frozen_export/
#   PAPER_ARTIFACT_MANIFEST.json                  (required artifact manifest)
#   global_paper/
#     PAPER_PRIMARY_CONTRASTS.csv                 (audit/reference; not required for calculations)
#     PAPER_MAIN_FINAL_TEST_TABLE.csv             (audit/reference; not required for calculations)
#     Q1_POSTHOC_SUPPLEMENTARY_PACKAGE.zip        (optional; may contain runtime CSV)
#   campaigns/<dataset>/final_test_runs/<dataset>/<method>/seed_<seed>/
#     test_logits.npy, test_labels.npy, test_sample_ids.json,
#     test_metrics.json, FINAL_TEST_COMPLETE.json, config.json

required = [GLOBAL_PAPER, CAMPAIGNS]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        'Project export folders were not found. Check PROJECT_ROOT and the local folder layout. Missing: ' + ', '.join(missing)
    )

print('PROJECT_ROOT :', PROJECT_ROOT)
print('GLOBAL_PAPER :', GLOBAL_PAPER)
print('CAMPAIGNS    :', CAMPAIGNS)
print('OUTDIR       :', OUTDIR)
for p in [GLOBAL_PAPER/'PAPER_PRIMARY_CONTRASTS.csv',
          GLOBAL_PAPER/'PAPER_MAIN_FINAL_TEST_TABLE.csv',
          GLOBAL_PAPER/'Q1_POSTHOC_SUPPLEMENTARY_PACKAGE.zip',
          PROJECT_ROOT/'PAPER_ARTIFACT_MANIFEST.json']:
    print(('FOUND ' if p.exists() else 'MISSING'), p)

# Source archive checksum recorded in the experiment manifest.
EXPECTED_SOURCE_BUNDLE_SHA256 = '634d510a6855e345140d5bfa0ed7eb8f7621374713188f7ef684b67dcb696ede'
manifest_path = PROJECT_ROOT / 'PAPER_ARTIFACT_MANIFEST.json'
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        seen = manifest.get('bundle_sha256')
        print('Drive manifest bundle SHA256:', seen)
        if seen and seen != EXPECTED_SOURCE_BUNDLE_SHA256:
            raise RuntimeError('Unexpected training source bundle hash')
    except Exception as e:
        print('WARNING: Could not parse the artifact manifest:', repr(e))
else:
    print('WARNING: global artifact manifest is absent; run-level checks will still be performed.')
# Analysis settings and run discovery.
DATASETS = ['reuters21578', 'aapd', 'eurlex57k']
COMPARATORS = [
    'pairwise',                         # reproduce the predeclared primary comparison
    'pairwise_cardinality_penalty',     # direct cardinality regularization
    'asl',                              # contextual imbalance-aware baseline
    'wasl',                             # contextual imbalance-aware baseline
    'normmatch',                        # direction-preserving norm control
    'param_pcgrad',                     # only where the common 3-seed subset exists
    'score_auxproj',                    # one-sided score-space negative control; Reuters/AAPD only
]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_BASE_SEED = 20261205
MIN_MATCHED_SEEDS = 3
METRICS = ['ranking_loss', 'soft_cardinality_mae', 'nll', 'brier']

METHOD_LABEL = {
    'pairwise':'Pairwise',
    'pairwise_cardinality_penalty':'Direct cardinality penalty',
    'asl':'ASL', 'wasl':'WASL', 'normmatch':'NormMatch',
    'param_pcgrad':'Param-PCGrad', 'score_auxproj':'Score-AuxProj'
}

# Directory layout produced by export_campaign_artifacts().
# campaigns/<dataset>/final_test_runs/<dataset>/<method>/seed_<seed>/test_logits.npy

def _seed_from_path(path):
    for part in path.parts:
        if part.startswith('seed_'):
            try: return int(part.split('_',1)[1])
            except Exception: pass
    return None

def find_run_files(dataset, method):
    base = CAMPAIGNS / dataset / 'final_test_runs' / dataset / method
    out = {}
    if not base.exists():
        return out
    for lp in base.rglob('test_logits.npy'):
        seed = _seed_from_path(lp)
        if seed is None:
            continue
        run_dir = lp.parent
        required_names = ['test_labels.npy','test_sample_ids.json','test_metrics.json','FINAL_TEST_COMPLETE.json','config.json']
        missing_names = [n for n in required_names if not (run_dir/n).exists()]
        if missing_names:
            print(f'WARNING incomplete run skipped: {dataset}/{method}/seed_{seed}: {missing_names}')
            continue
        if seed in out:
            raise RuntimeError(f'Duplicate final-test run for {dataset}/{method}/seed_{seed}')
        out[seed] = {
            'dir':run_dir, 'logits':lp, 'labels':run_dir/'test_labels.npy',
            'ids':run_dir/'test_sample_ids.json', 'metrics':run_dir/'test_metrics.json',
            'complete':run_dir/'FINAL_TEST_COMPLETE.json', 'config':run_dir/'config.json'
        }
    return out

RUN_INDEX = {}
for ds in DATASETS:
    RUN_INDEX[ds] = {}
    for method in ['ctpu'] + COMPARATORS:
        RUN_INDEX[ds][method] = find_run_files(ds, method)

inventory = []
for ds in DATASETS:
    for method, runs in RUN_INDEX[ds].items():
        seeds = sorted(runs)
        inventory.append({
            'dataset':ds, 'method':method, 'n_final_test_runs':len(seeds),
            'seeds':';'.join(map(str,seeds)),
            'run_root':str(CAMPAIGNS/ds/'final_test_runs'/ds/method)
        })
INV = pd.DataFrame(inventory)
display(INV)
INV.to_csv(OUTDIR/'final_test_artifact_inventory.csv', index=False)

if not any(RUN_INDEX[ds].get('ctpu') for ds in DATASETS):
    raise FileNotFoundError('No complete CTPU final-test run folders were found. Raw final-test artefacts may not have been exported to Drive.')

# Validate artifact identity before using prediction arrays. The hashes below
# belong to the frozen export, rather than to the analysis software.
EXPECTED_CODE_HASH = 'a5a64e2351cf9eb55141ffa3c96892b37ce752153d6dc1d9093e2f805e7ce8c4'
manifest = json.loads((PROJECT_ROOT/'PAPER_ARTIFACT_MANIFEST.json').read_text())
if manifest.get('bundle_sha256') != EXPECTED_SOURCE_BUNDLE_SHA256:
    raise RuntimeError('Training source bundle does not match frozen export')
manifest_entries = {r['path']:r for r in manifest['files']}
provenance_rows = []
def check_export_file(path):
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    row = manifest_entries.get(relative)
    if row is None:
        raise RuntimeError(f'File absent from export manifest: {relative}')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            digest.update(block)
    if path.stat().st_size != row['bytes'] or digest.hexdigest() != row['sha256']:
        raise RuntimeError(f'Export checksum mismatch: {relative}')

for ds, methods in RUN_INDEX.items():
    for method, runs in methods.items():
        for seed, files in runs.items():
            for key in ['logits','labels','ids','metrics','complete','config']:
                check_export_file(files[key])
            conf = json.loads(files['config'].read_text())
            complete = json.loads(files['complete'].read_text())
            if (conf['dataset'],conf['method'],conf['seed']) != (ds,method,seed):
                raise RuntimeError('Final-test run identity mismatch')
            expected = {'epochs':4, 'lambda_pair':.25, 'pair_k_pos':3, 'pair_k_neg':3,
                        'model_revision':'e2da8e2f811d1448a5b465c236feacd80ffbac7b'}
            if any(conf.get(k) != v for k,v in expected.items()):
                raise RuntimeError(f'Unexpected frozen configuration: {files["dir"]}')
            if method == 'pairwise_cardinality_penalty' and conf['cardinality_penalty_mu'] != .1:
                raise RuntimeError('Final penalty model is not the fixed mu=.1 control')
            if complete['code_hash'] != EXPECTED_CODE_HASH or complete['status'] != 'FINAL_TEST_COMPLETE':
                raise RuntimeError('Frozen executable hash or completion status mismatch')
            dev = CAMPAIGNS/ds/'runs'/ds/method/f'seed_{seed}'/files['dir'].name
            dev_done = json.loads((dev/'COMPLETE.json').read_text())
            dev_conf = json.loads((dev/'config.json').read_text())
            if dev_conf != conf or dev_done['config_hash'] != complete['config_hash']:
                raise RuntimeError('Final-test/development configuration mismatch')
            if dev_done['final_model_tree_hash'] != complete['source_final_model_tree_hash']:
                raise RuntimeError('Final-test weights do not match retained development model')
            if dev_done['final_epoch'] != 4 or dev_done['early_stopping']:
                raise RuntimeError('Unexpected checkpoint policy')
            if complete['retrained_for_test'] or not complete['exact_frozen_development_weights_used']:
                raise RuntimeError('Test model identity policy failed')
            files['dev_dir'] = dev
            provenance_rows.append({'dataset':ds,'method':method,'seed':seed,
                'run_id':files['dir'].name,'code_hash':complete['code_hash'],
                'config_hash':complete['config_hash'],'pass':True})
pd.DataFrame(provenance_rows).to_csv(OUTDIR/'frozen_artifact_provenance.csv',index=False)
print('Frozen provenance validated for',len(provenance_rows),'test runs',flush=True)

# Metric definitions and per-document validation.
def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def ranking_loss_per_document(y, p):
    """Per-document label-ranking loss; ties count as ranking violations, matching sklearn."""
    n, n_labels = y.shape
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        mask_pos = y[i].astype(bool)
        n_pos = int(mask_pos.sum())
        n_neg = n_labels - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        pos_scores = p[i, mask_pos]
        neg_sorted = np.sort(p[i, ~mask_pos])
        # Number of negatives with score >= each positive score; ties are counted as errors.
        n_ge = n_neg - np.searchsorted(neg_sorted, pos_scores, side='left')
        out[i] = float(n_ge.mean() / n_neg)
    return out


def per_document_metrics(y, logits):
    y = np.asarray(y)
    if not np.isin(y, [0,1]).all():
        raise ValueError('Labels must be binary before any dtype conversion')
    y = y.astype(np.uint8)
    z = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite logits')
    if z.shape != y.shape:
        raise ValueError(f'logits/labels shape mismatch: {z.shape} vs {y.shape}')
    if not np.isin(y, [0,1]).all():
        raise ValueError('test_labels.npy contains values other than 0/1')
    p = sigmoid_np(z)
    eps = 1e-8
    pc = np.clip(p, eps, 1-eps)
    nll = -(y*np.log(pc) + (1-y)*np.log(1-pc)).mean(axis=1)
    brier = ((p-y)**2).mean(axis=1)
    soft = p.sum(axis=1)
    true = y.sum(axis=1)
    ranking = ranking_loss_per_document(y, p)
    return {
        'ranking_loss':ranking,
        'soft_cardinality_mae':np.abs(soft-true).astype(np.float64),
        'nll':nll.astype(np.float64),
        'brier':brier.astype(np.float64),
    }


def stable_int_seed(*parts, bits=63):
    s = '||'.join(map(str, parts)).encode('utf-8')
    h = hashlib.blake2b(s, digest_size=16).digest()
    return int.from_bytes(h, 'little') & ((1 << bits)-1)


def _mean_finite(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float('nan')


def verify_against_saved_metrics(doc_metrics, metrics_path, dataset, method, seed):
    saved = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    rows=[]
    for metric in METRICS:
        if metric not in saved:
            rows.append({'dataset':dataset,'method':method,'seed':seed,'metric':metric,
                         'recomputed':_mean_finite(doc_metrics[metric]),'saved':None,'abs_difference':None,'pass':False,
                         'note':'metric not present in test_metrics.json'})
            continue
        calc = _mean_finite(doc_metrics[metric])
        ref = float(saved[metric])
        delta = abs(calc-ref)
        ok = bool(np.isclose(calc, ref, rtol=2e-4, atol=2e-6, equal_nan=True))
        rows.append({'dataset':dataset,'method':method,'seed':seed,'metric':metric,
                     'recomputed':calc,'saved':ref,'abs_difference':delta,'pass':ok,
                     'note':'per-document recomputation compared with saved final-test metric'})
    return rows

print('Metric functions ready. No model, tokenizer, or GPU is loaded.')
# Paired crossed-bootstrap calculations from per-document metrics.
all_boot_rows = []
all_seed_rows = []
all_validation_rows = []
all_skip_rows = []
all_alignment_rows = []


def load_dataset_metric_cache(dataset):
    methods = ['ctpu'] + COMPARATORS
    ctpu_runs = RUN_INDEX[dataset].get('ctpu', {})
    if not ctpu_runs:
        return {}, None, None
    first_seed = sorted(ctpu_runs)[0]
    ref = ctpu_runs[first_seed]
    canonical_ids = json.loads(ref['ids'].read_text(encoding='utf-8'))
    ref_y = np.load(ref['labels'], mmap_mode='r')
    cache = {}
    for method in methods:
        cache[method] = {}
        for seed, files in sorted(RUN_INDEX[dataset].get(method, {}).items()):
            ids = json.loads(files['ids'].read_text(encoding='utf-8'))
            if ids != canonical_ids:
                all_alignment_rows.append({'dataset':dataset,'method':method,'seed':seed,'check':'sample_id_order','pass':False,
                                           'note':'Document IDs/order differ from CTPU reference; run excluded.'})
                continue
            y = np.load(files['labels'], mmap_mode='r')
            if y.shape != ref_y.shape or not np.array_equal(y, ref_y):
                all_alignment_rows.append({'dataset':dataset,'method':method,'seed':seed,'check':'test_labels','pass':False,
                                           'note':'Test-label matrix differs from CTPU reference; run excluded.'})
                continue
            logits = np.load(files['logits'], mmap_mode='r')
            if logits.shape != y.shape:
                all_alignment_rows.append({'dataset':dataset,'method':method,'seed':seed,'check':'logit_label_shape','pass':False,
                                           'note':f'{logits.shape} != {y.shape}; run excluded.'})
                continue
            dm = per_document_metrics(y, logits)
            checks = verify_against_saved_metrics(dm, files['metrics'], dataset, method, seed)
            if any(not row['pass'] for row in checks):
                raise RuntimeError(f'Saved metric verification failed: {dataset}/{method}/{seed}')
            cache[method][seed] = dm
            all_validation_rows.extend(checks)
            all_alignment_rows.append({'dataset':dataset,'method':method,'seed':seed,'check':'sample_ids_labels_shapes','pass':True,
                                       'note':f'{y.shape[0]} documents x {y.shape[1]} labels'})
            # The distribution is across documents after averaging the absolute
            # error over the available seeds for each document.
            del logits, y, dm
    return cache, canonical_ids, ref_y.shape


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


metric_caches = {}
analysis_started = time.time()
for ds in DATASETS:
    cache, canonical_ids, shape = load_dataset_metric_cache(ds)
    metric_caches[ds] = cache
    if not cache or not cache.get('ctpu'):
        all_skip_rows.append({'dataset':ds,'comparator':'ALL','reason':'No valid CTPU final-test predictions.'})
        continue
    print(f'\n{ds}: loaded metrics for {len(canonical_ids)} test documents; label dimension={shape[1]}')
    for comp in COMPARATORS:
        cseeds=set(cache['ctpu'])
        mseeds=set(cache.get(comp,{}))
        common=sorted(cseeds & mseeds)
        if len(common) < MIN_MATCHED_SEEDS:
            all_skip_rows.append({'dataset':ds,'comparator':comp,'reason':f'Only {len(common)} matched seeds; minimum is {MIN_MATCHED_SEEDS}.',
                                  'matched_seeds':';'.join(map(str,common))})
            continue
        pairs=[(cache['ctpu'][s],cache[comp][s]) for s in common]
        boot_seed = stable_int_seed(BOOTSTRAP_BASE_SEED, ds, comp)
        # Preserve original predeclared random stream for the primary CTPU-Pairwise analysis.
        if comp == 'pairwise':
            boot_seed = stable_int_seed(BOOTSTRAP_BASE_SEED, ds)
        results = crossed_paired_bootstrap_arrays(pairs, reps=BOOTSTRAP_REPS, seed=boot_seed)
        for metric, res in results.items():
            lo, hi = res['ci_2_5'], res['ci_97_5']
            direction = ('CTPU lower' if hi < 0 else 'Comparator lower' if lo > 0 else 'Interval includes zero')
            all_boot_rows.append({
                'dataset':ds,'comparison':f'CTPU - {METHOD_LABEL.get(comp,comp)}','comparator':comp,
                'metric':metric,'n_paired_seeds':len(common),'matched_seeds':';'.join(map(str,common)),
                'n_documents':len(canonical_ids),'difference_ctpu_minus_comparator':res['observed_mean_difference'],
                'ci_95_lower':lo,'ci_95_upper':hi,'interval_excludes_zero':bool(hi<0 or lo>0),
                'bootstrap_fraction_difference_lt_0_descriptive':res['bootstrap_fraction_difference_lt_0'],
                'interpretation_for_loss_metric':direction,
                'analysis_status':'posthoc_exploratory_except_predeclared_ctpu_vs_pairwise',
                'bootstrap_reps':BOOTSTRAP_REPS,'bootstrap_design':'paired_crossed_seed_by_document',
                'not_a_p_value':True,'multiplicity_adjustment':'none; exploratory comparisons'
            })
            for s,(a,b) in zip(common,pairs):
                d=np.asarray(a[metric],dtype=float)-np.asarray(b[metric],dtype=float)
                mean_d=_mean_finite(d)
                all_seed_rows.append({
                    'dataset':ds,'comparator':comp,'metric':metric,'seed':s,
                    'mean_ctpu_minus_comparator':mean_d,
                    'ctpu_better_on_loss_metric':bool(mean_d<0) if np.isfinite(mean_d) else None,
                    'n_documents_used':int(np.isfinite(d).sum())
                })
        print(f'  CTPU vs {comp}: matched seeds={common}, bootstrap complete')

print(f'\nElapsed analysis time: {(time.time()-analysis_started)/60:.2f} minutes')
BOOT = pd.DataFrame(all_boot_rows)
SEED_DIFF = pd.DataFrame(all_seed_rows)
ALIGN = pd.DataFrame(all_alignment_rows)
VERIFY = pd.DataFrame(all_validation_rows)
SKIPS = pd.DataFrame(all_skip_rows)

for name,df in [('paired_crossed_bootstrap_comparisons.csv',BOOT),
                ('paired_seed_level_differences.csv',SEED_DIFF),
                ('run_alignment_audit.csv',ALIGN),
                ('recomputed_vs_saved_metric_check.csv',VERIFY),
                ('skipped_comparisons.csv',SKIPS)]:
    df.to_csv(OUTDIR/name,index=False)

if BOOT.empty:
    raise RuntimeError('No paired comparison completed. Check the Drive campaign files listed in final_test_artifact_inventory.csv.')

print('\nBootstrap comparison output:')
display(BOOT.sort_values(['dataset','comparator','metric']))
print('\nRecomputed metric check:')
display(VERIFY.groupby(['metric','pass']).size().rename('n').reset_index())

# Pair elapsed time using the exact development run referenced by each final-test
# run. The older flattened runtime CSV contains three Reuters penalty settings
# for one seed and therefore must not be reduced to a seed-keyed dictionary.
runtime_rows=[]
for ds in DATASETS:
    for seed in sorted(set(RUN_INDEX[ds]['ctpu']) & set(RUN_INDEX[ds]['pairwise_cardinality_penalty'])):
        times = {}
        for method in ['ctpu','pairwise_cardinality_penalty']:
            f = RUN_INDEX[ds][method][seed]
            done = json.loads((f['dev_dir']/'COMPLETE.json').read_text())
            times[method] = float(done['elapsed_seconds'])
        runtime_rows.append({'dataset':ds,'seed':seed,'ctpu_seconds':times['ctpu'],
            'direct_penalty_seconds':times['pairwise_cardinality_penalty'],
            'ctpu_over_penalty_ratio':times['ctpu']/times['pairwise_cardinality_penalty'],
            'ctpu_run_id':RUN_INDEX[ds]['ctpu'][seed]['dir'].name,
            'penalty_run_id':RUN_INDEX[ds]['pairwise_cardinality_penalty'][seed]['dir'].name})
RUNTIME = pd.DataFrame(runtime_rows)
RUNTIME.to_csv(OUTDIR/'paired_training_runtime_ctpu_vs_direct_penalty.csv',index=False)
display(RUNTIME.groupby('dataset').agg(n=('seed','size'),ratio_mean=('ctpu_over_penalty_ratio','mean')).reset_index())

quantiles=[]
for ds in DATASETS:
    for method in ['ctpu','pairwise','pairwise_cardinality_penalty']:
        values=np.stack([m['soft_cardinality_mae'] for _,m in sorted(metric_caches[ds][method].items())])
        per_doc=np.mean(values,axis=0)
        qs=np.quantile(per_doc,[.05,.25,.5,.75,.95])
        quantiles.append({'dataset':ds,'method':method,'n_seeds':len(values),'n_documents':len(per_doc),
            'mean_abs_error':per_doc.mean(),'q05':qs[0],'q25':qs[1],'median':qs[2],'q75':qs[3],'q95':qs[4]})
pd.DataFrame(quantiles).to_csv(OUTDIR/'document_error_quantiles.csv',index=False)

subset=[]
for ds in ['reuters21578','aapd']:
    common=sorted(set(metric_caches[ds]['ctpu']) & set(metric_caches[ds]['param_pcgrad']))
    for method in ['ctpu','pairwise','param_pcgrad']:
        vals=np.array([[np.nanmean(metric_caches[ds][method][s][m]) for m in METRICS] for s in common])
        for j,m in enumerate(METRICS):
            subset.append({'dataset':ds,'method':method,'metric':m,'n_seeds':len(common),
                'seeds':';'.join(map(str,common)),'mean':vals[:,j].mean(),'sd':vals[:,j].std(ddof=1)})
pd.DataFrame(subset).to_csv(OUTDIR/'common_three_seed_summary.csv',index=False)

# An existing development-only penalty sensitivity; it is not a CTPU lambda sweep.
penalty=[]
base=CAMPAIGNS/'reuters21578'/'runs'/'reuters21578'/'pairwise_cardinality_penalty'/'seed_20260805'
for path in sorted(base.glob('*/config.json')):
    conf=json.loads(path.read_text()); done=json.loads((path.parent/'COMPLETE.json').read_text())
    val=done['validation_metrics']
    penalty.append({'dataset':'reuters21578','seed':20260805,'mu':conf['cardinality_penalty_mu'],
        'lambda_pair':conf['lambda_pair'],'epochs':conf['epochs'],
        'validation_ranking_loss':val['ranking_loss'],'validation_soft_cardinality_mae':val['soft_cardinality_mae'],
        'run_id':path.parent.name,'analysis_status':'existing_development_only_single_seed'})
pd.DataFrame(penalty).sort_values('mu').to_csv(OUTDIR/'development_penalty_sensitivity.csv',index=False)

# Forest plots: positive values mean CTPU has a higher (worse) loss/error.
for metric in ['ranking_loss','soft_cardinality_mae']:
    d=BOOT[BOOT['metric']==metric].copy()
    if d.empty: continue
    d['row_label']=d['dataset']+' | '+d['comparator'].map(METHOD_LABEL).fillna(d['comparator'])
    d=d.sort_values(['dataset','comparator']).reset_index(drop=True)
    y=np.arange(len(d))
    est=d['difference_ctpu_minus_comparator'].to_numpy(float)
    lo=est-d['ci_95_lower'].to_numpy(float)
    hi=d['ci_95_upper'].to_numpy(float)-est
    fig,ax=plt.subplots(figsize=(10,max(4.5,0.42*len(d)+1.5)))
    ax.errorbar(est,y,xerr=[lo,hi],fmt='o',capsize=3)
    ax.axvline(0,linewidth=1)
    ax.set_yticks(y,d['row_label'].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(f'CTPU minus comparator {metric} (negative favors CTPU)')
    ax.set_title(f'Post-hoc paired crossed-bootstrap: {metric}')
    ax.grid(True,axis='x',alpha=.25)
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(OUTDIR/f'forest_{metric}.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.show()
    plt.close(fig)

print('Figures and optional training-time comparison saved to:', OUTDIR)