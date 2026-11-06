from pathlib import Path
import json
import numpy as np
from ctpu.statistics import crossed_paired_bootstrap


def _write_run(root: Path, logits, y, ids):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root/'validation_logits.npy', np.asarray(logits, dtype=np.float32))
    np.save(root/'validation_labels.npy', np.asarray(y, dtype=np.uint8))
    (root/'validation_sample_ids.json').write_text(json.dumps(ids), encoding='utf-8')


def test_crossed_bootstrap_is_deterministic_and_paired(tmp_path):
    y=np.asarray([[1,0],[0,1],[1,0],[0,1]],dtype=np.uint8); ids=['a','b','c','d']
    pairs=[]
    for seed in range(3):
        a=tmp_path/f'a{seed}'; b=tmp_path/f'b{seed}'
        _write_run(a, [[3,-3],[-3,3],[2,-2],[-2,2]], y, ids)
        _write_run(b, [[2,-2],[-2,2],[1,-1],[-1,1]], y, ids)
        pairs.append((a,b))
    x=crossed_paired_bootstrap(pairs,reps=50,seed=9)
    z=crossed_paired_bootstrap(pairs,reps=50,seed=9)
    assert x==z
    assert x['bootstrap_design']=='crossed_seed_by_document_paired'
    assert x['metrics']['nll']['mean_bootstrap_difference'] < 0

def test_crossed_bootstrap_requires_same_documents_across_seeds(tmp_path):
    y=np.asarray([[1,0],[0,1]],dtype=np.uint8)
    pairs=[]
    for seed,ids in [(0,['a','b']),(1,['a','x']),(2,['a','b'])]:
        a=tmp_path/f'a{seed}'; b=tmp_path/f'b{seed}'
        _write_run(a, [[2,-2],[-2,2]], y, ids); _write_run(b, [[1,-1],[-1,1]], y, ids)
        pairs.append((a,b))
    import pytest
    with pytest.raises(RuntimeError, match='identical document order'):
        crossed_paired_bootstrap(pairs,reps=5,seed=1)
