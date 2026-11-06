from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import label_ranking_loss

from .metrics import sigmoid_np
from .utils import atomic_write_json, stable_int_seed


def _load_run_arrays(run_dir: str | Path, prefix: str = "validation"):
    p = Path(run_dir)
    logits = np.load(p / f"{prefix}_logits.npy", mmap_mode="r")
    y = np.load(p / f"{prefix}_labels.npy", mmap_mode="r")
    ids = json.loads((p / f"{prefix}_sample_ids.json").read_text(encoding="utf-8"))
    return np.asarray(logits), np.asarray(y), ids


def per_document_metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, np.ndarray]:
    p = sigmoid_np(logits)
    eps = 1e-8
    pc = np.clip(p, eps, 1 - eps)
    nll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean(axis=1)
    brier = ((p - y) ** 2).mean(axis=1)
    soft_card = p.sum(axis=1)
    true_card = y.sum(axis=1)
    card_abs = np.abs(soft_card - true_card)
    card_signed = soft_card - true_card
    ranking = np.full(len(y), np.nan, dtype=np.float64)
    for i in range(len(y)):
        npos = int(y[i].sum())
        if 0 < npos < y.shape[1]:
            ranking[i] = float(label_ranking_loss(y[i:i+1], p[i:i+1]))
    return {
        "nll": nll.astype(np.float64),
        "brier": brier.astype(np.float64),
        "soft_cardinality_mae": card_abs.astype(np.float64),
        "soft_cardinality_signed_bias": card_signed.astype(np.float64),
        "ranking_loss": ranking,
    }


def crossed_paired_bootstrap(seed_run_pairs: list[tuple[str | Path, str | Path]],
                             reps: int = 2000, seed: int = 20261205,
                             prefix: str = "validation") -> dict[str, Any]:
    """Paired crossed bootstrap over seed × document.

    The same evaluation documents recur under every random seed, so documents are
    crossed with seeds rather than nested inside them. Each replicate therefore
    resamples seeds and draws ONE shared document bootstrap sample that is applied
    to every selected seed and both methods. A-B differences are returned; negative
    is favorable for loss/error metrics. Non-decomposable AP metrics are intentionally
    excluded and must be summarized as paired seed-wise differences.
    """
    prepared = []
    canonical_ids = None
    canonical_y = None
    for a_dir, b_dir in seed_run_pairs:
        al, ay, aid = _load_run_arrays(a_dir, prefix)
        bl, by, bid = _load_run_arrays(b_dir, prefix)
        if aid != bid:
            raise RuntimeError("Paired bootstrap sample-id order mismatch within seed")
        if ay.shape != by.shape or not np.array_equal(ay, by):
            raise RuntimeError("Paired bootstrap label matrix mismatch within seed")
        if canonical_ids is None:
            canonical_ids = list(aid)
            canonical_y = np.asarray(ay)
        else:
            if list(aid) != canonical_ids:
                raise RuntimeError("Crossed bootstrap requires identical document order across seeds")
            if canonical_y.shape != ay.shape or not np.array_equal(canonical_y, ay):
                raise RuntimeError("Crossed bootstrap requires identical labels across seeds")
        prepared.append((per_document_metrics(ay, al), per_document_metrics(by, bl)))
    if not prepared:
        raise ValueError("No paired runs")
    n_docs = len(canonical_ids)
    if n_docs == 0:
        raise ValueError("No evaluation documents")
    rng = np.random.default_rng(seed)
    metric_names = list(prepared[0][0])
    draws = {m: np.empty(reps, dtype=np.float64) for m in metric_names}
    S = len(prepared)
    for r in range(reps):
        sampled_seeds = rng.integers(0, S, size=S)
        # ONE document draw per replicate, shared by every selected seed and method.
        doc_idx = rng.integers(0, n_docs, size=n_docs)
        for m in metric_names:
            seed_effects = []
            for sidx in sampled_seeds:
                a, b = prepared[int(sidx)]
                diff = np.asarray(a[m], dtype=np.float64) - np.asarray(b[m], dtype=np.float64)
                selected = diff[doc_idx]
                selected = selected[np.isfinite(selected)]
                if selected.size:
                    seed_effects.append(float(selected.mean()))
            draws[m][r] = float(np.mean(seed_effects)) if seed_effects else np.nan
    out = {
        "reps": reps,
        "seed": seed,
        "bootstrap_design": "crossed_seed_by_document_paired",
        "document_resample_shared_across_seeds_and_methods": True,
        "difference_definition": "method_A_minus_method_B",
        "interval": "percentile_95pct",
        "metrics": {},
    }
    for m, d in draws.items():
        d = d[np.isfinite(d)]
        if d.size == 0:
            raise RuntimeError(f"No finite bootstrap draws for metric {m}")
        observed_seed_effects = []
        for a, b in prepared:
            diff = np.asarray(a[m], dtype=np.float64) - np.asarray(b[m], dtype=np.float64)
            diff = diff[np.isfinite(diff)]
            if diff.size:
                observed_seed_effects.append(float(diff.mean()))
        observed = float(np.mean(observed_seed_effects)) if observed_seed_effects else float("nan")
        out["metrics"][m] = {
            "observed_mean_difference": observed,
            "mean_bootstrap_difference": float(np.mean(d)),
            "ci_2_5": float(np.percentile(d, 2.5)),
            "ci_97_5": float(np.percentile(d, 97.5)),
            "bootstrap_fraction_difference_lt_0": float(np.mean(d < 0)),
            "bootstrap_fraction_is_not_a_p_value": True,
        }
    return out


def hierarchical_paired_bootstrap(seed_run_pairs: list[tuple[str | Path, str | Path]],
                                  reps: int = 2000, seed: int = 20261205,
                                  prefix: str = "validation") -> dict[str, Any]:
    """Backward-compatible alias. v1.5 uses the correct crossed seed×document design."""
    return crossed_paired_bootstrap(seed_run_pairs, reps=reps, seed=seed, prefix=prefix)

def save_bootstrap(path: str | Path, result: dict[str, Any]) -> None:
    atomic_write_json(path, result)
