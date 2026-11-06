from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, label_ranking_loss


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def safe_macro_ap(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    vals = [float(average_precision_score(y[:, j], p[:, j]))
            for j in range(y.shape[1]) if y[:, j].sum() > 0]
    return (float(np.mean(vals)) if vals else float("nan"), len(vals))


def label_group_indices(train_support: Sequence[int]) -> Dict[str, np.ndarray]:
    s = np.asarray(train_support, dtype=np.int64)
    seen = np.where(s > 0)[0]
    unseen = np.where(s == 0)[0]
    order = seen[np.argsort(s[seen], kind="stable")]
    chunks = np.array_split(order, 3)
    return {
        "tail": chunks[0],
        "middle": chunks[1],
        "head": chunks[2],
        "zero_train_support": unseen,
        "frequent_support_gt_50": np.where(s > 50)[0],
        "few_shot_support_1_to_50": np.where((s >= 1) & (s <= 50))[0],
        "train_seen": seen,
    }


def group_macro_ap(y: np.ndarray, p: np.ndarray, indices: Sequence[int]) -> tuple[float, int]:
    vals = []
    for j in np.asarray(indices, dtype=int).tolist():
        if y[:, j].sum() > 0:
            vals.append(float(average_precision_score(y[:, j], p[:, j])))
    return (float(np.mean(vals)) if vals else float("nan"), len(vals))


def binary_nll(y: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece_binary_flat(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    yv = y.reshape(-1).astype(float)
    pv = p.reshape(-1).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    total = len(pv)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (pv >= lo) & ((pv <= hi) if b == n_bins - 1 else (pv < hi))
        n = int(mask.sum())
        if n:
            e += (n / total) * abs(float(pv[mask].mean()) - float(yv[mask].mean()))
    return float(e)


def macro_label_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, int]:
    vals = [ece_binary_flat(y[:, [j]], p[:, [j]], n_bins)
            for j in range(y.shape[1]) if y[:, j].sum() > 0]
    return (float(np.mean(vals)) if vals else float("nan"), len(vals))


def tune_global_threshold(y: np.ndarray, p: np.ndarray, start: float = .05, stop: float = .95,
                          step: float = .01, metric: str = "macro_f1") -> tuple[float, float]:
    best_t, best = .5, -float("inf")
    for t in np.round(np.arange(start, stop + 1e-12, step), 10):
        pred = (p >= t).astype(np.uint8)
        score = f1_score(y, pred, average="macro" if metric == "macro_f1" else "micro", zero_division=0)
        if score > best + 1e-15:
            best, best_t = float(score), float(t)
    return best_t, best


def topk_flat_ece_diagnostic(y: np.ndarray, p: np.ndarray, k: int = 5, n_bins: int = 15) -> dict[str, float]:
    """Simple flattened ECE over per-document top-k scores.

    This flattened diagnostic differs from dedicated calibration@k/ECE@k
    estimators and is reported separately from the main predictive metrics.
    """
    k = min(int(k), p.shape[1])
    idx = np.argpartition(-p, kth=k - 1, axis=1)[:, :k]
    row = np.arange(p.shape[0])[:, None]
    return {f"topk_flat_ece_{k}_diagnostic": ece_binary_flat(y[row, idx], p[row, idx], n_bins)}


def precision_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    k = min(int(k), p.shape[1])
    idx = np.argpartition(-p, kth=k - 1, axis=1)[:, :k]
    row = np.arange(y.shape[0])[:, None]
    return float(y[row, idx].sum(axis=1).mean() / float(k))


def ndcg_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    k = min(int(k), p.shape[1])
    order = np.argsort(-p, axis=1, kind="stable")[:, :k]
    row = np.arange(y.shape[0])[:, None]
    gains = y[row, order].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = (gains * discounts[None, :]).sum(axis=1)
    npos = y.sum(axis=1).astype(int)
    ideal_counts = np.minimum(npos, k)
    idcg = np.asarray([discounts[:c].sum() if c > 0 else 0.0 for c in ideal_counts], dtype=np.float64)
    valid = idcg > 0
    return float(np.mean(dcg[valid] / idcg[valid])) if valid.any() else float("nan")


def _metric_block(y: np.ndarray, p: np.ndarray, prefix: str = "") -> dict[str, float]:
    pred = (p >= .5).astype(np.uint8)
    macro_ap, nobs = safe_macro_ap(y, p)
    valid = (y.sum(1) > 0) & (y.sum(1) < y.shape[1])
    rank = float(label_ranking_loss(y[valid], p[valid])) if valid.any() else float("nan")
    soft, true = p.sum(1), y.sum(1)
    return {
        prefix + "nll": binary_nll(y, p),
        prefix + "brier": brier(y, p),
        prefix + "macro_ap": macro_ap,
        prefix + "macro_ap_num_observed_labels": int(nobs),
        prefix + "micro_ap": float(average_precision_score(y.reshape(-1), p.reshape(-1))),
        prefix + "macro_f1_at_0_5": float(f1_score(y, pred, average="macro", zero_division=0)),
        prefix + "micro_f1_at_0_5": float(f1_score(y, pred, average="micro", zero_division=0)),
        prefix + "ranking_loss": rank,
        prefix + "soft_cardinality_mean": float(soft.mean()),
        prefix + "true_cardinality_mean": float(true.mean()),
        prefix + "soft_cardinality_signed_bias": float((soft - true).mean()),
        prefix + "soft_cardinality_mae": float(np.abs(soft - true).mean()),
    }


def compute_metrics(y: np.ndarray, logits: np.ndarray, train_support: Sequence[int],
                    threshold_fixed: float = .5, tune_threshold: bool = True,
                    threshold_grid: tuple[float, float, float] = (.05, .95, .01)) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    logits = np.asarray(logits, dtype=np.float64)
    p = sigmoid_np(logits)
    pred = (p >= threshold_fixed).astype(np.uint8)
    groups = label_group_indices(train_support)
    macro_ap, macro_ap_n = safe_macro_ap(y, p)
    macro_ece, macro_ece_n = macro_label_ece(y, p)
    tail_ap, tail_n = group_macro_ap(y, p, groups["tail"])
    mid_ap, mid_n = group_macro_ap(y, p, groups["middle"])
    head_ap, head_n = group_macro_ap(y, p, groups["head"])
    frequent_ap, frequent_n = group_macro_ap(y, p, groups["frequent_support_gt_50"])
    few_ap, few_n = group_macro_ap(y, p, groups["few_shot_support_1_to_50"])
    zero_ap, zero_n = group_macro_ap(y, p, groups["zero_train_support"])
    valid = (y.sum(1) > 0) & (y.sum(1) < y.shape[1])
    ranking = float(label_ranking_loss(y[valid], p[valid])) if valid.any() else float("nan")
    soft, true = p.sum(1), y.sum(1)
    out: Dict[str, float] = {
        "n": int(y.shape[0]),
        "num_labels": int(y.shape[1]),
        "nll": binary_nll(y, p),
        "brier": brier(y, p),
        "micro_ece_15_diagnostic": ece_binary_flat(y, p, 15),
        "macro_label_ece_15_diagnostic": macro_ece,
        "macro_label_ece_num_observed_labels": int(macro_ece_n),
        "micro_ap": float(average_precision_score(y.reshape(-1), p.reshape(-1)),
        ),
        "macro_ap": macro_ap,
        "macro_ap_num_observed_labels": int(macro_ap_n),
        "tail_ap": tail_ap,
        "tail_ap_num_observed_labels": int(tail_n),
        "middle_ap": mid_ap,
        "middle_ap_num_observed_labels": int(mid_n),
        "head_ap": head_ap,
        "head_ap_num_observed_labels": int(head_n),
        "frequent_ap_support_gt_50": frequent_ap,
        "frequent_ap_num_observed_labels": int(frequent_n),
        "few_shot_ap_support_1_to_50": few_ap,
        "few_shot_ap_num_observed_labels": int(few_n),
        "zero_shot_ap_support_0": zero_ap,
        "zero_shot_ap_num_observed_labels": int(zero_n),
        "macro_f1_at_0_5": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1_at_0_5": float(f1_score(y, pred, average="micro", zero_division=0)),
        "hamming_loss_at_0_5": float(hamming_loss(y, pred)),
        "ranking_loss": ranking,
        "soft_cardinality_mean": float(soft.mean()),
        "true_cardinality_mean": float(true.mean()),
        "soft_cardinality_signed_bias": float((soft - true).mean()),
        "soft_cardinality_mae": float(np.abs(soft - true).mean()),
        "hard_cardinality_mean_at_0_5": float(pred.sum(1).mean()),
        "hard_cardinality_signed_bias_at_0_5": float((pred.sum(1).astype(np.int64) - true.astype(np.int64)).mean()),
    }
    out.update(topk_flat_ece_diagnostic(y, p, k=5, n_bins=15))
    for k in (1, 3, 5):
        out[f"precision_at_{k}"] = precision_at_k(y, p, k)
        out[f"ndcg_at_{k}"] = ndcg_at_k(y, p, k)

    seen = groups["train_seen"]
    if 0 < len(seen) < y.shape[1]:
        out.update(_metric_block(y[:, seen], p[:, seen], prefix="seen_"))

    if tune_threshold:
        a, b, c = threshold_grid
        t, score = tune_global_threshold(y, p, a, b, c, "macro_f1")
        pred_t = (p >= t).astype(np.uint8)
        out.update({
            "validation_selected_global_threshold_macro_f1": float(t),
            "macro_f1_at_selected_threshold": float(score),
            "micro_f1_at_selected_threshold": float(f1_score(y, pred_t, average="micro", zero_division=0)),
            "hamming_at_selected_threshold": float(hamming_loss(y, pred_t)),
        })
    return out


def per_label_metrics(y: np.ndarray, logits: np.ndarray, train_support: Sequence[int], threshold: float = .5) -> list[dict]:
    y = np.asarray(y, dtype=np.uint8)
    p = sigmoid_np(np.asarray(logits, dtype=np.float64))
    support = np.asarray(train_support, dtype=np.int64)
    groups = label_group_indices(support)
    group_name = np.full(len(support), "unknown", dtype=object)
    for name in ("tail", "middle", "head", "zero_train_support"):
        group_name[groups[name]] = name
    out = []
    for j in range(y.shape[1]):
        positives = int(y[:, j].sum())
        pred = (p[:, j] >= threshold).astype(np.uint8)
        ap = float(average_precision_score(y[:, j], p[:, j])) if positives > 0 else float("nan")
        f1 = float(f1_score(y[:, j], pred, zero_division=0)) if positives > 0 else float("nan")
        out.append({
            "label_index": j,
            "train_support": int(support[j]),
            "support_group": str(group_name[j]),
            "frequent_gt_50": bool(support[j] > 50),
            "few_shot_1_to_50": bool(1 <= support[j] <= 50),
            "zero_shot": bool(support[j] == 0),
            "eval_positives": positives,
            "average_precision": ap,
            "f1_at_0_5": f1,
            "mean_probability": float(p[:, j].mean()),
            "mean_target": float(y[:, j].mean()),
            "signed_probability_bias": float((p[:, j] - y[:, j]).mean()),
            "brier": float(np.mean((p[:, j] - y[:, j]) ** 2)),
        })
    return out
