from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .utils import stable_int_seed


@dataclass
class AuxOutput:
    grad: torch.Tensor
    active_mask: torch.Tensor
    pairwise_loss_mean: float
    diagnostics: Dict[str, float]


def _choose_pairs(pos: Sequence[int], eligible_labels: Sequence[int], sample_id: str, epoch: int, seed: int,
                  k_pos: int, k_neg: int) -> tuple[list[int], list[int]]:
    """Deterministic positive/negative sampling without materializing an O(L) complement per sample.

    For sparse multi-label datasets, at most K_neg labels are obtained by deterministic
    rejection sampling from the TRAIN-derived eligible label inventory. A complement
    fallback is used only in the unusual dense-label case where rejection sampling
    cannot fill the requested negatives quickly.
    """
    eligible = np.asarray(eligible_labels, dtype=np.int64)
    if eligible.ndim != 1 or eligible.size == 0:
        return [], []
    eligible_set = set(map(int, eligible.tolist()))
    pos = sorted(set(int(x) for x in pos if int(x) in eligible_set))
    if not pos or len(pos) >= len(eligible):
        return [], []
    rng = np.random.default_rng(stable_int_seed(seed, epoch, sample_id, "pair-sampling"))
    kp = min(int(k_pos), len(pos))
    selected_pos = rng.choice(np.asarray(pos, dtype=np.int64), size=kp, replace=False).astype(int).tolist()

    pos_set = set(pos)
    available_neg = int(len(eligible) - len(pos_set))
    kn = min(int(k_neg), available_neg)
    if kn <= 0:
        return [], []

    selected_neg: list[int] = []
    selected_neg_set: set[int] = set()
    # Sparse-label fast path. The cap is deterministic and ample for K_neg<=3.
    max_attempts = max(32, kn * 16)
    attempts = 0
    while len(selected_neg) < kn and attempts < max_attempts:
        candidate = int(eligible[int(rng.integers(0, len(eligible)))])
        attempts += 1
        if candidate in pos_set or candidate in selected_neg_set:
            continue
        selected_neg.append(candidate)
        selected_neg_set.add(candidate)

    if len(selected_neg) < kn:
        # Dense-label fallback. This is rarely reached in the intended benchmarks.
        neg_pool = np.asarray([int(x) for x in eligible.tolist() if int(x) not in pos_set], dtype=np.int64)
        remaining = np.asarray([int(x) for x in neg_pool.tolist() if int(x) not in selected_neg_set], dtype=np.int64)
        need = kn - len(selected_neg)
        if need > 0:
            selected_neg.extend(rng.choice(remaining, size=need, replace=False).astype(int).tolist())
    return list(map(int, selected_pos)), list(map(int, selected_neg))

def pairwise_score_gradient(logits: torch.Tensor, label_indices: Sequence[Sequence[int]],
                            sample_ids: Sequence[str], epoch: int, seed: int,
                            k_pos: int = 3, k_neg: int = 3,
                            eligible_labels: Sequence[int] | None = None
                            ) -> tuple[torch.Tensor, torch.Tensor, float, Dict[str, float]]:
    """Deterministic analytic gradient of per-sample mean softplus(z_neg-z_pos).

    Pair selection is CPU/deterministic. Score arithmetic is vectorized on the
    logits device. The dense padded/one-hot accumulation intentionally avoids
    CUDA atomic scatter accumulation, which can undermine strict reproducibility.
    With K_pos=K_neg=3, at most 9 pairs/sample are materialized.
    """
    z = logits.detach().float()
    bsz, num_labels = z.shape
    if eligible_labels is None:
        eligible_labels = list(range(num_labels))

    selected: list[list[tuple[int, int]]] = []
    pair_counts = np.zeros(bsz, dtype=np.int64)
    max_pairs = 0
    for i in range(bsz):
        pos, neg = _choose_pairs(
            label_indices[i], eligible_labels, sample_ids[i], epoch, seed, k_pos, k_neg
        )
        pairs = [(int(p), int(n)) for p in pos for n in neg]
        selected.append(pairs)
        pair_counts[i] = len(pairs)
        max_pairs = max(max_pairs, len(pairs))

    g = torch.zeros_like(z)
    active = torch.zeros_like(z)
    if max_pairs == 0:
        return g, active, 0.0, {"valid_pair_samples": 0.0, "mean_pairs_per_valid_sample": 0.0}

    # Padded pair matrices. Padding indices are masked out before contributing.
    pos_np = np.zeros((bsz, max_pairs), dtype=np.int64)
    neg_np = np.zeros((bsz, max_pairs), dtype=np.int64)
    mask_np = np.zeros((bsz, max_pairs), dtype=np.bool_)
    for i, pairs in enumerate(selected):
        if not pairs:
            continue
        n = len(pairs)
        pos_np[i, :n] = [p for p, _ in pairs]
        neg_np[i, :n] = [nidx for _, nidx in pairs]
        mask_np[i, :n] = True

    device = z.device
    pi = torch.as_tensor(pos_np, device=device, dtype=torch.long)
    ni = torch.as_tensor(neg_np, device=device, dtype=torch.long)
    mask = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
    counts = torch.as_tensor(pair_counts, device=device, dtype=z.dtype).clamp_min(1.0)

    z_pos = torch.gather(z, 1, pi)
    z_neg = torch.gather(z, 1, ni)
    d = z_neg - z_pos
    t = torch.sigmoid(d) / counts.unsqueeze(1)
    t = torch.where(mask, t, torch.zeros_like(t))

    # one_hot is modest here (B x <=9 x L) and deterministic under the strict
    # PyTorch deterministic-algorithm regime used by the study.
    pos_oh = F.one_hot(pi, num_classes=num_labels).to(dtype=z.dtype)
    neg_oh = F.one_hot(ni, num_classes=num_labels).to(dtype=z.dtype)
    signed = (-t.unsqueeze(-1) * pos_oh) + (t.unsqueeze(-1) * neg_oh)
    g = signed.sum(dim=1)
    active = (((pos_oh + neg_oh) * mask.unsqueeze(-1).to(z.dtype)).sum(dim=1) > 0).to(z.dtype)

    per_pair_loss = F.softplus(d) * mask.to(z.dtype)
    per_sample_loss = per_pair_loss.sum(dim=1) / counts
    valid = torch.as_tensor(pair_counts > 0, device=device)
    mean_loss = float(per_sample_loss[valid].mean().detach().cpu().item())
    valid_n = int(np.count_nonzero(pair_counts))
    total_pairs = int(pair_counts.sum())
    return g, active, mean_loss, {
        "valid_pair_samples": float(valid_n),
        "mean_pairs_per_valid_sample": float(total_pairs / max(1, valid_n)),
    }


def ctpu_project(g: torch.Tensor, logits: torch.Tensor, active_mask: torch.Tensor,
                 eps: float = 1e-12) -> tuple[torch.Tensor, Dict[str, float]]:
    """Exact Euclidean score-space tangent projection with a degenerate fallback.

    For each sample, if ||h||^2 > eps, u = g - (h·g / h·h) h. If the
    cardinality-gradient norm is numerically degenerate, u=g and the event is
    exposed through ``tiny_den_fraction``. ``eps`` is a threshold, not an
    additive denominator term.
    """
    p = torch.sigmoid(logits.detach().float())
    h = active_mask.float() * p * (1.0 - p)
    dot = (h * g).sum(dim=1, keepdim=True)
    den = (h * h).sum(dim=1, keepdim=True)
    safe = den > eps
    coeff = torch.where(safe, dot / den.clamp_min(eps), torch.zeros_like(dot))
    u = g - coeff * h
    gnorm = torch.linalg.vector_norm(g, dim=1)
    unorm = torch.linalg.vector_norm(u, dim=1)
    hdotu = (h * u).sum(dim=1)
    gd_u = (g * u).sum(dim=1)
    u2 = (u * u).sum(dim=1)
    return u, {
        "mean_abs_h_dot_g": float(dot.abs().mean().detach().cpu().item()),
        "mean_abs_h_dot_u": float(hdotu.abs().mean().detach().cpu().item()),
        "mean_g_norm": float(gnorm.mean().detach().cpu().item()),
        "mean_u_norm": float(unorm.mean().detach().cpu().item()),
        "mean_u_to_g_norm_ratio": float((unorm / gnorm.clamp_min(eps)).mean().detach().cpu().item()),
        "mean_pairwise_descent_identity_error": float((gd_u - u2).abs().mean().detach().cpu().item()),
        "tiny_den_fraction": float((~safe).float().mean().detach().cpu().item()),
    }


def normmatch_gradient(g: torch.Tensor, u: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, Dict[str, float]]:
    gnorm = torch.linalg.vector_norm(g, dim=1, keepdim=True)
    unorm = torch.linalg.vector_norm(u, dim=1, keepdim=True)
    alpha = torch.where(gnorm > eps, unorm / gnorm.clamp_min(eps), torch.ones_like(gnorm))
    out = alpha * g
    return out, {
        "mean_normmatch_scale": float(alpha.mean().detach().cpu().item()),
        "min_normmatch_scale": float(alpha.min().detach().cpu().item()),
        "max_normmatch_scale": float(alpha.max().detach().cpu().item()),
    }


def score_auxiliary_projection(g_pair: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor,
                 active_mask: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, Dict[str, float]]:
    """One-sided score-space auxiliary projection control.

    This control protects the primary BCE score gradient by removing the
    conflicting component of the Pairwise auxiliary on active labels. Standard
    symmetric parameter-space PCGrad is implemented separately in train.py.
    """
    p = torch.sigmoid(logits.detach().float())
    num_labels = logits.shape[1]
    g_primary = active_mask.float() * (p - targets.detach().float()) / float(num_labels)
    dot = (g_pair * g_primary).sum(dim=1, keepdim=True)
    den = (g_primary * g_primary).sum(dim=1, keepdim=True)
    conflict = dot < 0
    coeff = torch.where(conflict & (den > eps), dot / den.clamp_min(eps), torch.zeros_like(dot))
    out = g_pair - coeff * g_primary
    return out, {
        "conflict_fraction": float(conflict.float().mean().detach().cpu().item()),
        "mean_abs_primary_pair_dot": float(dot.abs().mean().detach().cpu().item()),
    }


def build_auxiliary_gradient(method: str, logits: torch.Tensor, targets: torch.Tensor,
                             label_indices: Sequence[Sequence[int]], sample_ids: Sequence[str],
                             epoch: int, seed: int, k_pos: int = 3, k_neg: int = 3,
                             eps: float = 1e-12,
                             eligible_labels: Sequence[int] | None = None) -> AuxOutput:
    g, active, pair_loss, diag = pairwise_score_gradient(
        logits, label_indices, sample_ids, epoch, seed, k_pos=k_pos, k_neg=k_neg,
        eligible_labels=eligible_labels,
    )
    if method == "pairwise":
        out = g
    elif method == "ctpu":
        out, d = ctpu_project(g, logits, active, eps=eps); diag.update(d)
        # The projection is computed in FP32, but autograd receives grad_outputs in
        # logits.dtype (BF16 under the main A100 protocol). Record the actual
        # post-cast tangency error rather than claiming exact orthogonality after cast.
        p32 = torch.sigmoid(logits.detach().float())
        h32 = active.float() * p32 * (1.0 - p32)
        out_after_cast = out.to(logits.dtype).float()
        cast_dot = (h32 * out_after_cast).sum(dim=1).abs()
        diag["mean_abs_h_dot_u_after_autograd_dtype_cast"] = float(cast_dot.mean().detach().cpu().item())
        diag["max_abs_h_dot_u_after_autograd_dtype_cast"] = float(cast_dot.max().detach().cpu().item())
    elif method == "normmatch":
        u, d1 = ctpu_project(g, logits, active, eps=eps)
        out, d2 = normmatch_gradient(g, u, eps=eps); diag.update(d1); diag.update(d2)
    elif method == "score_auxproj":
        out, d = score_auxiliary_projection(g, logits, targets, active, eps=eps); diag.update(d)
    else:
        raise ValueError(f"Unknown auxiliary method: {method}")
    return AuxOutput(grad=out, active_mask=active, pairwise_loss_mean=pair_loss, diagnostics=diag)


def synthetic_mechanism_self_test(seed: int = 7) -> Dict[str, float]:
    torch.manual_seed(seed)
    logits = torch.randn(8, 13, dtype=torch.float64)
    labels = [sorted(set([i % 13, (i * 3 + 2) % 13])) for i in range(8)]
    sample_ids = [f"synthetic-{i}" for i in range(8)]
    eligible = list(range(13))
    g, active, _, _ = pairwise_score_gradient(
        logits, labels, sample_ids, epoch=0, seed=seed, k_pos=2, k_neg=3, eligible_labels=eligible
    )
    u, diag = ctpu_project(g, logits, active)
    p = torch.sigmoid(logits)
    h = active * p * (1.0 - p)
    eps_step = 1e-6
    c0 = p.sum(dim=1)
    c_pair = torch.sigmoid(logits - eps_step * g).sum(dim=1)
    c_ctpu = torch.sigmoid(logits - eps_step * u).sum(dim=1)
    pred_pair = -(h * g).sum(dim=1)
    obs_pair = (c_pair - c0) / eps_step
    obs_ctpu = (c_ctpu - c0) / eps_step
    max_pred_err = float((pred_pair - obs_pair).abs().max().item())
    max_ctpu_drift = float(obs_ctpu.abs().max().item())
    max_tangent = float((h * u).sum(dim=1).abs().max().item())
    max_descent_id = float(((g * u).sum(dim=1) - (u * u).sum(dim=1)).abs().max().item())
    assert max_pred_err < 2e-5, max_pred_err
    assert max_ctpu_drift < 2e-5, max_ctpu_drift
    assert max_tangent < 1e-6, max_tangent
    assert max_descent_id < 1e-6, max_descent_id
    return {
        "status": "PASS",
        "max_first_order_pairwise_drift_prediction_error": max_pred_err,
        "max_first_order_ctpu_drift": max_ctpu_drift,
        "max_tangency_error": max_tangent,
        "max_pairwise_descent_identity_error": max_descent_id,
        **diag,
    }
