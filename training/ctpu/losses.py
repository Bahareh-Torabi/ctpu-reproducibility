from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Normalized ASL formulation used as a baseline.

    Element-wise form follows the standard ASL defaults gamma_pos=1, gamma_neg=4, clip=0.05.
    The default mean reduction matches the BCE scaling used in the experiments.
    Results therefore concern this reduction convention rather than a bitwise
    reproduction of the reference implementation.
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, clip: float = 0.05,
                 eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.float()
        p_pos = torch.sigmoid(logits.float())
        p_neg = 1.0 - p_pos
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
        log_pos = torch.log(p_pos.clamp_min(self.eps))
        log_neg = torch.log(p_neg.clamp_min(self.eps))
        with torch.no_grad():
            pt = p_pos * y + p_neg * (1.0 - y)
            gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            weight = torch.pow(1.0 - pt, gamma)
        loss = -(y * log_pos + (1.0 - y) * log_neg) * weight
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


def build_ppmi(label_lists: Sequence[Sequence[int]], num_labels: int, eps: float = 1e-12,
               zero_diagonal: bool = True) -> np.ndarray:
    """Build a PPMI label co-occurrence matrix from TRAIN labels only.

    Complexity is O(L^2) memory. For EURLEX57K with ~4.3k labels this is ~73 MB float32.
    """
    n = len(label_lists)
    support = np.zeros(num_labels, dtype=np.float64)
    cooc = np.zeros((num_labels, num_labels), dtype=np.float32)
    for labs in label_lists:
        idx = np.asarray(sorted(set(map(int, labs))), dtype=np.int64)
        if idx.size == 0:
            continue
        support[idx] += 1.0
        cooc[np.ix_(idx, idx)] += 1.0
    p_i = support / max(1, n)
    p_ij = cooc.astype(np.float64) / max(1, n)
    denom = np.outer(p_i, p_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((p_ij + eps) / (denom + eps))
    ppmi = np.maximum(pmi, 0.0).astype(np.float32)
    if zero_diagonal:
        np.fill_diagonal(ppmi, 0.0)
    return ppmi


class WeightedAsymmetricLoss(nn.Module):
    """WASL reimplementation from Yasuda et al. (2024), equations 11--18.

    Class-balanced weights and PPMI are computed from training labels only.
    The PPMI diagonal is zero to avoid self-smoothing. This convention is
    explicit here because diagonal handling is not specified in the published
    text. Labels with zero training support have zero class-balanced weight
    and make no WASL training contribution; they remain in evaluation when
    included in the benchmark schema.
    """

    def __init__(self, train_support: Sequence[int], ppmi: np.ndarray, alpha: float, beta: float,
                 gamma_pos: float = 1.0, gamma_neg: float = 4.0, m: float = 0.05,
                 eps: float = 1e-8):
        super().__init__()
        support = np.asarray(train_support, dtype=np.float64)
        if ppmi.shape != (len(support), len(support)):
            raise ValueError("PPMI shape mismatch")
        if not (0 <= beta < 1):
            raise ValueError("beta must be in [0,1)")
        raw_w = np.zeros_like(support, dtype=np.float64)
        seen = support > 0
        raw_w[seen] = (1.0 - beta) / (1.0 - np.power(beta, support[seen]))
        if raw_w.sum() <= 0:
            raise ValueError("No seen labels for WASL")
        w_prime = raw_w * len(raw_w) / raw_w.sum()
        self.register_buffer("w_prime", torch.tensor(w_prime, dtype=torch.float32))
        self.register_buffer("gamma_neg_label", torch.tensor(w_prime * gamma_neg, dtype=torch.float32))
        self.register_buffer("ppmi", torch.tensor(ppmi, dtype=torch.float32))
        self.alpha = float(alpha)
        self.gamma_pos = float(gamma_pos)
        self.m = float(m)
        self.eps = float(eps)

    def _smoothed_targets(self, targets: torch.Tensor) -> torch.Tensor:
        y = targets.float()
        ppmi = self.ppmi.to(y.device)
        out = torch.empty_like(y)
        # Sparse row-sum implementation avoids dense BxL @ LxL for EURLEX57K.
        for i in range(y.shape[0]):
            pos = torch.nonzero(y[i] > 0.5, as_tuple=False).flatten()
            if pos.numel() == 0:
                o = torch.zeros(y.shape[1], device=y.device, dtype=y.dtype)
            else:
                o = ppmi.index_select(0, pos).sum(dim=0).to(y.dtype)
            omin, omax = o.min(), o.max()
            span = omax - omin
            norm = (o - omin) / span.clamp_min(self.eps)
            norm = torch.where(span > self.eps, norm, torch.zeros_like(norm))
            out[i] = (1.0 - self.alpha) * y[i] + self.alpha * norm
        return out

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        z = logits.float()
        y_prime = self._smoothed_targets(targets)
        p = torch.sigmoid(z)
        pm = torch.clamp(p - self.m, min=0.0, max=1.0 - self.eps)
        wp = self.w_prime.to(z.device, z.dtype)
        gn = self.gamma_neg_label.to(z.device, z.dtype)
        l_pos = torch.pow(1.0 - p, self.gamma_pos) * torch.log(p.clamp_min(self.eps))
        l_neg = torch.pow(pm, gn.unsqueeze(0)) * torch.log((1.0 - pm).clamp_min(self.eps))
        elem = -wp.unsqueeze(0) * (y_prime * l_pos + (1.0 - y_prime) * l_neg)
        return elem.mean()


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="mean")


def cardinality_penalty_loss(logits: torch.Tensor, targets: torch.Tensor, mu: float = 0.1) -> torch.Tensor:
    base = bce_loss(logits, targets)
    p = torch.sigmoid(logits.float())
    soft_card = p.sum(dim=1)
    true_card = targets.float().sum(dim=1)
    # Divide by number of labels so mu remains interpretable across label-space sizes.
    penalty = (((soft_card - true_card) ** 2) / float(logits.shape[1])).mean()
    return base + float(mu) * penalty


def cardinality_penalty_term(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-example soft-cardinality squared error, normalized by label count."""
    p = torch.sigmoid(logits.float())
    soft_card = p.sum(dim=1)
    true_card = targets.float().sum(dim=1)
    return (((soft_card - true_card) ** 2) / float(logits.shape[1])).mean()
