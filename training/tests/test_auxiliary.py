import numpy as np
import torch

from ctpu.auxiliary import (
    ctpu_project,
    normmatch_gradient,
    pairwise_score_gradient,
    synthetic_mechanism_self_test,
)


def test_synthetic_mechanism():
    report = synthetic_mechanism_self_test()
    assert report["status"] == "PASS"


def test_pair_sampling_deterministic():
    z = torch.tensor([[1.0, -0.2, 0.4, -1.0]], dtype=torch.float32)
    labs = [[0, 2]]
    ids = ["a"]
    a = pairwise_score_gradient(z, labs, ids, epoch=2, seed=123, k_pos=2, k_neg=2)
    b = pairwise_score_gradient(z, labs, ids, epoch=2, seed=123, k_pos=2, k_neg=2)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_ctpu_tangent_and_normmatch():
    torch.manual_seed(5)
    z = torch.randn(4, 10)
    labs = [[0, 2], [1], [3, 4, 5], [7]]
    ids = [str(i) for i in range(4)]
    g, active, _, _ = pairwise_score_gradient(z, labs, ids, 1, 11, 3, 3)
    u, _ = ctpu_project(g, z, active)
    h = active * torch.sigmoid(z) * (1 - torch.sigmoid(z))
    assert float((h * u).sum(1).abs().max()) < 1e-6
    nm, _ = normmatch_gradient(g, u)
    assert torch.allclose(torch.linalg.vector_norm(nm, dim=1), torch.linalg.vector_norm(u, dim=1), atol=1e-6)
