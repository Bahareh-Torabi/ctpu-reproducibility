import torch
import torch.nn.functional as F

from ctpu.auxiliary import pairwise_score_gradient, ctpu_project


def test_pairwise_analytic_gradient_matches_autograd_for_same_pairs():
    # k_pos/k_neg exceed available counts, so all possible pairs are selected and
    # there is no ambiguity from the deterministic sampler.
    z = torch.tensor([[0.7, -0.4, 1.2, -1.1]], dtype=torch.float32, requires_grad=True)
    labels = [[0, 2]]
    ids = ["sample"]
    g, active, loss_mean, _ = pairwise_score_gradient(
        z.detach(), labels, ids, epoch=3, seed=99, k_pos=10, k_neg=10
    )
    pos = [0, 2]
    neg = [1, 3]
    terms = [F.softplus(z[0, n] - z[0, p]) for p in pos for n in neg]
    loss = torch.stack(terms).mean()
    (grad,) = torch.autograd.grad(loss, z)
    assert torch.allclose(g, grad, atol=1e-6, rtol=1e-6)
    assert abs(loss_mean - float(loss.detach())) < 1e-7
    assert torch.equal(active.bool(), torch.tensor([[1, 1, 1, 1]], dtype=torch.bool))


def test_pair_sampling_is_batch_order_invariant_by_sample_id():
    z = torch.tensor([
        [0.1, -0.2, 0.3, -0.4, 0.5],
        [-0.7, 0.6, -0.5, 0.4, -0.3],
    ], dtype=torch.float32)
    labels = [[0, 2], [1, 3]]
    ids = ["A", "B"]
    g1, a1, _, _ = pairwise_score_gradient(z, labels, ids, 2, 12345, 2, 2)
    order = [1, 0]
    g2, a2, _, _ = pairwise_score_gradient(z[order], [labels[i] for i in order], [ids[i] for i in order], 2, 12345, 2, 2)
    reverse = [1, 0]
    assert torch.equal(g1, g2[reverse])
    assert torch.equal(a1, a2[reverse])


def test_ctpu_is_minimum_change_tangent_projection_numerically():
    torch.manual_seed(123)
    z = torch.randn(3, 7)
    labels = [[0, 2], [1, 5], [3]]
    ids = ["x", "y", "z"]
    g, active, _, _ = pairwise_score_gradient(z, labels, ids, 1, 7, 3, 3)
    u, _ = ctpu_project(g, z, active)
    p = torch.sigmoid(z)
    h = active * p * (1 - p)
    assert float((h * u).sum(1).abs().max()) < 1e-6
    # Orthogonal projection residual must be parallel to h wherever h is nonzero.
    residual = g - u
    for i in range(z.shape[0]):
        if float((h[i] * h[i]).sum()) > 1e-12:
            # Component of residual orthogonal to h should vanish.
            coeff = (residual[i] * h[i]).sum() / (h[i] * h[i]).sum()
            err = residual[i] - coeff * h[i]
            assert float(torch.linalg.vector_norm(err)) < 1e-6


def test_pairwise_eligible_labels_exclude_zero_support_candidates():
    z = torch.tensor([[0.4, -0.3, 0.1, 0.8]], dtype=torch.float32)
    # Label 3 is deliberately positive in the provided target list but is not
    # eligible from TRAIN; label 2 is also ineligible and must not be sampled as a negative.
    labels = [[0, 3]]
    g, active, _, _ = pairwise_score_gradient(
        z, labels, ["sample"], epoch=1, seed=123,
        k_pos=3, k_neg=3, eligible_labels=[0, 1]
    )
    assert active[0, 2].item() == 0.0
    assert active[0, 3].item() == 0.0
    assert g[0, 2].item() == 0.0
    assert g[0, 3].item() == 0.0
    # The only possible eligible pair is positive 0 vs negative 1.
    assert active[0, 0].item() == 1.0
    assert active[0, 1].item() == 1.0
