import torch
from ctpu.auxiliary import build_auxiliary_gradient


def test_score_auxproj_is_available_under_honest_name():
    logits = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    out = build_auxiliary_gradient(
        "score_auxproj", logits, targets, [[0]], ["x"], epoch=1, seed=7,
        k_pos=1, k_neg=1, eligible_labels=[0,1,2],
    )
    assert out.grad.shape == logits.shape
    assert torch.isfinite(out.grad).all()
