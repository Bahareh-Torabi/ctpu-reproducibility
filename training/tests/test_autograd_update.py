import torch
from ctpu.auxiliary import build_auxiliary_gradient
from ctpu.losses import bce_loss


def test_bce_plus_ctpu_score_injection_backpropagates():
    torch.manual_seed(3)
    model = torch.nn.Linear(5, 4)
    x = torch.randn(3, 5)
    y = torch.tensor([[1,0,0,0],[0,1,0,0],[1,0,1,0]], dtype=torch.float32)
    label_lists = [[0],[1],[0,2]]
    ids = ["a","b","c"]
    z = model(x)
    base = bce_loss(z, y)
    aux = build_auxiliary_gradient("ctpu", z, y, label_lists, ids, 1, 123, 3, 3)
    base.backward(retain_graph=True)
    torch.autograd.backward(z, grad_tensors=0.25 * aux.grad / 3.0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
