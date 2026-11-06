import pytest
pytest.importorskip("transformers")
import torch

from ctpu.train import _parameter_pcgrad_step


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.0, 0.0]))


def test_two_task_parameter_pcgrad_projects_both_conflicting_gradients():
    model = Tiny()
    logits = model.w.view(1, 2)
    # Primary gradient wrt w is p=[1,0].
    scalar_loss = model.w[0]
    # Auxiliary parameter gradient is a=[-1,1].
    aux_score_grad = torch.tensor([[-1.0, 1.0]])
    diag = {"param_pcgrad_conflict": [], "param_pcgrad_primary_aux_dot": [],
            "param_pcgrad_primary_norm": [], "param_pcgrad_aux_norm": []}
    _parameter_pcgrad_step(
        model, scalar_loss, logits, aux_score_grad,
        scalar_scale=1.0, aux_scale=1.0, diagnostics=diag,
    )
    # Symmetric two-task PCGrad:
    # p' = [1,0] - (-1/2)*[-1,1] = [.5,.5]
    # a' = [-1,1] - (-1/1)*[1,0] = [0,1]
    # sum = [.5,1.5]
    assert torch.allclose(model.w.grad, torch.tensor([0.5, 1.5]), atol=1e-7)
    assert diag["param_pcgrad_conflict"] == [1.0]
