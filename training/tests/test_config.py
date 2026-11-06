import dataclasses
import pytest

from ctpu.config import PRIMARY_METHODS, FULL_METHODS, PROFILES, TrainConfig, RuntimeConfig


def test_profiles_have_core_methods_and_fixed_quick_endpoint():
    assert PRIMARY_METHODS == ("bce", "pairwise", "normmatch", "ctpu")
    for m in PRIMARY_METHODS:
        assert m in FULL_METHODS
    assert PROFILES["QUICK"].epochs_override == 4
    assert PROFILES["QUICK"].seeds == (20260805,)


def test_scientific_and_runtime_configs_are_immutable_and_separate():
    cfg = TrainConfig()
    runtime = RuntimeConfig(
        effective_batch_size=32,
        micro_batch_size=16,
        gradient_accumulation_steps=2,
        eval_batch_size=32,
        gpu_name="A100",
        gpu_total_memory_gb=40.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.epochs = 99
    with pytest.raises(dataclasses.FrozenInstanceError):
        runtime.micro_batch_size = 8
    assert "micro_batch_size" not in cfg.to_dict()
    assert runtime.to_dict()["micro_batch_size"] == 16
