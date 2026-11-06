import random
import numpy as np
import torch

from ctpu.utils import capture_rng_state, restore_rng_state, set_global_seed


def test_rng_capture_restore_repeats_next_draws():
    set_global_seed(20260805, deterministic=True)
    _ = random.random(); _ = np.random.rand(); _ = torch.rand(2)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(4))
    # Consume more RNG then restore.
    _ = random.random(); _ = np.random.rand(); _ = torch.rand(9)
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(4))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
