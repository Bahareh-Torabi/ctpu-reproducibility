from pathlib import Path
import pytest

from ctpu.multisession import delete_local_heavy_models_after_export


def test_multisession_refuses_model_deletion_before_dataset_test(tmp_path: Path):
    with pytest.raises(RuntimeError, match='before dataset final TEST completion'):
        delete_local_heavy_models_after_export(tmp_path, 'reuters21578', require_test_executed=True)
