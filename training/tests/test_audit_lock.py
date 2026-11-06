import pytest

from pathlib import Path
from ctpu.data import _persist_development_audit, _persist_final_test_audit, _aapd_layout_manifest_path
from ctpu.utils import sha256_file


def test_development_audit_is_immutable_and_final_audit_is_separate(tmp_path: Path):
    prep = tmp_path / "prepared" / "toy"
    prep.mkdir(parents=True)
    dev = {"dataset": "toy", "n": 3, "test_accessed": False}
    _persist_development_audit(prep, dev)
    before = sha256_file(prep / "audit.json")
    _persist_final_test_audit(prep, {**dev, "test_rows": 2})
    after = sha256_file(prep / "audit.json")
    assert before == after
    assert (prep / "final_test_audit.json").exists()
    with pytest.raises(RuntimeError):
        _persist_development_audit(prep, {"dataset": "toy", "n": 4, "test_accessed": False})


def test_aapd_dev_and_final_layout_manifests_are_distinct(tmp_path: Path):
    raw = tmp_path / "raw"
    dev = raw / "extracted_dev_only"
    test = raw / "extracted_test_final_only"
    dev.mkdir(parents=True)
    test.mkdir(parents=True)
    assert _aapd_layout_manifest_path(dev) != _aapd_layout_manifest_path(test)
