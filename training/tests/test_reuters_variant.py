from ctpu.config import DATASET_DEFAULTS
from ctpu.data import REUTERS_SHA256


def test_reuters_raw_distribution_variant_is_explicit_and_frozen():
    cfg = DATASET_DEFAULTS["reuters21578"]
    assert cfg["expected_num_labels"] == 90
    assert cfg["expected_train_labeled"] == 7770
    assert "Distribution 1.0" in cfg["source_name"]


def test_reuters_raw_archive_content_hash_is_pinned():
    assert REUTERS_SHA256 == "3bae43c9b14e387f76a61b6d82bf98a4fb5d3ef99ef7e7075ff2ccbcf59f9d30"
