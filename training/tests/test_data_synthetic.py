import pytest
pytest.importorskip("iterstrat")
from ctpu.data import _multilabel_group_split_indices


def test_group_split_keeps_duplicate_text_together():
    rows = [
        {"id":"a1","text":"Same text", "labels":["x"]},
        {"id":"a2","text":" same   text ", "labels":["x"]},
        {"id":"b","text":"other b", "labels":["y"]},
        {"id":"c","text":"other c", "labels":["x","y"]},
        {"id":"d","text":"other d", "labels":["x"]},
        {"id":"e","text":"other e", "labels":["y"]},
    ]
    tr, va, manifest = _multilabel_group_split_indices(rows, ["x","y"], 0.33, 42)
    tr, va = set(tr.tolist()), set(va.tolist())
    assert ({0,1} <= tr) or ({0,1} <= va)
    assert not (tr & va)
    assert manifest["num_unique_text_groups"] == 5
