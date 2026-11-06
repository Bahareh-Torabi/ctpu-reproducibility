import csv
from pathlib import Path

import pytest

from ctpu.data import _read_aapd_split


def _write_wide(path: Path, *, bad_value=None, n_labels=54):
    labels = [f"label-{i:02d}" for i in range(n_labels)]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["abstract", *labels])
        w.writeheader()
        r1 = {"abstract": "first abstract", **{k: "0" for k in labels}}
        r1[labels[1]] = "1"
        r1[labels[7]] = "1.0"
        if bad_value is not None:
            r1[labels[3]] = bad_value
        w.writerow(r1)
        r2 = {"abstract": "second abstract", **{k: "0.0" for k in labels}}
        r2[labels[-1]] = "1"
        w.writerow(r2)
    return labels


def test_aapd_wide_54_binary_columns(tmp_path):
    p = tmp_path / "train.csv"
    labels = _write_wide(p)
    rows = _read_aapd_split(None, None, p, "train-source")
    assert len(rows) == 2
    assert rows[0]["text"] == "first abstract"
    assert rows[0]["labels"] == [labels[1], labels[7]]
    assert rows[1]["labels"] == [labels[-1]]


def test_aapd_wide_nonbinary_hard_fails(tmp_path):
    p = tmp_path / "train.csv"
    _write_wide(p, bad_value="0.25")
    with pytest.raises(RuntimeError, match="non-binary value"):
        _read_aapd_split(None, None, p, "train-source")


def test_aapd_wide_wrong_label_count_hard_fails(tmp_path):
    p = tmp_path / "train.csv"
    _write_wide(p, n_labels=53)
    with pytest.raises(RuntimeError, match="expected 54-column binary layout"):
        _read_aapd_split(None, None, p, "train-source")


def test_aapd_serialized_labels_layout_still_supported(tmp_path):
    p = tmp_path / "train.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["abstract", "labels"])
        w.writeheader()
        w.writerow({"abstract": "x", "labels": "cs.ai cs.lg"})
    rows = _read_aapd_split(None, None, p, "train-source")
    assert rows == [{"id": "aapd-train-source-0", "text": "x", "labels": ["cs.ai", "cs.lg"]}]
