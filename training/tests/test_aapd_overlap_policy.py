from ctpu.data import _decontaminate_source_validation_overlap, text_hash


def row(i, text, labels):
    return {"id": i, "text": text, "labels": labels}


def test_aapd_source_validation_overlap_removes_all_train_copies_and_preserves_validation():
    train = [
        row("t0", "same abstract", ["a"]),
        row("t1", " same   abstract ", ["a"]),
        row("t2", "unique train", ["b"]),
    ]
    val = [row("v0", "same abstract", ["a"]), row("v1", "unique val", ["b"])]
    cleaned, manifest = _decontaminate_source_validation_overlap(train, val)
    assert [r["id"] for r in cleaned] == ["t2"]
    assert [r["id"] for r in val] == ["v0", "v1"]
    assert manifest["removed_train_row_count"] == 2
    assert manifest["removed_unique_text_hash_count"] == 1
    assert manifest["remaining_train_validation_exact_duplicate_pair_count"] == 0


def test_aapd_overlap_manifest_records_label_conflict_without_leaking_validation_into_schema():
    train = [row("t0", "same", ["a"]), row("t1", "unique", ["b"])]
    val = [row("v0", "same", ["c"])]
    cleaned, manifest = _decontaminate_source_validation_overlap(train, val)
    assert [r["id"] for r in cleaned] == ["t1"]
    assert manifest["conflicting_label_hash_count"] == 1
    assert manifest["removed_train_rows"][0]["labels_consistent"] is False
    assert manifest["removed_train_rows"][0]["sha256"] == text_hash("same")
