from pathlib import Path

from ctpu.utils import tree_hash


def test_tree_hash_changes_with_file_content_and_is_order_stable(tmp_path: Path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "b.txt").write_text("B", encoding="utf-8")
    (d / "a.txt").write_text("A", encoding="utf-8")
    h1 = tree_hash(d)
    h2 = tree_hash(d)
    assert h1 == h2
    (d / "a.txt").write_text("A2", encoding="utf-8")
    assert tree_hash(d) != h1
