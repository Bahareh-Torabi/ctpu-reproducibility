import pytest
from ctpu.data import _parse_reuters_sgm_file


def _doc(newid: int, text_markup: str, topic: str = "earn", split: str = "TRAIN") -> str:
    return (
        f'<REUTERS TOPICS="YES" LEWISSPLIT="{split}" CGISPLIT="TRAINING-SET" OLDID="{newid}" NEWID="{newid}">\n'
        f'<TOPICS><D>{topic}</D></TOPICS>\n{text_markup}\n</REUTERS>'
    )


def test_reuters_parser_preserves_norm_brief_and_unproc(tmp_path):
    sgm = tmp_path / "mini.sgm"
    sgm.write_text("\n".join([
        _doc(1, "<TEXT><TITLE>Normal title</TITLE><BODY>Normal body</BODY></TEXT>"),
        _doc(2, '<TEXT TYPE="BRIEF"><TITLE>Brief story only</TITLE></TEXT>'),
        _doc(3, '<TEXT TYPE="UNPROC">Raw unprocessed story text with no BODY tag.</TEXT>'),
        _doc(4, '<TEXT><TITLE>Outside vocabulary</TITLE></TEXT>', topic="not-an-r90-topic"),
    ]), encoding="latin-1")
    rows = _parse_reuters_sgm_file(sgm, "TRAIN")
    assert [r["id"] for r in rows] == ["reuters-1", "reuters-2", "reuters-3"]
    assert rows[0]["source_text_extraction"] == "structured_title_body"
    assert rows[1]["source_text_extraction"] == "structured_title_only"
    assert rows[2]["source_text_extraction"] == "text_container_fallback"
    assert "Raw unprocessed story text" in rows[2]["text"]


def test_reuters_parser_never_silently_drops_labeled_empty_text(tmp_path):
    sgm = tmp_path / "empty.sgm"
    sgm.write_text(_doc(9, '<TEXT></TEXT>'), encoding="latin-1")
    with pytest.raises(RuntimeError, match="no recoverable TEXT payload"):
        _parse_reuters_sgm_file(sgm, "TRAIN")
