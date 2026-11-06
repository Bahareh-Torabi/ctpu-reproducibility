from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import requests
from tqdm.auto import tqdm

from .config import DATASET_DEFAULTS
from .utils import (
    atomic_write_json,
    ensure_dir,
    flatten_text_field,
    md5_file,
    read_jsonl,
    sha256_file,
    stable_int_seed,
    text_hash,
    write_jsonl,
)

REUTERS_URL = "https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz"
REUTERS_SHA256 = "3bae43c9b14e387f76a61b6d82bf98a4fb5d3ef99ef7e7075ff2ccbcf59f9d30"
AAPD_URL = "https://zenodo.org/records/6344750/files/AAPD.zip?download=1"
AAPD_MD5 = "e1419b5d03be572115287fcb80577c48"
EURLEX_OFFICIAL_URL = "http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip"
EURLEX_FALLBACK_URL = "https://archive.org/download/EURLEX57K/dataset.zip"
EURLEX_SHA256 = "dc6830ce8ea49aea049989c4a8253b12d0fb3665fc336d830cb07b9bd28c0f92"

REUTERS90 = [
    "acq", "alum", "barley", "bop", "carcass", "castor-oil", "cocoa", "coconut",
    "coconut-oil", "coffee", "copper", "copra-cake", "corn", "cotton", "cotton-oil",
    "cpi", "cpu", "crude", "dfl", "dlr", "dmk", "earn", "fuel", "gas", "gnp", "gold",
    "grain", "groundnut", "groundnut-oil", "heat", "hog", "housing", "income",
    "instal-debt", "interest", "ipi", "iron-steel", "jet", "jobs", "l-cattle", "lead",
    "lei", "lin-oil", "livestock", "lumber", "meal-feed", "money-fx", "money-supply",
    "naphtha", "nat-gas", "nickel", "nkr", "nzdlr", "oat", "oilseed", "orange",
    "palladium", "palm-oil", "palmkernel", "pet-chem", "platinum", "potato", "propane",
    "rand", "rape-oil", "rapeseed", "reserves", "retail", "rice", "rubber", "rye", "ship",
    "silver", "sorghum", "soy-meal", "soy-oil", "soybean", "strategic-metal", "sugar",
    "sun-meal", "sun-oil", "sunseed", "tea", "tin", "trade", "veg-oil", "wheat", "wpi",
    "yen", "zinc",
]


def download_file(url: str, dest: str | Path, expected_sha256: str | None = None,
                  expected_md5: str | None = None, chunk_size: int = 1 << 20) -> Path:
    dest = Path(dest)
    ensure_dir(dest.parent)
    if dest.exists():
        ok = True
        if expected_sha256:
            ok &= sha256_file(dest) == expected_sha256
        if expected_md5:
            ok &= md5_file(dest) == expected_md5
        if ok:
            return dest
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with requests.get(url, stream=True, timeout=(30, 300), allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"download {dest.name}") as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp, dest)
    if expected_sha256 and sha256_file(dest) != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {dest}")
    if expected_md5 and md5_file(dest) != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {dest}")
    return dest


def download_from_mirrors(urls: Sequence[str], dest: str | Path, expected_sha256: str | None = None,
                          expected_md5: str | None = None) -> tuple[Path, str]:
    """Try ordered mirrors, require pinned content, and persist the actual source URL.

    A verified cache is safe to reuse, but publication provenance should still retain
    the URL that originally produced the cached bytes. A small sidecar therefore
    survives notebook/runtime restarts.
    """
    errors = []
    dest = Path(dest)
    sidecar = dest.with_name(dest.name + ".source.json")
    if dest.exists():
        ok = ((expected_sha256 is None or sha256_file(dest) == expected_sha256) and
              (expected_md5 is None or md5_file(dest) == expected_md5))
        if ok:
            source = "cached_verified_source_unknown"
            if sidecar.exists():
                try:
                    obj = json.loads(sidecar.read_text(encoding="utf-8"))
                    source = str(obj.get("source_url") or source)
                except Exception:
                    pass
            return dest, source
        dest.unlink()
        if sidecar.exists():
            sidecar.unlink()
    for url in urls:
        try:
            path = download_file(url, dest, expected_sha256=expected_sha256, expected_md5=expected_md5)
            atomic_write_json(sidecar, {
                "source_url": url,
                "sha256": sha256_file(path),
                "md5": md5_file(path),
            })
            return path, url
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
            if dest.exists():
                dest.unlink()
            if sidecar.exists():
                sidecar.unlink()
            part = dest.with_suffix(dest.suffix + ".part")
            if part.exists():
                part.unlink()
    raise RuntimeError("All download mirrors failed or produced the wrong pinned hash:\n" + "\n".join(errors))


def _safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise RuntimeError(f"Unsafe archive member: {name}")
    return target


def safe_extract_tar(path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with tarfile.open(path, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            _safe_target(out_dir, m.name)
            if m.issym() or m.islnk() or m.isdev():
                raise RuntimeError(f"Refusing non-regular TAR member: {m.name}")
            members.append(m)
        tf.extractall(out_dir, members=members)


def safe_extract_zip(path: Path, out_dir: Path, member_filter=None) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            if member_filter is not None and not member_filter(info.filename):
                continue
            _safe_target(out_dir, info.filename)
            # Unix symlinks are encoded in the upper mode bits of external_attr.
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RuntimeError(f"Refusing ZIP symlink member: {info.filename}")
            zf.extract(info, out_dir)


def _clean_sgml_text(x: str) -> str:
    x = re.sub(r"<[^>]+>", " ", x, flags=re.S)
    x = html.unescape(x)
    return " ".join(x.split())


def _extract_tag(block: str, tag: str) -> str:
    m = re.search(fr"<{tag}[^>]*>(.*?)</{tag}>", block, flags=re.I | re.S)
    return m.group(1) if m else ""


def _extract_reuters_text(body: str) -> tuple[str, str]:
    """Recover Reuters story text without leaking metadata into the text field.

    Reuters-21578 TEXT may be NORM, BRIEF, or UNPROC. NORM stories usually
    expose TITLE/BODY, BRIEF stories often contain TITLE only, while UNPROC
    stories may contain free text directly inside TEXT with no BODY tag.

    Crucially, the fallback is restricted to the *inside of the TEXT element*.
    Falling back to the full REUTERS body is forbidden because that body also
    contains metadata such as TOPICS/PLACES; an empty TEXT element must not be
    converted into a fake document whose content is merely a topic label.
    """
    m = re.search(r"<TEXT\b[^>]*>(.*?)</TEXT>", body, flags=re.I | re.S)
    if not m:
        return "", "missing_text_container"
    text_container = m.group(1)
    scope = text_container
    title = _clean_sgml_text(_extract_tag(scope, "TITLE"))
    text_body = _clean_sgml_text(_extract_tag(scope, "BODY"))
    structured = (title + "\n" + text_body).strip()
    if structured:
        if title and text_body:
            return structured, "structured_title_body"
        if title:
            return structured, "structured_title_only"
        return structured, "structured_body_only"

    fallback = _clean_sgml_text(text_container)
    if fallback:
        return fallback, "text_container_fallback"
    return "", "unrecoverable"


def _parse_reuters_sgm_file(path: Path, wanted_split: str) -> list[dict]:
    raw = path.read_text(encoding="latin-1", errors="ignore")
    blocks = re.findall(r"<REUTERS\b(.*?)>(.*?)</REUTERS>", raw, flags=re.I | re.S)
    rows = []
    allowed = set(REUTERS90)
    for attrs, body in blocks:
        def attr(name: str) -> str:
            m = re.search(fr'{name}="([^"]*)"', attrs, flags=re.I)
            return m.group(1) if m else ""

        if attr("LEWISSPLIT").upper() != wanted_split.upper():
            continue
        if attr("TOPICS").upper() != "YES":
            continue
        topics_raw = _extract_tag(body, "TOPICS")
        labels = re.findall(r"<D>(.*?)</D>", topics_raw, flags=re.I | re.S)
        labels = [html.unescape(x.strip()) for x in labels if html.unescape(x.strip()) in allowed]
        if not labels:
            continue

        text, extraction_mode = _extract_reuters_text(body)
        if not text:
            raise RuntimeError(
                f"Reuters R(90) document NEWID={attr('NEWID')} in {wanted_split} has valid R90 labels "
                "but no recoverable TEXT payload. Refusing to silently delete a benchmark row."
            )
        rows.append({
            "id": f"reuters-{attr('NEWID')}",
            "text": text,
            "labels": sorted(set(labels)),
            "source_text_extraction": extraction_mode,
        })
    return rows

def _read_all_reuters_split(extract_dir: Path, split: str) -> list[dict]:
    rows = []
    sgm_files = sorted(list(extract_dir.rglob("*.sgm")) + list(extract_dir.rglob("*.SGM")))
    if not sgm_files:
        raise FileNotFoundError(f"No Reuters SGML files found under {extract_dir}")
    for path in sgm_files:
        rows.extend(_parse_reuters_sgm_file(path, split))
    return rows


def _multilabel_split_indices(label_lists: Sequence[Sequence[str]], label_names: Sequence[str],
                              validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    y = np.zeros((len(label_lists), len(label_names)), dtype=np.uint8)
    for i, labs in enumerate(label_lists):
        for lab in labs:
            if lab in label_to_idx:
                y[i, label_to_idx[lab]] = 1
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    tr, va = next(splitter.split(np.zeros(len(y)), y))
    tr, va = list(map(int, tr)), list(map(int, va))
    tr_set = set(tr)
    va_set = set(va)
    moved = []
    support_tr = y[tr].sum(axis=0)
    support_all = y.sum(axis=0)
    for j in np.where((support_all > 0) & (support_tr == 0))[0]:
        candidates = [idx for idx in va if y[idx, j] == 1]
        if not candidates:
            raise RuntimeError(f"Could not preserve training support for label {label_names[j]}")
        idx = candidates[0]
        va_set.remove(idx)
        tr_set.add(idx)
        moved.append({"sample_index": idx, "label": label_names[j]})
    tr = np.array(sorted(tr_set), dtype=np.int64)
    va = np.array(sorted(va_set), dtype=np.int64)
    manifest = {
        "method": "MultilabelStratifiedShuffleSplit + support-preserving repair",
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "actual_validation_fraction": len(va) / len(y),
        "moved_validation_to_train": moved,
    }
    return tr, va, manifest



def _multilabel_group_split_indices(rows: Sequence[dict], label_names: Sequence[str],
                                    validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Group-aware multilabel split using normalized-text hashes as groups.

    This is used only when the source dataset does not provide a validation split.
    Exact duplicate texts are never allowed to straddle train/validation.  Group labels
    are the union of all labels attached to any row in the group, then iterative
    stratification is applied at group level.  Any label that would disappear from
    training triggers a whole-group repair from validation back to training.
    """
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    groups: Dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(text_hash(row["text"]), []).append(i)
    group_keys = sorted(groups)
    gy = np.zeros((len(group_keys), len(label_names)), dtype=np.uint8)
    for gi, key in enumerate(group_keys):
        labs = set()
        for ridx in groups[key]:
            labs.update(rows[ridx]["labels"])
        for lab in labs:
            if lab in label_to_idx:
                gy[gi, label_to_idx[lab]] = 1
    # Load the stratifier only for split generation so parser tests can run
    # without the optional stratification dependency.
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    gtr, gva = next(splitter.split(np.zeros(len(group_keys)), gy))
    gtr_set, gva_set = set(map(int, gtr)), set(map(int, gva))
    moved = []
    support_tr = gy[list(gtr_set)].sum(axis=0) if gtr_set else np.zeros(len(label_names))
    support_all = gy.sum(axis=0)
    for j in np.where((support_all > 0) & (support_tr == 0))[0]:
        candidates = [gi for gi in sorted(gva_set) if gy[gi, j] == 1]
        if not candidates:
            raise RuntimeError(f"Could not preserve training support for label {label_names[j]}")
        gi = candidates[0]
        gva_set.remove(gi)
        gtr_set.add(gi)
        moved.append({"group_sha256": group_keys[gi], "label": label_names[j], "rows": groups[group_keys[gi]]})
        support_tr += gy[gi]
    tr = np.asarray(sorted(i for gi in gtr_set for i in groups[group_keys[gi]]), dtype=np.int64)
    va = np.asarray(sorted(i for gi in gva_set for i in groups[group_keys[gi]]), dtype=np.int64)
    # Hard invariant: no normalized-text hash crosses the split.
    tr_hash = {text_hash(rows[i]["text"]) for i in tr}
    va_hash = {text_hash(rows[i]["text"]) for i in va}
    if tr_hash & va_hash:
        raise AssertionError("Group-aware split failed: duplicate text crosses train/validation")
    return tr, va, {
        "method": "group-aware MultilabelStratifiedShuffleSplit + support-preserving whole-group repair",
        "group_key": "sha256(normalized_text)",
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "actual_validation_fraction": len(va) / len(rows),
        "num_rows": len(rows),
        "num_unique_text_groups": len(group_keys),
        "duplicate_row_count": len(rows) - len(group_keys),
        "moved_validation_groups_to_train": moved,
    }

def _dataset_audit(rows_train: list[dict], rows_val: list[dict], label_names: list[str]) -> dict:
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    train_support = np.zeros(len(label_names), dtype=np.int64)
    val_support = np.zeros(len(label_names), dtype=np.int64)
    for row in rows_train:
        for l in row["labels"]:
            if l in label_to_idx:
                train_support[label_to_idx[l]] += 1
    for row in rows_val:
        for l in row["labels"]:
            if l in label_to_idx:
                val_support[label_to_idx[l]] += 1
    train_hashes = {text_hash(r["text"]): r["id"] for r in rows_train}
    cross_dups = []
    for r in rows_val:
        h = text_hash(r["text"])
        if h in train_hashes:
            cross_dups.append({"train_id": train_hashes[h], "validation_id": r["id"], "sha256": h})
    return {
        "train_rows": len(rows_train),
        "validation_rows": len(rows_val),
        "num_labels": len(label_names),
        "train_mean_cardinality": float(np.mean([len(r["labels"]) for r in rows_train])) if rows_train else 0.0,
        "validation_mean_cardinality": float(np.mean([len(r["labels"]) for r in rows_val])) if rows_val else 0.0,
        "train_support_min": int(train_support.min()) if len(train_support) else 0,
        "train_support_max": int(train_support.max()) if len(train_support) else 0,
        "zero_train_support_labels": [label_names[i] for i in np.where(train_support == 0)[0]],
        "zero_validation_support_labels": [label_names[i] for i in np.where(val_support == 0)[0]],
        "train_validation_exact_text_duplicates": cross_dups,
        "train_support": train_support.tolist(),
        "validation_support": val_support.tolist(),
    }



def _cross_split_exact_duplicate_pairs(left_rows: list[dict], right_rows: list[dict],
                                      left_name: str, right_name: str) -> list[dict]:
    """Return exact normalized-text duplicate pairs across two frozen splits."""
    left_by_hash: dict[str, list[str]] = {}
    for r in left_rows:
        left_by_hash.setdefault(text_hash(r["text"]), []).append(str(r["id"]))
    pairs: list[dict] = []
    for r in right_rows:
        h = text_hash(r["text"])
        for lid in left_by_hash.get(h, []):
            pairs.append({f"{left_name}_id": lid, f"{right_name}_id": str(r["id"]), "sha256": h})
    return pairs


def _attach_final_duplicate_audit(final_audit: dict, train_rows: list[dict],
                                  val_rows: list[dict], test_rows: list[dict]) -> None:
    train_test = _cross_split_exact_duplicate_pairs(train_rows, test_rows, "train", "test")
    val_test = _cross_split_exact_duplicate_pairs(val_rows, test_rows, "validation", "test")
    final_audit["train_test_exact_text_duplicates"] = train_test
    final_audit["validation_test_exact_text_duplicates"] = val_test
    final_audit["train_test_exact_text_duplicate_pair_count"] = len(train_test)
    final_audit["validation_test_exact_text_duplicate_pair_count"] = len(val_test)
    final_audit["test_ids_with_any_development_exact_duplicate"] = sorted({
        str(x["test_id"]) for x in train_test + val_test
    })
    final_audit["duplicate_clean_test_sensitivity_predeclared"] = True

def _persist_development_audit(prep_dir: Path, audit: dict) -> None:
    """Write the immutable development audit, or verify it is byte-for-byte equivalent.

    Final-test preparation must never mutate this file because its SHA256 is part
    of FINAL_PROTOCOL_LOCK and is also used to permit safe final-campaign resume.
    """
    audit = dict(audit)
    audit["test_accessed"] = False
    path = prep_dir / "audit.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != audit:
            raise RuntimeError(
                f"Development audit changed for {prep_dir.name}. "
                "Refusing to overwrite a locked provenance record; use a fresh experiment root if the data pipeline changed."
            )
        return
    atomic_write_json(path, audit)


def _persist_final_test_audit(prep_dir: Path, audit: dict) -> None:
    """Persist final-only test provenance separately from immutable development audit."""
    obj = dict(audit)
    obj["test_accessed"] = True
    atomic_write_json(prep_dir / "final_test_audit.json", obj)


def prepare_reuters(root: str | Path, validation_seed: int = 20260805, include_test: bool = False) -> dict:
    root = Path(root)
    raw_dir = ensure_dir(root / "raw" / "reuters21578")
    prep_dir = ensure_dir(root / "prepared" / "reuters21578")
    archive = download_file(REUTERS_URL, raw_dir / "reuters21578.tar.gz", expected_sha256=REUTERS_SHA256)
    extract_dir = raw_dir / "extracted"
    if not extract_dir.exists() or not any(extract_dir.rglob("*.sgm")):
        safe_extract_tar(archive, extract_dir)

    train_all = _read_all_reuters_split(extract_dir, "TRAIN")
    train_extraction_counts = dict(sorted(Counter(
        r.get("source_text_extraction", "unknown") for r in train_all
    ).items()))
    if len(train_all) != 7770:
        raise RuntimeError(
            f"Reuters Distribution 1.0 raw ModApte R(90) TRAIN audit failed: expected 7770, found {len(train_all)}; "
            f"text_extraction_counts={train_extraction_counts}. "
            "Do not train until parser/split is audited."
        )
    tr_idx, va_idx, split_manifest = _multilabel_group_split_indices(train_all, REUTERS90, 0.10, validation_seed)
    train_rows = [train_all[i] for i in tr_idx]
    val_rows = [train_all[i] for i in va_idx]
    write_jsonl(prep_dir / "train.jsonl", train_rows)
    write_jsonl(prep_dir / "validation.jsonl", val_rows)
    atomic_write_json(prep_dir / "labels.json", REUTERS90)
    audit = _dataset_audit(train_rows, val_rows, REUTERS90)
    audit.update({
        "dataset": "reuters21578",
        "source_url": REUTERS_URL,
        "raw_sha256": sha256_file(archive),
        "source_sha256_expected": REUTERS_SHA256,
        "benchmark_note": ("Reuters-21578 Distribution 1.0 raw SGML, ModApte-derived R(90) labeled subset: "
                           "7,770 source TRAIN and 3,019 source TEST rows. This raw reconstruction is "
                           "explicitly distinguished from the 10,788-document NLTK ApteMod corpus."),
        "source_variant": "raw_distribution_1_0_modapte_r90_7770_3019",
        "source_train_r90_rows_before_validation": len(train_all),
        "source_train_text_extraction_counts": train_extraction_counts,
        "split_manifest": split_manifest,
        "test_accessed": False,
    })
    _persist_development_audit(prep_dir, audit)
    if include_test:
        test_rows = _read_all_reuters_split(extract_dir, "TEST")
        test_extraction_counts = dict(sorted(Counter(
            r.get("source_text_extraction", "unknown") for r in test_rows
        ).items()))
        if len(test_rows) != 3019:
            raise RuntimeError(
                f"Reuters R(90) TEST audit failed: expected 3019, found {len(test_rows)}; "
                f"text_extraction_counts={test_extraction_counts}"
            )
        write_jsonl(prep_dir / "test.jsonl", test_rows)
        final_audit = dict(audit)
        final_audit["test_rows"] = len(test_rows)
        final_audit["source_test_text_extraction_counts"] = test_extraction_counts
        _attach_final_duplicate_audit(final_audit, train_rows, val_rows, test_rows)
        _persist_final_test_audit(prep_dir, final_audit)
        return {**final_audit, "test_accessed": True}
    return audit


def _split_alias_score(name: str, aliases: Sequence[str]) -> bool:
    low = name.lower()
    return any(re.search(fr"(^|[^a-z]){re.escape(a)}([^a-z]|$)", low) for a in aliases)


def _discover_aapd_pair(extract_dir: Path, split_aliases: Sequence[str]) -> tuple[Path | None, Path | None, Path | None]:
    """Discover an unambiguous AAPD source layout or hard-fail.

    Silent "last match wins" discovery is intentionally forbidden.
    """
    files = [p for p in extract_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    relevant = [p for p in files if _split_alias_score(p.name, split_aliases)]
    text_candidates, label_candidates, table_candidates = [], [], []
    for fp in relevant:
        low = fp.name.lower()
        if fp.suffix.lower() in {".csv", ".tsv"}:
            table_candidates.append(fp)
        if any(k in low for k in ("text", "src", "abstract")):
            text_candidates.append(fp)
        if any(k in low for k in ("label", "tgt", "target")):
            label_candidates.append(fp)

    # Prefer a unique text+label pair. A unique table is used only when no pair exists.
    if text_candidates or label_candidates:
        if len(text_candidates) != 1 or len(label_candidates) != 1:
            raise RuntimeError(
                "Ambiguous AAPD split discovery. "
                f"text={[str(x.relative_to(extract_dir)) for x in text_candidates]}, "
                f"labels={[str(x.relative_to(extract_dir)) for x in label_candidates]}, "
                f"tables={[str(x.relative_to(extract_dir)) for x in table_candidates]}"
            )
        return text_candidates[0], label_candidates[0], None
    if len(table_candidates) > 1:
        raise RuntimeError(
            "Ambiguous AAPD table discovery: " +
            str([str(x.relative_to(extract_dir)) for x in table_candidates])
        )
    if len(table_candidates) == 1:
        return None, None, table_candidates[0]
    return None, None, None



def _aapd_layout_manifest_path(extract_dir: Path) -> Path:
    # Development and final-test extraction trees are intentionally separate.
    # Their source-layout manifests must also be separate so opening the final
    # test archive cannot mutate the development provenance record.
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", extract_dir.name)
    return extract_dir.parent / f"source_layout__{safe_name}.json"


def _resolve_aapd_pair_locked(extract_dir: Path, split_name: str, aliases: Sequence[str],
                              required: bool = True) -> tuple[Path | None, Path | None, Path | None]:
    """Discover once, then persist exact relative source paths for reproducibility.

    Validation is optional for AAPD because some public layouts contain only
    train/test; in that case the absence itself is persisted and validation is
    carved from source TRAIN. Train and final TEST remain required.
    """
    manifest_path = _aapd_layout_manifest_path(extract_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"splits": {}}
    if split_name in manifest.get("splits", {}):
        spec = manifest["splits"][split_name]
        def p(key):
            rel = spec.get(key)
            return (extract_dir / rel) if rel else None
        trio = (p("text"), p("labels"), p("table"))
        if not all(x is None or x.exists() for x in trio):
            raise RuntimeError(f"AAPD locked source layout no longer matches extracted archive for split={split_name}")
        for key, fp in zip(("text", "labels", "table"), trio):
            expected = spec.get(f"{key}_sha256")
            if fp is not None and expected and sha256_file(fp) != expected:
                raise RuntimeError(f"AAPD locked source file hash changed: split={split_name}, file={fp}")
        return trio
    trio = _discover_aapd_pair(extract_dir, aliases)
    if not any(trio) and required:
        raise RuntimeError(f"Could not discover AAPD source layout for required split={split_name}")
    manifest.setdefault("splits", {})[split_name] = {
        "text": str(trio[0].relative_to(extract_dir)) if trio[0] else None,
        "labels": str(trio[1].relative_to(extract_dir)) if trio[1] else None,
        "table": str(trio[2].relative_to(extract_dir)) if trio[2] else None,
        "absent": not any(trio),
        "text_sha256": sha256_file(trio[0]) if trio[0] else None,
        "labels_sha256": sha256_file(trio[1]) if trio[1] else None,
        "table_sha256": sha256_file(trio[2]) if trio[2] else None,
    }
    manifest["archive_files"] = sorted(str(p.relative_to(extract_dir)) for p in extract_dir.rglob("*") if p.is_file())
    atomic_write_json(manifest_path, manifest)
    return trio


def _parse_label_line(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    # JSON-style list.
    if line.startswith("["):
        try:
            arr = json.loads(line)
            return [str(x).strip() for x in arr]
        except Exception:
            pass
    toks = [x for x in re.split(r"[\s,;]+", line) if x]
    # Dense binary vector -> indices of positive entries.
    if len(toks) >= 20 and set(toks).issubset({"0", "1", "0.0", "1.0"}):
        return [str(i) for i, x in enumerate(toks) if float(x) > 0.5]
    return toks



def _decontaminate_source_validation_overlap(train_rows: list[dict], val_rows: list[dict]) -> tuple[list[dict], dict]:
    """Remove TRAIN rows whose normalized text appears in a source-provided validation split.

    Policy rationale: preserve the publisher/source validation split exactly and prevent
    exact-text leakage from TRAIN into validation.  Removal is hash-based on the same
    normalized-text SHA256 used by the global audit.  All affected TRAIN rows are
    removed, including multiple TRAIN copies of the same validation text.
    """
    val_by_hash: dict[str, list[dict]] = {}
    for r in val_rows:
        val_by_hash.setdefault(text_hash(r["text"]), []).append(r)

    kept: list[dict] = []
    removed: list[dict] = []
    conflicting_hashes: list[dict] = []
    seen_conflict: set[str] = set()
    for r in train_rows:
        h = text_hash(r["text"])
        matches = val_by_hash.get(h)
        if not matches:
            kept.append(r)
            continue
        train_labels = sorted(map(str, r.get("labels", [])))
        val_label_sets = [sorted(map(str, v.get("labels", []))) for v in matches]
        consistent = all(vls == train_labels for vls in val_label_sets)
        removed.append({
            "train_id": str(r["id"]),
            "sha256": h,
            "train_labels": train_labels,
            "validation_ids": [str(v["id"]) for v in matches],
            "validation_label_sets": val_label_sets,
            "labels_consistent": bool(consistent),
        })
        if not consistent and h not in seen_conflict:
            seen_conflict.add(h)
            conflicting_hashes.append({
                "sha256": h,
                "train_id": str(r["id"]),
                "validation_ids": [str(v["id"]) for v in matches],
            })

    remaining_pairs = _cross_split_exact_duplicate_pairs(kept, val_rows, "train", "validation")
    if remaining_pairs:
        raise AssertionError("AAPD overlap decontamination failed: exact TRAIN-validation duplicates remain")

    manifest = {
        "policy": "preserve source validation; remove every TRAIN row whose normalized-text SHA256 occurs in validation",
        "normalization_and_hash": "sha256(normalized_text) via ctpu.data.text_hash",
        "source_train_rows_before_decontamination": len(train_rows),
        "validation_rows_preserved": len(val_rows),
        "removed_train_row_count": len(removed),
        "removed_unique_text_hash_count": len({x["sha256"] for x in removed}),
        "removed_train_rows": removed,
        "conflicting_label_hash_count": len(conflicting_hashes),
        "conflicting_label_hashes": conflicting_hashes,
        "train_rows_after_decontamination": len(kept),
        "remaining_train_validation_exact_duplicate_pair_count": 0,
    }
    return kept, manifest

def _read_aapd_split(text_file: Path | None, label_file: Path | None, table_file: Path | None,
                     split_name: str) -> list[dict]:
    if text_file and label_file:
        texts = text_file.read_text(encoding="utf-8", errors="replace").splitlines()
        labels = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(texts) != len(labels):
            raise RuntimeError(f"AAPD {split_name}: text/label line-count mismatch {len(texts)} != {len(labels)}")
        return [
            {"id": f"aapd-{split_name}-{i}", "text": t.strip(), "labels": _parse_label_line(l)}
            for i, (t, l) in enumerate(zip(texts, labels)) if t.strip()
        ]
    if table_file:
        delim = "\t" if table_file.suffix.lower() == ".tsv" else ","
        rows = []
        with open(table_file, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            fieldnames = list(reader.fieldnames or [])
            text_key = next((orig for orig in fieldnames if orig.lower() in {"text", "abstract", "sentence", "content"}), None)
            label_key = next((orig for orig in fieldnames if orig.lower() in {"labels", "label", "targets", "target"}), None)
            if not text_key:
                raise RuntimeError(f"AAPD table {table_file} lacks a recognized text column: {reader.fieldnames}")

            # Public AAPD layouts occur in two encodings:
            #   (1) one text column + one serialized labels/targets column; or
            #   (2) one `abstract` column + exactly 54 binary one-hot label columns.
            # The second form is the current Zenodo AAPD.zip layout.  Treat the
            # header names themselves as the canonical label strings and hard-fail
            # if the non-text columns are not exactly the expected 54 binary labels.
            wide_label_keys = [k for k in fieldnames if k != text_key] if label_key is None else []
            if label_key is None and len(wide_label_keys) != 54:
                raise RuntimeError(
                    f"AAPD table {table_file} has no serialized label column and does not match the "
                    f"expected 54-column binary layout: text={text_key!r}, other_columns={len(wide_label_keys)}, "
                    f"fields={reader.fieldnames}"
                )

            for i, row in enumerate(reader):
                text = str(row.get(text_key, "")).strip()
                if not text:
                    continue
                if label_key is not None:
                    labels = _parse_label_line(str(row.get(label_key, "")))
                else:
                    labels = []
                    for key in wide_label_keys:
                        raw = str(row.get(key, "")).strip()
                        if raw not in {"0", "1", "0.0", "1.0"}:
                            raise RuntimeError(
                                f"AAPD wide table {table_file} contains non-binary value {raw!r} "
                                f"at row={i}, column={key!r}"
                            )
                        if float(raw) > 0.5:
                            labels.append(str(key))
                rows.append({"id": f"aapd-{split_name}-{i}", "text": text, "labels": labels})
        return rows
    raise RuntimeError(f"Could not discover AAPD files for split {split_name}")


def _sort_labels(labels: Iterable[str]) -> list[str]:
    labels = sorted(set(map(str, labels)))
    if labels and all(re.fullmatch(r"-?\d+", x) for x in labels):
        return sorted(labels, key=lambda x: int(x))
    return labels


def prepare_aapd(root: str | Path, validation_seed: int = 20260805, include_test: bool = False) -> dict:
    root = Path(root)
    raw_dir = ensure_dir(root / "raw" / "aapd")
    prep_dir = ensure_dir(root / "prepared" / "aapd")
    archive = download_file(AAPD_URL, raw_dir / "AAPD.zip", expected_md5=AAPD_MD5)
    # During development, extract only non-test members. The raw archive necessarily
    # contains test bytes, but test files are not instantiated or parsed.
    extract_dir = raw_dir / "extracted_dev_only"
    marker = extract_dir / ".extracted_ok"
    if not marker.exists():
        safe_extract_zip(
            archive, extract_dir,
            member_filter=lambda name: not bool(re.search(r"(^|[/_.-])test([/_.-]|$)", name.lower().replace("\\", "/")))
        )
        marker.write_text("ok\n", encoding="utf-8")

    train_pair = _resolve_aapd_pair_locked(extract_dir, "train", ("train", "training"))
    val_pair = _resolve_aapd_pair_locked(extract_dir, "validation", ("valid", "validation", "dev"), required=False)
    train_source = _read_aapd_split(*train_pair, split_name="train-source")
    if any(val_pair):
        # If a source validation split is discovered, any parse error is fatal.
        # We only generate validation from TRAIN when no validation source exists.
        val_source = _read_aapd_split(*val_pair, split_name="validation-source")
    else:
        val_source = None

    if val_source:
        # The public source split contains a small number of exact normalized-text
        # overlaps between TRAIN and validation.  Preserve validation exactly and
        # decontaminate TRAIN before any schema construction/tokenization/training.
        val_rows = val_source
        train_rows, overlap_manifest = _decontaminate_source_validation_overlap(train_source, val_rows)
        split_manifest = {
            "method": "source-provided validation split + precommitted exact-overlap decontamination of TRAIN",
            "validation_seed": None,
            "train_validation_overlap_policy": overlap_manifest,
        }
    else:
        labels_provisional = _sort_labels(l for r in train_source for l in r["labels"])
        val_n = 2000 if len(train_source) >= 40000 else max(1, round(0.10 * len(train_source)))
        val_frac = val_n / len(train_source)
        tr_idx, va_idx, split_manifest = _multilabel_group_split_indices(train_source, labels_provisional, val_frac, validation_seed)
        train_rows = [train_source[i] for i in tr_idx]
        val_rows = [train_source[i] for i in va_idx]
        split_manifest["requested_validation_rows"] = val_n

    # Leakage-resistant classifier schema: validation annotations must not create output labels.
    # AAPD is expected to have all 54 labels represented in TRAIN. If not, stop rather than
    # silently deriving the classifier dimensionality from validation/test.
    label_names = _sort_labels(l for r in train_rows for l in r["labels"])
    if len(label_names) != 54:
        listing = sorted(str(p.relative_to(extract_dir)) for p in extract_dir.rglob("*") if p.is_file())[:100]
        raise RuntimeError(
            f"AAPD TRAIN-seen label audit failed: expected 54 labels in TRAIN, found {len(label_names)}. "
            f"Archive files (first 100): {listing}. Do not train until source/split is audited."
        )
    unknown_val = sorted(set(l for r in val_rows for l in r["labels"]) - set(label_names))
    if unknown_val:
        raise RuntimeError(
            f"AAPD validation contains labels absent from TRAIN: {unknown_val}. "
            "Validation is not allowed to define the classifier schema."
        )
    write_jsonl(prep_dir / "train.jsonl", train_rows)
    write_jsonl(prep_dir / "validation.jsonl", val_rows)
    atomic_write_json(prep_dir / "labels.json", label_names)
    audit = _dataset_audit(train_rows, val_rows, label_names)
    audit.update({
        "dataset": "aapd",
        "source_url": AAPD_URL,
        "source_doi": "10.5281/zenodo.6344750",
        "raw_md5": md5_file(archive),
        "raw_sha256": sha256_file(archive),
        "source_train_rows_before_optional_split": len(train_source),
        "source_validation_rows": len(val_source) if val_source else 0,
        "source_validation_preserved_exactly": bool(val_source),
        "train_validation_overlap_policy_applied": bool(val_source),
        "train_validation_overlap_removed_train_rows": (
            split_manifest.get("train_validation_overlap_policy", {}).get("removed_train_row_count", 0)
        ),
        "train_validation_overlap_removed_unique_hashes": (
            split_manifest.get("train_validation_overlap_policy", {}).get("removed_unique_text_hash_count", 0)
        ),
        "train_validation_overlap_conflicting_label_hashes": (
            split_manifest.get("train_validation_overlap_policy", {}).get("conflicting_label_hash_count", 0)
        ),
        "split_manifest": split_manifest,
        "label_schema_mode": "TRAIN-seen labels only; validation/test do not define classifier schema",
        "validation_labels_absent_from_train": unknown_val,
        "known_literature_total_counts": [55840, 55837],
        "test_accessed": False,
        "archive_file_listing": sorted(str(p.relative_to(extract_dir)) for p in extract_dir.rglob("*") if p.is_file()),
        "development_source_layout_manifest": str(_aapd_layout_manifest_path(extract_dir)),
    })
    _persist_development_audit(prep_dir, audit)
    if include_test:
        test_extract = raw_dir / "extracted_test_final_only"
        marker_test = test_extract / ".extracted_ok"
        if not marker_test.exists():
            safe_extract_zip(
                archive, test_extract,
                member_filter=lambda name: bool(re.search(r"(^|[/_.-])test([/_.-]|$)", name.lower().replace("\\", "/")))
            )
            marker_test.write_text("ok\n", encoding="utf-8")
        test_pair = _resolve_aapd_pair_locked(test_extract, "test", ("test", "testing"))
        test_rows = _read_aapd_split(*test_pair, split_name="test")
        unknown = sorted(set(l for r in test_rows for l in r["labels"]) - set(label_names))
        if unknown:
            raise RuntimeError(f"AAPD test contains labels absent from TRAIN schema: {unknown}")
        write_jsonl(prep_dir / "test.jsonl", test_rows)
        final_audit = dict(audit)
        final_audit["test_rows"] = len(test_rows)
        final_audit["total_rows_after_test"] = len(train_rows) + len(val_rows) + len(test_rows)
        final_audit["test_source_layout_manifest"] = str(_aapd_layout_manifest_path(test_extract))
        _attach_final_duplicate_audit(final_audit, train_rows, val_rows, test_rows)
        _persist_final_test_audit(prep_dir, final_audit)
        return {**final_audit, "test_accessed": True}
    return audit


def _is_test_member(name: str) -> bool:
    low = name.lower().replace("\\", "/")
    return bool(re.search(r"(^|[/_.-])test([/_.-]|$)", low))


def _find_eurlex_split_files(extract_dir: Path, aliases: Sequence[str]) -> tuple[list[Path], list[Path]]:
    json_files, jsonl_files = [], []
    for p in extract_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(extract_dir)).lower().replace("\\", "/")
        if not any(re.search(fr"(^|[/_.-]){re.escape(a)}([/_.-]|$)", rel) for a in aliases):
            continue
        if p.suffix.lower() == ".json":
            json_files.append(p)
        elif p.suffix.lower() in {".jsonl", ".jl"}:
            jsonl_files.append(p)
    return sorted(json_files), sorted(jsonl_files)


def _eurlex_obj_to_record(obj: dict, fallback_id: str) -> dict:
    sample_id = str(obj.get("celex_id") or obj.get("id") or obj.get("celex") or fallback_id)
    labels = obj.get("eurovoc_concepts")
    if labels is None:
        labels = obj.get("concepts")
    if labels is None:
        labels = obj.get("labels")
    if labels is None:
        raise RuntimeError(f"EURLEX record {sample_id} has no recognized label field")
    labels = [str(x) for x in labels]
    if "text" in obj and isinstance(obj["text"], str) and obj["text"].strip():
        text = obj["text"].strip()
    else:
        parts = []
        for key in ("title", "header", "recitals", "main_body", "attachments"):
            if key in obj:
                s = flatten_text_field(obj[key]).strip()
                if s:
                    parts.append(s)
        text = "\n".join(parts)
    if not text.strip():
        raise RuntimeError(f"EURLEX record {sample_id} has empty text")
    return {"id": sample_id, "text": text, "labels": sorted(set(labels))}


def _read_eurlex_files(json_files: list[Path], jsonl_files: list[Path]) -> list[dict]:
    rows = []
    for p in json_files:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for i, x in enumerate(obj):
                rows.append(_eurlex_obj_to_record(x, f"{p.stem}-{i}"))
        else:
            rows.append(_eurlex_obj_to_record(obj, p.stem))
    for p in jsonl_files:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if line.strip():
                    rows.append(_eurlex_obj_to_record(json.loads(line), f"{p.stem}-{i}"))
    return rows



def prepare_eurlex57k(root: str | Path, include_test: bool = False) -> dict:
    """Prepare EURLEX57K without using test annotations during development.

    Development head: 4,108 labels with >=1 TRAIN positive.
    Final one-shot evaluation: after test is unlocked, build the official full
    4,271-label inventory from the now-legitimate union of all official splits,
    and evaluate the seen-only model with probability 0 for unsupported
    zero-shot labels. The full inventory is NEVER used for training/tuning.
    """
    root=Path(root); raw_dir=ensure_dir(root/"raw"/"eurlex57k"); prep_dir=ensure_dir(root/"prepared"/"eurlex57k")
    archive, download_source_url = download_from_mirrors((EURLEX_OFFICIAL_URL, EURLEX_FALLBACK_URL), raw_dir/"dataset.zip", expected_sha256=EURLEX_SHA256)
    dev_extract=raw_dir/"extracted_dev_only"; marker=dev_extract/".extracted_ok"
    if not marker.exists():
        safe_extract_zip(archive,dev_extract,member_filter=lambda name:not _is_test_member(name)); marker.write_text("ok\n",encoding="utf-8")
    train_json,train_jsonl=_find_eurlex_split_files(dev_extract,("train","training")); val_json,val_jsonl=_find_eurlex_split_files(dev_extract,("dev","valid","validation"))
    train_rows=_read_eurlex_files(train_json,train_jsonl); val_rows_raw=_read_eurlex_files(val_json,val_jsonl)
    if len(train_rows)!=45000 or len(val_rows_raw)!=6000:
        listing=sorted(str(x.relative_to(dev_extract)) for x in dev_extract.rglob("*") if x.is_file())[:200]
        raise RuntimeError(f"EURLEX57K split audit failed: expected 45000/6000, found {len(train_rows)}/{len(val_rows_raw)}. Files:{listing}")
    label_names=_sort_labels(l for r in train_rows for l in r["labels"])
    if len(label_names)!=4108:
        raise RuntimeError(f"EURLEX57K TRAIN-seen label audit failed: expected 4,108, found {len(label_names)}")
    known=set(label_names); val_unseen=sorted(set(l for r in val_rows_raw for l in r["labels"])-known)
    val_rows=[{"id":r["id"],"text":r["text"],"labels":[l for l in r["labels"] if l in known]} for r in val_rows_raw]
    write_jsonl(prep_dir/"train.jsonl",train_rows); write_jsonl(prep_dir/"validation.jsonl",val_rows); write_jsonl(prep_dir/"validation_full_labels_dev.jsonl",val_rows_raw)
    atomic_write_json(prep_dir/"labels.json",label_names)
    audit=_dataset_audit(train_rows,val_rows,label_names)
    audit.update({"dataset":"eurlex57k","source_url_primary":EURLEX_OFFICIAL_URL,"source_url_fallback":EURLEX_FALLBACK_URL,"download_source_url":download_source_url,"source_sha256_expected":EURLEX_SHA256,"raw_sha256":sha256_file(archive),
                  "label_schema_mode":"TRAIN-seen 4,108-label training head; final-only full-task evaluation after test unlock",
                  "label_schema_count":len(label_names),"original_paper_total_label_count":4271,"original_paper_zero_train_support_label_count":163,
                  "validation_labels_excluded_not_seen_in_train":val_unseen,"validation_excluded_label_count":len(val_unseen),
                  "validation_positive_assignments_excluded":int(sum(1 for r in val_rows_raw for l in r["labels"] if l not in known)),
                  "wasl_2024_reported_label_count":4256,"wasl_exact_4256_filter_rule_known":False,"test_accessed":False})
    _persist_development_audit(prep_dir, audit)
    if include_test:
        test_extract=raw_dir/"extracted_test_final_only"; marker_test=test_extract/".extracted_ok"
        if not marker_test.exists():
            safe_extract_zip(archive,test_extract,member_filter=lambda name:_is_test_member(name)); marker_test.write_text("ok\n",encoding="utf-8")
        test_json,test_jsonl=_find_eurlex_split_files(test_extract,("test","testing")); test_rows_raw=_read_eurlex_files(test_json,test_jsonl)
        if len(test_rows_raw)!=6000: raise RuntimeError(f"EURLEX57K test audit failed: expected 6000, found {len(test_rows_raw)}")
        full_labels=_sort_labels(list(known)+[l for r in val_rows_raw for l in r["labels"]]+[l for r in test_rows_raw for l in r["labels"]])
        if len(full_labels)!=4271:
            raise RuntimeError(f"Final-only EURLEX full-task inventory audit failed: expected 4,271, found {len(full_labels)}")
        zero_labels=sorted(set(full_labels)-known)
        if len(zero_labels)!=163:
            raise RuntimeError(f"Final-only EURLEX zero-shot audit failed: expected 163, found {len(zero_labels)}")
        test_rows=[{"id":r["id"],"text":r["text"],"labels":[l for l in r["labels"] if l in known]} for r in test_rows_raw]
        write_jsonl(prep_dir/"test.jsonl",test_rows); write_jsonl(prep_dir/"test_full_labels_final_only.jsonl",test_rows_raw)
        atomic_write_json(prep_dir/"full_task_labels_final_only.json",full_labels)
        final_audit=dict(audit)
        final_audit.update({"test_rows":len(test_rows),"final_only_full_task_label_count":len(full_labels),"final_only_zero_shot_label_count":len(zero_labels),
                      "final_only_zero_shot_labels":zero_labels,
                      "test_positive_assignments_on_zero_shot_labels":int(sum(1 for r in test_rows_raw for l in r["labels"] if l in set(zero_labels)))})
        _attach_final_duplicate_audit(final_audit, train_rows, val_rows, test_rows)
        _persist_final_test_audit(prep_dir, final_audit)
        return {**final_audit, "test_accessed": True}
    return audit

def prepare_dataset(dataset: str, root: str | Path, include_test: bool = False,
                    validation_seed: int = 20260805) -> dict:
    if dataset == "reuters21578":
        return prepare_reuters(root, validation_seed=validation_seed, include_test=include_test)
    if dataset == "aapd":
        return prepare_aapd(root, validation_seed=validation_seed, include_test=include_test)
    if dataset == "eurlex57k":
        return prepare_eurlex57k(root, include_test=include_test)
    raise ValueError(dataset)


def load_prepared_records(root: str | Path, dataset: str, split: str) -> tuple[list[dict], list[str]]:
    root = Path(root)
    prep = root / "prepared" / dataset
    if split == "test" and not (prep / "test.jsonl").exists():
        raise RuntimeError("Test split is not prepared/unlocked. Do not access test during development.")
    rows = read_jsonl(prep / f"{split}.jsonl")
    labels = json.loads((prep / "labels.json").read_text(encoding="utf-8"))
    return rows, labels


def tokenize_split(root: str | Path, dataset: str, split: str, tokenizer, max_length: int,
                   batch_size: int = 512, force: bool = False) -> Path:
    root = Path(root)
    rows, labels = load_prepared_records(root, dataset, split)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    out = ensure_dir(root / "tokenized" / dataset / f"{tokenizer.name_or_path.replace('/', '__')}__len{max_length}" / split)
    manifest_path = out / "manifest.json"
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    prepared_split = root / "prepared" / dataset / f"{split}.jsonl"
    labels_path = root / "prepared" / dataset / "labels.json"
    expected_manifest = {
        "tokenization_schema_version": 3,
        "dataset": dataset,
        "split": split,
        "n": len(rows),
        "num_labels": len(labels),
        "max_length": max_length,
        "tokenizer": tokenizer.name_or_path,
        "tokenizer_commit_hash": tokenizer_commit,
        "prepared_split_sha256": sha256_file(prepared_split),
        "labels_sha256": sha256_file(labels_path),
    }
    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in expected_manifest.items()):
            required = [out / "input_ids.npy", out / "attention_mask.npy", out / "label_ptr.npy", out / "label_idx.npy", out / "sample_ids.json"]
            if all(fp.exists() for fp in required):
                recorded = old.get("cache_files_sha256", {})
                current = {fp.name: sha256_file(fp) for fp in required}
                if recorded == current:
                    return out
                raise RuntimeError(f"Token-cache integrity failure for {dataset}/{split}; delete/rebuild the cache explicitly")

    n = len(rows)
    input_ids = np.lib.format.open_memmap(out / "input_ids.npy", mode="w+", dtype=np.int32, shape=(n, max_length))
    attention_mask = np.lib.format.open_memmap(out / "attention_mask.npy", mode="w+", dtype=np.uint8, shape=(n, max_length))
    ptr = np.zeros(n + 1, dtype=np.int64)
    flat_labels: list[int] = []
    sample_ids = []
    original_token_lengths: list[int] = []
    for start in tqdm(range(0, n, batch_size), desc=f"tokenize {dataset}/{split}"):
        end = min(n, start + batch_size)
        texts = [rows[i]["text"] for i in range(start, end)]
        # Record sequence lengths before truncation for the data audit.
        raw_enc = tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        original_token_lengths.extend(len(x) for x in raw_enc["input_ids"])
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        input_ids[start:end] = np.asarray(enc["input_ids"], dtype=np.int32)
        attention_mask[start:end] = np.asarray(enc["attention_mask"], dtype=np.uint8)
        for i in range(start, end):
            idx = sorted({label_to_idx[l] for l in rows[i]["labels"] if l in label_to_idx})
            flat_labels.extend(idx)
            ptr[i + 1] = len(flat_labels)
            sample_ids.append(rows[i]["id"])
    input_ids.flush(); attention_mask.flush()
    np.save(out / "label_ptr.npy", ptr)
    np.save(out / "label_idx.npy", np.asarray(flat_labels, dtype=np.int32))
    (out / "sample_ids.json").write_text(json.dumps(sample_ids, ensure_ascii=False) + "\n", encoding="utf-8")
    lens = np.asarray(original_token_lengths, dtype=np.int64)
    cache_files = [out / "input_ids.npy", out / "attention_mask.npy", out / "label_ptr.npy", out / "label_idx.npy", out / "sample_ids.json"]
    expected_manifest.update({
        "cache_files_sha256": {fp.name: sha256_file(fp) for fp in cache_files},
        "total_positive_assignments": len(flat_labels),
        "original_token_length_mean": float(lens.mean()) if lens.size else 0.0,
        "original_token_length_median": float(np.median(lens)) if lens.size else 0.0,
        "original_token_length_p95": float(np.percentile(lens, 95)) if lens.size else 0.0,
        "original_token_length_max": int(lens.max()) if lens.size else 0,
        "num_sequences_exceeding_max_length": int((lens > max_length).sum()),
        "fraction_sequences_exceeding_max_length": float((lens > max_length).mean()) if lens.size else 0.0,
    })
    atomic_write_json(manifest_path, expected_manifest)
    return out
