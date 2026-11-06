from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import collect_results, project_paths
from .utils import atomic_write_json


def compare_wasl_published(root: str | Path, code_root: str | Path) -> list[dict[str, Any]]:
    """Compare the optional 50-epoch compatibility track with published WASL F1 means.

    This report is contextual only. It does not replace matched-condition causal comparisons.
    """
    root = Path(root)
    paths = project_paths(root)
    ref = json.loads((Path(code_root) / "references" / "wasl_reference.json").read_text(encoding="utf-8"))
    rows = collect_results(root, final=False)
    out = []
    for ds, methods in ref["reported_mean_f1"].items():
        for method, pub in methods.items():
            candidates = [r for r in rows if r["dataset"] == ds and r["method"] == method and int(r.get("epochs") or 0) == 50]
            if not candidates:
                out.append({
                    "dataset": ds, "method": method, "n_local_seeds": 0,
                    "published_macro_f1": pub["macro_f1"], "published_micro_f1": pub["micro_f1"],
                    "local_macro_f1_mean": None, "local_micro_f1_mean": None,
                    "note": "WASL_REUTERS_REPRO not completed for this dataset/method"
                })
                continue
            ma = np.asarray([float(r["macro_f1_at_0_5"]) for r in candidates])
            mi = np.asarray([float(r["micro_f1_at_0_5"]) for r in candidates])
            out.append({
                "dataset": ds, "method": method, "n_local_seeds": len(candidates),
                "published_macro_f1": pub["macro_f1"], "published_micro_f1": pub["micro_f1"],
                "local_macro_f1_mean": float(ma.mean()), "local_micro_f1_mean": float(mi.mean()),
                "macro_difference_local_minus_published": float(ma.mean() - pub["macro_f1"]),
                "micro_difference_local_minus_published": float(mi.mean() - pub["micro_f1"]),
                "note": "Contextual reproduction check; preprocessing/implementation details can differ."
            })
    atomic_write_json(paths["reports"] / "wasl_published_comparison.json", out)
    if out:
        keys = sorted({k for r in out for k in r})
        with open(paths["reports"] / "wasl_published_comparison.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    return out
