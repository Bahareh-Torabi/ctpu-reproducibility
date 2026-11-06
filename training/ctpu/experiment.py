from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .auxiliary import pairwise_score_gradient, ctpu_project, synthetic_mechanism_self_test
from .config import (
    DATASET_DEFAULTS,
    FINAL_TEST_AUTH_PHRASE,
    PROFILES,
    PROJECT_VERSION,
    SEEDS_3,
    SEEDS_5,
    TrainConfig,
    WASL_REPRO_LR,
    QUICK_PAIRWISE_GAIN_FLOOR,
)
from .data import load_prepared_records, prepare_dataset, tokenize_split
from .dataset import TokenizedMultiLabelDataset, make_collate_fn
from .train import evaluate_saved_model_on_test, run_training
from .utils import (
    atomic_write_json,
    ensure_dir,
    environment_report,
    object_hash,
    read_json,
    reproducibility_environment_fingerprint,
    sha256_file,
    stable_int_seed,
    set_global_seed,
)


def project_paths(root: str | Path) -> dict[str, Path]:
    root = ensure_dir(root)
    return {
        "root": root,
        "data": ensure_dir(root / "data"),
        "runs": ensure_dir(root / "runs"),
        "reports": ensure_dir(root / "reports"),
        "protocol": ensure_dir(root / "protocol"),
        "final": ensure_dir(root / "final_test_runs"),
    }


def compute_code_hash(code_root: str | Path) -> str:
    """Hash executable research code plus pinned environment/reference metadata."""
    code_root = Path(code_root)
    files = sorted((code_root / "ctpu").glob("*.py"))
    for rel in ("run_ctpu.py", "requirements-colab.txt", "VERSION"):
        q = code_root / rel
        if q.exists():
            files.append(q)
    ref_dir = code_root / "references"
    if ref_dir.exists():
        files.extend(sorted(ref_dir.glob("*.json")))
    payload = {str(q.relative_to(code_root)): sha256_file(q) for q in sorted(set(files))}
    return object_hash(payload)

def resolve_model_revision(root: str | Path, model_name: str = "FacebookAI/roberta-base") -> str:
    paths = project_paths(root)
    p = paths["protocol"] / "model_revision.json"
    if p.exists():
        obj = read_json(p)
        if obj.get("model_name") != model_name:
            raise RuntimeError("Protocol already pins a different model name")
        return str(obj["revision"])
    info = HfApi().model_info(model_name)
    revision = str(info.sha)
    atomic_write_json(p, {
        "model_name": model_name,
        "revision": revision,
        "resolved_at_unix": time.time(),
        "rule": "First successful run pins the Hugging Face commit; all later runs reuse it.",
    })
    return revision


def _audit_guard(audit: dict) -> None:
    dups = audit.get("train_validation_exact_text_duplicates", [])
    if dups:
        raise RuntimeError(
            f"Cross-split exact text duplicates detected ({len(dups)}). "
            "Training is blocked until an explicit, documented duplicate policy is approved."
        )
    if audit.get("dataset") == "reuters21578" and int(audit.get("num_labels", -1)) != 90:
        raise RuntimeError("Reuters R(90) audit failed")
    if audit.get("dataset") == "aapd" and int(audit.get("num_labels", -1)) != 54:
        raise RuntimeError("AAPD label-count audit failed")
    if audit.get("dataset") == "eurlex57k" and int(audit.get("label_schema_count", -1)) != 4108:
        raise RuntimeError("EURLEX57K TRAIN-seen supervised head audit failed: expected 4,108 labels")


def prepare_and_tokenize(root: str | Path, code_root: str | Path, dataset: str,
                         include_test: bool = False, validation_seed: int = SEEDS_5[0]) -> dict[str, Any]:
    paths = project_paths(root)
    revision = resolve_model_revision(root)
    audit = prepare_dataset(dataset, paths["data"], include_test=include_test, validation_seed=validation_seed)
    if not include_test:
        _audit_guard(audit)
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", revision=revision, use_fast=True)
    max_length = int(DATASET_DEFAULTS[dataset]["max_length"])
    splits = ["train", "validation"] + (["test"] if include_test else [])
    tok = {}
    for split in splits:
        tok[split] = tokenize_split(paths["data"], dataset, split, tokenizer, max_length=max_length)
    report = {
        "dataset": dataset,
        "audit": audit,
        "tokenized": {k: str(v) for k, v in tok.items()},
        "token_manifest_sha256": {k: sha256_file(Path(v) / "manifest.json") for k, v in tok.items()},
        "model_revision": revision,
        "max_length": max_length,
        "code_hash": compute_code_hash(code_root),
        # This is deliberately the immutable development audit, even during
        # final-test preparation. Final-only provenance is stored separately.
        "dataset_audit_sha256": sha256_file(paths["data"] / "prepared" / dataset / "audit.json"),
    }
    final_audit_path = paths["data"] / "prepared" / dataset / "final_test_audit.json"
    if include_test:
        if not final_audit_path.exists():
            raise RuntimeError(f"Final-test audit was not persisted for {dataset}")
        report["final_test_audit_sha256"] = sha256_file(final_audit_path)
        report["final_test_audit_path"] = str(final_audit_path)
    report_name = f"{dataset}_prepare_report_final_test.json" if include_test else f"{dataset}_prepare_report.json"
    atomic_write_json(paths["reports"] / report_name, report)
    return report


def make_train_config(root: str | Path, dataset: str, method: str, seed: int,
                      epochs_override: int | None = None, wasl_repro: bool = False,
                      cardinality_penalty_mu: float = 0.10) -> TrainConfig:
    defaults = DATASET_DEFAULTS[dataset]
    cfg = TrainConfig(
        dataset=dataset,
        method=method,
        seed=int(seed),
        model_name="FacebookAI/roberta-base",
        model_revision=resolve_model_revision(root),
        max_length=int(defaults["max_length"]),
        epochs=int(epochs_override if epochs_override is not None else defaults["epochs"]),
        lr=float(WASL_REPRO_LR[dataset] if wasl_repro else defaults["lr"]),
        cardinality_penalty_mu=float(cardinality_penalty_mu),
    )
    return cfg


def _run_dir(paths: dict[str, Path], cfg: TrainConfig, final: bool = False) -> Path:
    short = object_hash({"project_version": PROJECT_VERSION, "config": cfg.to_dict()})[:12]
    base = paths["final"] if final else paths["runs"]
    return base / cfg.dataset / cfg.method / f"seed_{cfg.seed}" / short


def real_logit_mechanism_test(root: str | Path, code_root: str | Path,
                              dataset: str = "reuters21578", seed: int = SEEDS_5[0]) -> dict[str, Any]:
    """No training: verify the first-order mechanism on real transformer logits."""
    paths = project_paths(root)
    prep = prepare_and_tokenize(root, code_root, dataset, include_test=False)
    labels = json.loads((paths["data"] / "prepared" / dataset / "labels.json").read_text(encoding="utf-8"))
    ds = TokenizedMultiLabelDataset(prep["tokenized"]["train"], len(labels))
    collate = make_collate_fn(len(labels))
    batch = collate([ds[i] for i in range(min(16, len(ds)))])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed, deterministic=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base", revision=prep["model_revision"],
        num_labels=len(labels), problem_type="multi_label_classification", attn_implementation="eager",
    ).to(device).eval()
    with torch.inference_mode():
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits.detach().float().cpu()
    g, active, _, _ = pairwise_score_gradient(
        logits, batch["label_indices"], batch["sample_ids"], epoch=1, seed=seed, k_pos=3, k_neg=3
    )
    u, udiag = ctpu_project(g, logits, active)
    p = torch.sigmoid(logits.double())
    h = active.double() * p * (1 - p)
    step = 1e-5
    c0 = p.sum(1)
    pair_obs = (torch.sigmoid(logits.double() - step * g.double()).sum(1) - c0) / step
    pair_pred = -(h * g.double()).sum(1)
    ctpu_obs = (torch.sigmoid(logits.double() - step * u.double()).sum(1) - c0) / step
    valid = torch.isfinite(pair_obs) & torch.isfinite(pair_pred)
    corr = float(np.corrcoef(pair_obs[valid].numpy(), pair_pred[valid].numpy())[0, 1]) if int(valid.sum()) > 2 else float("nan")
    report = {
        "synthetic": synthetic_mechanism_self_test(seed),
        "real_logits": {
            "dataset": dataset,
            "n_examples": len(batch["sample_ids"]),
            "finite_difference_step": step,
            "pairwise_predicted_vs_observed_correlation": corr,
            "pairwise_max_abs_prediction_error": float((pair_obs - pair_pred).abs().max().item()),
            "ctpu_max_abs_first_order_drift": float(ctpu_obs.abs().max().item()),
            "ctpu_max_abs_tangency_error": float((h * u.double()).sum(1).abs().max().item()),
            **udiag,
        },
        "code_hash": compute_code_hash(code_root),
        "model_revision": prep["model_revision"],
        "environment_fingerprint": reproducibility_environment_fingerprint(),
        "environment_fingerprint_hash": object_hash(reproducibility_environment_fingerprint()),
        "dataset_audit_sha256": sha256_file(paths["data"] / "prepared" / dataset / "audit.json"),
    }
    # Check the analytic identities for basic finite-difference consistency.
    if not math.isnan(corr) and corr < 0.99:
        raise RuntimeError(f"Mechanism preflight failed: finite-difference correlation={corr}")
    atomic_write_json(paths["reports"] / "mechanism_preflight.json", report)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _verify_quick_gate_binding(root: str | Path, code_root: str | Path) -> dict:
    paths = project_paths(root)
    gate_path = paths["reports"] / "QUICK_GATE.json"
    if not gate_path.exists():
        raise RuntimeError("Expensive profile is blocked until QUICK_GATE.json exists")
    gate = read_json(gate_path)
    recorded_gate_hash = gate.get("gate_hash")
    gate_body = dict(gate); gate_body.pop("gate_hash", None)
    if not recorded_gate_hash or object_hash(gate_body) != recorded_gate_hash:
        raise RuntimeError("QUICK_GATE.json integrity hash mismatch")
    if gate.get("status") != "PASS":
        raise RuntimeError("Expensive profile is blocked because QUICK engineering/readiness gate did not PASS")
    current_code = compute_code_hash(code_root)
    current_revision = resolve_model_revision(root)
    audit_path = paths["data"] / "prepared" / "reuters21578" / "audit.json"
    mech_path = paths["reports"] / "mechanism_preflight.json"
    prep_report_path = paths["reports"] / "reuters21578_prepare_report.json"
    prep_report = read_json(prep_report_path)
    current_data_provenance = object_hash({
        "dataset_audit_sha256": sha256_file(audit_path),
        "train_token_manifest_sha256": prep_report["token_manifest_sha256"]["train"],
        "validation_token_manifest_sha256": prep_report["token_manifest_sha256"]["validation"],
    })
    expected = {
        "code_hash": current_code,
        "model_revision": current_revision,
        "reuters_audit_sha256": sha256_file(audit_path),
        "mechanism_preflight_sha256": sha256_file(mech_path),
        "quick_data_provenance_hash": current_data_provenance,
        "environment_fingerprint_hash": object_hash(reproducibility_environment_fingerprint()),
    }
    for k, v in expected.items():
        if gate.get(k) != v:
            raise RuntimeError(
                f"QUICK gate binding mismatch for {k}: gate={gate.get(k)} current={v}. "
                "Any code/model/data change requires rerunning mechanism + QUICK under the new protocol."
            )
    return gate


def verify_quick_gate_binding(root: str | Path, code_root: str | Path) -> dict:
    """Read-only verification of protocol identities before campaign execution."""
    return _verify_quick_gate_binding(root, code_root)


def run_profile(root: str | Path, code_root: str | Path, profile_name: str,
                cardinality_penalty_mu: float = 0.10) -> list[dict[str, Any]]:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")
    profile = PROFILES[profile_name]
    if profile.allow_test:
        raise RuntimeError("Development profiles may not allow test")
    paths = project_paths(root)
    code_hash = compute_code_hash(code_root)
    atomic_write_json(paths["reports"] / "environment_latest.json", environment_report())

    if profile_name == "QUICK":
        mech_path = paths["reports"] / "mechanism_preflight.json"
        if not mech_path.exists():
            raise RuntimeError("QUICK is blocked until mechanism_preflight.json exists")
        mech = read_json(mech_path)
        audit_path = paths["data"] / "prepared" / "reuters21578" / "audit.json"
        if mech.get("code_hash") != code_hash:
            raise RuntimeError("Mechanism preflight was produced by different code; rerun it")
        if mech.get("model_revision") != resolve_model_revision(root):
            raise RuntimeError("Mechanism preflight model revision mismatch")
        if mech.get("dataset_audit_sha256") != sha256_file(audit_path):
            raise RuntimeError("Mechanism preflight Reuters audit mismatch")
        current_env_hash = object_hash(reproducibility_environment_fingerprint())
        if mech.get("environment_fingerprint_hash") != current_env_hash:
            raise RuntimeError("Mechanism preflight environment fingerprint mismatch; rerun under the current software/CUDA environment")
    elif profile_name not in {"AUDIT_ONLY", "ASL_OFFICIAL_REDUCTION_CHECK"}:
        _verify_quick_gate_binding(root, code_root)

    outputs=[]
    if not profile.methods:
        for dataset in profile.datasets:
            outputs.append(prepare_and_tokenize(root,code_root,dataset,include_test=False))
        return outputs
    for dataset in profile.datasets:
        prep=prepare_and_tokenize(root,code_root,dataset,include_test=False)
        for method in profile.methods:
            for seed in profile.seeds:
                cfg=make_train_config(root,dataset,method,seed,epochs_override=profile.epochs_override,
                                      wasl_repro=(profile_name=="WASL_REUTERS_REPRO"),cardinality_penalty_mu=cardinality_penalty_mu)
                out=run_training(cfg,data_root=paths["data"],tokenized_train=prep["tokenized"]["train"],
                                 tokenized_validation=prep["tokenized"]["validation"],
                                 run_dir=_run_dir(paths,cfg,final=False),code_hash=code_hash)
                outputs.append(out)
    collect_results(root,final=False)
    if profile_name=="QUICK": quick_gate(root,code_root)
    return outputs

def _scan_complete(root: Path, final: bool = False) -> list[dict[str, Any]]:
    base = root / ("final_test_runs" if final else "runs")
    name = "FINAL_TEST_COMPLETE.json" if final else "COMPLETE.json"
    rows = []
    if not base.exists():
        return rows
    for p in base.rglob(name):
        obj = read_json(p)
        metrics = obj.get("test_metrics" if final else "validation_metrics", {})
        cfg_path = p.parent / "config.json"
        cfg = read_json(cfg_path) if cfg_path.exists() else {}
        row = {
            "dataset": obj.get("dataset"),
            "method": obj.get("method"),
            "seed": obj.get("seed"),
            "epochs": cfg.get("epochs"),
            "lr": cfg.get("lr"),
            "lambda_pair": cfg.get("lambda_pair"),
            "cardinality_penalty_mu": cfg.get("cardinality_penalty_mu"),
            "config_hash": obj.get("config_hash"),
            "code_hash": obj.get("code_hash"),
            "provenance_hash": obj.get("provenance_hash"),
            "data_provenance_hash": obj.get("data_provenance_hash"),
            "run_dir": str(p.parent),
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        rows.append(row)
    return sorted(rows, key=lambda x: (str(x["dataset"]), str(x["method"]), int(x["seed"])))



def _aggregate_seed_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        key = (
            r.get("dataset"), r.get("method"), r.get("epochs"), r.get("lr"),
            r.get("lambda_pair"), r.get("cardinality_penalty_mu"),
        )
        groups.setdefault(key, []).append(r)
    out = []
    identity = {"dataset", "method", "epochs", "lr", "lambda_pair", "cardinality_penalty_mu", "seed", "config_hash", "code_hash", "run_dir"}
    for key, vals in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {
            "dataset": key[0], "method": key[1], "epochs": key[2], "lr": key[3],
            "lambda_pair": key[4], "cardinality_penalty_mu": key[5], "n_seeds": len(vals),
        }
        metric_keys = sorted({k for v in vals for k in v if k not in identity and isinstance(v.get(k), (int, float))})
        for m in metric_keys:
            arr = np.asarray([float(v[m]) for v in vals if isinstance(v.get(m), (int, float)) and math.isfinite(float(v[m]))], dtype=float)
            if arr.size:
                row[f"{m}__mean"] = float(arr.mean())
                row[f"{m}__sd"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(row)
    return out

def collect_results(root: str | Path, final: bool = False) -> list[dict[str, Any]]:
    paths = project_paths(root)
    rows = _scan_complete(paths["root"], final=final)
    out_csv = paths["reports"] / ("final_test_results.csv" if final else "development_results.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    atomic_write_json(paths["reports"] / ("final_test_results.json" if final else "development_results.json"), rows)
    summary = _aggregate_seed_summary(rows)
    summary_csv = paths["reports"] / ("final_test_summary.csv" if final else "development_summary.csv")
    if summary:
        keys = sorted({k for r in summary for k in r})
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(summary)
    atomic_write_json(paths["reports"] / ("final_test_summary.json" if final else "development_summary.json"), summary)
    return rows


def quick_gate(root: str | Path, code_root: str | Path) -> dict[str, Any]:
    """Outcome-independent QUICK readiness gate plus descriptive effect report.

    QUICK can block the expensive campaign only for engineering/provenance failures.
    The direction or magnitude of Reuters validation effects is never a continuation
    criterion, preventing outcome-dependent study continuation.
    """
    paths = project_paths(root)
    current_code = compute_code_hash(code_root)
    revision = resolve_model_revision(root)
    by: dict[str, dict[str, Any]] = {}
    readiness_checks: dict[str, bool] = {}
    finite_metric_names = ("ranking_loss", "soft_cardinality_mae", "nll", "brier", "macro_ap", "micro_ap")
    for method in ("bce", "pairwise", "normmatch", "ctpu"):
        cfg = make_train_config(root, "reuters21578", method, seed=SEEDS_5[0], epochs_override=4)
        run_dir = _run_dir(paths, cfg, final=False)
        cp = run_dir / "COMPLETE.json"
        if not cp.exists():
            raise RuntimeError(f"QUICK readiness missing exact expected run: {run_dir}")
        obj = read_json(cp)
        if obj.get("code_hash") != current_code or obj.get("config_hash") != object_hash(cfg.to_dict()):
            raise RuntimeError(f"QUICK run identity mismatch: {run_dir}")
        if cfg.model_revision != revision or cfg.attn_implementation != "eager":
            raise RuntimeError("QUICK model revision/attention implementation mismatch")
        metrics = read_json(run_dir / "validation_metrics.json")
        by[method] = {**metrics, "config_hash": obj["config_hash"], "code_hash": obj["code_hash"],
                      "provenance_hash": obj.get("provenance_hash"),
                      "data_provenance_hash": obj.get("data_provenance_hash"),
                      "environment_fingerprint_hash": obj.get("environment_fingerprint_hash"),
                      "run_dir": str(run_dir)}
        readiness_checks[f"{method}_complete_final_epoch_4_no_early_stopping"] = bool(
            obj.get("status") == "COMPLETE" and int(obj.get("final_epoch", -1)) == 4 and not bool(obj.get("early_stopping", True))
        )
        readiness_checks[f"{method}_exact_final_model_retained"] = bool(
            obj.get("exact_validation_model_retained") and obj.get("final_model_tree_hash")
        )
        readiness_checks[f"{method}_core_metrics_finite"] = all(
            isinstance(metrics.get(k), (int, float)) and math.isfinite(float(metrics[k])) for k in finite_metric_names
        )

    mech_path = paths["reports"] / "mechanism_preflight.json"
    audit_path = paths["data"] / "prepared" / "reuters21578" / "audit.json"
    if not mech_path.exists():
        raise RuntimeError("Missing mechanism preflight")
    mech = read_json(mech_path)
    real = mech.get("real_logits", {})
    real_vals = [real.get(k) for k in ("pairwise_max_abs_prediction_error", "ctpu_max_abs_first_order_drift", "ctpu_max_abs_tangency_error")]
    readiness_checks["mechanism_preflight_pass"] = bool(
        mech.get("synthetic", {}).get("status") == "PASS" and
        all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in real_vals)
    )
    audit = read_json(audit_path)
    readiness_checks["development_cross_split_exact_duplicate_free"] = not bool(audit.get("train_validation_exact_text_duplicates"))

    provenance_values = {m: by[m].get("data_provenance_hash") for m in by}
    readiness_checks["common_data_token_provenance"] = bool(None not in provenance_values.values() and len(set(provenance_values.values())) == 1)
    env_values = {m: by[m].get("environment_fingerprint_hash") for m in by}
    current_env_hash = object_hash(reproducibility_environment_fingerprint())
    readiness_checks["common_current_environment_fingerprint"] = bool(
        None not in env_values.values() and len(set(env_values.values())) == 1 and next(iter(env_values.values())) == current_env_hash
    )
    if not readiness_checks["common_data_token_provenance"]:
        raise RuntimeError(f"QUICK runs do not share one data/token provenance: {provenance_values}")
    if not readiness_checks["common_current_environment_fingerprint"]:
        raise RuntimeError(f"QUICK runs are not bound to one current software/CUDA environment: {env_values}, current={current_env_hash}")

    b, p, n, c = (by[x] for x in ("bce", "pairwise", "normmatch", "ctpu"))
    pair_gain = float(b["ranking_loss"] - p["ranking_loss"])
    ctpu_gain = float(b["ranking_loss"] - c["ranking_loss"])
    measurable = pair_gain > QUICK_PAIRWISE_GAIN_FLOOR
    retention = ctpu_gain / pair_gain if measurable else float("nan")
    pair_mae = float(p["soft_cardinality_mae"]); ctpu_mae = float(c["soft_cardinality_mae"]); norm_mae = float(n["soft_cardinality_mae"])
    descriptive = {
        "performance_is_descriptive_only_never_a_gate": True,
        "pairwise_ranking_gain_vs_bce": pair_gain,
        "ctpu_ranking_gain_vs_bce": ctpu_gain,
        "pairwise_gain_numerical_floor": QUICK_PAIRWISE_GAIN_FLOOR,
        "pairwise_gain_exceeds_numerical_floor": bool(measurable),
        "ctpu_ranking_gain_retention_if_defined": retention,
        "bce_soft_cardinality_mae": float(b["soft_cardinality_mae"]),
        "pairwise_soft_cardinality_mae": pair_mae,
        "normmatch_soft_cardinality_mae": norm_mae,
        "ctpu_soft_cardinality_mae": ctpu_mae,
        "ctpu_nll_relative_to_pairwise": float(c["nll"]) / max(float(p["nll"]), 1e-15),
        "ctpu_brier_relative_to_pairwise": float(c["brier"]) / max(float(p["brier"]), 1e-15),
    }
    status = "PASS" if all(readiness_checks.values()) else "FAIL"
    report = {
        "status": status,
        "gate_scope": "engineering_readiness_and_provenance_only",
        "outcome_dependent_continuation_forbidden": True,
        "readiness_checks": readiness_checks,
        "descriptive_effects": descriptive,
        "code_hash": current_code,
        "model_revision": revision,
        "attention_implementation": "eager",
        "reuters_audit_sha256": sha256_file(audit_path),
        "mechanism_preflight_sha256": sha256_file(mech_path),
        "quick_run_config_hashes": {m: by[m]["config_hash"] for m in by},
        "quick_data_provenance_hash": next(iter(provenance_values.values())),
        "environment_fingerprint_hash": current_env_hash,
        "quick_run_dirs": {m: by[m]["run_dir"] for m in by},
    }
    report["gate_hash"] = object_hash(report)
    atomic_write_json(paths["reports"] / "QUICK_GATE.json", report)
    return report

def _find_dev_run(root: Path, cfg: TrainConfig) -> Path:
    paths = project_paths(root)
    p = _run_dir(paths, cfg, final=False)
    if not (p / "COMPLETE.json").exists():
        raise RuntimeError(f"Required development run is incomplete: {p}")
    return p


def _verify_lock_integrity(lock: dict[str, Any]) -> None:
    recorded = lock.get("lock_hash")
    if not recorded:
        raise RuntimeError("Protocol lock has no lock_hash")
    body = dict(lock)
    body.pop("lock_hash", None)
    if object_hash(body) != recorded:
        raise RuntimeError("FINAL_PROTOCOL_LOCK.json integrity hash mismatch")


def _frozen_validation_artifact_hashes(dev: str | Path) -> dict[str, str]:
    dev = Path(dev)
    required = (
        "config.json", "validation_metrics.json", "validation_logits.npy",
        "validation_labels.npy", "validation_sample_ids.json", "train_support.npy",
        "runtime_config.json", "gradient_diagnostics.json", "COMPLETE.json",
    )
    missing = [name for name in required if not (dev / name).exists()]
    if missing:
        raise RuntimeError(f"Cannot freeze incomplete validation artifact set at {dev}: {missing}")
    return {name: sha256_file(dev / name) for name in required}


def _verify_frozen_validation_artifacts(dev: str | Path, expected: dict[str, str]) -> None:
    dev = Path(dev)
    current = _frozen_validation_artifact_hashes(dev)
    if current != expected:
        changed = sorted(set(current) | set(expected))
        changed = [k for k in changed if current.get(k) != expected.get(k)]
        raise RuntimeError(f"Frozen validation artifacts changed after lock at {dev}: {changed}")


def _verify_existing_lock(root: str | Path, code_root: str | Path, lock: dict[str, Any]) -> None:
    _verify_lock_integrity(lock)
    paths = project_paths(root)
    if compute_code_hash(code_root) != lock.get("code_hash"):
        raise RuntimeError("Existing final lock belongs to different code")
    if resolve_model_revision(root) != lock.get("model_revision"):
        raise RuntimeError("Existing final lock belongs to a different model revision")
    if object_hash(reproducibility_environment_fingerprint()) != lock.get("environment_fingerprint_hash"):
        raise RuntimeError("Existing final lock belongs to a different software/CUDA environment fingerprint")
    for dataset, meta in lock.get("dataset_audits", {}).items():
        ap = paths["data"] / "prepared" / dataset / "audit.json"
        if not ap.exists() or sha256_file(ap) != meta.get("sha256"):
            raise RuntimeError(f"Existing final lock dataset audit mismatch: {dataset}")
    for item in lock.get("final_plan", []):
        dev = Path(item["validation_run_dir"])
        complete = read_json(dev / "COMPLETE.json")
        if complete.get("provenance_hash") != item.get("provenance_hash"):
            raise RuntimeError(f"Existing final lock run provenance mismatch: {dev}")
        model_dir = dev / "final_model"
        from .utils import tree_hash
        if (not model_dir.exists() or complete.get("final_model_tree_hash") != item.get("final_model_tree_hash") or
                tree_hash(model_dir) != item.get("final_model_tree_hash")):
            raise RuntimeError(f"Existing final lock retained-model mismatch: {dev}")
        if not item.get("validation_artifact_hashes"):
            raise RuntimeError(f"Existing final lock lacks validation artifact hashes: {dev}")
        _verify_frozen_validation_artifacts(dev, item["validation_artifact_hashes"])


def freeze_final_protocol(root: str | Path, code_root: str | Path) -> dict[str, Any]:
    """Freeze a one-shot test plan that points to exact retained development weights."""
    paths=project_paths(root); lock_path=paths["protocol"]/"FINAL_PROTOCOL_LOCK.json"
    if lock_path.exists():
        lock = read_json(lock_path)
        _verify_existing_lock(root, code_root, lock)
        return lock
    gate=_verify_quick_gate_binding(root,code_root); code_hash=compute_code_hash(code_root)
    current_env_hash = object_hash(reproducibility_environment_fingerprint())
    mech_path=paths["reports"]/"mechanism_preflight.json"
    plan=[]
    # Main suite: five Reuters/AAPD seeds and three EURLEX seeds.
    # Parameter-space PCGrad uses three Reuters/AAPD seeds.
    main_methods=["bce","pairwise","normmatch","ctpu","pairwise_cardinality_penalty","score_auxproj","asl","wasl"]
    for dataset,seeds in (("reuters21578",SEEDS_5),("aapd",SEEDS_5),("eurlex57k",SEEDS_3)):
        for method in main_methods:
            if dataset=="eurlex57k" and method=="score_auxproj":
                continue
            for seed in seeds:
                cfg=make_train_config(root,dataset,method,seed=seed)
                dev=_find_dev_run(paths["root"],cfg); complete=read_json(dev/"COMPLETE.json")
                model_dir=dev/"final_model"
                if not model_dir.exists() or not complete.get("exact_validation_model_retained"):
                    raise RuntimeError(f"Exact final model not retained: {dev}")
                if complete.get("final_model_tree_hash") is None:
                    raise RuntimeError(f"Missing final model hash: {dev}")
                if not complete.get("provenance_hash"):
                    raise RuntimeError(f"Missing run provenance hash: {dev}")
                if complete.get("environment_fingerprint_hash") != current_env_hash:
                    raise RuntimeError(f"Run environment fingerprint differs from current freeze environment: {dev}")
                metrics=read_json(dev/"validation_metrics.json")
                plan.append({"config":cfg.to_dict(),"config_hash":object_hash(cfg.to_dict()),"validation_run_dir":str(dev),
                             "frozen_global_threshold":float(metrics.get("validation_selected_global_threshold_macro_f1",0.5)),
                             "final_model_tree_hash":complete["final_model_tree_hash"],
                             "provenance_hash":complete.get("provenance_hash"),
                             "validation_artifact_hashes":_frozen_validation_artifact_hashes(dev)})
    for dataset in ("reuters21578","aapd"):
        for seed in SEEDS_3:
            cfg=make_train_config(root,dataset,"param_pcgrad",seed=seed); dev=_find_dev_run(paths["root"],cfg); complete=read_json(dev/"COMPLETE.json")
            if not (dev/"final_model").exists(): raise RuntimeError(f"Mandatory parameter-PCGrad control missing model: {dev}")
            if not complete.get("provenance_hash"): raise RuntimeError(f"Mandatory parameter-PCGrad run lacks provenance: {dev}")
            if complete.get("environment_fingerprint_hash") != current_env_hash:
                raise RuntimeError(f"Mandatory parameter-PCGrad environment fingerprint differs from freeze environment: {dev}")
            metrics=read_json(dev/"validation_metrics.json")
            plan.append({"config":cfg.to_dict(),"config_hash":object_hash(cfg.to_dict()),"validation_run_dir":str(dev),
                         "frozen_global_threshold":float(metrics.get("validation_selected_global_threshold_macro_f1",0.5)),
                         "final_model_tree_hash":complete["final_model_tree_hash"],
                         "provenance_hash":complete.get("provenance_hash"),
                         "validation_artifact_hashes":_frozen_validation_artifact_hashes(dev)})
    audits={}
    for dataset in ("reuters21578","aapd","eurlex57k"):
        ap=paths["data"]/"prepared"/dataset/"audit.json"
        if not ap.exists(): raise RuntimeError(f"Missing audit {ap}")
        a=read_json(ap); _audit_guard(a)
        if bool(a.get("test_accessed",False)): raise RuntimeError(f"Cannot freeze: test already accessed for {dataset}")
        audits[dataset]={"sha256":sha256_file(ap),"test_accessed":False}
    obj={"status":"FROZEN","created_unix":time.time(),"project_version":PROJECT_VERSION,"code_hash":code_hash,
         "model_revision":resolve_model_revision(root),
         "environment_fingerprint_hash":current_env_hash,
         "quick_gate_sha256":sha256_file(paths["reports"]/"QUICK_GATE.json"),
         "mechanism_preflight_sha256":sha256_file(mech_path),"dataset_audits":audits,"final_plan":plan,
         "rules":{"test_one_shot":True,"no_hyperparameter_changes_after_lock":True,"no_early_stopping":True,
                  "fixed_final_epoch":True,"final_test_uses_exact_retained_validation_model":True,"final_test_retraining":False,
                  "validation_thresholds_frozen_before_test":True,
                  "all_validation_decision_artifacts_sha256_locked":True,
                  "duplicate_clean_test_sensitivity_predeclared":True,
                  "primary_inference_estimation_first":True,
                  "bootstrap_design":"crossed_seed_by_document_paired"}}
    obj["lock_hash"]=object_hash(obj); atomic_write_json(lock_path,obj); return obj


def run_final_test(root: str | Path, code_root: str | Path, auth_phrase: str) -> list[dict[str, Any]]:
    paths=project_paths(root)
    if auth_phrase!=FINAL_TEST_AUTH_PHRASE: raise RuntimeError("Final-test authorization phrase is incorrect")
    executed=paths["protocol"]/"FINAL_TEST_EXECUTED.json"; started=paths["protocol"]/"FINAL_TEST_STARTED.json"; lock_path=paths["protocol"]/"FINAL_PROTOCOL_LOCK.json"
    if not lock_path.exists(): raise RuntimeError("FINAL_PROTOCOL_LOCK.json does not exist")
    lock=read_json(lock_path)
    _verify_lock_integrity(lock)
    if compute_code_hash(code_root)!=lock["code_hash"]: raise RuntimeError("Code changed after final lock")
    if resolve_model_revision(root)!=lock["model_revision"]: raise RuntimeError("Model revision changed after final lock")
    if object_hash(reproducibility_environment_fingerprint()) != lock.get("environment_fingerprint_hash"):
        raise RuntimeError("Software/CUDA environment fingerprint changed after final lock")
    for dataset,meta in lock["dataset_audits"].items():
        ap=paths["data"]/"prepared"/dataset/"audit.json"
        if sha256_file(ap)!=meta["sha256"]: raise RuntimeError(f"Dataset audit changed after lock: {dataset}")
    # Verify every frozen validation artifact BEFORE any test member is extracted or parsed.
    for item in lock["final_plan"]:
        dev = Path(item["validation_run_dir"])
        if object_hash(item["config"]) != item.get("config_hash"):
            raise RuntimeError(f"Frozen config hash mismatch inside lock: {dev}")
        _verify_frozen_validation_artifacts(dev, item.get("validation_artifact_hashes", {}))
    if executed.exists():
        prior_exec = read_json(executed)
        if prior_exec.get("lock_hash") != lock.get("lock_hash"):
            raise RuntimeError("Existing FINAL_TEST_EXECUTED.json belongs to another lock")
        print("FINAL TEST already completed under the verified lock; returning existing results.")
        return collect_results(root,final=True)
    if started.exists():
        prior=read_json(started)
        if prior.get("lock_hash")!=lock.get("lock_hash"): raise RuntimeError("Final campaign started under different lock")
        print("RESUME FINAL TEST campaign under same frozen lock")
    else:
        atomic_write_json(started,{"status":"FINAL_TEST_STARTED","started_unix":time.time(),"lock_hash":lock["lock_hash"],"code_hash":lock["code_hash"],
                                   "rule":"One frozen campaign; may resume after interruption; no retuning/reselection."})
    prepared={}
    for dataset in sorted({x["config"]["dataset"] for x in lock["final_plan"]}):
        prepared[dataset]=prepare_and_tokenize(root,code_root,dataset,include_test=True)
    outputs=[]
    for item in lock["final_plan"]:
        cfg=TrainConfig(**item["config"]); dev=Path(item["validation_run_dir"])
        complete=read_json(dev/"COMPLETE.json")
        from .utils import tree_hash
        if complete.get("final_model_tree_hash")!=item["final_model_tree_hash"] or tree_hash(dev/"final_model")!=item["final_model_tree_hash"]:
            raise RuntimeError(f"Frozen model hash mismatch before test: {dev}")
        if complete.get("provenance_hash") != item.get("provenance_hash"):
            raise RuntimeError(f"Frozen run provenance mismatch before test: {dev}")
        out=evaluate_saved_model_on_test(cfg,data_root=paths["data"],tokenized_test=prepared[cfg.dataset]["tokenized"]["test"],
                                         development_run_dir=dev,final_run_dir=_run_dir(paths,cfg,final=True),code_hash=lock["code_hash"],
                                         frozen_threshold=float(item["frozen_global_threshold"]),
                                         campaign_lock_hash=str(lock["lock_hash"]))
        outputs.append(out)
    atomic_write_json(executed,{"status":"FINAL_TEST_EXECUTED","completed_unix":time.time(),"lock_hash":lock["lock_hash"],"number_of_runs":len(outputs)})
    collect_results(root,final=True); return outputs

def estimate_model_storage(root: str | Path) -> dict[str, Any]:
    """Estimate persistent Drive storage required by the predeclared study.

    Exact final-epoch model retention is a scientific requirement because the
    one-shot final test evaluates those exact weights rather than retraining.
    This report is advisory only; it never changes the protocol.
    """
    paths = project_paths(root)
    model_dirs = [p for p in paths["runs"].rglob("final_model") if p.is_dir()]
    sizes = []
    for d in model_dirs:
        total = sum(x.stat().st_size for x in d.rglob("*") if x.is_file())
        if total > 0:
            sizes.append(total)
    # Planned unique retained models: Reuters 8x5 + param-PCGrad 3;
    # AAPD 8x5 + param-PCGrad 3; EURLEX 7x3. QUICK/FULL3/PAPER5 reuse paths.
    planned_models = 43 + 43 + 21
    fallback_bytes = 500 * 1024**2
    typical = int(np.median(sizes)) if sizes else fallback_bytes
    current = int(sum(sizes))
    projected = int(planned_models * typical)
    report = {
        "retained_models_current": len(sizes),
        "retained_models_planned_unique": planned_models,
        "typical_model_bytes": typical,
        "typical_model_gib": typical / 1024**3,
        "current_model_storage_gib": current / 1024**3,
        "projected_model_storage_gib": projected / 1024**3,
        "projected_remaining_model_storage_gib": max(0, projected-current) / 1024**3,
        "note": (
            "Advisory estimate only. Do not delete final_model directories or switch to test-time retraining to save space. "
            "Choose a Drive root with sufficient quota before expensive profiles."
        ),
    }
    atomic_write_json(paths["reports"] / "storage_estimate.json", report)
    return report


def _paired_seedwise_non_decomposable(rows: list[dict[str, Any]], common: list[int],
                                        method_a: str, method_b: str) -> dict[str, Any]:
    by = {(r["method"], int(r["seed"])): r for r in rows}
    out: dict[str, Any] = {}
    for metric in ("macro_ap", "tail_ap", "head_ap", "micro_ap"):
        diffs = []
        for sd in common:
            a = by[(method_a, sd)].get(metric); b = by[(method_b, sd)].get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(float(a)) and math.isfinite(float(b)):
                diffs.append({"seed": sd, "A_minus_B": float(a) - float(b)})
        if diffs:
            arr = np.asarray([x["A_minus_B"] for x in diffs], dtype=float)
            out[metric] = {
                "paired_seed_differences": diffs,
                "mean_A_minus_B": float(arr.mean()),
                "sd_A_minus_B": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "note": "Seed-wise paired estimate; AP is non-decomposable and is not document-bootstrapped.",
            }
    return out


def paired_bootstrap_report(root: str | Path, dataset: str, method_a: str = "ctpu", method_b: str = "pairwise",
                            reps: int = 2000, seed: int = 20261205, final: bool = False) -> dict[str, Any]:
    """Paired crossed seed×document bootstrap on saved predictions.

    Development uses validation predictions; final inference uses the one-shot frozen
    test predictions. Non-decomposable AP metrics are reported only as paired seed-wise
    differences.
    """
    from .statistics import crossed_paired_bootstrap, save_bootstrap
    paths = project_paths(root)
    rows = [r for r in _scan_complete(paths["root"], final=final)
            if r["dataset"] == dataset and int(r.get("epochs") or 0) == 4 and r["method"] in {method_a, method_b}]
    by = {(r["method"], int(r["seed"])): r for r in rows}
    common = sorted(set(sd for m,sd in by if m == method_a) & set(sd for m,sd in by if m == method_b))
    if len(common) < 3:
        raise RuntimeError(f"Need at least 3 matched seeds; found {common}")
    pairs = [(by[(method_a,sd)]["run_dir"], by[(method_b,sd)]["run_dir"]) for sd in common]
    prefix = "test" if final else "validation"
    result = crossed_paired_bootstrap(pairs, reps=reps, seed=seed, prefix=prefix)
    result.update({
        "dataset": dataset, "method_A": method_a, "method_B": method_b,
        "matched_seeds": common, "split": prefix, "final_one_shot": bool(final),
        "non_decomposable_seedwise": _paired_seedwise_non_decomposable(rows, common, method_a, method_b),
    })
    tag = "final_test" if final else "development"
    out = paths["reports"] / f"bootstrap_{tag}_{dataset}_{method_a}_vs_{method_b}.json"
    save_bootstrap(out, result)
    return result


def final_confirmatory_report(root: str | Path, reps: int = 5000, seed: int = 20261205) -> dict[str, Any]:
    """Generate the predeclared estimation-first primary final report after one-shot test completion."""
    paths = project_paths(root)
    executed = paths["protocol"] / "FINAL_TEST_EXECUTED.json"
    if not executed.exists():
        raise RuntimeError("Final confirmatory report requires a completed frozen final-test campaign")
    collect_results(root, final=True)
    comparisons = {}
    for dataset in ("reuters21578", "aapd", "eurlex57k"):
        comparisons[dataset] = paired_bootstrap_report(
            root, dataset, "ctpu", "pairwise", reps=reps, seed=stable_int_seed(seed, dataset), final=True
        )
    report = {
        "analysis_type": "predeclared_estimation_first_confirmatory",
        "primary_contrast": "ctpu_minus_pairwise",
        "primary_endpoint": "ranking_loss",
        "mechanism_endpoint": "soft_cardinality_mae",
        "safety_endpoints": ["nll", "brier"],
        "bootstrap_design": "paired_crossed_seed_by_document",
        "multiplicity_note": "CIs/effect estimates are primary; no post-test model or threshold selection is permitted.",
        "comparisons": comparisons,
    }
    atomic_write_json(paths["reports"] / "FINAL_CONFIRMATORY_REPORT.json", report)
    return report

