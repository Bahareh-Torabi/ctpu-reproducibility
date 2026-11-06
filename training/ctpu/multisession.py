from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .config import FULL_METHODS, PROJECT_VERSION, SEEDS_3, SEEDS_5, TrainConfig
from .utils import atomic_write_json, object_hash, read_json, reproducibility_environment_fingerprint, sha256_file, stable_int_seed, tree_hash

MULTISESSION_PROTOCOL_VERSION = "ctpu-multisession-r7-20260808"

def _exp():
    from . import experiment as e
    return e

def _train():
    from . import train as t
    return t
GLOBAL_COMMITMENT_FILENAME = "GLOBAL_PROTOCOL_COMMITMENT.json"
DATASET_LOCK_FILENAME = "DATASET_PROTOCOL_LOCK.json"
DATASET_TEST_STARTED_FILENAME = "DATASET_FINAL_TEST_STARTED.json"
DATASET_TEST_EXECUTED_FILENAME = "DATASET_FINAL_TEST_EXECUTED.json"

# Campaign orchestration retains the configured methods and seed counts.
# Local model cleanup is permitted only after final evaluation and export.
_DATASET_MAIN_METHODS = {
    "reuters21578": tuple(FULL_METHODS),
    "aapd": tuple(FULL_METHODS),
    "eurlex57k": tuple(m for m in FULL_METHODS if m != "score_auxproj"),
}
_DATASET_MAIN_SEEDS = {
    "reuters21578": tuple(SEEDS_5),
    "aapd": tuple(SEEDS_5),
    "eurlex57k": tuple(SEEDS_3),
}
_DATASET_PARAM_PCGRAD_SEEDS = {
    "reuters21578": tuple(SEEDS_3),
    "aapd": tuple(SEEDS_3),
    "eurlex57k": tuple(),
}


def campaign_plan(dataset: str) -> list[tuple[str, int]]:
    if dataset not in _DATASET_MAIN_METHODS:
        raise ValueError(f"Unknown campaign dataset: {dataset}")
    plan: list[tuple[str, int]] = []
    for method in _DATASET_MAIN_METHODS[dataset]:
        for seed in _DATASET_MAIN_SEEDS[dataset]:
            plan.append((method, int(seed)))
    for seed in _DATASET_PARAM_PCGRAD_SEEDS[dataset]:
        plan.append(("param_pcgrad", int(seed)))
    return plan


def expected_campaign_run_count(dataset: str) -> int:
    return len(campaign_plan(dataset))


def _scientific_commitment_payload(root: str | Path, code_root: str | Path) -> dict[str, Any]:
    root = Path(root)
    code_root = Path(code_root)
    revision = _exp().resolve_model_revision(root)
    code_hash = _exp().compute_code_hash(code_root)
    all_configs: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("reuters21578", "aapd", "eurlex57k"):
        cfgs: list[dict[str, Any]] = []
        for method, seed in campaign_plan(dataset):
            cfg = _exp().make_train_config(root, dataset, method, seed=seed)
            cfgs.append(cfg.to_dict())
        all_configs[dataset] = cfgs
    return {
        "multisession_protocol_version": MULTISESSION_PROTOCOL_VERSION,
        "project_version": PROJECT_VERSION,
        "code_hash": code_hash,
        "model_name": "FacebookAI/roberta-base",
        "model_revision": revision,
        "environment_fingerprint": reproducibility_environment_fingerprint(),
        "environment_fingerprint_hash": object_hash(reproducibility_environment_fingerprint()),
        "datasets_in_fixed_order": ["reuters21578", "aapd", "eurlex57k"],
        "campaigns": {
            ds: {
                "main_methods": list(_DATASET_MAIN_METHODS[ds]),
                "main_seeds": list(_DATASET_MAIN_SEEDS[ds]),
                "param_pcgrad_seeds": list(_DATASET_PARAM_PCGRAD_SEEDS[ds]),
                "expected_final_test_runs": expected_campaign_run_count(ds),
                "scientific_configs": all_configs[ds],
            }
            for ds in ("reuters21578", "aapd", "eurlex57k")
        },
        "primary_contrast": "ctpu_minus_pairwise",
        "primary_endpoint": "ranking_loss",
        "mechanism_endpoint": "soft_cardinality_mae",
        "safety_endpoints": ["nll", "brier"],
        "rules": {
            "all_methods_seeds_configs_committed_before_first_dataset_test": True,
            "no_outcome_dependent_method_seed_or_hyperparameter_changes": True,
            "fixed_final_epoch_four": True,
            "no_early_stopping": True,
            "no_best_checkpoint_selection": True,
            "final_test_uses_exact_same_session_development_weights": True,
            "no_test_time_retraining": True,
            "dataset_test_is_one_shot_under_dataset_lock": True,
            "dataset_campaigns_may_run_in_separate_colab_sessions": True,
            "all_sessions_must_match_global_code_model_environment_fingerprint": True,
            "test_results_from_an_earlier_dataset_must_not_change_later_campaigns": True,
            "bootstrap_design": "crossed_seed_by_document_paired",
        },
    }


def create_or_verify_global_commitment(root: str | Path, code_root: str | Path,
                                       persistent_protocol_dir: str | Path) -> dict[str, Any]:
    """Create the immutable global plan before any campaign test, or verify it exactly.

    The persistent file is intentionally small and must live on Drive. It commits
    all datasets/methods/seeds/scientific configs before the first dataset-specific
    final test can be opened. Later sessions may not alter the committed plan.
    """
    root = Path(root)
    code_root = Path(code_root)
    persistent_protocol_dir = Path(persistent_protocol_dir)
    persistent_protocol_dir.mkdir(parents=True, exist_ok=True)
    path = persistent_protocol_dir / GLOBAL_COMMITMENT_FILENAME
    current = _scientific_commitment_payload(root, code_root)
    current_core_hash = object_hash(current)
    if path.exists():
        prior = read_json(path)
        if prior.get("commitment_hash") != object_hash({k: v for k, v in prior.items() if k != "commitment_hash"}):
            raise RuntimeError("Persistent GLOBAL_PROTOCOL_COMMITMENT integrity check failed")
        prior_core = {k: v for k, v in prior.items() if k not in {"created_unix", "commitment_hash"}}
        current_core = dict(current)
        if object_hash(prior_core) != object_hash(current_core):
            raise RuntimeError(
                "GLOBAL protocol commitment mismatch. Code/model/environment/config changed after commitment; "
                "do not continue or overwrite the commitment."
            )
        return prior
    obj = dict(current)
    obj["created_unix"] = time.time()
    obj["commitment_hash"] = object_hash({k: v for k, v in obj.items() if k != "commitment_hash"})
    atomic_write_json(path, obj)
    return obj


def install_global_commitment_into_session(root: str | Path, code_root: str | Path,
                                           persistent_protocol_dir: str | Path) -> dict[str, Any]:
    """Pin the committed HF revision locally, then verify code/environment/config identity."""
    root = Path(root)
    paths = _exp().project_paths(root)
    p = Path(persistent_protocol_dir) / GLOBAL_COMMITMENT_FILENAME
    if not p.exists():
        raise RuntimeError("GLOBAL_PROTOCOL_COMMITMENT.json does not exist; run INIT first")
    obj = read_json(p)
    expected_hash = object_hash({k: v for k, v in obj.items() if k != "commitment_hash"})
    if obj.get("commitment_hash") != expected_hash:
        raise RuntimeError("GLOBAL commitment file was modified")
    atomic_write_json(paths["protocol"] / "model_revision.json", {
        "model_name": obj["model_name"],
        "revision": obj["model_revision"],
        "resolved_at_unix": obj.get("created_unix"),
        "rule": "Pinned by persistent GLOBAL_PROTOCOL_COMMITMENT before dataset final tests.",
    })
    current = _scientific_commitment_payload(root, code_root)
    prior_core = {k: v for k, v in obj.items() if k not in {"created_unix", "commitment_hash"}}
    if object_hash(prior_core) != object_hash(current):
        raise RuntimeError("Current session does not match the immutable GLOBAL protocol commitment")
    atomic_write_json(paths["protocol"] / GLOBAL_COMMITMENT_FILENAME, obj)
    return obj


def run_dataset_development(root: str | Path, code_root: str | Path, dataset: str) -> list[dict[str, Any]]:
    """Run the prespecified matched development suite for exactly one dataset."""
    root = Path(root)
    code_root = Path(code_root)
    _exp().verify_quick_gate_binding(root, code_root)
    paths = _exp().project_paths(root)
    prep = _exp().prepare_and_tokenize(root, code_root, dataset, include_test=False)
    if prep["audit"].get("test_accessed"):
        raise RuntimeError("Development preparation unexpectedly accessed TEST")
    _exp()._audit_guard(prep["audit"])
    code_hash = _exp().compute_code_hash(code_root)
    outputs: list[dict[str, Any]] = []
    for method, seed in campaign_plan(dataset):
        cfg = _exp().make_train_config(root, dataset, method, seed=seed)
        out = _train().run_training(
            cfg,
            data_root=paths["data"],
            tokenized_train=prep["tokenized"]["train"],
            tokenized_validation=prep["tokenized"]["validation"],
            run_dir=_exp()._run_dir(paths, cfg, final=False),
            code_hash=code_hash,
        )
        outputs.append(out)
    _exp().collect_results(root, final=False)
    _exp().paired_bootstrap_report(root, dataset, "ctpu", "pairwise", reps=5000,
                            seed=stable_int_seed(20261205, dataset), final=False)
    return outputs


def _target_runs(root: Path, dataset: str) -> list[tuple[TrainConfig, Path, dict[str, Any]]]:
    out = []
    for method, seed in campaign_plan(dataset):
        cfg = _exp().make_train_config(root, dataset, method, seed=seed)
        dev = _exp()._find_dev_run(root, cfg)
        cp = dev / "COMPLETE.json"
        if not cp.exists():
            raise RuntimeError(f"Missing completed development run: {dev}")
        complete = read_json(cp)
        out.append((cfg, dev, complete))
    return out


def freeze_dataset_campaign(root: str | Path, code_root: str | Path, dataset: str,
                            persistent_protocol_dir: str | Path) -> dict[str, Any]:
    """Freeze all exact weights and validation decision artifacts for one dataset.

    This is completeness/integrity based, never performance-selection based.
    """
    root = Path(root)
    code_root = Path(code_root)
    paths = _exp().project_paths(root)
    global_obj = install_global_commitment_into_session(root, code_root, persistent_protocol_dir)
    lock_path = paths["protocol"] / DATASET_LOCK_FILENAME
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock.get("lock_hash") != object_hash({k: v for k, v in lock.items() if k != "lock_hash"}):
            raise RuntimeError("Existing dataset lock integrity check failed")
        return lock
    current_env_hash = object_hash(reproducibility_environment_fingerprint())
    if current_env_hash != global_obj["environment_fingerprint_hash"]:
        raise RuntimeError("Current environment differs from globally committed environment")
    audit_path = paths["data"] / "prepared" / dataset / "audit.json"
    if not audit_path.exists():
        raise RuntimeError(f"Missing dataset audit before freeze: {audit_path}")
    audit = read_json(audit_path)
    _exp()._audit_guard(audit)
    if bool(audit.get("test_accessed", False)):
        raise RuntimeError("Cannot freeze dataset campaign after TEST has already been accessed")
    plan = []
    for cfg, dev, complete in _target_runs(paths["root"], dataset):
        model_dir = dev / "final_model"
        if not model_dir.exists() or not complete.get("exact_validation_model_retained"):
            raise RuntimeError(f"Exact final-epoch development model missing: {dev}")
        if complete.get("final_model_tree_hash") != tree_hash(model_dir):
            raise RuntimeError(f"Retained model hash mismatch before dataset freeze: {dev}")
        if complete.get("environment_fingerprint_hash") != current_env_hash:
            raise RuntimeError(f"Run environment mismatch before dataset freeze: {dev}")
        metrics = read_json(dev / "validation_metrics.json")
        plan.append({
            "config": cfg.to_dict(),
            "config_hash": object_hash(cfg.to_dict()),
            "validation_run_dir": str(dev),
            "frozen_global_threshold": float(metrics.get("validation_selected_global_threshold_macro_f1", 0.5)),
            "final_model_tree_hash": complete["final_model_tree_hash"],
            "provenance_hash": complete.get("provenance_hash"),
            "validation_artifact_hashes": _exp()._frozen_validation_artifact_hashes(dev),
        })
    if len(plan) != expected_campaign_run_count(dataset):
        raise RuntimeError("Dataset final plan is incomplete")
    obj = {
        "status": "DATASET_FROZEN",
        "dataset": dataset,
        "created_unix": time.time(),
        "multisession_protocol_version": MULTISESSION_PROTOCOL_VERSION,
        "global_commitment_hash": global_obj["commitment_hash"],
        "project_version": PROJECT_VERSION,
        "code_hash": _exp().compute_code_hash(code_root),
        "model_revision": global_obj["model_revision"],
        "environment_fingerprint_hash": current_env_hash,
        "dataset_audit_sha256": sha256_file(audit_path),
        "final_plan": plan,
        "rules": {
            "freeze_is_completeness_not_performance_selection": True,
            "all_prespecified_methods_and_seeds_included": True,
            "test_one_shot_for_this_dataset": True,
            "exact_retained_development_weights": True,
            "no_retraining": True,
            "validation_artifacts_sha256_locked": True,
            "global_plan_was_committed_before_first_dataset_test": True,
        },
    }
    obj["lock_hash"] = object_hash({k: v for k, v in obj.items() if k != "lock_hash"})
    atomic_write_json(lock_path, obj)
    persistent_locks = Path(persistent_protocol_dir) / "dataset_locks"
    persistent_locks.mkdir(parents=True, exist_ok=True)
    persistent_lock = persistent_locks / f"{dataset}__{DATASET_LOCK_FILENAME}"
    if persistent_lock.exists():
        prior = read_json(persistent_lock)
        if prior.get("lock_hash") != obj["lock_hash"]:
            raise RuntimeError("A different persistent dataset lock already exists")
    else:
        atomic_write_json(persistent_lock, obj)
    return obj


def _verify_dataset_lock(root: Path, code_root: Path, lock: dict[str, Any],
                         persistent_protocol_dir: Path) -> None:
    if lock.get("lock_hash") != object_hash({k: v for k, v in lock.items() if k != "lock_hash"}):
        raise RuntimeError("Dataset lock was modified")
    global_obj = install_global_commitment_into_session(root, code_root, persistent_protocol_dir)
    if lock.get("global_commitment_hash") != global_obj.get("commitment_hash"):
        raise RuntimeError("Dataset lock belongs to another global commitment")
    if _exp().compute_code_hash(code_root) != lock.get("code_hash"):
        raise RuntimeError("Code changed after dataset lock")
    if object_hash(reproducibility_environment_fingerprint()) != lock.get("environment_fingerprint_hash"):
        raise RuntimeError("Environment changed after dataset lock")
    audit_path = _exp().project_paths(root)["data"] / "prepared" / lock["dataset"] / "audit.json"
    if sha256_file(audit_path) != lock.get("dataset_audit_sha256"):
        raise RuntimeError("Dataset development audit changed after lock")
    for item in lock["final_plan"]:
        dev = Path(item["validation_run_dir"])
        if object_hash(item["config"]) != item["config_hash"]:
            raise RuntimeError("Frozen config hash mismatch")
        _exp()._verify_frozen_validation_artifacts(dev, item["validation_artifact_hashes"])
        if tree_hash(dev / "final_model") != item["final_model_tree_hash"]:
            raise RuntimeError(f"Frozen exact model changed: {dev}")


def run_dataset_final_test(root: str | Path, code_root: str | Path, dataset: str,
                           persistent_protocol_dir: str | Path) -> list[dict[str, Any]]:
    """One-shot exact-weight final test for one globally precommitted dataset campaign."""
    root = Path(root)
    code_root = Path(code_root)
    persistent_protocol_dir = Path(persistent_protocol_dir)
    paths = _exp().project_paths(root)
    lock_path = paths["protocol"] / DATASET_LOCK_FILENAME
    if not lock_path.exists():
        raise RuntimeError("Dataset campaign must be frozen before TEST")
    lock = read_json(lock_path)
    if lock.get("dataset") != dataset:
        raise RuntimeError("Dataset lock does not match requested campaign")
    _verify_dataset_lock(root, code_root, lock, persistent_protocol_dir)
    started = paths["protocol"] / DATASET_TEST_STARTED_FILENAME
    executed = paths["protocol"] / DATASET_TEST_EXECUTED_FILENAME
    persistent_state = persistent_protocol_dir / "dataset_test_state"
    persistent_state.mkdir(parents=True, exist_ok=True)
    persistent_started = persistent_state / f"{dataset}__{DATASET_TEST_STARTED_FILENAME}"
    persistent_executed = persistent_state / f"{dataset}__{DATASET_TEST_EXECUTED_FILENAME}"
    if persistent_executed.exists() and not executed.exists():
        raise RuntimeError("Persistent record says this dataset final TEST already completed in another session; refusing duplicate TEST")
    if persistent_started.exists() and not started.exists() and not persistent_executed.exists():
        raise RuntimeError(
            "Persistent record says this dataset final TEST began in a prior/lost runtime, but the exact local model session is unavailable. "
            "Refusing a silent repeat; this requires explicit protocol accounting."
        )
    if executed.exists():
        prior = read_json(executed)
        if prior.get("lock_hash") != lock["lock_hash"]:
            raise RuntimeError("Existing dataset TEST execution belongs to another lock")
        return _exp().collect_results(root, final=True)
    if started.exists():
        prior = read_json(started)
        if prior.get("lock_hash") != lock["lock_hash"]:
            raise RuntimeError("Dataset TEST was started under another lock")
    else:
        started_obj = {
            "status": "DATASET_FINAL_TEST_STARTED",
            "dataset": dataset,
            "started_unix": time.time(),
            "lock_hash": lock["lock_hash"],
            "global_commitment_hash": lock["global_commitment_hash"],
            "rule": "One frozen dataset campaign; interruption may resume under the exact same lock only.",
        }
        atomic_write_json(started, started_obj)
        atomic_write_json(persistent_started, started_obj)
    # No TEST parsing happens before every frozen validation artifact/model above is verified.
    prepared = _exp().prepare_and_tokenize(root, code_root, dataset, include_test=True)
    outputs: list[dict[str, Any]] = []
    for item in lock["final_plan"]:
        cfg = TrainConfig(**item["config"])
        dev = Path(item["validation_run_dir"])
        out = _train().evaluate_saved_model_on_test(
            cfg,
            data_root=paths["data"],
            tokenized_test=prepared["tokenized"]["test"],
            development_run_dir=dev,
            final_run_dir=_exp()._run_dir(paths, cfg, final=True),
            code_hash=lock["code_hash"],
            frozen_threshold=float(item["frozen_global_threshold"]),
            campaign_lock_hash=str(lock["lock_hash"]),
        )
        outputs.append(out)
    executed_obj = {
        "status": "DATASET_FINAL_TEST_EXECUTED",
        "dataset": dataset,
        "completed_unix": time.time(),
        "lock_hash": lock["lock_hash"],
        "global_commitment_hash": lock["global_commitment_hash"],
        "number_of_runs": len(outputs),
    }
    atomic_write_json(executed, executed_obj)
    atomic_write_json(persistent_executed, executed_obj)
    _exp().collect_results(root, final=True)
    _exp().paired_bootstrap_report(root, dataset, "ctpu", "pairwise", reps=5000,
                            seed=stable_int_seed(20261205, dataset), final=True)
    return outputs


def campaign_status(root: str | Path, dataset: str) -> dict[str, Any]:
    paths = _exp().project_paths(root)
    expected = expected_campaign_run_count(dataset)
    completed = 0
    with_model = 0
    for _, dev, _ in _target_runs(paths["root"], dataset):
        completed += int((dev / "COMPLETE.json").exists())
        with_model += int((dev / "final_model").exists())
    return {
        "dataset": dataset,
        "expected_runs": expected,
        "completed_runs": completed,
        "retained_exact_models": with_model,
        "dataset_lock_exists": (paths["protocol"] / DATASET_LOCK_FILENAME).exists(),
        "final_test_started": (paths["protocol"] / DATASET_TEST_STARTED_FILENAME).exists(),
        "final_test_executed": (paths["protocol"] / DATASET_TEST_EXECUTED_FILENAME).exists(),
    }


def delete_local_heavy_models_after_export(root: str | Path, dataset: str,
                                           require_test_executed: bool = True) -> dict[str, Any]:
    """Free ephemeral local storage only after the dataset one-shot TEST is complete/exported.

    This never deletes validation/test prediction artifacts or audit metadata. It is
    an orchestration utility and must not be called before final test completion.
    """
    root = Path(root)
    protocol_path = root / "protocol" / DATASET_TEST_EXECUTED_FILENAME
    if require_test_executed and not protocol_path.exists():
        raise RuntimeError("Refusing to delete exact models before dataset final TEST completion")
    paths = _exp().project_paths(root)
    removed = 0
    bytes_removed = 0
    for cfg, dev, _ in _target_runs(paths["root"], dataset):
        model_dir = dev / "final_model"
        if model_dir.exists():
            bytes_removed += sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
            shutil.rmtree(model_dir)
            removed += 1
        ckpt = dev / "resume_checkpoint.pt"
        if ckpt.exists():
            bytes_removed += ckpt.stat().st_size
            ckpt.unlink()
    return {"dataset": dataset, "model_dirs_removed": removed, "bytes_removed": bytes_removed}
