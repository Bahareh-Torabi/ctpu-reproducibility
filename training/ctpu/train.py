from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

from .auxiliary import build_auxiliary_gradient, pairwise_score_gradient
from .config import ASL_HPARAMS, WASL_HPARAMS, RuntimeConfig, TrainConfig
from .dataset import TokenizedMultiLabelDataset, make_collate_fn
from .losses import (
    AsymmetricLoss,
    WeightedAsymmetricLoss,
    bce_loss,
    build_ppmi,
    cardinality_penalty_term,
)
from .metrics import compute_metrics, per_label_metrics, sigmoid_np
from .utils import (
    atomic_torch_save,
    atomic_write_json,
    capture_rng_state,
    ensure_dir,
    environment_report,
    object_hash,
    restore_rng_state,
    reproducibility_environment_fingerprint,
    set_global_seed,
    sha256_file,
    stable_int_seed,
    tree_hash,
)

AUX_METHODS = {"pairwise", "normmatch", "ctpu", "score_auxproj", "param_pcgrad", "pairwise_cardinality_penalty"}


def compute_train_support(ds: TokenizedMultiLabelDataset, num_labels: int) -> np.ndarray:
    idx = np.asarray(ds.label_idx, dtype=np.int64)
    if idx.size == 0:
        return np.zeros(num_labels, dtype=np.int64)
    return np.bincount(idx, minlength=num_labels).astype(np.int64)


def dataset_label_lists(ds: TokenizedMultiLabelDataset) -> list[list[int]]:
    ptr = np.asarray(ds.label_ptr, dtype=np.int64)
    idx = np.asarray(ds.label_idx, dtype=np.int64)
    return [idx[ptr[i]:ptr[i + 1]].astype(int).tolist() for i in range(len(ds))]


def _autocast_ctx(cfg: TrainConfig):
    enabled = bool(cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def _worker_init(worker_id: int):
    # Data loading itself is deterministic, but this prevents future transforms from
    # silently inheriting non-reproducible worker RNG states.
    seed = torch.initial_seed() % (2**32 - 1)
    np.random.seed(seed)


def _candidate_micro_batches(effective: int, max_candidate: int) -> list[int]:
    vals = []
    for m in range(min(effective, max_candidate), 0, -1):
        if effective % m == 0 and (m & (m - 1) == 0):
            vals.append(m)
    if 1 not in vals:
        vals.append(1)
    return vals


def probe_micro_batch(model, ds: TokenizedMultiLabelDataset, cfg: TrainConfig, device: torch.device,
                      asl: AsymmetricLoss | None = None, wasl: WeightedAsymmetricLoss | None = None,
                      eligible_labels: Sequence[int] | None = None) -> tuple[int, int]:
    """Method-aware reversible OOM probe.

    Each candidate executes the same backward pathway as the requested method,
    including pairwise auxiliary injection or two-task parameter PCGrad. RNG state
    is restored after every candidate so the probe cannot perturb the study.
    """
    if not torch.cuda.is_available():
        micro = min(4, cfg.effective_batch_size)
        return micro, max(1, cfg.effective_batch_size // micro)
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    heuristic = (32 if gb >= 35 else 16) if cfg.max_length <= 256 else (16 if gb >= 35 else 8)
    candidates = _candidate_micro_batches(cfg.effective_batch_size, heuristic)
    initial_state = capture_rng_state()
    collate = make_collate_fn(ds.num_labels)
    for micro in candidates:
        restore_rng_state(initial_state)
        model.train()
        model.zero_grad(set_to_none=True)
        try:
            batch = collate([ds[i] for i in range(min(micro, len(ds)))])
            ids = batch["input_ids"].to(device, non_blocking=False)
            mask = batch["attention_mask"].to(device, non_blocking=False)
            y = batch["targets"].to(device, non_blocking=False)
            with _autocast_ctx(cfg):
                logits = model(input_ids=ids, attention_mask=mask).logits
                scalar_loss, _ = _scalar_training_loss(cfg.method, logits, y, cfg, asl, wasl)
            if cfg.method == "param_pcgrad":
                g, _, _, _ = pairwise_score_gradient(
                    logits, batch["label_indices"], batch["sample_ids"], 1, cfg.seed,
                    k_pos=cfg.pair_k_pos, k_neg=cfg.pair_k_neg, eligible_labels=eligible_labels,
                )
                dummy: dict[str, list[float]] = defaultdict(list)
                _parameter_pcgrad_step(model, scalar_loss, logits, g.to(device), 1.0,
                                       cfg.lambda_pair / float(max(1, micro)), dummy)
            else:
                has_aux = cfg.method in AUX_METHODS
                scalar_loss.backward(retain_graph=has_aux)
                if has_aux:
                    aux = build_auxiliary_gradient(
                        cfg.method if cfg.method != "pairwise_cardinality_penalty" else "pairwise",
                        logits, y, batch["label_indices"], batch["sample_ids"], 1, cfg.seed,
                        k_pos=cfg.pair_k_pos, k_neg=cfg.pair_k_neg, eligible_labels=eligible_labels,
                    )
                    torch.autograd.backward(
                        logits, grad_tensors=(cfg.lambda_pair * aux.grad.to(device) / float(max(1, micro))).to(logits.dtype)
                    )
            model.zero_grad(set_to_none=True)
            del ids, mask, y, logits, scalar_loss, batch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            restore_rng_state(initial_state)
            return micro, cfg.effective_batch_size // micro
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
    restore_rng_state(initial_state)
    raise RuntimeError("No safe method-aware microbatch found, even microbatch=1")


def _make_loader(ds: TokenizedMultiLabelDataset, batch_size: int, shuffle: bool, seed: int,
                 num_workers: int, pin_memory: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=make_collate_fn(ds.num_labels),
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=False,
        drop_last=False,
    )


def _accumulate_diag(acc: dict[str, list[float]], diag: dict[str, float]) -> None:
    for k, v in diag.items():
        try:
            x = float(v)
            if math.isfinite(x):
                acc[k].append(x)
        except Exception:
            pass


def _mean_diag(acc: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) for k, v in sorted(acc.items()) if v}


def _label_structure_fingerprint(ds: TokenizedMultiLabelDataset) -> str:
    """Fingerprint the exact TRAIN label incidence used by a cached structured baseline."""
    h = hashlib.sha256()
    h.update(np.asarray(ds.label_ptr, dtype=np.int64).tobytes(order="C"))
    h.update(np.asarray(ds.label_idx, dtype=np.int64).tobytes(order="C"))
    h.update(str(int(ds.num_labels)).encode("ascii"))
    return h.hexdigest()


def _build_wasl(train_ds: TokenizedMultiLabelDataset, train_support: np.ndarray, dataset: str,
                cache_root: Path) -> WeightedAsymmetricLoss:
    hp = WASL_HPARAMS[dataset]
    ppmi_dir = ensure_dir(cache_root / "wasl")
    train_fp = _label_structure_fingerprint(train_ds)
    ppmi_path = ppmi_dir / f"{dataset}_ppmi_zero_diag_{train_fp[:16]}.npy"
    manifest_path = ppmi_dir / f"{dataset}_ppmi_manifest_{train_fp[:16]}.json"
    if ppmi_path.exists():
        ppmi = np.load(ppmi_path, mmap_mode=None)
        if not manifest_path.exists():
            raise RuntimeError("WASL PPMI cache exists without its provenance manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("train_label_structure_sha256") != train_fp:
            raise RuntimeError("WASL PPMI cache fingerprint mismatch")
    else:
        ppmi = build_ppmi(dataset_label_lists(train_ds), train_ds.num_labels, zero_diagonal=True)
        np.save(ppmi_path, ppmi)
        atomic_write_json(manifest_path, {
            "dataset": dataset,
            "num_labels": int(train_ds.num_labels),
            "n_train": len(train_ds),
            "zero_diagonal": True,
            "train_label_structure_sha256": train_fp,
            "note": "Transparent WASL reimplementation; paper text did not specify PPMI diagonal handling.",
        })
    return WeightedAsymmetricLoss(train_support=train_support, ppmi=ppmi, **hp)


def _scalar_training_loss(method: str, logits: torch.Tensor, targets: torch.Tensor,
                          cfg: TrainConfig, asl: AsymmetricLoss | None,
                          wasl: WeightedAsymmetricLoss | None) -> tuple[torch.Tensor, dict[str, float]]:
    if method in {"bce", "pairwise", "normmatch", "ctpu", "score_auxproj", "param_pcgrad"}:
        loss = bce_loss(logits, targets)
        return loss, {"bce_loss": float(loss.detach().item())}
    if method == "pairwise_cardinality_penalty":
        base = bce_loss(logits, targets)
        pen = cardinality_penalty_term(logits, targets)
        loss = base + cfg.cardinality_penalty_mu * pen
        return loss, {
            "bce_loss": float(base.detach().item()),
            "cardinality_penalty": float(pen.detach().item()),
            "scalar_loss": float(loss.detach().item()),
        }
    if method in {"asl", "asl_official_sum"}:
        assert asl is not None
        loss = asl(logits, targets)
        raw = float(loss.detach().item())
        denom = float(max(1, logits.shape[0] * logits.shape[1]))
        return loss, {
            "asl_loss_normalized_per_element_diagnostic": raw / denom if method == "asl_official_sum" else raw,
            "asl_reference_sum_raw_diagnostic": raw if method == "asl_official_sum" else float("nan"),
            "asl_reduction_is_official_sum": float(method == "asl_official_sum"),
        }
    if method == "wasl":
        assert wasl is not None
        loss = wasl(logits, targets)
        return loss, {"wasl_loss": float(loss.detach().item())}
    raise ValueError(method)


def _parameter_task_grads(model, scalar_loss: torch.Tensor, logits: torch.Tensor,
                          aux_score_grad: torch.Tensor, scalar_scale: float,
                          aux_scale: float):
    params = [p for p in model.parameters() if p.requires_grad]
    pgrads = torch.autograd.grad(scalar_loss * scalar_scale, params, retain_graph=True, allow_unused=True)
    agrads = torch.autograd.grad(
        logits, params, grad_outputs=(aux_score_grad * aux_scale).to(logits.dtype),
        retain_graph=False, allow_unused=True,
    )
    return params, pgrads, agrads


def _accumulate_task_grads(accum, grads):
    if accum is None:
        return [None if g is None else g.detach().clone() for g in grads]
    for i, g in enumerate(grads):
        if g is None:
            continue
        if accum[i] is None:
            accum[i] = g.detach().clone()
        else:
            accum[i].add_(g.detach())
    return accum


def _scale_task_grads(grads, scale: float) -> None:
    if scale == 1.0 or grads is None:
        return
    for g in grads:
        if g is not None:
            g.mul_(float(scale))


def _apply_two_task_pcgrad(params, pgrads, agrads, diagnostics: dict[str, list[float]]) -> None:
    """Apply deterministic symmetric two-task PCGrad to accumulated task gradients."""
    device = next(p.device for p in params)
    dot = torch.zeros((), device=device, dtype=torch.float32)
    pnorm = torch.zeros_like(dot)
    anorm = torch.zeros_like(dot)
    for pg, ag in zip(pgrads, agrads):
        if pg is not None:
            pnorm = pnorm + (pg.detach().float() ** 2).sum()
        if ag is not None:
            anorm = anorm + (ag.detach().float() ** 2).sum()
        if pg is not None and ag is not None:
            dot = dot + (pg.detach().float() * ag.detach().float()).sum()
    conflict = bool(dot.item() < 0.0 and pnorm.item() > 1e-30 and anorm.item() > 1e-30)
    p_coeff = float((dot / anorm).item()) if conflict else 0.0
    a_coeff = float((dot / pnorm).item()) if conflict else 0.0
    for param, pg, ag in zip(params, pgrads, agrads):
        if pg is None and ag is None:
            continue
        if pg is None:
            total = ag
        elif ag is None:
            total = pg
        elif conflict:
            total = (pg - p_coeff * ag) + (ag - a_coeff * pg)
        else:
            total = pg + ag
        param.grad = total.detach().clone()
    diagnostics["param_pcgrad_conflict"].append(float(conflict))
    diagnostics["param_pcgrad_primary_aux_dot"].append(float(dot.item()))
    diagnostics["param_pcgrad_primary_norm"].append(float(torch.sqrt(pnorm).item()))
    diagnostics["param_pcgrad_aux_norm"].append(float(torch.sqrt(anorm).item()))


def _parameter_pcgrad_step(model, scalar_loss: torch.Tensor, logits: torch.Tensor,
                            aux_score_grad: torch.Tensor, scalar_scale: float,
                            aux_scale: float, diagnostics: dict[str, list[float]]) -> None:
    """Single-batch wrapper used by the OOM probe and unit tests.

    The training loop accumulates both task gradients over the complete effective
    batch and applies the same symmetric projection only at the optimizer boundary.
    """
    params, pgrads, agrads = _parameter_task_grads(
        model, scalar_loss, logits, aux_score_grad, scalar_scale, aux_scale
    )
    _apply_two_task_pcgrad(params, list(pgrads), list(agrads), diagnostics)


def _rescale_partial_window(model, effective_batch: int, actual_examples: int) -> None:
    if actual_examples <= 0 or actual_examples == effective_batch:
        return
    scale = float(effective_batch) / float(actual_examples)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


def evaluate_model(model, ds: TokenizedMultiLabelDataset, cfg: TrainConfig, device: torch.device,
                   train_support: Sequence[int], split_name: str = "validation",
                   tune_threshold: bool = True, eval_batch_size: int | None = None) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str]]:
    model.eval()
    # Inference allows a larger batch than training; cap at 2x selected train microbatch elsewhere.
    loader = _make_loader(
        ds, batch_size=int(eval_batch_size or cfg.eval_batch_size_cap), shuffle=False,
        seed=cfg.seed, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    logits_all, y_all, ids_all = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{cfg.dataset} {cfg.method} seed={cfg.seed} {split_name}", leave=False):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            with _autocast_ctx(cfg):
                logits = model(input_ids=ids, attention_mask=mask).logits
            logits_all.append(logits.detach().float().cpu().numpy())
            y_all.append(batch["targets"].numpy().astype(np.uint8, copy=False))
            ids_all.extend(batch["sample_ids"])
    logits_np = np.concatenate(logits_all, axis=0).astype(np.float32, copy=False)
    y_np = np.concatenate(y_all, axis=0).astype(np.uint8, copy=False)
    metrics = compute_metrics(
        y_np, logits_np, train_support,
        threshold_fixed=cfg.threshold_fixed,
        tune_threshold=tune_threshold,
        threshold_grid=(cfg.threshold_grid_start, cfg.threshold_grid_stop, cfg.threshold_grid_step),
    )
    return metrics, logits_np, y_np, ids_all



def _save_per_label_csv(run_dir: Path, prefix: str, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(run_dir / f"{prefix}_per_label.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def _save_predictions(run_dir: Path, prefix: str, logits: np.ndarray, y: np.ndarray, sample_ids: list[str]) -> None:
    np.save(run_dir / f"{prefix}_logits.npy", logits.astype(np.float32, copy=False))
    np.save(run_dir / f"{prefix}_labels.npy", y.astype(np.uint8, copy=False))
    (run_dir / f"{prefix}_sample_ids.json").write_text(
        json.dumps(sample_ids, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _load_checkpoint(path: Path, model, optimizer, scheduler, expected_hash: str, expected_code_hash: str, expected_provenance_hash: str) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if (ckpt.get("config_hash") != expected_hash or ckpt.get("code_hash") != expected_code_hash or
            ckpt.get("provenance_hash") != expected_provenance_hash):
        raise RuntimeError("Checkpoint config/code/data provenance mismatch; refusing unsafe resume")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    restore_rng_state(ckpt["rng_state"])
    return int(ckpt["completed_epoch"])


def _run_provenance(cfg: TrainConfig, data_root: str | Path, tokenized_train: str | Path,
                    tokenized_validation: str | Path, config_hash: str, code_hash: str) -> dict[str, str]:
    data_root = Path(data_root)
    audit_path = data_root / "prepared" / cfg.dataset / "audit.json"
    train_manifest = Path(tokenized_train) / "manifest.json"
    val_manifest = Path(tokenized_validation) / "manifest.json"
    for fp in (audit_path, train_manifest, val_manifest):
        if not fp.exists():
            raise RuntimeError(f"Missing provenance file: {fp}")
    data_obj = {
        "dataset_audit_sha256": sha256_file(audit_path),
        "train_token_manifest_sha256": sha256_file(train_manifest),
        "validation_token_manifest_sha256": sha256_file(val_manifest),
    }
    data_hash = object_hash(data_obj)
    env_fp = reproducibility_environment_fingerprint()
    env_hash = object_hash(env_fp)
    obj = {**data_obj, "data_provenance_hash": data_hash,
           "config_hash": config_hash, "code_hash": code_hash,
           "environment_fingerprint": env_fp,
           "environment_fingerprint_hash": env_hash}
    obj["provenance_hash"] = object_hash(obj)
    return obj


def _history_diagnostics(history: list[dict]) -> dict[str, Any]:
    excluded = {"epoch", "train_scalar_loss", "train_pairwise_loss_diagnostic",
                "epoch_elapsed_seconds", "elapsed_seconds_cumulative"}
    by_epoch = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    for rec in sorted(history, key=lambda x: int(x.get("epoch", 0))):
        d = {k: float(v) for k, v in rec.items()
             if k not in excluded and isinstance(v, (int, float)) and math.isfinite(float(v))}
        by_epoch.append({"epoch": int(rec.get("epoch", 0)), **d})
        for k, v in d.items():
            aggregate[k].append(v)
    return {"by_epoch": by_epoch, "mean_across_epochs": _mean_diag(aggregate)}


def _scalar_display_value(method: str, scalar_loss: torch.Tensor, bsz: int, num_labels: int) -> float:
    value = float(scalar_loss.detach().item())
    if method == "asl_official_sum":
        return value / float(max(1, bsz * num_labels))
    return value


def run_training(cfg: TrainConfig, data_root: str | Path, tokenized_train: str | Path,
                 tokenized_validation: str | Path, run_dir: str | Path, code_hash: str) -> dict[str, Any]:
    """Train one development run and retain the exact final-epoch model weights.

    Scientific invariants:
    - test is never accepted by this function;
    - exactly cfg.epochs are run (unless resuming completed epochs);
    - no early stopping and no best-checkpoint selection;
    - final validation logits and the exact model producing them are persisted.
    """
    run_dir = ensure_dir(run_dir)
    scientific = cfg.to_dict()
    atomic_write_json(run_dir / "config.json", scientific)
    config_hash = object_hash(scientific)
    completed_path = run_dir / "COMPLETE.json"
    final_model_dir = run_dir / "final_model"
    provenance = _run_provenance(cfg, data_root, tokenized_train, tokenized_validation, config_hash, code_hash)
    if completed_path.exists():
        done = json.loads(completed_path.read_text(encoding="utf-8"))
        if (done.get("config_hash") == config_hash and done.get("code_hash") == code_hash and
                done.get("provenance_hash") == provenance["provenance_hash"]):
            if cfg.retain_final_model:
                if not final_model_dir.exists() or done.get("final_model_tree_hash") != tree_hash(final_model_dir):
                    raise RuntimeError(
                        f"Completed run lacks the exact retained final model required for one-shot test: {run_dir}"
                    )
            return done
        raise RuntimeError(f"Completed run exists with incompatible config/code/data provenance: {run_dir}")

    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = json.loads((Path(data_root) / "prepared" / cfg.dataset / "labels.json").read_text(encoding="utf-8"))
    train_ds = TokenizedMultiLabelDataset(tokenized_train, num_labels=len(label_names))
    val_ds = TokenizedMultiLabelDataset(tokenized_validation, num_labels=train_ds.num_labels)
    train_support = compute_train_support(train_ds, train_ds.num_labels)
    np.save(run_dir / "train_support.npy", train_support)
    aux_eligible_labels = np.where(train_support > 0)[0].astype(int).tolist()
    if not aux_eligible_labels:
        raise RuntimeError("No TRAIN-seen labels; auxiliary ranking is undefined")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, revision=cfg.model_revision, num_labels=train_ds.num_labels,
        problem_type="multi_label_classification", attn_implementation=cfg.attn_implementation,
    ).to(device)
    model.config.problem_type = "multi_label_classification"

    asl = None
    if cfg.method == "asl":
        asl = AsymmetricLoss(**ASL_HPARAMS, reduction="mean")
    elif cfg.method == "asl_official_sum":
        asl = AsymmetricLoss(**ASL_HPARAMS, reduction="sum")
    wasl = _build_wasl(train_ds, train_support, cfg.dataset, Path(data_root) / "cache") if cfg.method == "wasl" else None
    if wasl is not None:
        wasl = wasl.to(device)

    # Matched BCE, Pairwise, NormMatch, and CTPU runs use the same conservative
    # CTPU memory path to select the microbatch. This keeps microbatch sizes
    # and dropout shapes consistent across the comparison.
    causal_core = {"bce", "pairwise", "normmatch", "ctpu"}
    if cfg.method in causal_core:
        probe_cfg = replace(cfg, method="ctpu")
        probe_asl = None
        probe_wasl = None
        probe_method = "ctpu-common-core"
        matched_core = True
    else:
        probe_cfg = cfg
        probe_asl = asl
        probe_wasl = wasl
        probe_method = cfg.method
        matched_core = False
    micro, accum = probe_micro_batch(
        model, train_ds, probe_cfg, device, asl=probe_asl, wasl=probe_wasl, eligible_labels=aux_eligible_labels
    )
    eval_batch = min(cfg.eval_batch_size_cap, max(micro, micro * 2))
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else None
    runtime = RuntimeConfig(
        effective_batch_size=cfg.effective_batch_size,
        micro_batch_size=int(micro), gradient_accumulation_steps=int(accum),
        eval_batch_size=int(eval_batch), gpu_name=gpu_name, gpu_total_memory_gb=gpu_gb,
        probe_method=probe_method, matched_causal_core_microbatch=matched_core,
    )
    atomic_write_json(run_dir / "runtime_config.json", runtime.to_dict())

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / cfg.effective_batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(round(total_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    atomic_write_json(run_dir / "environment.json", environment_report())
    atomic_write_json(run_dir / "manifest.json", {
        "config_hash": config_hash, "code_hash": code_hash, "provenance": provenance,
        "no_early_stopping": True, "checkpoint_selection": "final fixed epoch only",
        "test_loaded_during_development": False, "attention_implementation": cfg.attn_implementation,
        "train_rows": len(train_ds), "validation_rows": len(val_ds), "num_labels": train_ds.num_labels,
        "train_seen_labels": int((train_support > 0).sum()), "zero_train_support_labels": int((train_support == 0).sum()),
        "pairwise_auxiliary_eligible_labels": len(aux_eligible_labels),
        "pairwise_auxiliary_excludes_zero_train_support_labels": True,
        "matched_causal_core_microbatch_policy": bool(matched_core),
        "runtime_probe_method": probe_method,
        "optimizer_steps_per_epoch": steps_per_epoch, "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "asl_variant": (
            "normalized-elementwise-mean" if cfg.method == "asl" else
            "official-sum-reduction-check" if cfg.method == "asl_official_sum" else None
        ),
    })

    checkpoint_path = run_dir / "resume_checkpoint.pt"
    start_epoch = 0
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    if checkpoint_path.exists():
        start_epoch = _load_checkpoint(checkpoint_path, model, optimizer, scheduler, config_hash, code_hash, provenance["provenance_hash"])
        print(f"RESUME {cfg.dataset}/{cfg.method}/seed={cfg.seed} from completed epoch {start_epoch}")

    t0 = time.time()
    for epoch_idx in range(start_epoch, cfg.epochs):
        epoch_num = epoch_idx + 1
        model.train()
        epoch_t0 = time.time()
        epoch_loader = _make_loader(
            train_ds, batch_size=micro, shuffle=True,
            seed=stable_int_seed(cfg.seed, epoch_num, "dataloader"),
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )
        optimizer.zero_grad(set_to_none=True)
        pc_primary_accum = None; pc_aux_accum = None; pc_params = None
        window_examples=0; epoch_examples=0; scalar_weighted=0.0; pair_weighted=0.0; pair_valid_total=0.0
        epoch_diag: dict[str, list[float]] = defaultdict(list)
        progress=tqdm(epoch_loader, desc=f"{cfg.dataset} | {cfg.method} | seed={cfg.seed} | epoch={epoch_num}/{cfg.epochs}")
        for micro_idx,batch in enumerate(progress):
            bsz=len(batch["sample_ids"])
            ids=batch["input_ids"].to(device,non_blocking=True); mask=batch["attention_mask"].to(device,non_blocking=True); targets=batch["targets"].to(device,non_blocking=True)
            with _autocast_ctx(cfg):
                logits=model(input_ids=ids,attention_mask=mask).logits
                scalar_loss,scalar_diag=_scalar_training_loss(cfg.method,logits,targets,cfg,asl,wasl)
            # Mean-reduced matched-condition losses are accumulated as an exact
            # effective-batch mean. The optional ASL official-sum check instead
            # preserves the reference SUM reduction across accumulated microbatches.
            scalar_scale = (1.0 if cfg.method == "asl_official_sum"
                            else float(bsz)/float(cfg.effective_batch_size))

            if cfg.method == "param_pcgrad":
                g,active,pair_loss,pdiag=pairwise_score_gradient(
                    logits,batch["label_indices"],batch["sample_ids"],epoch_num,cfg.seed,
                    k_pos=cfg.pair_k_pos,k_neg=cfg.pair_k_neg,eligible_labels=aux_eligible_labels,
                )
                with torch.no_grad():
                    gbce=(torch.sigmoid(logits.detach().float())-targets.float())/float(logits.shape[1])
                    ratio=(cfg.lambda_pair*torch.linalg.vector_norm(g,dim=1)/torch.linalg.vector_norm(gbce,dim=1).clamp_min(1e-12)).mean()
                    epoch_diag["lambda_aux_to_bce_score_norm_ratio"].append(float(ratio.item()))
                params, pgrads, agrads = _parameter_task_grads(
                    model, scalar_loss, logits, g.to(device), scalar_scale,
                    cfg.lambda_pair/float(cfg.effective_batch_size)
                )
                pc_params = params
                pc_primary_accum = _accumulate_task_grads(pc_primary_accum, pgrads)
                pc_aux_accum = _accumulate_task_grads(pc_aux_accum, agrads)
                _accumulate_diag(epoch_diag,pdiag)
                valid_pairs = float(pdiag.get("valid_pair_samples", 0.0))
                pair_weighted += pair_loss * valid_pairs; pair_valid_total += valid_pairs
            else:
                has_aux=cfg.method in AUX_METHODS
                (scalar_loss*scalar_scale).backward(retain_graph=has_aux)
                if has_aux:
                    aux=build_auxiliary_gradient(
                        cfg.method if cfg.method != "pairwise_cardinality_penalty" else "pairwise",
                        logits,targets,batch["label_indices"],batch["sample_ids"],epoch_num,cfg.seed,
                        k_pos=cfg.pair_k_pos,k_neg=cfg.pair_k_neg,eligible_labels=aux_eligible_labels,
                    )
                    with torch.no_grad():
                        gbce=(torch.sigmoid(logits.detach().float())-targets.float())/float(logits.shape[1])
                        ratio=(cfg.lambda_pair*torch.linalg.vector_norm(aux.grad,dim=1)/torch.linalg.vector_norm(gbce,dim=1).clamp_min(1e-12)).mean()
                        epoch_diag["lambda_aux_to_bce_score_norm_ratio"].append(float(ratio.item()))
                    torch.autograd.backward(logits,grad_tensors=(cfg.lambda_pair*aux.grad.to(device)/float(cfg.effective_batch_size)).to(logits.dtype))
                    valid_pairs = float(aux.diagnostics.get("valid_pair_samples", 0.0))
                    pair_weighted += aux.pairwise_loss_mean * valid_pairs; pair_valid_total += valid_pairs
                    _accumulate_diag(epoch_diag,aux.diagnostics)

            scalar_weighted += _scalar_display_value(cfg.method, scalar_loss, bsz, logits.shape[1]) * bsz
            epoch_examples += bsz; window_examples += bsz; _accumulate_diag(epoch_diag,scalar_diag)
            is_boundary=window_examples>=cfg.effective_batch_size or micro_idx==len(epoch_loader)-1
            if is_boundary:
                if cfg.method == "param_pcgrad":
                    if pc_params is None or pc_primary_accum is None or pc_aux_accum is None:
                        raise RuntimeError("PCGrad optimizer boundary reached without accumulated task gradients")
                    if window_examples != cfg.effective_batch_size:
                        scale = float(cfg.effective_batch_size) / float(window_examples)
                        _scale_task_grads(pc_primary_accum, scale); _scale_task_grads(pc_aux_accum, scale)
                    _apply_two_task_pcgrad(pc_params, pc_primary_accum, pc_aux_accum, epoch_diag)
                    pc_primary_accum = None; pc_aux_accum = None; pc_params = None
                elif cfg.method != "asl_official_sum":
                    _rescale_partial_window(model,cfg.effective_batch_size,window_examples)
                grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm).item())
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                epoch_diag["preclip_grad_norm"].append(grad_norm); epoch_diag["learning_rate"].append(float(scheduler.get_last_lr()[0])); window_examples=0
            progress.set_postfix({"scalar":f"{scalar_weighted/max(1,epoch_examples):.5f}","pair":f"{pair_weighted/max(1.0,pair_valid_total):.5f}" if cfg.method in AUX_METHODS else "-","lr":f"{scheduler.get_last_lr()[0]:.2e}"})
            del ids,mask,targets,logits,scalar_loss

        epoch_elapsed = time.time() - epoch_t0
        prior_elapsed = sum(float(x.get("epoch_elapsed_seconds", 0.0)) for x in history if int(x.get("epoch", -1)) != epoch_num)
        epoch_record={"epoch":epoch_num,"train_scalar_loss":scalar_weighted/max(1,epoch_examples),
                      "train_pairwise_loss_diagnostic":pair_weighted/max(1.0,pair_valid_total),
                      "valid_pair_sample_rate":pair_valid_total/max(1,epoch_examples),
                      "epoch_elapsed_seconds":epoch_elapsed,"elapsed_seconds_cumulative":prior_elapsed+epoch_elapsed,
                      **_mean_diag(epoch_diag)}
        if cfg.validation_each_epoch:
            vm,_,_,_=evaluate_model(model,val_ds,cfg,device,train_support,tune_threshold=False,eval_batch_size=runtime.eval_batch_size)
            epoch_record.update({f"validation_{k}":v for k,v in vm.items() if isinstance(v,(int,float))})
        history=[x for x in history if int(x.get("epoch",-1))!=epoch_num]+[epoch_record]; history.sort(key=lambda x:int(x["epoch"])); atomic_write_json(history_path,history)
        atomic_torch_save(checkpoint_path,{"completed_epoch":epoch_num,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"rng_state":capture_rng_state(),"config_hash":config_hash,"code_hash":code_hash,"provenance_hash":provenance["provenance_hash"]})

    metrics,logits_np,y_np,sample_ids=evaluate_model(model,val_ds,cfg,device,train_support,split_name="validation",tune_threshold=True,eval_batch_size=runtime.eval_batch_size)
    atomic_write_json(run_dir/"validation_metrics.json",metrics); _save_predictions(run_dir,"validation",logits_np,y_np,sample_ids)
    _save_per_label_csv(run_dir,"validation",per_label_metrics(y_np,logits_np,train_support,threshold=0.5))
    atomic_write_json(run_dir/"gradient_diagnostics.json", _history_diagnostics(history))

    final_model_hash=None
    if cfg.retain_final_model:
        if final_model_dir.exists():
            import shutil; shutil.rmtree(final_model_dir)
        model.save_pretrained(final_model_dir, safe_serialization=True, max_shard_size="2GB")
        final_model_hash=tree_hash(final_model_dir)
    done={"status":"COMPLETE","dataset":cfg.dataset,"method":cfg.method,"seed":cfg.seed,"config_hash":config_hash,"code_hash":code_hash,"provenance_hash":provenance["provenance_hash"],"data_provenance_hash":provenance["data_provenance_hash"],"environment_fingerprint_hash":provenance["environment_fingerprint_hash"],"provenance":provenance,"validation_metrics":metrics,"elapsed_seconds":sum(float(x.get("epoch_elapsed_seconds",0.0)) for x in history),"final_epoch":cfg.epochs,"early_stopping":False,"final_model_tree_hash":final_model_hash,"exact_validation_model_retained":bool(cfg.retain_final_model)}
    atomic_write_json(completed_path,done)
    if checkpoint_path.exists(): checkpoint_path.unlink()
    gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return done


def evaluate_saved_model_on_test(cfg: TrainConfig, data_root: str | Path, tokenized_test: str | Path,
                                 development_run_dir: str | Path, final_run_dir: str | Path,
                                 code_hash: str, frozen_threshold: float,
                                 campaign_lock_hash: str) -> dict[str, Any]:
    """Evaluate the exact retained development model once; NO retraining occurs here.

    A completed run is safely reusable only inside the exact same frozen final
    campaign. This makes interruption recovery possible without repeatedly
    evaluating already completed test runs.
    """
    dev=Path(development_run_dir); out=ensure_dir(final_run_dir); complete=json.loads((dev/"COMPLETE.json").read_text(encoding="utf-8"))
    done_path = out / "FINAL_TEST_COMPLETE.json"
    if done_path.exists():
        prior = json.loads(done_path.read_text(encoding="utf-8"))
        if (prior.get("campaign_lock_hash") == campaign_lock_hash and
            prior.get("code_hash") == code_hash and
            prior.get("source_final_model_tree_hash") == complete.get("final_model_tree_hash")):
            return prior
        raise RuntimeError(f"Existing final-test result is incompatible with current frozen campaign: {out}")
    if complete.get("code_hash")!=code_hash: raise RuntimeError("Development model code hash mismatch")
    current_env_hash = object_hash(reproducibility_environment_fingerprint())
    if complete.get("environment_fingerprint_hash") != current_env_hash:
        raise RuntimeError(
            "Current software/CUDA compute-capability fingerprint differs from the environment that produced the frozen development model. "
            "Refusing numerically unmatched final evaluation."
        )
    final_model=dev/"final_model"
    if not final_model.exists() or complete.get("final_model_tree_hash")!=tree_hash(final_model): raise RuntimeError("Frozen development model hash mismatch")
    train_support=np.load(dev/"train_support.npy")
    labels=json.loads((Path(data_root)/"prepared"/cfg.dataset/"labels.json").read_text(encoding="utf-8"))
    test_ds=TokenizedMultiLabelDataset(tokenized_test,num_labels=len(labels))
    runtime=json.loads((dev/"runtime_config.json").read_text(encoding="utf-8"))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AutoModelForSequenceClassification.from_pretrained(final_model, attn_implementation=cfg.attn_implementation).to(device)
    tm,tlogits,ty,tids=evaluate_model(model,test_ds,cfg,device,train_support,split_name="test-final-one-shot",tune_threshold=False,eval_batch_size=int(runtime["eval_batch_size"]))
    # EURLEX57K: after the one-shot test is legitimately unlocked, add a
    # transparent full-4,271 benchmark view. The trained head remains the
    # leakage-resistant 4,108 TRAIN-seen head; unsupported zero-shot labels
    # receive near-zero probability (logit=-30) only for reporting.
    if cfg.dataset == "eurlex57k":
        prep=Path(data_root)/"prepared"/cfg.dataset
        full_labels_path=prep/"full_task_labels_final_only.json"
        raw_test_path=prep/"test_full_labels_final_only.jsonl"
        if full_labels_path.exists() and raw_test_path.exists():
            full_labels=json.loads(full_labels_path.read_text(encoding="utf-8")); seen_labels=labels
            full_index={lab:i for i,lab in enumerate(full_labels)}
            seen_to_full=np.asarray([full_index[x] for x in seen_labels],dtype=np.int64)
            raw_rows=[]
            with open(raw_test_path,"r",encoding="utf-8") as f:
                for line in f:
                    if line.strip(): raw_rows.append(json.loads(line))
            raw_by_id={str(r["id"]):r for r in raw_rows}
            y_full=np.zeros((len(tids),len(full_labels)),dtype=np.uint8)
            for i,sid in enumerate(tids):
                r=raw_by_id[str(sid)]
                for lab in r["labels"]: y_full[i,full_index[lab]]=1
            logits_full=np.full((len(tids),len(full_labels)),-30.0,dtype=np.float32)
            logits_full[:,seen_to_full]=tlogits
            support_full=np.zeros(len(full_labels),dtype=np.int64); support_full[seen_to_full]=train_support
            fm=compute_metrics(y_full,logits_full,support_full,threshold_fixed=cfg.threshold_fixed,tune_threshold=False)
            tm.update({f"full_task_supervised_no_zeroshot_diagnostic_{k}":v for k,v in fm.items() if isinstance(v,(int,float))})
            tm["full_task_supervised_no_zeroshot_diagnostic_note"]="4,271-label final-only padded view; 4,108 TRAIN-seen logits retained, 163 unsupported zero-shot labels assigned logit -30."
    p=sigmoid_np(tlogits); pred=(p>=float(frozen_threshold)).astype(np.uint8)
    from sklearn.metrics import f1_score,hamming_loss
    tm.update({"frozen_validation_threshold":float(frozen_threshold),"macro_f1_at_frozen_threshold":float(f1_score(ty,pred,average="macro",zero_division=0)),"micro_f1_at_frozen_threshold":float(f1_score(ty,pred,average="micro",zero_division=0)),"hamming_at_frozen_threshold":float(hamming_loss(ty,pred))})
    # Predeclared exact-duplicate-clean sensitivity: benchmark-as-is remains primary.
    final_audit_path = Path(data_root)/"prepared"/cfg.dataset/"final_test_audit.json"
    if final_audit_path.exists():
        final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
        dup_ids = set(map(str, final_audit.get("test_ids_with_any_development_exact_duplicate", [])))
        keep = np.asarray([str(x) not in dup_ids for x in tids], dtype=bool)
        tm["duplicate_clean_sensitivity_excluded_test_rows"] = int((~keep).sum())
        tm["duplicate_clean_sensitivity_retained_test_rows"] = int(keep.sum())
        if keep.any():
            dm = compute_metrics(ty[keep], tlogits[keep], train_support, threshold_fixed=cfg.threshold_fixed, tune_threshold=False)
            tm.update({f"duplicate_clean_sensitivity_{k}":v for k,v in dm.items() if isinstance(v,(int,float))})
            dpred = (sigmoid_np(tlogits[keep]) >= float(frozen_threshold)).astype(np.uint8)
            tm.update({
                "duplicate_clean_sensitivity_macro_f1_at_frozen_threshold":float(f1_score(ty[keep],dpred,average="macro",zero_division=0)),
                "duplicate_clean_sensitivity_micro_f1_at_frozen_threshold":float(f1_score(ty[keep],dpred,average="micro",zero_division=0)),
                "duplicate_clean_sensitivity_hamming_at_frozen_threshold":float(hamming_loss(ty[keep],dpred)),
            })
    atomic_write_json(out/"config.json",cfg.to_dict()); atomic_write_json(out/"test_metrics.json",tm); _save_predictions(out,"test",tlogits,ty,tids)
    _save_per_label_csv(out,"test",per_label_metrics(ty,tlogits,train_support,threshold=0.5))
    _save_per_label_csv(out,"test_frozen_threshold",per_label_metrics(ty,tlogits,train_support,threshold=float(frozen_threshold)))
    done={"status":"FINAL_TEST_COMPLETE","dataset":cfg.dataset,"method":cfg.method,"seed":cfg.seed,"config_hash":object_hash(cfg.to_dict()),
          "code_hash":code_hash,"campaign_lock_hash":campaign_lock_hash,"test_metrics":tm,
          "source_development_run":str(dev),"source_final_model_tree_hash":complete["final_model_tree_hash"],
          "retrained_for_test":False,"exact_frozen_development_weights_used":True}
    atomic_write_json(done_path,done)
    del model; gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return done

