from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

PROJECT_VERSION = "ctpu-article-v1.5.7-20260808"

# Methods in the QUICK protocol check.
PRIMARY_METHODS = ("bce", "pairwise", "normmatch", "ctpu")

# Matched-condition suite. Parameter-space PCGrad is evaluated on Reuters and
# AAPD before the final protocol freeze; the EURLEX57K suite excludes this control.
FULL_METHODS = (
    "bce",
    "pairwise",
    "normmatch",
    "ctpu",
    "pairwise_cardinality_penalty",
    "score_auxproj",
    "asl",
    "wasl",
)

SEEDS_5 = (20260805, 20260817, 20260829, 20260910, 20260922)
SEEDS_3 = SEEDS_5[:3]

DATASET_DEFAULTS: Dict[str, dict] = {
    "reuters21578": {
        "max_length": 256,
        "lr": 2e-5,
        "epochs": 4,
        "expected_num_labels": 90,
        "expected_train_labeled": 7770,
        "source_name": "Reuters-21578 Distribution 1.0 raw ModApte R(90)",
    },
    "aapd": {
        "max_length": 256,
        "lr": 2e-5,
        "epochs": 4,
        "expected_num_labels": 54,
        "source_name": "AAPD (Yang et al., COLING 2018)",
    },
    "eurlex57k": {
        "max_length": 512,
        "lr": 2e-5,
        "epochs": 4,
        # Training head is TRAIN-seen only (4,108 labels). This avoids reading
        # test annotations to construct the classifier vocabulary. Once the
        # one-shot test is unlocked, a separate full-task 4,271-label padded
        # evaluation is generated for transparent benchmark reporting.
        "expected_num_labels": 4108,
        "expected_full_task_labels": 4271,
        "expected_zero_shot_labels": 163,
        "expected_train": 45000,
        "expected_validation": 6000,
        "source_name": "EURLEX57K (Chalkidis et al., ACL 2019)",
    },
}


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "reuters21578"
    method: str = "ctpu"
    seed: int = SEEDS_5[0]
    model_name: str = "FacebookAI/roberta-base"
    model_revision: str | None = None
    attn_implementation: str = "eager"
    max_length: int = 256
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    effective_batch_size: int = 32
    eval_batch_size_cap: int = 64
    max_grad_norm: float = 1.0
    lambda_pair: float = 0.25
    pair_k_pos: int = 3
    pair_k_neg: int = 3
    threshold_fixed: float = 0.5
    threshold_grid_start: float = 0.05
    threshold_grid_stop: float = 0.95
    threshold_grid_step: float = 0.01
    cardinality_penalty_mu: float = 0.10
    bf16: bool = True
    deterministic: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    validation_each_epoch: bool = False
    # Development models are retained because final test MUST evaluate the
    # exact frozen validation model weights, not a retrained approximation.
    retain_final_model: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    effective_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    eval_batch_size: int
    gpu_name: str | None
    gpu_total_memory_gb: float | None
    probe_method: str = ""
    matched_causal_core_microbatch: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StudyProfile:
    name: str
    datasets: Tuple[str, ...]
    methods: Tuple[str, ...]
    seeds: Tuple[int, ...]
    epochs_override: int | None = None
    allow_test: bool = False
    description: str = ""


PROFILES: Dict[str, StudyProfile] = {
    "AUDIT_ONLY": StudyProfile(
        name="AUDIT_ONLY", datasets=("reuters21578",), methods=(), seeds=(SEEDS_5[0],),
        description="Download/audit/prepare only; no model training.",
    ),
    "QUICK": StudyProfile(
        name="QUICK", datasets=("reuters21578",), methods=PRIMARY_METHODS,
        seeds=(SEEDS_5[0],), epochs_override=4,
        description="Four-epoch, one-seed Reuters engineering/readiness screen; effect estimates are descriptive only and never gate continuation; test inaccessible.",
    ),
    "FULL_3SEED": StudyProfile(
        name="FULL_3SEED", datasets=("reuters21578", "aapd"), methods=FULL_METHODS,
        seeds=SEEDS_3,
        description="Three-seed matched development/replication on Reuters and AAPD; test inaccessible.",
    ),
    "PAPER_5SEED": StudyProfile(
        name="PAPER_5SEED", datasets=("reuters21578", "aapd"), methods=FULL_METHODS,
        seeds=SEEDS_5,
        description="Five-seed paper-grade development/replication; test inaccessible.",
    ),
    "EURLEX_CONFIRM_3SEED": StudyProfile(
        name="EURLEX_CONFIRM_3SEED", datasets=("eurlex57k",),
        methods=("bce", "pairwise", "normmatch", "ctpu", "pairwise_cardinality_penalty", "asl", "wasl"),
        seeds=SEEDS_3,
        description="Three-seed EURLEX57K confirmation on the 4,108 TRAIN-seen supervised head; the 4,271-label full-task view is generated only in the frozen final test.",
    ),
    "PARAM_PCGRAD_3SEED": StudyProfile(
        name="PARAM_PCGRAD_3SEED", datasets=("reuters21578", "aapd"), methods=("param_pcgrad",),
        seeds=SEEDS_3,
        description="Mandatory reviewer control before final freeze: exact parameter-space PCGrad on Reuters/AAPD.",
    ),
    "PENALTY_SENSITIVITY": StudyProfile(
        name="PENALTY_SENSITIVITY", datasets=("reuters21578",), methods=("pairwise_cardinality_penalty",),
        seeds=(SEEDS_5[0],),
        description="Diagnostic only; run with explicit mu values 0.01/0.1/1.0 if needed.",
    ),
    "ASL_OFFICIAL_REDUCTION_CHECK": StudyProfile(
        name="ASL_OFFICIAL_REDUCTION_CHECK", datasets=("reuters21578",), methods=("asl_official_sum",),
        seeds=(SEEDS_5[0],),
        description="Optional exact-reduction ASL sanity track. Main matched-condition ASL remains normalized mean reduction.",
    ),
    "WASL_REUTERS_REPRO": StudyProfile(
        name="WASL_REUTERS_REPRO", datasets=("reuters21578",),
        methods=("bce", "asl", "wasl"), seeds=SEEDS_3, epochs_override=50,
        description="Optional Reuters-only contextual reproduction track for the reported WASL 50-epoch setup; never part of the causal claim.",
    ),
}

WASL_REPRO_LR = {"reuters21578": 2e-5, "aapd": 2e-5, "eurlex57k": 2e-4}
WASL_HPARAMS = {
    "reuters21578": {"alpha": 0.2, "beta": 0.2, "gamma_pos": 1.0, "gamma_neg": 4.0, "m": 0.05},
    "aapd": {"alpha": 0.2, "beta": 0.3, "gamma_pos": 1.0, "gamma_neg": 4.0, "m": 0.05},
    "eurlex57k": {"alpha": 0.3, "beta": 0.3, "gamma_pos": 1.0, "gamma_neg": 4.0, "m": 0.05},
}
ASL_HPARAMS = {"gamma_pos": 1.0, "gamma_neg": 4.0, "clip": 0.05}
FINAL_TEST_AUTH_PHRASE = "I_UNDERSTAND_FINAL_TEST_IS_ONE_SHOT"

# Descriptive numerical floor only. QUICK performance never gates continuation.
QUICK_PAIRWISE_GAIN_FLOOR = 1e-4
