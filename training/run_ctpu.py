#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctpu.config import FINAL_TEST_AUTH_PHRASE, PROFILES
from ctpu.experiment import (
    collect_results,
    estimate_model_storage,
    freeze_final_protocol,
    prepare_and_tokenize,
    quick_gate,
    real_logit_mechanism_test,
    run_final_test,
    run_profile,
)


def main():
    ap = argparse.ArgumentParser(description="CTPU article experiment runner")
    ap.add_argument("action", choices=[
        "audit", "mechanism", "quick", "full3", "paper5", "eurlex3", "wasl_reuters_repro",
        "penalty_sensitivity", "param_pcgrad3", "asl_official_check", "gate", "collect", "storage", "bootstrap", "wasl_compare", "freeze", "final_test", "final_analysis"
    ])
    ap.add_argument("--root", required=True, help="Persistent experiment root, e.g. Google Drive path")
    ap.add_argument("--code-root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--dataset", choices=["reuters21578", "aapd", "eurlex57k"], default="reuters21578")
    ap.add_argument("--mu", type=float, default=0.10, help="Cardinality-penalty diagnostic baseline coefficient")
    ap.add_argument("--auth", default="", help="Required only for one-shot final test")
    ap.add_argument("--method-a", default="ctpu")
    ap.add_argument("--method-b", default="pairwise")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--final", action="store_true", help="For bootstrap: use frozen final-test predictions instead of validation")
    args = ap.parse_args()

    mapping = {
        "audit": "AUDIT_ONLY",
        "quick": "QUICK",
        "full3": "FULL_3SEED",
        "paper5": "PAPER_5SEED",
        "eurlex3": "EURLEX_CONFIRM_3SEED",
        "wasl_reuters_repro": "WASL_REUTERS_REPRO",
        "penalty_sensitivity": "PENALTY_SENSITIVITY",
        "param_pcgrad3": "PARAM_PCGRAD_3SEED",
        "asl_official_check": "ASL_OFFICIAL_REDUCTION_CHECK",
    }
    if args.action in mapping:
        out = run_profile(args.root, args.code_root, mapping[args.action], cardinality_penalty_mu=args.mu)
    elif args.action == "mechanism":
        out = real_logit_mechanism_test(args.root, args.code_root, dataset=args.dataset)
    elif args.action == "gate":
        out = quick_gate(args.root, args.code_root)
    elif args.action == "collect":
        out = collect_results(args.root, final=False)
    elif args.action == "storage":
        out = estimate_model_storage(args.root)
    elif args.action == "bootstrap":
        from ctpu.experiment import paired_bootstrap_report
        out = paired_bootstrap_report(args.root, args.dataset, args.method_a, args.method_b, reps=args.reps, final=args.final)
    elif args.action == "wasl_compare":
        from ctpu.reporting import compare_wasl_published
        out = compare_wasl_published(args.root, args.code_root)
    elif args.action == "freeze":
        out = freeze_final_protocol(args.root, args.code_root)
    elif args.action == "final_test":
        out = run_final_test(args.root, args.code_root, args.auth)
    elif args.action == "final_analysis":
        from ctpu.experiment import final_confirmatory_report
        out = final_confirmatory_report(args.root, reps=max(args.reps, 2000))
    else:
        raise AssertionError(args.action)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
