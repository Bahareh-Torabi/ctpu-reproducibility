# CTPU source code and experiment records

Cardinality-Tangent Pairwise Updates (CTPU) constrain the first-order effect of a
sampled Pairwise score direction on the current soft label count. This repository
contains the training implementation and reproducibility materials for the
Reuters-21578, AAPD, and EURLEX57K-Seen experiments.

## Contents

| Path | Purpose |
| --- | --- |
| `training/ctpu/` and `training/run_ctpu.py` | Training, data preparation, metrics, and evaluation |
| `training/tests/` | Tests of the research implementation |
| `training/references/` | Dataset sources and baseline reference metadata |
| `frozen_protocol/` | Committed design, dataset locks, software environments, and export manifest |
| `run_records/` | Configurations, histories, completion records, and training times |
| `development_penalty_records/` | Additional Reuters validation runs for the direct penalty |
| `original_summary/` | Run-level and aggregate experiment results |
| `compact_document_metrics/` | Document-level metrics and checksums for 94 final-test runs |
| `analysis_results/` | Paired contrasts, seed summaries, error distributions, timing, and consistency checks |
| `replay_statistics.py` | CPU reproduction of the 76 reported metric contrasts |
| `audit_predictions.py` | Audit using the original dense prediction export |
| `SOURCE_PROVENANCE.json` | Training and publication source identities |
| `MANIFEST_SHA256.json` | Checksums of the repository files |

## Replay

Using Python 3.12, run these commands from the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python replay_statistics.py --output replayed_contrasts.csv
```

The script validates metric-array checksums and computes 5,000 paired crossed
seed-by-document bootstrap replicates. Its 76 contrasts can be compared with
`analysis_results/paired_crossed_bootstrap_comparisons.csv`. This replay requires
no GPU or model weights. The result of the repository replay check is recorded
in `analysis_results/replay_validation.json`.

The primary contrast is CTPU minus Pairwise. Other method contrasts are post hoc
and exploratory, with no multiplicity adjustment. Bootstrap sign fractions are
descriptive, not p-values. The number of independent trained seeds remains three
or five. Param-PCGrad comparisons use only the three common seeds. The validation
comparison of the direct-penalty coefficient μ is not a sensitivity study of the
CTPU coefficient λ.

## Training

The experiment version is `ctpu-article-v1.5.7-20260808`, defined in
`training/ctpu/config.py`. The study uses `FacebookAI/roberta-base` revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`, four epochs, λ = 0.25, and at most
3 × 3 positive–negative pairs per eligible document. Reuters and AAPD use seeds
20260805, 20260817, 20260829, 20260910, and 20260922. EURLEX57K-Seen and
Param-PCGrad use the first three.

Training requires a CUDA-enabled PyTorch environment and the dependencies in
`training/requirements-colab.txt`. Recorded environments are provided with the
protocol and run records. Dataset acquisition sources are specified in
`training/references/data_sources.json`. The supervised EURLEX classifier uses
the 4,108 labels observed in training; the separate 4,271-label view is diagnostic.
Dataset texts and model checkpoints are not redistributed.

To inspect available training commands in the compatible environment:

```bash
python training/run_ctpu.py --help
```

The staged campaign API in `training/ctpu/multisession.py` provides the original
execution sequence: create the global commitment, install it in the session,
run dataset development, freeze the retained model identities, and evaluate
those exact models once on the final test. The relevant functions are
`create_or_verify_global_commitment`, `install_global_commitment_into_session`,
`run_dataset_development`, `freeze_dataset_campaign`, and
`run_dataset_final_test`. Use `training/` as `code_root`.

The archived locks identify the reported experiments. A new replication records
its own source, environment, model, and data identities. Comments, docstrings,
and the VERSION text have been edited for publication; the executable training
statements and dependency specifications are unchanged. `SOURCE_PROVENANCE.json` lists both source hashes and the per-file
checksums. The original run records have not been rewritten.

## Audit of dense predictions

Install matplotlib in addition to the analysis dependencies, then run:

```bash
python audit_predictions.py /path/to/frozen_export --output new_audit
```

The original export must contain `campaigns/`, `global_paper/`, and
`PAPER_ARTIFACT_MANIFEST.json`. The audit verifies artifact hashes, run identity,
document and label alignment, saved metrics, and the development-run timing
associated with each final-test model. Dense logits are not included in this
compact repository.

## Data and model terms

The dataset and model terms specified by their original providers continue to
apply. Third-party references are identified in `training/references/`.
