# TRACE-ECG

Triplet-based Reliability and Artifact-Controlled Evaluation for ECG MLLMs.

TRACE-ECG is an audit framework, not a training method. It asks whether an ECG
multimodal large language model gives stable and correct clinical answers when
the same raw ECG signal is rendered under controlled display changes.

The central principle is:

```text
Consistency is not reliability.
A model can become more consistent by becoming consistently wrong.
```

## What TRACE-ECG Evaluates

TRACE-ECG separates three sources of behavior that are often conflated:

1. Same-signal layout variation.
2. Rendering artifact policy.
3. Correctness-aware reliability status.

For each eligible ECG sample, the same raw waveform is paired with ECG images
rendered as:

- `3R4C`: 3 rows x 4 columns
- `6R2C`: 6 rows x 2 columns
- `12R1C`: 12 rows x 1 column

The model is then evaluated with a reliability taxonomy:

- `Consistent-Correct`: all displays are correct.
- `Inconsistency`: display answers differ, or correctness is mixed.
- `Consistent-Error`: all displays agree on the same wrong answer.

This repository contains the public project layer for the TRACE-ECG audit:

- evaluation and scoring tools
- training launchers for experimental conditions evaluated by the audit
- result tables
- dataset preparation notes
- GEM patch files used by the experiments

Large ECG datasets, rendered PNGs, and model checkpoints are not included.

## Experimental Conditions

TRACE-ECG is the framework name. The rows below are model conditions evaluated
by the framework:

| Paper name | Short name | Role |
| --- | --- | --- |
| PULSE (Image-only ECG MLLM) | PULSE | Image-only baseline |
| GEM (Signal+Image ECG MLLM) | GEM | Signal+image baseline |
| GEM-like Single-Layout LoRA | GL-SL-LoRA | View-matched single-layout control |
| GEM-like Triplet-Layout LoRA | GL-TL-LoRA | Triplet-layout LoRA SFT |
| GEM-like Triplet-Layout Full SFT | GL-TL-Full | Full-parameter triplet-layout SFT |
| Clean1008 Single-Layout SFT | CL-SL-SFT | Controlled clean single-layout SFT |
| Clean1008 Triplet-Layout Subset SFT | CL-TL-Subset-SFT | Controlled clean triplet-layout SFT |
| Target-Margin Triplet LoRA | TM-TL-LoRA | Appendix-only diagnostic |

`TM-TL-LoRA` is not the TRACE-ECG framework. It is a diagnostic training
variant that adds a row-wise target-margin loss on bind-eligible closed-ended
rows.

## Repository Layout

```text
configs/      Example path configuration
docs/         Method, data, experiment, and metric documentation
scripts/      Public training and scoring entrypoints
tools/        Standalone scoring utilities
patches/      GEM patch and new files used by the experiments
results/      Curated result tables
reports/      Code inventory and release reports
```

## Quick Start

1. Clone or prepare a compatible GEM checkout.
2. Apply the GEM patch used by these experiments:

```bash
bash scripts/apply_gem_patch.sh /path/to/GEM
```

3. Edit paths:

```bash
cp configs/paths.example.env paths.env
vim paths.env
source paths.env
```

4. Score a close-ended ECGBench-L3 triplet prediction file:

```bash
bash scripts/score_ecgbench_l3_choice.sh \
  --predictions /path/to/predictions.jsonl \
  --out_dir /path/to/output_dir
```

5. Run a LoRA-SFT experimental condition:

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_lora_sft_gemlike.sh \
  --full_train --setting l3 --frac 005
```

6. Run the appendix-only Target-Margin diagnostic:

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_target_margin_triplet_lora.sh \
  --full_train --frac 005
```

## Data

The public repository does not include raw ECG data, rendered images, model
weights, or full JSONL manifests. See `docs/DATA.md` and
`docs/DATA_GENERATION.md` for the controlled dataset definitions and validation
reports.

## License and Upstream

This project builds on GEM/LLaVA-style code. Check upstream licenses and dataset
licenses before redistribution. ECG data, rendered images, and model checkpoints
are intentionally not committed here.
