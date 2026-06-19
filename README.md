# TRACE-ECG

Target-Anchored Counterfactual Display Calibration for ECG MLLMs.

TRACE-ECG calibrates each counterfactual ECG display view toward the correct
clinical target, rather than forcing layout predictions to agree with each
other. This matters because display consistency can be false: a model can be
consistent across ECG layouts while consistently wrong.

This repository contains the public project layer for TRACE-ECG:

- training launchers
- evaluation/scoring tools
- result tables
- dataset preparation notes
- GEM patch files needed to reproduce the method

The method is implemented on top of GEM. We do not vendor the full GEM source as
the top-level project because this repository is intended to present TRACE-ECG
as the primary contribution. Use the patch files in `patches/` to apply the
method to a GEM checkout.

Implementation note: the public method name is TRACE-ECG. The GEM patch still
uses internal `anchor_ecg_*` flag names for backward compatibility with the
earlier development branch; the public launchers expose TRACE-ECG names and map
them to those internal flags.

## Method

Main objective:

```text
L_total = L_CE_all + lambda_margin * L_TargetMargin_bind
```

Where:

- `L_CE_all` is ordinary supervised fine-tuning cross entropy on every row.
- `TargetMargin` is applied only to bind-eligible close-ended rows.
- Candidate answers are scored by teacher-forced likelihood under the current
  model.
- No teacher model is used.
- No layout-to-layout KL or consistency loss is used by the main method.

Default TRACE-ECG hyperparameters:

```text
lambda_margin = 0.05
margin_delta = 0.5
length_normalize = True
strict_eligibility = True
```

## Repository Layout

```text
configs/      Example path configuration
docs/         Method, data, experiment, and metric documentation
scripts/      Public training and scoring entrypoints
tools/        Standalone scoring utilities
patches/      GEM patch and new files required by TRACE-ECG
results/      Curated result tables
reports/      Code inventory and release reports
```

## Quick Start

1. Clone or prepare a compatible GEM checkout.
2. Apply the TRACE-ECG GEM patch:

```bash
bash scripts/apply_gem_patch.sh /path/to/GEM
```

3. Edit paths:

```bash
cp configs/paths.example.env paths.env
vim paths.env
source paths.env
```

4. Run a LoRA-SFT control:

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_lora_sft_gemlike.sh \
  --full_train --setting l3 --frac 005
```

5. Run TRACE-ECG:

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_trace_ecg_gemlike.sh \
  --full_train --frac 005
```

## Display Robustness Metrics

Public tables use:

- `Consistent-Correct`: all layouts predict the correct target.
- `Inconsistency`: layout predictions are not all identical.
- `Consistent-Error`: all layouts agree on the same wrong answer.
- `Correctable Inconsistency`: inconsistent and at least one layout is correct.
- `Always-wrong Inconsistency`: inconsistent and no layout is correct.
- `Worst Acc`: minimum per-layout accuracy.
- `Oracle Acc`: correctness if any layout is correct.

## Data

Large ECG data and rendered images are not included in this repository. See
`docs/DATA.md`.

## License and Upstream

This project builds on GEM/LLaVA-style code. Check upstream licenses and dataset
licenses before redistribution. ECG data, rendered images, and model checkpoints
are intentionally not committed here.
