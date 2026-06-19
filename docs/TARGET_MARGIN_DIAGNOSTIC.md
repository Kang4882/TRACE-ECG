# Target-Margin Triplet LoRA Diagnostic

`TM-TL-LoRA` is an appendix-only training diagnostic evaluated inside the
TRACE-ECG audit framework.

It is not the TRACE-ECG framework itself.

## Objective

For bind-eligible closed-ended rows, the diagnostic adds a row-wise target
margin term:

```text
L = L_CE_all + lambda_margin * L_TargetMargin
```

where candidate answers are scored by teacher-forced likelihood under the
current model:

```text
score(c) = -normalized_NLL(c | prompt, ECG image, ECG time series)
margin = score(target) - max_{wrong c} score(c)
L_TargetMargin = ReLU(margin_delta - margin)
```

## Defaults

```text
lambda_margin = 0.05
margin_delta = 0.5
length_normalize = true
strict_eligibility = true
```

## Launcher

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_target_margin_triplet_lora.sh \
  --full_train --frac 005
```

The deprecated `scripts/train_trace_ecg_gemlike.sh` wrapper forwards here for
backward compatibility.
