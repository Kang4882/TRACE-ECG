# Experiments

TRACE-ECG is the audit framework. The training scripts below create model
conditions evaluated by the framework.

## Main GEM-like LoRA Conditions

All LoRA experiments use the same effective global batch by default:

```text
effective_global_batch = 256
per_device_train_batch_size = 8
gradient_accumulation_steps = 256 / (num_gpus * 8)
```

### GL-SL-LoRA: GEM-like Single-Layout LoRA

Single-layout control. Each selected training group uses one fixed layout.

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_lora_sft_gemlike.sh \
  --full_train --setting l1 --frac 005
```

### GL-TL-LoRA: GEM-like Triplet-Layout LoRA

Triplet-layout CE/SFT control. Each selected training group expands to
`3R4C`, `6R2C`, and `12R1C`.

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_lora_sft_gemlike.sh \
  --full_train --setting l3 --frac 005
```

Both LoRA conditions disable:

- TargetMargin
- legacy LayoutBind/TargetBind
- grouped triplet losses

## GL-TL-Full: Full-Parameter Triplet-Layout SFT

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_full_l3_sft_gemlike.sh --full_train
```

This baseline is full-parameter row-level CE/SFT only. TargetMargin and legacy
LayoutBind/TargetBind losses are disabled.

## Appendix Diagnostic: TM-TL-LoRA

Target-Margin Triplet LoRA is an appendix-only diagnostic. It is not the
TRACE-ECG framework.

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_target_margin_triplet_lora.sh \
  --full_train --frac 005
```

This diagnostic uses:

```text
L = L_CE_all + lambda_margin * L_TargetMargin
```

on bind-eligible closed-ended rows. It uses no frozen teacher distribution and
no layout-to-layout KL.

The legacy wrapper `scripts/train_trace_ecg_gemlike.sh` is retained only for
backward compatibility and forwards to this diagnostic launcher.
