# Experiments

## Main LoRA Triangle

All LoRA experiments use the same effective global batch by default:

```text
effective_global_batch = 256
per_device_train_batch_size = 8
gradient_accumulation_steps = 256 / (num_gpus * 8)
```

### L1 LoRA-SFT

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_lora_sft_gemlike.sh \
  --full_train --setting l1 --frac 005
```

### L3 LoRA-SFT

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_lora_sft_gemlike.sh \
  --full_train --setting l3 --frac 005
```

### TRACE-ECG L3

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_trace_ecg_gemlike.sh \
  --full_train --frac 005
```

## Full L3 SFT Baseline

```bash
USER_APPROVED_FULL_TRAIN=1 bash scripts/train_full_l3_sft_gemlike.sh --full_train
```

This baseline is full-parameter row-level CE/SFT only. TRACE-ECG and legacy
LayoutBind/TargetBind losses are disabled.

