TRACE_ECG_FRAMEWORK_ALIGNMENT = COMPLETE

# TRACE-ECG Framework Alignment Report

## Summary

The public overlay has been aligned with the final TRACE-ECG framing:

```text
TRACE-ECG = Triplet-based Reliability and Artifact-Controlled Evaluation for ECG MLLMs
```

TRACE-ECG is now documented as an audit framework, not a training method.

## Main Changes

- `README.md` now defines TRACE-ECG as a same-signal reliability audit protocol.
- `docs/METHOD.md` now describes triplet rendering, artifact-control settings,
  and reliability taxonomy.
- `docs/EXPERIMENTS.md` separates main SFT conditions from appendix
  Target-Margin diagnostics.
- `scripts/train_target_margin_triplet_lora.sh` is the appendix diagnostic
  launcher.
- `scripts/train_trace_ecg_gemlike.sh` is retained only as a deprecated
  compatibility wrapper.
- Result tables have been renamed to paper-facing model names.

## Naming Rule

TRACE-ECG is not a row in model-comparison tables.

Model rows should use:

- `PULSE`
- `GEM`
- `GL-SL-LoRA-*`
- `GL-TL-LoRA-*`
- `GL-TL-Full`
- `CL-SL-SFT`
- `CL-TL-Subset-SFT`
- `TM-TL-LoRA-*` only for appendix Target-Margin diagnostics

## Compatibility Note

The underlying patched GEM trainer still contains internal `anchor_ecg_*` names
for historical compatibility. Those flags now refer to the appendix TargetMargin
diagnostic path and should not be described as the TRACE-ECG framework.
