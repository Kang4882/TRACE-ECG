# TRACE-ECG Framework Update Summary 2026-06-19

## Status

The public overlay has been updated to match the final TRACE-ECG framing:

```text
TRACE-ECG = Triplet-based Reliability and Artifact-Controlled Evaluation for ECG MLLMs
```

TRACE-ECG is now described as an audit framework, not a training method.

## Updated

- Rewrote `README.md`.
- Rewrote `docs/METHOD.md`.
- Rewrote `docs/EXPERIMENTS.md`.
- Rewrote `docs/DATA.md`.
- Added `docs/DATA_GENERATION.md`.
- Added `docs/TARGET_MARGIN_DIAGNOSTIC.md`.
- Fixed `docs/INSTALL.md` patch filename.
- Updated `configs/paths.example.env` output-root default.
- Added `scripts/train_target_margin_triplet_lora.sh`.
- Converted `scripts/train_trace_ecg_gemlike.sh` into a deprecated compatibility wrapper.
- Updated LoRA and full-SFT script descriptions.
- Replaced stale rename reports with framework alignment reports.
- Updated code inventory to distinguish framework code from appendix diagnostics.
- Renamed public result rows to final paper-facing names.
- Added `results/final_method_name_registry.csv`.
- Removed generated `__pycache__` directories.

## Naming Changes

TRACE-ECG is no longer used as a model row.

TargetMargin rows are now:

- `TM-TL-LoRA-1%`
- `TM-TL-LoRA-5%`

Main LoRA rows are now:

- `GL-SL-LoRA-1%`
- `GL-TL-LoRA-1%`
- `GL-SL-LoRA-3%`
- `GL-TL-LoRA-3%`
- `GL-SL-LoRA-5%`
- `GL-TL-LoRA-5%`

Clean controlled rows are now:

- `CL-SL-SFT`
- `CL-TL-Subset-SFT`

PTB-XL pilot rows are now:

- `P4-LoRA`
- `P4-LayoutBind`

## Validation

Completed:

- shell syntax check for all public `.sh` files
- Python compile check for all public `.py` files
- JSON validation for public summary reports
- stale phrase grep for old TRACE-as-method wording
- `__pycache__` cleanup after compile check

The grep check found no remaining stale TRACE-as-training-method wording,
legacy TRACE-labeled model rows, stale patch filenames, or old local result
source paths in public-facing files.

## Remaining Caveat

The GEM patch and current GEM checkout still contain historical internal
`anchor_ecg_*` flag names because they are used by the TargetMargin diagnostic
implementation. Public docs now describe these as compatibility names, not as
TRACE-ECG framework names.
