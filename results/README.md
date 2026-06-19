# Result Tables

These tables are grouped by evaluation dataset. The incorrectly rendered legacy
PTB-XL 2200x1700 PoC experiment is excluded.

## Main file

- `all_main_eval_datasets_presentation.csv`

## Per-dataset files

- `official_ecg_bench_broad.csv`: Official ECG-Bench Broad.
- `gem_like_ecgbench_l3_choice_csn_g12ec.csv`: GEM-like ECGBench-L3 Choice.
- `clean_ecgbench_l3_choice_csn_g12ec.csv`: Clean ECGBench-L3 Choice.
- `ptb_xl_core7_clean1008.csv`: PTB-XL Core7 clean1008.
- `ptb_xl_core4_fold9_clean1008.csv`: PTB-XL Core4 fold9 clean1008.
- `ptb_xl_core4_fold10_clean1008.csv`: PTB-XL Core4 fold10 clean1008.

## Naming

TRACE-ECG is the audit framework, not a model row. Paper-facing rows use:

- `PULSE`
- `GEM`
- `GL-SL-LoRA-1%`, `GL-SL-LoRA-3%`, `GL-SL-LoRA-5%`
- `GL-TL-LoRA-1%`, `GL-TL-LoRA-3%`, `GL-TL-LoRA-5%`
- `GL-TL-Full`
- `CL-SL-SFT`
- `CL-TL-Subset-SFT`
- `TM-TL-LoRA-1%`, `TM-TL-LoRA-5%` for appendix Target-Margin diagnostics only
- `P4-LoRA` and `P4-LayoutBind` for appendix PTB-XL Core4 pilots

## Separate diagnostics

- `diagnostic_adapter_scale_sweep_presentation.csv`
- `training_efficiency_presentation.csv`

Metric terminology:

- `DI` = Inconsistency
- `DCC` = Consistent-Correct
- `DCE` = Consistent-Error
- `CDI` = Correctable Inconsistency
- `ADI` = Always-wrong Inconsistency
