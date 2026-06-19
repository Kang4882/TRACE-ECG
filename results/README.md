# Presentation Tables By Evaluation Dataset

These simplified tables are grouped by evaluation dataset. The incorrectly rendered PTB-XL 2200x1700 legacy experiment is excluded. Frac003 rows were imported from A100/A6000 remote results.

## Main file
- `all_main_eval_datasets_presentation.csv`

## Per-dataset files
- `official_ecg_bench_broad.csv`: Official ECG-Bench Broad (16 models)
- `gem_like_ecgbench_l3_choice_csn_g12ec.csv`: GEM-like ECGBench-L3 Choice: CSN+G12EC (11 models)
- `clean_ecgbench_l3_choice_csn_g12ec.csv`: Clean ECGBench-L3 Choice: CSN+G12EC (4 models)
- `ptb_xl_core7_clean1008.csv`: PTB-XL Core7 clean1008 (5 models)
- `ptb_xl_core4_fold9_clean1008.csv`: PTB-XL Core4 fold9 clean1008 (2 models)
- `ptb_xl_core4_fold10_clean1008.csv`: PTB-XL Core4 fold10 clean1008 (2 models)

## Separate diagnostics
- `diagnostic_adapter_scale_sweep_presentation.csv`
- `training_efficiency_presentation.csv`

Metric terminology: DI=old Flip, DCC=old Stable-Correct, DCE=old Stable-Wrong, CDI=old Recoverable Flip, ADI=old Unstable-Wrong.
