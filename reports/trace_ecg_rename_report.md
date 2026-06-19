TRACE_ECG_RENAME = COMPLETE

# TRACE-ECG Rename Report

## Summary

The public framework name has been changed from ANCHOR-ECG to TRACE-ECG in the
overlay repository.

## Public Changes

- Repository title: `TRACE-ECG`
- Main training wrapper:
  - `scripts/train_trace_ecg_gemlike.sh`
- Patch files:
  - `patches/gem_trace_ecg_tracked_changes.patch`
  - `patches/gem_trace_ecg_diff_stat.txt`
- Public utility re-export:
  - `patches/new_files/llava/trace_ecg_utils.py`
- Result rows now use:
  - `TRACE-ECG_L3_frac001`
  - `TRACE-ECG_L3_frac005`

## Compatibility Note

The underlying GEM trainer still uses internal `anchor_ecg_*` flag names and the
`ANCHOR_ECG_TRAIN_LOG_CSV` environment variable. These are implementation
compatibility names only. Public scripts and documentation use TRACE-ECG.

## Metrics

The display robustness metric names remain:

- `Consistent-Correct`
- `Inconsistency`
- `Consistent-Error`
- `Correctable Inconsistency`
- `Always-wrong Inconsistency`
- `Worst Acc`
- `Oracle Acc`

