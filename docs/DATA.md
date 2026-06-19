# Data

This repository does not include raw ECG data, rendered ECG images, or model
checkpoints.

Expected local paths used by the provided scripts:

```text
/data/ecg_l3_gemlike_v1/
/data/ecg_l3_clean_1008/
/data/ecg_timeseries/
/data/model_checkpoints/LANSG_GEM
```

## GEM-like L3 Training Data

The GEM-like L3 dataset is generated from GEM training sources:

- ECGInstruct
- ECG-Grounding

Each selected training unit is expanded to three layout rows:

- `3R4C`
- `6R2C`
- `12R1C`

The rendered images are too large for Git and are not redistributed in this
repository.

## Controlled clean1008 Evaluation Data

The controlled clean1008 data is used for display robustness evaluation. It is
also not included in this repository.

## Recommended Release Practice

Use this repository for:

- code
- scripts
- configs
- small examples
- result tables

Use external dataset hosting or regeneration instructions for:

- rendered PNGs
- large JSONL/parquet manifests
- ECG time-series files
- model checkpoints

