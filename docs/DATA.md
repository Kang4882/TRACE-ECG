# Data

This repository does not include raw ECG data, rendered ECG images, model
checkpoints, or full training JSONLs.

The experiments use three dataset families:

## 1. Official ECG-Bench Broad

The original ECG-Bench-style broad evaluation is used to check retention and
general benchmark behavior. This is separate from TRACE-ECG triplet auditing.

## 2. Clean1008 Controlled Triplets

Clean1008 is the controlled clean rendering used to isolate layout-equivalence.

Properties:

- same raw ECG signal
- same instruction and target
- three layouts: `3R4C`, `6R2C`, `12R1C`
- no stochastic visual artifacts
- 1008 x 1008 PNG rendering
- raw time-series references retained for GEM-style signal+image models

This setting supports clean controlled evaluation and clean SFT baselines:

- `CL-SL-SFT`
- `CL-TL-Subset-SFT`

## 3. GEM-like Source-Matched Triplets

GEM-like v1 is a distribution-matched rendering emulator for GEM training
sources:

- ECGInstruct
- ECG-Grounding

It is not an exact reproduction of the hidden original augmentation recipe.
Instead, it matches audited image statistics well enough to test whether
triplet exposure behaves differently under artifact-containing inputs.

Properties:

- one common image pool
- no image duplication per fraction subset
- deterministic fraction sampling by original training group
- three layouts per selected group for L3 conditions
- one fixed layout per selected signal/group for L1 conditions
- same artifact bundle across the three layouts of a render group

## Expected Local Paths

```text
/data/ecg_l3_gemlike_v1/
/data/ecg_l3_clean_1008/
/data/ecg_timeseries/
/data/model_checkpoints/LANSG_GEM
```

Override paths through `configs/paths.example.env`.

## Release Practice

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

See `docs/DATA_GENERATION.md` for the current internal generation provenance.
