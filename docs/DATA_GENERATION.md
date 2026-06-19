# Data Generation Provenance

This file documents the dataset-generation status used by the current
TRACE-ECG audit experiments. The full rendered datasets are not included in the
public repository.

## Clean1008

Internal root:

```text
/data/ecg_l3_clean_1008/
```

Purpose:

- controlled clean triplet rendering
- clean ECGBench-L3 choice evaluation
- clean PTB-XL triplet evaluation
- clean single-layout/triplet SFT baselines

Validation reports:

```text
/data/ecg_l3_clean_1008/reports/index_sanity_check_report.md
/data/ecg_l3_clean_1008/reports/gem_json_sanity_check_report.md
/data/ecg_l3_clean_1008/reports/pure_lora_protocol_audit.md
```

Key invariants:

- L1 has one fixed layout per signal.
- L3 has exactly three layouts per group.
- Training rows with train/eval signal overlap are excluded.
- Raw time-series references are retained.
- CODE-15 short-signal padding metadata is preserved.

## GEM-like v1

Internal root:

```text
/data/ecg_l3_gemlike_v1/
```

Purpose:

- artifact-containing source-matched triplet training/evaluation
- GEM-like single-layout and triplet-layout LoRA experiments
- full triplet-layout SFT baseline

Validation reports:

```text
/data/ecg_l3_gemlike_v1/reports/lead_order_audit.md
/data/ecg_l3_gemlike_v1/reports/full_render_final_summary.md
/data/ecg_l3_gemlike_v1/reports/full_l3_jsonl_readiness.md
```

Key invariants:

- corrected lead-order logic is reused from the clean L3 renderer.
- artifact policy is sampled once per render group and shared across `3R4C`,
  `6R2C`, and `12R1C`.
- fraction sampling is by original training group before layout expansion.
- all images live in one common pool.

## Artifact-Control Status

Current public result tables cover:

- clean triplets
- GEM-like source-matched triplets with same artifact bundle across layouts

The broader TRACE-ECG artifact-control protocol also defines:

- independent-artifact seed
- same-layout multi-artifact

Those modes should be generated and reported as separate audit settings when
used. They should not be silently mixed into the current clean or GEM-like
tables.
