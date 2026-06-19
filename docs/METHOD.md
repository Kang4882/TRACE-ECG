# Method

TRACE-ECG stands for Triplet-based Reliability and Artifact-Controlled
Evaluation for ECG MLLMs.

TRACE-ECG is an audit protocol. It is not a new training objective.

## Audit Question

Given the same raw ECG waveform, should an ECG multimodal LLM preserve the same
clinical answer when the accompanying ECG display changes?

TRACE-ECG evaluates this by holding the signal, instruction, and target fixed
while changing the rendered display according to controlled policies.

## Same-Signal Triplet Rendering

Each eligible sample is rendered into three standard ECG display geometries:

- `3R4C`: 3 rows x 4 columns
- `6R2C`: 6 rows x 2 columns
- `12R1C`: 12 rows x 1 column

The triplet shares:

- raw ECG waveform reference
- signal segment
- instruction
- target
- source metadata

Only the display condition changes.

## Artifact-Control Protocol

The framework separates display effects into explicit audit settings:

| Setting | Purpose |
| --- | --- |
| `clean` | Isolate layout variation without stochastic visual artifacts. |
| `same-artifact seed` | Change layout while holding the artifact bundle fixed. |
| `independent-artifact seed` | Measure combined layout and artifact instability. |
| `same-layout multi-artifact` | Isolate artifact sensitivity without layout change. |

The current public result tables include clean and GEM-like source-matched
triplet evaluations. Additional artifact-control modes should be reported
separately when generated.

## Reliability Taxonomy

For closed-ended tasks with one parsed answer per display, TRACE-ECG reports
group status over the three displays:

- `Consistent-Correct`: all three displays predict the correct answer.
- `Inconsistency`: predictions differ across displays, or correctness is mixed.
- `Consistent-Error`: all three displays produce the same wrong answer.

This distinction matters because lower inconsistency is not automatically better
if errors become stable.

## Experimental Conditions

The framework evaluates baselines and calibration conditions such as:

- PULSE
- GEM
- GL-SL-LoRA
- GL-TL-LoRA
- GL-TL-Full
- CL-SL-SFT
- CL-TL-Subset-SFT

Target-Margin Triplet LoRA (`TM-TL-LoRA`) is appendix-only under the current
paper story. It is useful as a diagnostic but should not be described as the
TRACE-ECG method.

## Scope Limitation

The main triplet reliability taxonomy is intended for closed-ended tasks with a
well-defined answer space. Report generation and multi-label outputs require
separate semantic or contradiction metrics and are not treated as main flip-rate
tasks.
