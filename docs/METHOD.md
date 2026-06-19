# Method

TRACE-ECG stands for Target-Anchored Counterfactual Display Calibration for ECG
MLLMs.

The main method uses same-signal multi-layout ECG displays but does not optimize
layout-to-layout agreement directly. Instead, each displayed view is anchored to
the same clinical target.

## Objective

```text
L_total = L_CE_all + lambda_margin * L_TargetMargin_bind
```

For each bind-eligible row with candidate answer set `C` and target `y`:

```text
score(c) = - normalized_NLL(c | prompt, ECG image, ECG time series)
margin = score(y) - max_{c != y} score(c)
L_TargetMargin = ReLU(margin_delta - margin)
```

`TargetMargin` backpropagates through the current LoRA model. There is no frozen
teacher and no detached candidate score.

## Main Mode

Main TRACE-ECG runs use:

```text
grouped_triplet_training = False
layoutbind_enable = False
anchor_ecg_enable = True
anchor_ecg_rowwise = True
targetmargin_enable = True
```

The main method does not use:

- legacy TargetBind
- pbar/q consensus distributions
- layout-to-layout KL
- grouped triplet consistency
- worst-layout max aggregation

