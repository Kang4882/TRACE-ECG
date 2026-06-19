# Metrics

For each evaluation sample, let:

- `y` be the ground-truth answer.
- `p3`, `p6`, `p12` be parsed predictions for layouts `3R4C`, `6R2C`, `12R1C`.

| Metric | Definition |
|---|---|
| `Consistent-Correct` | `p3 == p6 == p12 == y` |
| `Inconsistency` | not all of `p3`, `p6`, `p12` are identical |
| `Consistent-Error` | `p3 == p6 == p12` and `p3 != y` |
| `Correctable Inconsistency` | inconsistent and at least one layout prediction equals `y` |
| `Always-wrong Inconsistency` | inconsistent and none of the layout predictions equals `y` |
| `Consistency` | `Consistent-Correct + Consistent-Error` |
| `Worst Acc` | minimum per-layout accuracy |
| `Oracle Acc` | sample is correct if any layout prediction equals `y` |

Older internal reports may use legacy terms:

- `Flip` = `Inconsistency`
- `Stable-Correct` = `Consistent-Correct`
- `Stable-Wrong` = `Consistent-Error`
- `Recoverable Flip` = `Correctable Inconsistency`
- `Unstable-Wrong` = `Always-wrong Inconsistency`

