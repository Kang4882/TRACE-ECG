#!/usr/bin/env python3
"""Evaluate close-ended ECGBench-L3 layout stability from expanded predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


LAYOUTS = ("3R4C", "6R2C", "12R1C")
MAIN_SUBSETS = ("csn-test-no-cot", "g12-test-no-cot")
LETTER_RE = re.compile(r"\b([A-H])\b", re.IGNORECASE)
DATASET_ROOT = Path("/data/ecg_l3_clean_1008")
PUBLIC_METRIC_NAMES = {
    "stable_correct": "Consistent-Correct",
    "answer_flip": "Inconsistency",
    "stable_wrong": "Consistent-Error",
    "recoverable_flip": "Correctable Inconsistency",
    "unstable_wrong": "Always-wrong Inconsistency",
    "layout_agreement": "Consistency",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_options(prompt: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*([A-H])\.\s*(.+?)\s*$", prompt):
        letter = match.group(1).upper()
        text = match.group(2).strip()
        text = re.split(
            r"\n\s*(?:Only answer|The last line|Answer the following|Provide the correct)",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        options[letter] = text
    return options


def parse_ground_truth_letter(ground_truth: str) -> str | None:
    text = clean_text(ground_truth)
    match = re.match(r"^\s*([A-H])\s*[\.\)]\s*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if re.fullmatch(r"[A-H]", text, flags=re.IGNORECASE):
        return text.upper()
    return None


def parse_prediction_choice(prediction: str, prompt: str) -> tuple[str | None, str]:
    text = clean_text(prediction)
    if not text:
        return None, "empty_prediction"

    # Prefer explicit final-answer patterns.
    explicit_patterns = [
        r"(?is)(?:^|\b)answer\s*[:\-]?\s*(?:option\s*)?([A-H])\b",
        r"(?is)(?:correct\s+)?option\s*(?:is|:)?\s*([A-H])\b",
        r"(?is)(?:correct\s+)?answer\s*(?:is|:)?\s*([A-H])\b",
        r"(?is)(?:choose|select|selected)\s*(?:option\s*)?([A-H])\b",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper(), "explicit_letter"

    # Exact or first-token letter answers.
    stripped = text.strip()
    match = re.match(r"^\s*([A-H])\s*[\.\)]?\s*(?:$|\n)", stripped, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper(), "leading_letter"

    # Option text match.
    options = parse_options(prompt)
    norm_pred = normalize_for_match(text)
    matched_letters = []
    for letter, option_text in options.items():
        norm_option = normalize_for_match(option_text)
        if not norm_option or len(norm_option) < 4:
            continue
        if norm_option in norm_pred:
            matched_letters.append(letter)
    if len(matched_letters) == 1:
        return matched_letters[0], "option_text"
    if len(matched_letters) > 1:
        return None, "ambiguous_option_text"

    # Last-resort single option mention. Reject multiple unique letters.
    letters = [x.upper() for x in LETTER_RE.findall(text)]
    letters = [x for x in letters if x in set("ABCDEFGH")]
    unique_letters = list(dict.fromkeys(letters))
    if len(unique_letters) == 1:
        return unique_letters[0], "single_letter_mention"
    if len(unique_letters) > 1:
        return None, "ambiguous_letters"

    return None, "no_choice_found"


def prediction_to_row(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    prompt = clean_text(item.get("prompt") or metadata.get("question"))
    prediction_text = clean_text(item.get("text") or item.get("response"))
    ground_truth = clean_text(metadata.get("ground_truth"))
    parsed_truth = parse_ground_truth_letter(ground_truth)
    parsed_answer, parse_reason = parse_prediction_choice(prediction_text, prompt)
    correct = parsed_answer == parsed_truth if parsed_answer and parsed_truth else False
    return {
        "question_id": item.get("question_id") or item.get("id"),
        "benchmark_sample_id": metadata.get("benchmark_sample_id"),
        "group_key": metadata.get("group_key"),
        "signal_key": metadata.get("signal_key"),
        "signal_path_key": metadata.get("signal_path_key"),
        "layout_name": metadata.get("layout_name"),
        "benchmark_subset": metadata.get("benchmark_subset"),
        "task_type": metadata.get("task_type"),
        "answer_type": metadata.get("answer_type"),
        "ground_truth": ground_truth,
        "truth_answer": parsed_truth,
        "prediction_text": prediction_text,
        "parsed_answer": parsed_answer,
        "parse_success": parsed_answer is not None and parsed_truth is not None,
        "parse_reason": parse_reason if parsed_answer is not None else parse_reason,
        "correct": bool(correct),
    }


def safe_metric(value: float) -> float | None:
    if value is None or np.isnan(value) or np.isinf(value):
        return None
    return float(value)


def format_cell(value: Any, floatfmt: str = ".4f") -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return format(float(value), floatfmt)
    return str(value)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(row[col], floatfmt) for col in cols) + " |")
    return "\n".join(lines)


def compute_layout_metrics(rows: pd.DataFrame, valid_only: bool = False) -> pd.DataFrame:
    out = []
    for subset_name, subset_df in [("CSN+G12EC", rows), *list(rows.groupby("benchmark_subset"))]:
        for layout in LAYOUTS:
            df = subset_df[subset_df["layout_name"] == layout]
            parsed = df[df["parse_success"]].copy()
            n = int(len(df))
            parsed_n = int(len(parsed))
            accuracy = float(parsed["correct"].mean()) if parsed_n else None
            bal_acc = None
            macro_f1 = None
            if parsed_n and parsed["truth_answer"].nunique() > 1:
                labels = sorted(set(parsed["truth_answer"].dropna()) | set(parsed["parsed_answer"].dropna()))
                bal_acc = balanced_accuracy_score(parsed["truth_answer"], parsed["parsed_answer"])
                macro_f1 = f1_score(parsed["truth_answer"], parsed["parsed_answer"], labels=labels, average="macro", zero_division=0)
            out.append(
                {
                    "subset": subset_name,
                    "layout_name": layout,
                    "rows": n,
                    "parsed_rows": parsed_n,
                    "parse_rate": parsed_n / n if n else None,
                    "accuracy": accuracy,
                    "balanced_accuracy": safe_metric(bal_acc),
                    "macro_f1": safe_metric(macro_f1),
                }
            )
    return pd.DataFrame(out)


def compute_group_metrics(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    for group_key, group in rows.groupby("group_key", dropna=False):
        layouts_present = set(group["layout_name"])
        subset = clean_text(group["benchmark_subset"].iloc[0]) if len(group) else ""
        missing_layouts = sorted(set(LAYOUTS) - layouts_present)
        duplicate_layout = len(group) != len(layouts_present)
        complete = not missing_layouts and not duplicate_layout and len(group) == 3
        parse_success = bool(complete and group["parse_success"].all())
        row = {
            "group_key": group_key,
            "benchmark_sample_id": group["benchmark_sample_id"].iloc[0] if len(group) else "",
            "signal_key": group["signal_key"].iloc[0] if len(group) else "",
            "benchmark_subset": subset,
            "complete_triplet": complete,
            "missing_layouts": ",".join(missing_layouts),
            "duplicate_layout": duplicate_layout,
            "all_parse_success": parse_success,
        }
        for layout in LAYOUTS:
            layout_df = group[group["layout_name"] == layout]
            if len(layout_df) == 1:
                item = layout_df.iloc[0]
                row[f"pred_{layout}"] = item["parsed_answer"]
                row[f"correct_{layout}"] = bool(item["correct"]) if item["parse_success"] else None
                row[f"parse_success_{layout}"] = bool(item["parse_success"])
                row[f"parse_reason_{layout}"] = item["parse_reason"]
            else:
                row[f"pred_{layout}"] = None
                row[f"correct_{layout}"] = None
                row[f"parse_success_{layout}"] = False
                row[f"parse_reason_{layout}"] = "missing_or_duplicate_layout"
        if parse_success:
            preds = [row[f"pred_{layout}"] for layout in LAYOUTS]
            corrects = [bool(row[f"correct_{layout}"]) for layout in LAYOUTS]
            layout_agreement = len(set(preds)) == 1
            stable_correct = all(corrects)
            recoverable_flip = any(corrects) and not all(corrects)
            stable_wrong = (not any(corrects)) and layout_agreement
            unstable_wrong = (not any(corrects)) and not layout_agreement
            row.update(
                {
                    "layout_agreement": layout_agreement,
                    "stable_correct": stable_correct,
                    "stable_wrong": stable_wrong,
                    "recoverable_flip": recoverable_flip,
                    "unstable_wrong": unstable_wrong,
                    "answer_flip": not layout_agreement,
                    "oracle_correct": any(corrects),
                    "all_wrong": not any(corrects),
                }
            )
        else:
            row.update(
                {
                    "layout_agreement": None,
                    "stable_correct": None,
                    "stable_wrong": None,
                    "recoverable_flip": None,
                    "unstable_wrong": None,
                    "answer_flip": None,
                    "oracle_correct": None,
                    "all_wrong": None,
                }
            )
        group_rows.append(row)

    group_df = pd.DataFrame(group_rows)
    metric_rows = []
    for subset_name, subset_df in [("CSN+G12EC", group_df), *list(group_df.groupby("benchmark_subset"))]:
        valid = subset_df[subset_df["all_parse_success"] == True].copy()
        total = int(len(subset_df))
        valid_n = int(len(valid))
        layout_accs = []
        for layout in LAYOUTS:
            vals = valid[f"correct_{layout}"].dropna()
            layout_accs.append(float(vals.mean()) if len(vals) else None)
        metric_rows.append(
            {
                "subset": subset_name,
                "groups_total": total,
                "groups_valid": valid_n,
                "groups_excluded_parse_or_triplet": total - valid_n,
                "group_parse_rate": valid_n / total if total else None,
                "stable_correct": int(valid["stable_correct"].sum()) if valid_n else 0,
                "stable_correct_rate": float(valid["stable_correct"].mean()) if valid_n else None,
                "stable_wrong": int(valid["stable_wrong"].sum()) if valid_n else 0,
                "stable_wrong_rate": float(valid["stable_wrong"].mean()) if valid_n else None,
                "recoverable_flip": int(valid["recoverable_flip"].sum()) if valid_n else 0,
                "recoverable_flip_rate": float(valid["recoverable_flip"].mean()) if valid_n else None,
                "unstable_wrong": int(valid["unstable_wrong"].sum()) if valid_n else 0,
                "unstable_wrong_rate": float(valid["unstable_wrong"].mean()) if valid_n else None,
                "flip_rate": float(valid["answer_flip"].mean()) if valid_n else None,
                "layout_agreement": float(valid["layout_agreement"].mean()) if valid_n else None,
                "worst_layout_accuracy": min([x for x in layout_accs if x is not None], default=None),
                "oracle_layout_accuracy": float(valid["oracle_correct"].mean()) if valid_n else None,
            }
        )
    return group_df, pd.DataFrame(metric_rows)


def build_choice_metrics(layout_metrics: pd.DataFrame, group_metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, grow in group_metrics.iterrows():
        subset = grow["subset"]
        layout_subset = layout_metrics[layout_metrics["subset"] == subset]
        rec = {
            "subset": subset,
            "groups_total": int(grow["groups_total"]),
            "groups_valid": int(grow["groups_valid"]),
            "parse_rate": grow["group_parse_rate"],
            "Consistent-Correct": grow["stable_correct_rate"],
            "Inconsistency": grow["flip_rate"],
            "Consistent-Error": grow["stable_wrong_rate"],
            "Correctable Inconsistency": grow["recoverable_flip_rate"],
            "Always-wrong Inconsistency": grow["unstable_wrong_rate"],
            "Consistency": grow["layout_agreement"],
            "Worst Acc": grow["worst_layout_accuracy"],
            "Oracle Acc": grow["oracle_layout_accuracy"],
        }
        acc_values = []
        for layout in LAYOUTS:
            lrow = layout_subset[layout_subset["layout_name"] == layout]
            if len(lrow):
                item = lrow.iloc[0]
                rec[f"Acc_{layout}"] = item["accuracy"]
                rec[f"BalAcc_{layout}"] = item["balanced_accuracy"]
                rec[f"MacroF1_{layout}"] = item["macro_f1"]
                acc_values.append(item["accuracy"])
            else:
                rec[f"Acc_{layout}"] = None
                rec[f"BalAcc_{layout}"] = None
                rec[f"MacroF1_{layout}"] = None
        rec["Acc"] = float(np.mean([x for x in acc_values if x is not None])) if acc_values else None
        records.append(rec)
    return pd.DataFrame(records)


def build_examples(rows: pd.DataFrame, group_df: pd.DataFrame, per_outcome: int) -> list[dict[str, Any]]:
    outcome_cols = [
        ("Consistent-Correct", "stable_correct"),
        ("Consistent-Error", "stable_wrong"),
        ("Correctable Inconsistency", "recoverable_flip"),
        ("Always-wrong Inconsistency", "unstable_wrong"),
    ]
    row_groups = {key: group.copy() for key, group in rows.groupby("group_key", dropna=False)}
    examples: list[dict[str, Any]] = []
    for label, col in outcome_cols:
        subset = group_df[(group_df["all_parse_success"] == True) & (group_df[col] == True)]
        for _, group_row in subset.head(per_outcome).iterrows():
            group_key = group_row["group_key"]
            rgroup = row_groups.get(group_key, pd.DataFrame())
            preds = {}
            for layout in LAYOUTS:
                layout_rows = rgroup[rgroup["layout_name"] == layout]
                if len(layout_rows) == 1:
                    item = layout_rows.iloc[0]
                    preds[layout] = {
                        "parsed_answer": item["parsed_answer"],
                        "correct": bool(item["correct"]),
                        "prediction_text": clean_text(item["prediction_text"])[:1000],
                    }
            examples.append(
                {
                    "outcome": label,
                    "benchmark_subset": group_row["benchmark_subset"],
                    "benchmark_sample_id": group_row["benchmark_sample_id"],
                    "group_key": group_key,
                    "signal_key": group_row["signal_key"],
                    "ground_truth": clean_text(rgroup["ground_truth"].iloc[0]) if len(rgroup) else "",
                    "truth_answer": clean_text(rgroup["truth_answer"].iloc[0]) if len(rgroup) else "",
                    "predictions": preds,
                }
            )
    return examples


def write_report(
    path: Path,
    predictions: Path,
    rows: pd.DataFrame,
    layout_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    choice_metrics: pd.DataFrame,
    examples_path: Path,
) -> None:
    lines = [
        "# ECGBench-L3 Close-Ended Display Robustness Report",
        "",
        "## Scope",
        "Included subsets:",
        "- `csn-test-no-cot`",
        "- `g12-test-no-cot`",
        "",
        "Excluded subsets and reasons:",
        "- `ptb-test`: multilabel",
        "- `cpsc-test`: multilabel",
        "- `code15-test`: multilabel / label-set",
        "- `ptb-test-report`: open-ended report generation",
        "- `ecgqa-test`: optional secondary due to parsing noise",
        f"- Prediction file: `{predictions}`",
        "",
        "## Definitions",
        "- Let `y` be the ground truth and `p3`, `p6`, `p12` be parsed predictions for `3R4C`, `6R2C`, and `12R1C`.",
        "- `Consistent-Correct`: `p3 == p6 == p12 == y`.",
        "- `Inconsistency`: not all of `p3`, `p6`, and `p12` are identical.",
        "- `Consistent-Error`: `p3 == p6 == p12` and `p3 != y`.",
        "- `Correctable Inconsistency`: `Inconsistency` is true and at least one layout prediction equals `y`.",
        "- `Always-wrong Inconsistency`: `Inconsistency` is true and none of the layout predictions equals `y`.",
        "- `Consistency`: `Consistent-Correct + Consistent-Error`.",
        "- `Worst-layout Accuracy`: `min(Accuracy_3R4C, Accuracy_6R2C, Accuracy_12R1C)`.",
        "- `Oracle-layout Accuracy`: sample is correct if any of `p3`, `p6`, `p12` equals `y`.",
        "",
        "## Data",
        f"- Rows read: `{len(rows)}`",
        f"- Groups: `{rows['group_key'].nunique(dropna=False)}`",
        "",
        "## Main Metrics",
        df_to_markdown(choice_metrics, floatfmt=".4f"),
        "",
        "## Per-Layout Metrics",
        df_to_markdown(layout_metrics, floatfmt=".4f"),
        "",
        "## Grouped Stability Metrics",
        df_to_markdown(group_metrics, floatfmt=".4f"),
        "",
        "## Qualitative Examples",
        f"- Examples JSONL: `{examples_path}`",
        "- Includes examples of Consistent-Correct, Consistent-Error, Correctable Inconsistency, and Always-wrong Inconsistency when available.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--main-subsets", default=",".join(MAIN_SUBSETS))
    parser.add_argument("--examples-per-outcome", type=int, default=25)
    args = parser.parse_args()

    subsets = {x.strip() for x in args.main_subsets.split(",") if x.strip()}
    output_dir = args.output_dir or (args.dataset_root / "results" / f"{args.run_name}_ecgbench_choice")
    report_dir = args.dataset_root / "reports"
    result_dir = args.dataset_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    requested_metrics_csv = result_dir / f"{args.run_name}_ecgbench_choice_metrics.csv"
    requested_grouped_parquet = result_dir / f"{args.run_name}_ecgbench_choice_grouped_outcomes.parquet"
    requested_examples_jsonl = result_dir / f"{args.run_name}_ecgbench_choice_examples.jsonl"
    requested_report_md = report_dir / f"{args.run_name}_ecgbench_choice_metric_report.md"

    raw = read_jsonl(args.predictions)
    rows = pd.DataFrame(prediction_to_row(item) for item in raw)
    rows = rows[rows["benchmark_subset"].isin(subsets)].copy()
    rows.to_parquet(output_dir / "row_level_predictions.parquet", index=False)
    rows.to_csv(output_dir / "row_level_predictions.csv", index=False)

    parse_failures = rows[~rows["parse_success"]].copy()
    parse_failures.to_csv(output_dir / "parse_failures.csv", index=False)

    layout_metrics = compute_layout_metrics(rows)
    group_df, group_metrics = compute_group_metrics(rows)
    choice_metrics = build_choice_metrics(layout_metrics, group_metrics)
    examples = build_examples(rows, group_df, args.examples_per_outcome)

    layout_metrics.to_csv(output_dir / "layout_metrics.csv", index=False)
    group_df.to_parquet(output_dir / "group_level_predictions.parquet", index=False)
    group_df.to_csv(output_dir / "group_level_predictions.csv", index=False)
    group_metrics.to_csv(output_dir / "internal_group_metrics_legacy_columns.csv", index=False)
    choice_metrics.to_csv(output_dir / "choice_metrics.csv", index=False)
    (output_dir / "internal_group_metrics_legacy_columns.json").write_text(
        json.dumps(group_metrics.to_dict(orient="records"), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    choice_metrics.to_csv(requested_metrics_csv, index=False)
    group_df.to_parquet(requested_grouped_parquet, index=False)
    with requested_examples_jsonl.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n")
    write_report(requested_report_md, args.predictions, rows, layout_metrics, group_metrics, choice_metrics, requested_examples_jsonl)
    write_report(output_dir / "ecgbench_l3_close_ended_stability_report.md", args.predictions, rows, layout_metrics, group_metrics, choice_metrics, requested_examples_jsonl)

    print("# ECGBench-L3 Close-Ended Display Robustness Summary")
    print(df_to_markdown(choice_metrics, floatfmt=".4f"))
    print(f"metrics_csv={requested_metrics_csv}")
    print(f"grouped_outcomes={requested_grouped_parquet}")
    print(f"examples={requested_examples_jsonl}")
    print(f"report={requested_report_md}")


if __name__ == "__main__":
    main()
