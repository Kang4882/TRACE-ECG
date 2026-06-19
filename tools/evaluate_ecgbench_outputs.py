#!/usr/bin/env python
import argparse
import json
import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

random.seed(42)


def normalize_text(value):
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def extract_choice(text):
    if text in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        return text
    for char_dot in ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H."]:
        if char_dot in text:
            return char_dot[0]
    if "The correct option is " in text:
        predict_char = text.split("The correct option is ")[-1][0]
        return predict_char if predict_char in ["A", "B", "C", "D", "E", "F", "G", "H"] else None
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
        for char in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            if char in answer:
                return char
    return None


def compute_f1_auc(y_pred, y_true):
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    hl = hamming_loss(y_true_bin, y_pred_bin)
    f1_scores = f1_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    auc_scores = []
    for i in range(y_true_bin.shape[1]):
        try:
            auc_scores.append(roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i]))
        except ValueError:
            auc_scores.append(np.nan)
    return float(np.mean(f1_scores)), float(np.nanmean(auc_scores)), float(hl)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eval_multilabel(result_file, golden_file, label_space, split):
    golden = {}
    for item in read_json(golden_file):
        answer = normalize_text(item["conversations"][1]["value"])
        if split == "code15-test" and answer in ["NORM", "ABNORMAL"]:
            golden[item["id"]] = [answer]
        else:
            golden[item["id"]] = [label for label in label_space if label in answer]
    preds = []
    trues = []
    for item in read_jsonl(result_file):
        text = item.get("text", item.get("response", ""))
        if split == "code15-test":
            if "Answer:" in text:
                text = text.split("Answer:")[-1]
            if "NORM" in text and "ABNORMAL" not in text:
                pred = ["NORM"] + [label for label in label_space if label in text]
            elif "ABNORMAL" in text:
                pred = ["ABNORMAL"] + [label for label in label_space if label in text]
            else:
                pred = [label for label in label_space if label in text]
        else:
            pred = [label for label in label_space if label in text]
        qid = item.get("question_id", item.get("id"))
        preds.append(pred)
        trues.append(golden[qid])
    f1, auc, hl = compute_f1_auc(preds, trues)
    return {
        "split": split,
        "n": len(preds),
        "f1": f1,
        "auc": auc,
        "hamming_loss": hl,
    }


def eval_choice(result_file, golden_file, split):
    data = read_json(golden_file)
    answer_dict = {
        item["id"]: normalize_text(item["conversations"][1]["value"])[0]
        for item in data
    }
    preds = []
    trues = []
    for item in read_jsonl(result_file):
        pred = extract_choice(item.get("text", item.get("response", "")))
        if pred is None:
            pred = random.choice(["A", "B", "C", "D", "E", "F", "G", "H"])
        qid = item.get("question_id", item.get("id"))
        preds.append(pred)
        trues.append(answer_dict[qid])
    return {
        "split": split,
        "n": len(preds),
        "accuracy": float(accuracy_score(trues, preds)),
    }


def eval_ecgqa(result_file, golden_file):
    golden = {
        item["id"]: normalize_text(item["conversations"][1]["value"]).lower()
        for item in read_json(golden_file)
    }
    correct = 0
    total = 0
    for item in read_jsonl(result_file):
        prompt = item.get("prompt", "")
        text = item.get("text", item.get("response", "")).lower()
        if "Options:" in prompt:
            options_text = prompt.split("Options:")[-1].replace(
                "Only answer based on the given Options without any explanation.",
                "",
            )
            candidates = [candidate.strip() for candidate in options_text.split(",")]
            pred = "".join(candidate for candidate in candidates if candidate.lower() in text)
        else:
            pred = text
        qid = item.get("question_id", item.get("id"))
        correct += int(set(pred.lower()) == set(golden[qid]))
        total += 1
    return {"split": "ecgqa-test", "n": total, "accuracy": correct / total if total else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--golden-root", default="/data/ECGBench")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    specs = {
        "ptb-test": ("multilabel", ["NORM", "MI", "STTC", "CD", "HYP"]),
        "cpsc-test": ("multilabel", ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]),
        "code15-test": ("multilabel", ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]),
        "csn-test-no-cot": ("choice", None),
        "g12-test-no-cot": ("choice", None),
        "ecgqa-test": ("ecgqa", None),
    }

    results = []
    for split, (kind, labels) in specs.items():
        result_file = os.path.join(args.result_root, split, "step-final.jsonl")
        golden_file = os.path.join(args.golden_root, f"{split}.json")
        if not os.path.exists(result_file):
            continue
        if kind == "multilabel":
            results.append(eval_multilabel(result_file, golden_file, labels, split))
        elif kind == "choice":
            results.append(eval_choice(result_file, golden_file, split))
        else:
            results.append(eval_ecgqa(result_file, golden_file))

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for result in results:
        if "accuracy" in result:
            print(f"{result['split']}: n={result['n']} accuracy={result['accuracy']:.4f}")
        else:
            print(
                f"{result['split']}: n={result['n']} "
                f"f1={result['f1']:.4f} auc={result['auc']:.4f} "
                f"hamming_loss={result['hamming_loss']:.4f}"
            )


if __name__ == "__main__":
    main()
