import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


LAYOUTBIND_CATEGORIES = (
    "closed_yes_no",
    "closed_multiple_choice",
    "closed_categorical",
    "multi_label",
    "free_form_report",
    "open_ended_explanation",
    "grounding_free_form",
    "other_or_ambiguous",
)

MULTILABEL_MARKERS = (
    "select all",
    "all applicable",
    "every applicable",
    "all relevant options",
    "all pertinent options",
    "list your choices",
    "list your picks",
    "separated by",
    "commas-separated",
    "comma-separated",
    "semicolons",
    "slashes",
    "slash-separated",
)

REPORT_MARKERS = (
    "write a comprehensive report",
    "compose a diagnostic summary",
    "draft a comprehensive report",
    "provide the final diagnosis",
    "diagnostic summary",
    "ecg report",
)

GROUNDING_MARKERS = (
    "bounding box",
    "grounding",
    "locate",
    "localize",
    "highlight",
)

REASONING_PROMPT_MARKERS = (
    "explain",
    "reasoning",
    "rationale",
    "justify",
    "justification",
    "step-by-step",
    "step by step",
    "detailed explanation",
    "detailed justification",
    "why",
    "provide a reason",
    "provide reasons",
)

BARE_ANSWER_PROMPT_MARKERS = (
    "directly output",
    "direct output",
    "only output",
    "output only",
    "answer only",
    "only answer",
    "just answer",
    "short answer",
    "provide the answer",
    "output the answer",
    "direct provide",
    "fill in the blank",
)

FINAL_ANSWER_PROMPT_MARKERS = (
    "final answer",
    "final choice",
    "then choose",
    "then select",
    "before selecting",
    "before providing your answer",
    "after explaining",
)

STRICT_PROMPT_FORMATS = {
    "bare_answer",
    "final_answer",
    "unspecified_or_short_answer",
}

LAYOUTBIND_ELIGIBILITY_TIERS = ("strict", "broad")


def normalize_answer_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"<image>\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def _strip_image_token(text: str) -> str:
    return re.sub(r"^\s*<image>\s*", "", text or "", flags=re.IGNORECASE).strip()


def infer_prompt_format(prompt: str) -> str:
    prompt_clean = _strip_image_token(prompt)
    prompt_norm = normalize_answer_text(prompt_clean)
    has_reasoning = any(marker in prompt_norm for marker in REASONING_PROMPT_MARKERS)
    has_bare = any(marker in prompt_norm for marker in BARE_ANSWER_PROMPT_MARKERS)
    has_final = any(marker in prompt_norm for marker in FINAL_ANSWER_PROMPT_MARKERS)

    if has_reasoning and has_bare:
        return "mixed_reasoning_and_bare_answer"
    if has_reasoning and (has_final or "answer" in prompt_norm or "choice" in prompt_norm):
        return "explanation_plus_answer"
    if has_final:
        return "final_answer"
    if has_bare or "__" in prompt_clean or "(yes or no)" in prompt_norm:
        return "bare_answer"
    if len(prompt_norm.split()) <= 45:
        return "unspecified_or_short_answer"
    return "other_or_ambiguous"


def infer_target_form(target: str, inferred: Optional[Dict[str, Any]] = None) -> str:
    target_text = "" if target is None else str(target).strip()
    target_norm = normalize_answer_text(target_text)
    if not target_norm:
        return "ambiguous_or_unsupported"

    sentence_count = len([piece for piece in re.split(r"[.!?]+", target_text) if piece.strip()])
    word_count = len(target_norm.split())
    has_reasoning = any(marker in target_norm for marker in REASONING_PROMPT_MARKERS)
    canonical = normalize_answer_text((inferred or {}).get("target_canonical"))
    candidates = (inferred or {}).get("candidate_answer_strings") or []
    candidate_norms = {normalize_answer_text(candidate) for candidate in candidates}
    if target_norm in {"none", "null", "nan"} and canonical in {"uncertain", "unknown"}:
        return "bare_label"

    if word_count > 30 or sentence_count > 2:
        return "long_rationale_with_answer" if canonical and canonical in target_norm else "long_or_freeform_target"
    if has_reasoning and word_count > 8:
        return "long_rationale_with_answer"

    if re.fullmatch(r"[A-H](?:[\.\)])?", target_text.strip(), flags=re.IGNORECASE):
        return "bare_label"
    if target_norm in {"yes", "no", "norm", "normal", "abnormal", "abnorm"}:
        return "bare_label"
    if canonical and target_norm == canonical:
        return "bare_label"
    if target_norm in candidate_norms and word_count <= 12:
        return "bare_label"
    if re.match(r"^[A-H][\.\)]\s+.+", target_text, flags=re.IGNORECASE) and word_count <= 12:
        return "bare_label"

    if word_count > 12:
        return "long_rationale_with_answer" if canonical and canonical in target_norm else "long_or_freeform_target"
    return "long_or_freeform_target"


def extract_human_and_target(group: Dict[str, Any]) -> Tuple[str, str]:
    conversations = group.get("conversations") or []
    human = ""
    target = ""
    for conv in conversations:
        role = conv.get("from") or conv.get("role")
        value = conv.get("value") or ""
        if role in ("human", "user") and not human:
            human = value
        elif role in ("gpt", "assistant") and not target:
            target = value
    return human, target


def extract_lettered_options(prompt: str) -> List[Dict[str, str]]:
    options = []
    pattern = re.compile(r"(?m)^\s*([A-H])[\.\)]\s+(.+?)\s*$")
    for match in pattern.finditer(prompt or ""):
        text = match.group(2).strip()
        if text:
            options.append({"id": match.group(1).upper(), "text": text})
    return options


def _clean_option(option: str) -> str:
    option = re.sub(r"\s+", " ", option or "").strip()
    option = option.strip(" .;")
    return option


def extract_options_block(prompt: str) -> List[str]:
    text = prompt or ""
    match = re.search(
        r"Options:\s*(.+?)(?:\n\s*\n|Direct provide|Provide the accurate|Output should|Before selecting|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        list_match = re.search(
            r"(?:one of the following|following\s+\d+\s+options)\s*:?\s*(.+?)(?:\. Before|\. Direct| Direct|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not list_match:
            return []
        raw = list_match.group(1)
    else:
        raw = match.group(1)

    raw = raw.replace("\n", ", ")
    pieces = re.split(r"\s*,\s*|\s*;\s*", raw)
    options = [_clean_option(piece) for piece in pieces]
    return [option for option in options if option]


def _target_letter(target: str, options: List[Dict[str, str]]) -> Optional[int]:
    valid = {opt["id"]: idx for idx, opt in enumerate(options)}
    target_norm = normalize_answer_text(target)
    direct = re.match(r"^([A-H])(?:[\.\)]|\b)", target.strip(), flags=re.IGNORECASE)
    if direct and direct.group(1).upper() in valid:
        return valid[direct.group(1).upper()]
    for pattern in (
        r"(?:answer|choice|option|right choice|correct choice|would be|is)\s*(?:is|would be|:)?\s*([A-H])\b",
        r"\b([A-H])[\.\)]\s*",
    ):
        for match in re.finditer(pattern, target, flags=re.IGNORECASE):
            letter = match.group(1).upper()
            if letter in valid:
                return valid[letter]
    matches = []
    for idx, opt in enumerate(options):
        opt_norm = normalize_answer_text(opt["text"])
        if opt_norm and opt_norm in target_norm:
            matches.append(idx)
    if len(set(matches)) == 1:
        return matches[0]
    return None


def _target_option_index(target: str, options: List[str]) -> Optional[int]:
    target_norm = normalize_answer_text(target)
    if not target_norm:
        return None
    if target_norm in {"none", "null", "nan"}:
        for idx, option in enumerate(options):
            if normalize_answer_text(option) in {"uncertain", "unknown"}:
                return idx
    exact = [idx for idx, option in enumerate(options) if normalize_answer_text(option) == target_norm]
    if len(exact) == 1:
        return exact[0]
    contained = []
    for idx, option in enumerate(options):
        opt_norm = normalize_answer_text(option)
        if opt_norm and opt_norm in target_norm:
            contained.append(idx)
    if len(set(contained)) == 1:
        return contained[0]
    return None


def _metadata_answer_space(metadata: Dict[str, Any]) -> List[str]:
    raw = metadata.get("answer_space")
    if not isinstance(raw, (list, tuple)):
        return []
    options = []
    seen = set()
    for value in raw:
        option = _clean_option(str(value))
        norm = normalize_answer_text(option)
        if option and norm and norm not in seen:
            options.append(option)
            seen.add(norm)
    return options


def _yes_no_target(target: str) -> Optional[int]:
    target_norm = normalize_answer_text(target)
    if re.search(r"\byes\b", target_norm) and not re.search(r"\bno\b", target_norm):
        return 0
    if re.search(r"\bno\b", target_norm) and not re.search(r"\byes\b", target_norm):
        return 1
    return None


def _norm_abnormal_target(target: str) -> Optional[int]:
    target_norm = normalize_answer_text(target)
    if re.search(r"\b(norm|normal)\b", target_norm) and not re.search(r"\b(abnormal|abnorm)\b", target_norm):
        return 0
    if re.search(r"\b(abnormal|abnorm)\b", target_norm):
        return 1
    return None


def infer_layoutbind_answer_space(group: Dict[str, Any]) -> Dict[str, Any]:
    metadata = group.get("metadata") or {}
    prompt, target = extract_human_and_target(group)
    prompt_clean = _strip_image_token(prompt)
    prompt_norm = normalize_answer_text(prompt_clean)
    target_norm = normalize_answer_text(target)
    dataset_name = str(metadata.get("dataset_name") or "")
    task_category = str(metadata.get("task_category") or "")
    question_type = str(metadata.get("question_type") or "")

    result: Dict[str, Any] = {
        "category": "other_or_ambiguous",
        "finite_answer_space_exists": False,
        "targetbind_eligible": False,
        "targetmargin_eligible": False,
        "layoutbind_broad_eligible": False,
        "layoutbind_strict_eligible": False,
        "prompt_format": infer_prompt_format(prompt_clean),
        "target_form": "ambiguous_or_unsupported",
        "reasoning_or_long_target_risk": False,
        "answer_space_type": None,
        "answer_space": None,
        "candidate_answer_strings": None,
        "target_answer_id": None,
        "target_canonical": None,
        "exclusion_reason": "ambiguous_or_unsupported",
    }

    if "ground" in dataset_name.lower() or task_category.lower() == "grounding" or any(marker in prompt_norm for marker in GROUNDING_MARKERS):
        result.update(category="grounding_free_form", exclusion_reason="grounding_free_form")
        return result

    if task_category.lower() == "report" or any(marker in prompt_norm for marker in REPORT_MARKERS):
        result.update(category="free_form_report", exclusion_reason="free_form_report")
        return result

    if any(marker in prompt_norm for marker in MULTILABEL_MARKERS):
        options = extract_options_block(prompt_clean)
        result.update(
            category="multi_label",
            finite_answer_space_exists=bool(options),
            answer_space_type="multi_label_options" if options else None,
            answer_space=options or None,
            candidate_answer_strings=options or None,
            exclusion_reason="multi_label_excluded_v1",
        )
        return result

    metadata_options = _metadata_answer_space(metadata)
    if len(metadata_options) >= 2:
        target_idx = _target_option_index(metadata.get("target_value", target), metadata_options)
        option_norms = {normalize_answer_text(option) for option in metadata_options}
        if option_norms == {"yes", "no"}:
            category = "closed_yes_no"
            answer_space_type = "yes_no"
        elif option_norms in ({"norm", "abnormal"}, {"normal", "abnormal"}):
            category = "closed_categorical"
            answer_space_type = "norm_abnormal"
        else:
            category = "closed_categorical"
            answer_space_type = "categorical_options"
        result.update(
            category=category,
            finite_answer_space_exists=True,
            answer_space_type=answer_space_type,
            answer_space=metadata_options,
            candidate_answer_strings=metadata_options,
        )
        if target_idx is not None:
            result.update(
                targetbind_eligible=True,
                targetmargin_eligible=True,
                layoutbind_broad_eligible=True,
                target_answer_id=target_idx,
                target_canonical=metadata_options[target_idx],
                exclusion_reason=None,
            )
        else:
            result.update(exclusion_reason="metadata_answer_space_target_not_parsed")
        _finalize_layoutbind_tiers(result, target)
        return result

    lettered = extract_lettered_options(prompt_clean)
    if len(lettered) >= 2:
        target_idx = _target_letter(target, lettered)
        candidates = [f"{opt['id']}. {opt['text']}" for opt in lettered]
        result.update(
            category="closed_multiple_choice",
            finite_answer_space_exists=True,
            answer_space_type="multiple_choice_letter",
            answer_space=[opt["id"] for opt in lettered],
            candidate_answer_strings=candidates,
        )
        if target_idx is not None:
            result.update(
                targetbind_eligible=True,
                targetmargin_eligible=True,
                layoutbind_broad_eligible=True,
                target_answer_id=target_idx,
                target_canonical=lettered[target_idx]["id"],
                exclusion_reason=None,
            )
        else:
            result.update(exclusion_reason="target_letter_not_parsed")
        _finalize_layoutbind_tiers(result, target)
        return result

    if ("yes" in prompt_norm and "no" in prompt_norm) or question_type.lower() == "yes/no":
        target_idx = _yes_no_target(target)
        result.update(
            category="closed_yes_no",
            finite_answer_space_exists=True,
            answer_space_type="yes_no",
            answer_space=["yes", "no"],
            candidate_answer_strings=["yes", "no"],
        )
        if target_idx is not None:
            result.update(
                targetbind_eligible=True,
                targetmargin_eligible=True,
                layoutbind_broad_eligible=True,
                target_answer_id=target_idx,
                target_canonical=["yes", "no"][target_idx],
                exclusion_reason=None,
            )
        else:
            result.update(exclusion_reason="yes_no_target_not_parsed")
        _finalize_layoutbind_tiers(result, target)
        return result

    if ("norm" in prompt_norm and "abnormal" in prompt_norm) or ("normal" in prompt_norm and "abnormal" in prompt_norm):
        target_idx = _norm_abnormal_target(target)
        use_codes = "norm" in prompt_norm and "abnormal" in prompt_norm
        candidates = ["NORM", "ABNORMAL"] if use_codes else ["normal", "abnormal"]
        result.update(
            category="closed_categorical",
            finite_answer_space_exists=True,
            answer_space_type="norm_abnormal",
            answer_space=candidates,
            candidate_answer_strings=candidates,
        )
        if target_idx is not None:
            result.update(
                targetbind_eligible=True,
                targetmargin_eligible=True,
                layoutbind_broad_eligible=True,
                target_answer_id=target_idx,
                target_canonical=candidates[target_idx],
                exclusion_reason=None,
            )
        else:
            result.update(exclusion_reason="norm_abnormal_target_not_parsed")
        _finalize_layoutbind_tiers(result, target)
        return result

    options = extract_options_block(prompt_clean)
    if len(options) >= 2:
        target_idx = _target_option_index(target, options)
        result.update(
            category="closed_categorical",
            finite_answer_space_exists=True,
            answer_space_type="categorical_options",
            answer_space=options,
            candidate_answer_strings=options,
        )
        if target_idx is not None:
            result.update(
                targetbind_eligible=True,
                targetmargin_eligible=True,
                layoutbind_broad_eligible=True,
                target_answer_id=target_idx,
                target_canonical=options[target_idx],
                exclusion_reason=None,
            )
        else:
            result.update(exclusion_reason="categorical_target_not_parsed")
        _finalize_layoutbind_tiers(result, target)
        return result

    if len(target_norm.split()) > 20 or any(marker in prompt_norm for marker in ("explain", "reasoning", "step-by-step", "why")):
        result.update(category="open_ended_explanation", exclusion_reason="open_ended_or_explanation")
        return result

    return result


def _finalize_layoutbind_tiers(result: Dict[str, Any], target: str) -> None:
    result["target_form"] = infer_target_form(target, result)
    risk = (
        result.get("prompt_format") in {
            "long_rationale_with_answer",
            "explanation_plus_answer",
            "mixed_reasoning_and_bare_answer",
            "other_or_ambiguous",
        }
        or result.get("target_form") != "bare_label"
    )
    result["reasoning_or_long_target_risk"] = bool(risk)
    strict = (
        bool(result.get("layoutbind_broad_eligible"))
        and result.get("target_form") == "bare_label"
        and result.get("prompt_format") in STRICT_PROMPT_FORMATS
        and result.get("category") in {
            "closed_yes_no",
            "closed_multiple_choice",
            "closed_categorical",
        }
    )
    result["layoutbind_strict_eligible"] = bool(strict)


def is_layoutbind_tier_eligible(inferred: Dict[str, Any], tier: str) -> bool:
    tier = (tier or "strict").lower()
    if tier not in LAYOUTBIND_ELIGIBILITY_TIERS:
        raise ValueError(f"Unsupported LayoutBind eligibility tier: {tier}")
    if tier == "broad":
        return bool(inferred.get("layoutbind_broad_eligible") or inferred.get("targetbind_eligible"))
    return bool(inferred.get("layoutbind_strict_eligible"))


def compute_targetbind_margin_losses(
    scores_by_layout: torch.Tensor,
    target_index: int,
    *,
    alpha: float = 0.5,
    label_smoothing: float = 0.05,
    margin_delta: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scores_by_layout.dim() != 2:
        raise ValueError("scores_by_layout must have shape [num_layouts, num_candidates].")
    num_layouts, num_candidates = scores_by_layout.shape
    if num_layouts < 1 or num_candidates < 2:
        zero = scores_by_layout.sum() * 0.0
        return zero, zero
    if target_index < 0 or target_index >= num_candidates:
        raise ValueError("target_index is out of range.")

    log_probs = torch.log_softmax(scores_by_layout, dim=-1)
    probs = log_probs.exp()
    pbar = probs.mean(dim=0).detach()

    y_smooth = torch.full_like(pbar, label_smoothing / num_candidates)
    y_smooth[target_index] = (1.0 - label_smoothing) + (label_smoothing / num_candidates)
    q = ((1.0 - alpha) * y_smooth + alpha * pbar).detach()
    q = q / q.sum().clamp_min(1e-12)
    log_q = torch.log(q.clamp_min(1e-12))
    bind_loss = (q.unsqueeze(0) * (log_q.unsqueeze(0) - log_probs)).sum(dim=-1).mean()

    target_score = scores_by_layout[:, target_index]
    wrong_mask = torch.ones(num_candidates, dtype=torch.bool, device=scores_by_layout.device)
    wrong_mask[target_index] = False
    best_wrong = scores_by_layout[:, wrong_mask].max(dim=-1).values
    margin = target_score - best_wrong
    margin_loss = F.relu(torch.as_tensor(margin_delta, device=scores_by_layout.device, dtype=scores_by_layout.dtype) - margin).mean()
    return bind_loss, margin_loss
