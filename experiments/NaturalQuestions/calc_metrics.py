import json
import re
import string
from collections import Counter
from pathlib import Path


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def score_predictions(predictions: list[str], examples: list[dict]) -> tuple[float, float, list[dict]]:
    total = len(examples)
    f1_total = 0.0
    em_total = 0.0
    per_item = []

    for prediction, example in zip(predictions, examples):
        answers = example.get("answers", example.get("answer", []))
        item_f1 = metric_max_over_ground_truths(f1_score, prediction, answers)
        item_em = metric_max_over_ground_truths(exact_match_score, prediction, answers)
        f1_total += item_f1
        em_total += float(item_em)
        per_item.append(
            {
                "id": example.get("id"),
                "question": example["question"],
                "answers": answers,
                "prediction": prediction,
                "f1": item_f1,
                "em": bool(item_em),
            }
        )

    if total == 0:
        return 0.0, 0.0, per_item
    return 100.0 * f1_total / total, 100.0 * em_total / total, per_item


def write_tsv_artifacts(pred_path: Path, gold_path: Path, predictions: list[str], examples: list[dict]) -> None:
    pred_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    gold_lines = [
        f"{item['question']}\t{json.dumps(item.get('answers', item.get('answer', [])), ensure_ascii=False)}"
        for item in examples
    ]
    gold_path.write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
