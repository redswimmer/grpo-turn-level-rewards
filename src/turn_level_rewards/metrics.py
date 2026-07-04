"""SQuAD-style exact-match and F1 scoring (stdlib only, no dependencies)."""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punctuation(s: str) -> str:
        return "".join(ch for ch in s if ch not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    """True if prediction and ground_truth are identical after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 between prediction and ground_truth, after normalization."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)
