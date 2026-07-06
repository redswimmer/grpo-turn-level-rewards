"""Reward functions for GRPO training (see CLAUDE.md's "Reward design" section).

turn_reward implements turn-level credit assignment via reward density -- GRPO scores one
scalar per completed trajectory, so there is no per-timestep value function here, and this is
not a literal per-step RL change.
"""

import re
from collections.abc import Callable
from typing import Any, Literal

from turn_level_rewards.metrics import exact_match, f1_score

Completion = list[dict[str, Any]]
LogMetric = Callable[[str, float], None]

_ANSWER_RE = re.compile(r"<answer>(.+?)</answer>", re.DOTALL)


def _noop_log_metric(name: str, value: float) -> None:
    return None


def _extract_answer(completion: Completion) -> str | None:
    """Return the final answer text if the completion ends in one well-formed <answer> tag.

    Well-formed means: the last message has no unresolved tool_calls, and its content contains
    exactly one non-empty <answer>...</answer> pair.
    """
    if not completion:
        return None
    last = completion[-1]
    if last.get("tool_calls"):
        return None
    content = last.get("content")
    if not isinstance(content, str):
        return None
    matches = _ANSWER_RE.findall(content)
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def format_reward(
    completions: list[Completion], log_metric: LogMetric = _noop_log_metric, **kwargs: Any
) -> list[float]:
    """+0.1 for a well-formed single <answer> tag in the final message, -0.1 otherwise.

    Logs format_compliance_rate (1.0/0.0 per completion) -- see CLAUDE.md's "Experiment
    tracking" section.
    """
    rewards = []
    for completion in completions:
        compliant = _extract_answer(completion) is not None
        rewards.append(0.1 if compliant else -0.1)
        log_metric("format_compliance_rate", 1.0 if compliant else 0.0)
    return rewards


def outcome_reward(
    completions: list[Completion],
    golden_answers: list[list[str]],
    log_metric: LogMetric = _noop_log_metric,
    **kwargs: Any,
) -> list[float]:
    """SQuAD F1 + 0.5 exact-match bonus, maxed over each row's golden_answers list.

    Logs the winning answer's raw exact_match and f1 (unblended) -- see CLAUDE.md's "Experiment
    tracking" section.
    """
    rewards = []
    for completion, answers in zip(completions, golden_answers, strict=True):
        prediction = _extract_answer(completion) or ""
        scored = []
        for answer in answers:
            f1 = f1_score(prediction, answer)
            em = exact_match(prediction, answer)
            scored.append((f1 + (0.5 if em else 0.0), f1, em))
        best_reward, best_f1, best_em = max(scored, key=lambda item: item[0])
        rewards.append(best_reward)
        log_metric("exact_match", float(best_em))
        log_metric("f1", best_f1)
    return rewards


def turn_reward(
    environments: list[Any], log_metric: LogMetric = _noop_log_metric, **kwargs: Any
) -> list[float]:
    """0.4 * retrieval_fraction -- dense signal for surfacing gold supporting-fact passages.

    Logs the unscaled retrieval_fraction -- see CLAUDE.md's "Experiment tracking" section.
    """
    rewards = []
    for environment in environments:
        rewards.append(0.4 * environment.retrieval_fraction)
        log_metric("retrieval_fraction", environment.retrieval_fraction)
    return rewards


def get_reward_funcs(condition: Literal["outcome_only", "turn_level"]) -> list[Any]:
    """Return the reward function list for a training condition (CLAUDE.md's Reward design)."""
    if condition == "outcome_only":
        return [format_reward, outcome_reward]
    if condition == "turn_level":
        return [format_reward, outcome_reward, turn_reward]
    raise ValueError(f"Unknown condition: {condition!r}")
