"""Reward functions for GRPO training (see CLAUDE.md's "Reward design" section).

turn_reward implements turn-level credit assignment via reward density -- GRPO scores one
scalar per completed trajectory, so there is no per-timestep value function here, and this is
not a literal per-step RL change.
"""

import re
from typing import Any, Literal

from turn_level_rewards.metrics import exact_match, f1_score

Completion = list[dict[str, Any]]

_ANSWER_RE = re.compile(r"<answer>(.+?)</answer>", re.DOTALL)


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


def format_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    """+0.1 for a well-formed single <answer> tag in the final message, -0.1 otherwise."""
    return [0.1 if _extract_answer(c) is not None else -0.1 for c in completions]


def outcome_reward(
    completions: list[Completion], golden_answers: list[list[str]], **kwargs: Any
) -> list[float]:
    """SQuAD F1 + 0.5 exact-match bonus, maxed over each row's golden_answers list."""
    rewards = []
    for completion, answers in zip(completions, golden_answers, strict=True):
        prediction = _extract_answer(completion) or ""
        best = max(
            f1_score(prediction, answer) + (0.5 if exact_match(prediction, answer) else 0.0)
            for answer in answers
        )
        rewards.append(best)
    return rewards


def turn_reward(environments: list[Any], **kwargs: Any) -> list[float]:
    """0.4 * retrieval_fraction -- dense signal for surfacing gold supporting-fact passages."""
    return [0.4 * environment.retrieval_fraction for environment in environments]


def get_reward_funcs(condition: Literal["outcome_only", "turn_level"]) -> list[Any]:
    """Return the reward function list for a training condition (CLAUDE.md's Reward design)."""
    if condition == "outcome_only":
        return [format_reward, outcome_reward]
    if condition == "turn_level":
        return [format_reward, outcome_reward, turn_reward]
    raise ValueError(f"Unknown condition: {condition!r}")
