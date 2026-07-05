import pytest
from turn_level_rewards.rewards import (
    format_reward,
    get_reward_funcs,
    outcome_reward,
    turn_reward,
)


class FakeEnvironment:
    def __init__(self, retrieval_fraction: float) -> None:
        self.retrieval_fraction = retrieval_fraction


class _FakeLogMetric:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def __call__(self, name: str, value: float) -> None:
        self.calls.append((name, value))


def _search_tool_call(query: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "search", "arguments": {"query": query}}}
        ],
    }


def _tool_response(content: str) -> dict:
    return {"role": "tool", "name": "search", "content": content}


def _answer(text: str) -> dict:
    return {"role": "assistant", "content": f"<answer>{text}</answer>"}


def test_well_formed_correct_answer_full_retrieval():
    completions = [
        [
            _search_tool_call("127 hours"),
            _tool_response('Doc 1 (Title: "127 Hours"): A 2010 film.'),
            _answer("127 Hours"),
        ]
    ]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=1.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [1.5]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.4])


def test_well_formed_correct_answer_zero_retrieval():
    completions = [[_answer("127 Hours")]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [1.5]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.0])


def test_well_formed_wrong_answer():
    completions = [[_answer("Peter Schmeichel")]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=1.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [0.0]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.4])


def test_malformed_missing_answer_tag():
    completions = [[{"role": "assistant", "content": "I believe it is 127 Hours."}]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.5)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([-0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [0.0]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.2])


def test_hard_tool_call_cap_mid_call_unresolved_tool_calls_no_answer():
    completions = [[_search_tool_call("127 hours")]]  # cap hit: no trailing tool/answer message
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([-0.1])
    assert outcome_reward(completions=completions, golden_answers=golden_answers) == pytest.approx(
        [0.0]
    )
    assert turn_reward(completions=completions, environments=environments) == pytest.approx([0.0])


def test_get_reward_funcs_outcome_only_excludes_turn_reward():
    funcs = get_reward_funcs("outcome_only")
    assert [f.__name__ for f in funcs] == ["format_reward", "outcome_reward"]


def test_get_reward_funcs_turn_level_includes_turn_reward():
    funcs = get_reward_funcs("turn_level")
    assert [f.__name__ for f in funcs] == ["format_reward", "outcome_reward", "turn_reward"]


def test_get_reward_funcs_rejects_unknown_condition():
    with pytest.raises(ValueError):
        get_reward_funcs("bogus")  # type: ignore


def test_format_reward_logs_format_compliance_rate():
    log_metric = _FakeLogMetric()
    completions = [
        [_answer("127 Hours")],
        [{"role": "assistant", "content": "no tag here"}],
    ]

    format_reward(completions=completions, log_metric=log_metric)

    assert log_metric.calls == [
        ("format_compliance_rate", 1.0),
        ("format_compliance_rate", 0.0),
    ]


def test_outcome_reward_logs_exact_match_and_f1_per_completion():
    log_metric = _FakeLogMetric()
    completions = [[_answer("127 Hours")], [_answer("Peter Schmeichel")]]
    golden_answers = [["127 Hours"], ["127 Hours"]]

    outcome_reward(completions=completions, golden_answers=golden_answers, log_metric=log_metric)

    assert log_metric.calls == [
        ("exact_match", 1.0),
        ("f1", 1.0),
        ("exact_match", 0.0),
        ("f1", 0.0),
    ]


def test_turn_reward_logs_unscaled_retrieval_fraction():
    log_metric = _FakeLogMetric()
    environments = [
        FakeEnvironment(retrieval_fraction=1.0),
        FakeEnvironment(retrieval_fraction=0.5),
    ]

    turn_reward(environments=environments, log_metric=log_metric)

    assert log_metric.calls == [
        ("retrieval_fraction", 1.0),
        ("retrieval_fraction", 0.5),
    ]
