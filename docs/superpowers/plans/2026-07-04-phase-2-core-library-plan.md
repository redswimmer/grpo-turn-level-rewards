# Phase 2 Core Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `metrics.py`, `env.py`, `rewards.py` for the turn-level-rewards GRPO project, with a fast `tests/unit/` suite (no GPU, no network, no live retrieval server) and a `verify_phase2.py` gate that must print `PASS` before Phase 2 is considered done.

**Architecture:** Three independent, dependency-free-of-each-other-except-one-direction modules under `src/turn_level_rewards/`: `metrics.py` (stdlib-only SQuAD scoring, no dependencies), `env.py` (`SearchEnv`, depends on `requests` only inside its default retrieval seam), `rewards.py` (depends on `metrics.py` only — never imports `env.py`, since rewards operate on plain completions/duck-typed environments per CLAUDE.md's dependency-inversion principle). Every external boundary (the retrieval HTTP call) is injectable; tests never hit the network.

**Tech Stack:** Python 3.13, stdlib `re`/`string`/`collections.Counter` for metrics, `requests` for the retrieval seam, `pytest` for tests, `ruff`/`ty` for lint/type-check, `hatchling` as the build backend.

## Global Constraints

- Python 3.13 (`pyproject.toml`'s `requires-python`); `ty` is configured for `python-version = "3.13"`.
- `tests/unit/` is the only test tier — no live network, GPU, or retrieval server in any test.
- `ruff check` and `ty check` must be clean (both already configured as dev deps in `pyproject.toml`).
- Reward magnitudes are exact, from CLAUDE.md: `format_reward` = `+0.1` well-formed / `-0.1` malformed; `outcome_reward` = SQuAD F1 + `0.5` exact-match bonus, maxed over `golden_answers`; `turn_reward` = `0.4 * environment.retrieval_fraction`.
- The retrieval HTTP call must be genuinely injectable: `SearchEnv.search()` must never call `requests`/`httpx` directly — only `self._retrieve_fn`, so tests can fake it.
- `SearchEnv` instances are reused from a pool across episodes (TRL mechanic) — `reset()` must leave zero state from a prior episode.
- Fixture titles for `test_env.py` must be from the four titles Phase 1's handoff verified live against the real corpus: `"127 Hours"`, `"Big Stone Gap (film)"`, `"Peter Schmeichel"`, `"Virginia Commonwealth University"` — not the CLAUDE.md examples that turned out not to exist.
- `SearchEnv.reset()`'s contract is `reset(self, metadata, **kwargs)`, pulling `metadata["supporting_facts"]["title"]` — confirmed against a real `PeterJinGo/nq_hotpotqa_train` row and against Phase 3's own doc (see `docs/superpowers/specs/2026-07-04-phase-2-core-library-design.md`).
- Completion message shapes must match real TRL 1.7.1 internals: tool-call messages are `{"role": "assistant", "content": ..., "tool_calls": [{"type": "function", "function": {"name": ..., "arguments": {...}}}]}`; tool responses are `{"role": "tool", "name": ..., "content": ...}`; hitting the tool-call cap leaves the last message as an unresolved assistant `tool_calls` message with no trailing answer.

---

### Task 1: Package plumbing

**Files:**
- Modify: `pyproject.toml`
- Create: `src/turn_level_rewards/__init__.py`

**Interfaces:**
- Produces: an installed, importable `turn_level_rewards` package that all later tasks add modules to.

- [ ] **Step 1: Add the build-system table to `pyproject.toml`**

Add this table at the very top of `pyproject.toml`, before `[project]`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create the package directory**

```bash
mkdir -p src/turn_level_rewards
```

- [ ] **Step 3: Create `src/turn_level_rewards/__init__.py`**

```python
```

(empty file — just needs to exist so hatchling detects the `src/turn_level_rewards/` layout)

- [ ] **Step 4: Add `pytest` to dev dependencies**

In `pyproject.toml`, change:

```toml
[dependency-groups]
dev = [
    "pre-commit>=4.6.0",
    "ruff>=0.15.20",
    "ty>=0.0.56",
]
```

to:

```toml
[dependency-groups]
dev = [
    "pre-commit>=4.6.0",
    "pytest>=8.4.0",
    "ruff>=0.15.20",
    "ty>=0.0.56",
]
```

- [ ] **Step 5: Sync and verify the package installs**

Run: `uv sync`
Expected: completes without error.

Run: `uv run python -c "import turn_level_rewards"`
Expected: no output, exit code 0.

Run: `grep -A2 'name = "grpo-turn-level-rewards"' uv.lock`
Expected: the `source` line for this package is no longer `{ virtual = "." }` (it should now show a `source = { editable = "." }` or similar non-virtual source).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/turn_level_rewards/__init__.py
git commit -m "Make turn_level_rewards an installable package"
```

---

### Task 2: `metrics.py` — SQuAD-style EM/F1

**Files:**
- Create: `src/turn_level_rewards/metrics.py`
- Create: `tests/unit/__init__.py` (empty, so pytest can discover the package cleanly)
- Test: `tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces: `normalize_answer(text: str) -> str`, `exact_match(prediction: str, ground_truth: str) -> bool`, `f1_score(prediction: str, ground_truth: str) -> float` — all consumed by `rewards.py` in Task 4.

- [ ] **Step 1: Create the tests directory and write the failing test**

```bash
mkdir -p tests/unit
touch tests/unit/__init__.py
```

Create `tests/unit/test_metrics.py`:

```python
import pytest

from turn_level_rewards.metrics import exact_match, f1_score, normalize_answer


def test_normalize_answer_strips_case_punctuation_articles_and_whitespace():
    assert normalize_answer("The Beatles!") == "beatles"
    assert normalize_answer("  a   dog  ") == "dog"


def test_exact_match_identical_strings():
    assert exact_match("Paris", "Paris") is True


def test_exact_match_normalizes_before_comparing():
    assert exact_match("The Beatles", "beatles") is True


def test_exact_match_fully_disjoint_answers():
    assert exact_match("Paris", "London") is False


def test_f1_score_identical_strings_is_one():
    assert f1_score("Paris, France", "Paris, France") == 1.0


def test_f1_score_partial_overlap():
    assert f1_score("New", "New York") == pytest.approx(2 / 3)


def test_f1_score_fully_disjoint_is_zero():
    assert f1_score("Paris", "London") == 0.0


def test_f1_score_empty_prediction_against_nonempty_ground_truth_is_zero():
    assert f1_score("", "Paris") == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_level_rewards.metrics'`.

- [ ] **Step 3: Implement `src/turn_level_rewards/metrics.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix src/turn_level_rewards/metrics.py tests/unit/test_metrics.py`
(auto-fixes anything mechanical, e.g. import ordering)

Run: `uv run ruff check src/turn_level_rewards/metrics.py tests/unit/test_metrics.py`
Expected: `All checks passed!`

Run: `uv run ty check src/turn_level_rewards/metrics.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/turn_level_rewards/metrics.py tests/unit/__init__.py tests/unit/test_metrics.py
git commit -m "Add SQuAD-style EM/F1 metrics"
```

---

### Task 3: `env.py` — `SearchEnv`

**Files:**
- Create: `src/turn_level_rewards/env.py`
- Test: `tests/unit/test_env.py`

**Interfaces:**
- Consumes: nothing from other Phase 2 modules (no import of `metrics.py` or `rewards.py`).
- Produces: `SearchEnv` class with `__init__(self, retrieve_fn=None, topk=3, base_url="http://localhost:8000")`, `reset(self, metadata, **kwargs) -> None`, `search(self, query: str) -> str`, and property `retrieval_fraction -> float`. `rewards.py` (Task 4) consumes only the `.retrieval_fraction` attribute via duck typing, not the class itself.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_env.py`:

```python
from turn_level_rewards.env import SearchEnv

HOTPOT_ROW_1 = {
    "metadata": {
        "supporting_facts": {"title": ["127 Hours", "Peter Schmeichel"], "sent_id": [0, 0]},
    },
}

HOTPOT_ROW_2 = {
    "metadata": {
        "supporting_facts": {
            "title": ["Big Stone Gap (film)", "Virginia Commonwealth University"],
            "sent_id": [0, 0],
        },
    },
}


def _fake_retriever(docs_by_query):
    def retrieve(query, topk):
        return docs_by_query.get(query, [])

    return retrieve


def test_search_hitting_gold_title_updates_retrieval_fraction():
    retrieve = _fake_retriever(
        {
            "127 hours survival": [
                {
                    "title": "127 Hours",
                    "text": "A 2010 survival drama film.",
                    "contents": '"127 Hours"\nA 2010 survival drama film.',
                }
            ]
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    env.search("127 hours survival")

    assert env.retrieval_fraction == 0.5


def test_search_hitting_distractor_does_not_update_retrieval_fraction():
    retrieve = _fake_retriever(
        {
            "danish goalkeepers": [
                {
                    "title": "Football in Denmark",
                    "text": "Association football is the most popular sport in Denmark.",
                    "contents": '"Football in Denmark"\nAssociation football is the most popular sport in Denmark.',
                }
            ]
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    env.search("danish goalkeepers")

    assert env.retrieval_fraction == 0.0


def test_reset_twice_in_a_row_clears_prior_episode_state():
    retrieve = _fake_retriever(
        {
            "127 hours survival": [
                {
                    "title": "127 Hours",
                    "text": "A 2010 survival drama film.",
                    "contents": '"127 Hours"\nA 2010 survival drama film.',
                }
            ]
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)
    env.search("127 hours survival")
    assert env.retrieval_fraction == 0.5

    env.reset(**HOTPOT_ROW_2)

    assert env.retrieval_fraction == 0.0


def test_retrieval_fraction_caps_at_one_on_duplicate_hit():
    retrieve = _fake_retriever(
        {
            "127 hours": [
                {"title": "127 Hours", "text": "...", "contents": '"127 Hours"\n...'}
            ],
            "peter schmeichel": [
                {
                    "title": "Peter Schmeichel",
                    "text": "...",
                    "contents": '"Peter Schmeichel"\n...',
                }
            ],
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    env.search("127 hours")
    env.search("127 hours")  # duplicate hit, must not double-count
    env.search("peter schmeichel")

    assert env.retrieval_fraction == 1.0


def test_search_returns_readable_string_with_title_and_text():
    retrieve = _fake_retriever(
        {
            "127 hours": [
                {
                    "title": "127 Hours",
                    "text": "A 2010 survival drama film.",
                    "contents": '"127 Hours"\nA 2010 survival drama film.',
                }
            ]
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    result = env.search("127 hours")

    assert "127 Hours" in result
    assert "A 2010 survival drama film." in result


def test_search_with_no_results_returns_message_not_error():
    retrieve = _fake_retriever({})
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    result = env.search("nonexistent query")

    assert result == "No results found."
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_level_rewards.env'`.

- [ ] **Step 3: Implement `src/turn_level_rewards/env.py`**

```python
"""SearchEnv: a TRL environment_factory-compatible multi-turn search environment.

Instances are reused from a pool across episodes (see CLAUDE.md's "TRL mechanics being relied
on") -- reset() must fully reinitialize all mutable state, with zero leftover state from a
prior episode.
"""

from collections.abc import Callable

import requests

DocList = list[dict[str, str]]


def _default_retrieve_fn(query: str, topk: int, base_url: str) -> DocList:
    """POST a single query to the real retrieval server (scripts/retrieval_server.py).

    The server already splits each document's title/text out of its `contents` field
    server-side (see its `parse_title_text`) -- this trusts that and does no re-parsing.
    """
    response = requests.post(
        f"{base_url}/retrieve",
        json={"queries": [query], "topk": topk, "return_scores": False},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["result"][0]


class SearchEnv:
    """Multi-turn search environment exposing `search` as a tool for GRPO's environment_factory.

    Tracks the fraction of gold supporting-fact titles surfaced by search() calls during an
    episode -- this is the turn-level reward signal (CLAUDE.md's "Reward design" section).
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, int], DocList] | None = None,
        topk: int = 3,
        base_url: str = "http://localhost:8000",
    ) -> None:
        self._retrieve_fn = retrieve_fn or (
            lambda query, k: _default_retrieve_fn(query, k, base_url)
        )
        self._topk = topk
        self._gold_titles: frozenset[str] = frozenset()
        self._hit_titles: set[str] = set()

    def reset(self, metadata: dict, **kwargs: object) -> None:
        """Reset all per-episode state for a newly sampled row.

        Args:
            metadata (dict): The row's metadata; metadata["supporting_facts"]["title"] is the
                list of gold paragraph titles for this question.
        """
        self._gold_titles = frozenset(metadata["supporting_facts"]["title"])
        self._hit_titles = set()

    def search(self, query: str) -> str:
        """Search the wiki-18 corpus for passages relevant to a query.

        Args:
            query (str): The search query text.

        Returns:
            str: The top retrieved passages formatted as numbered documents with title and
                text, or a message indicating no results were found.
        """
        docs = self._retrieve_fn(query, self._topk)
        for doc in docs:
            if doc["title"] in self._gold_titles:
                self._hit_titles.add(doc["title"])
        if not docs:
            return "No results found."
        return "\n".join(
            f'Doc {i} (Title: "{doc["title"]}"): {doc["text"]}' for i, doc in enumerate(docs, 1)
        )

    @property
    def retrieval_fraction(self) -> float:
        """Fraction of gold supporting-fact titles surfaced by search() this episode."""
        if not self._gold_titles:
            return 0.0
        return len(self._hit_titles) / len(self._gold_titles)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_env.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix src/turn_level_rewards/env.py tests/unit/test_env.py`
(auto-fixes anything mechanical, e.g. import ordering)

Run: `uv run ruff check src/turn_level_rewards/env.py tests/unit/test_env.py`
Expected: `All checks passed!`

Run: `uv run ty check src/turn_level_rewards/env.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/turn_level_rewards/env.py tests/unit/test_env.py
git commit -m "Add SearchEnv with injectable retrieval seam"
```

---

### Task 4: `rewards.py` — reward functions

**Files:**
- Create: `src/turn_level_rewards/rewards.py`
- Test: `tests/unit/test_rewards.py`

**Interfaces:**
- Consumes: `normalize_answer` is not needed directly; consumes `exact_match(prediction: str, ground_truth: str) -> bool` and `f1_score(prediction: str, ground_truth: str) -> float` from Task 2's `turn_level_rewards.metrics`. Consumes only `environment.retrieval_fraction` (a `float` attribute) by duck typing — never imports `turn_level_rewards.env`.
- Produces: `format_reward(completions, **kwargs) -> list[float]`, `outcome_reward(completions, golden_answers, **kwargs) -> list[float]`, `turn_reward(environments, **kwargs) -> list[float]`, `get_reward_funcs(condition: Literal["outcome_only", "turn_level"]) -> list[Callable]` — consumed later by `train.py` (Phase 4).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_rewards.py`:

```python
import pytest

from turn_level_rewards.rewards import get_reward_funcs


class FakeEnvironment:
    def __init__(self, retrieval_fraction: float) -> None:
        self.retrieval_fraction = retrieval_fraction


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
    assert outcome_reward(
        completions=completions, golden_answers=golden_answers
    ) == pytest.approx([1.5])
    assert turn_reward(completions=completions, environments=environments) == pytest.approx(
        [0.4]
    )


def test_well_formed_correct_answer_zero_retrieval():
    completions = [[_answer("127 Hours")]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([0.1])
    assert outcome_reward(
        completions=completions, golden_answers=golden_answers
    ) == pytest.approx([1.5])
    assert turn_reward(completions=completions, environments=environments) == pytest.approx(
        [0.0]
    )


def test_well_formed_wrong_answer():
    completions = [[_answer("Peter Schmeichel")]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=1.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([0.1])
    assert outcome_reward(
        completions=completions, golden_answers=golden_answers
    ) == pytest.approx([0.0])
    assert turn_reward(completions=completions, environments=environments) == pytest.approx(
        [0.4]
    )


def test_malformed_missing_answer_tag():
    completions = [[{"role": "assistant", "content": "I believe it is 127 Hours."}]]
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.5)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([-0.1])
    assert outcome_reward(
        completions=completions, golden_answers=golden_answers
    ) == pytest.approx([0.0])
    assert turn_reward(completions=completions, environments=environments) == pytest.approx(
        [0.2]
    )


def test_hard_tool_call_cap_mid_call_unresolved_tool_calls_no_answer():
    completions = [[_search_tool_call("127 hours")]]  # cap hit: no trailing tool/answer message
    golden_answers = [["127 Hours"]]
    environments = [FakeEnvironment(retrieval_fraction=0.0)]

    format_reward, outcome_reward, turn_reward = get_reward_funcs("turn_level")

    assert format_reward(completions=completions) == pytest.approx([-0.1])
    assert outcome_reward(
        completions=completions, golden_answers=golden_answers
    ) == pytest.approx([0.0])
    assert turn_reward(completions=completions, environments=environments) == pytest.approx(
        [0.0]
    )


def test_get_reward_funcs_outcome_only_excludes_turn_reward():
    funcs = get_reward_funcs("outcome_only")
    assert [f.__name__ for f in funcs] == ["format_reward", "outcome_reward"]


def test_get_reward_funcs_turn_level_includes_turn_reward():
    funcs = get_reward_funcs("turn_level")
    assert [f.__name__ for f in funcs] == ["format_reward", "outcome_reward", "turn_reward"]


def test_get_reward_funcs_rejects_unknown_condition():
    with pytest.raises(ValueError):
        get_reward_funcs("bogus")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_rewards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turn_level_rewards.rewards'`.

- [ ] **Step 3: Implement `src/turn_level_rewards/rewards.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_rewards.py -v`
Expected: 8 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix src/turn_level_rewards/rewards.py tests/unit/test_rewards.py`
(auto-fixes anything mechanical, e.g. import ordering)

Run: `uv run ruff check src/turn_level_rewards/rewards.py tests/unit/test_rewards.py`
Expected: `All checks passed!`

Run: `uv run ty check src/turn_level_rewards/rewards.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/turn_level_rewards/rewards.py tests/unit/test_rewards.py
git commit -m "Add format/outcome/turn reward functions"
```

---

### Task 5: Validation script and Phase 2 sign-off

**Files:**
- Create: `scripts/verify_phase2.py`
- Modify: `docs/phase-2-core-library.md` (Handoff notes + task checkboxes)
- Modify: `CLAUDE.md` (roadmap table Status column)

**Interfaces:**
- Consumes: nothing from `src/turn_level_rewards/` at import time — it shells out to `uv run pytest`/`ruff`/`ty` as subprocesses and greps the `env.py` source text, so it stays decoupled from the library's internals.
- Produces: a `PASS`/`FAIL` exit-code gate that Phase 3 (and this task's own last step) uses before Phase 2 is declared done.

- [ ] **Step 1: Create `scripts/verify_phase2.py`**

```python
#!/usr/bin/env python3
"""Phase 2 exit-criteria check.

Mirrors scripts/verify_retrieval.py's pattern: prints exactly which check failed, or PASS and
exits 0, only if every check below passes. Run this after any change to metrics.py/env.py/
rewards.py, and again before marking Phase 2 done in docs/phase-2-core-library.md.

Usage: uv run python scripts/verify_phase2.py
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PY = REPO_ROOT / "src" / "turn_level_rewards" / "env.py"


def _run(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def check() -> list[str]:
    failures = []

    code, output = _run("uv", "run", "pytest", "tests/unit/", "-q")
    if code != 0:
        failures.append(f"pytest tests/unit/ failed:\n{output}")

    code, output = _run("uv", "run", "ruff", "check")
    if code != 0:
        failures.append(f"ruff check failed:\n{output}")

    code, output = _run("uv", "run", "ty", "check")
    if code != 0:
        failures.append(f"ty check failed:\n{output}")

    if not ENV_PY.exists():
        failures.append(f"{ENV_PY} does not exist yet.")
    else:
        env_source = ENV_PY.read_text()
        occurrences = len(re.findall(r"requests\.post\(|httpx\.", env_source))
        if occurrences == 0:
            failures.append(f"No requests.post/httpx call found in {ENV_PY} -- retrieval isn't wired.")
        elif occurrences > 1:
            failures.append(
                f"Found {occurrences} requests.post/httpx call sites in {ENV_PY} -- expected "
                "exactly one, inside the default retrieve_fn factory. SearchEnv.search() must "
                "only call self._retrieve_fn, never requests/httpx directly."
            )

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 2 is not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: unit tests, ruff, and ty are clean, and the retrieval seam is genuinely injectable.")
    print("Phase 2 exit criteria met -- safe to start Phase 3.")
    sys.exit(0)
```

- [ ] **Step 2: Run it and iterate until PASS**

Run: `uv run python scripts/verify_phase2.py`
Expected: `PASS: unit tests, ruff, and ty are clean, and the retrieval seam is genuinely injectable.`

If it prints `FAIL` instead, read exactly which check failed, fix the corresponding file from Tasks 2-4, and re-run this same command. Do not proceed to Step 3 until it prints `PASS`.

- [ ] **Step 3: Commit the validation script**

```bash
git add scripts/verify_phase2.py
git commit -m "Add Phase 2 exit-criteria validation script"
```

- [ ] **Step 4: Update `docs/phase-2-core-library.md`'s task checkboxes and Handoff notes**

Check off every completed task's `- [ ]` to `- [x]` in the Tasks and Exit criteria sections. Replace the Handoff notes section (currently `(not yet started)`) with:

```markdown
## Handoff notes

- **`SearchEnv.reset()`'s exact contract**: `reset(self, metadata, **kwargs)`, pulling
  `metadata["supporting_facts"]["title"]`. Confirmed directly against a real streamed row from
  `PeterJinGo/nq_hotpotqa_train` (nested `metadata` dict, not the phase doc's originally-drafted
  flat `context`/`supporting_facts` kwargs) and independently corroborated by Phase 3's own doc,
  which already commits to nesting `supporting_facts`/`context` under `metadata` for the eval
  set too. Phase 3's `data.py` needs no special-casing here -- just pass `metadata` through as-is.
- **The retrieval server already parses `title`/`text` server-side** (`parse_title_text()` in
  `scripts/retrieval_server.py`, confirmed by reading it directly) -- `SearchEnv.search()`
  trusts `document["title"]`/`document["text"]` directly and does no re-parsing of `contents`.
- **Answer format**: final answers are wrapped in `<answer>...</answer>` in the last assistant
  message (no unresolved `tool_calls`), checked by `rewards._extract_answer`. This convention is
  reused, unmodified, from this dataset's own baked-in `prompt` column.
- **Known gap for Phase 3/4, not fixed here**: `PeterJinGo/nq_hotpotqa_train`'s `prompt` column
  (confirmed by pulling a real row) is Search-R1's original text-tag ReAct prompt
  (`<search>...</search>` -> `<information>...</information>` -> `<answer>...</answer>`), which
  assumes a regex-based rollout loop -- not TRL's native `environment_factory` tool-calling
  (structured `tool_calls`, not text tags). Phase 3 or 4 will need to replace this `prompt`
  column with one that teaches native tool use, keeping only the `<answer>` convention for the
  final response. `rewards.py`/`env.py` do not depend on the exact prompt wording, so this did
  not block Phase 2.
- **Fixture titles**: `test_env.py` uses `"127 Hours"`, `"Big Stone Gap (film)"`,
  `"Peter Schmeichel"`, `"Virginia Commonwealth University"` -- the four titles Phase 1's
  handoff notes verified live against the real corpus, not the CLAUDE.md examples that turned
  out not to exist.
- **`scripts/verify_phase2.py`** is the exit-criteria gate (mirrors `verify_retrieval.py`'s
  pattern) -- re-run it after any future change to `metrics.py`/`env.py`/`rewards.py`.
```

- [ ] **Step 5: Update the roadmap table in `CLAUDE.md`**

Change the Phase 2 row in the roadmap table from:

```markdown
| 2 | Core library: `env.py`, `rewards.py`, `metrics.py` + `tests/unit/` | `docs/phase-2-core-library.md` | Not started |
```

to:

```markdown
| 2 | Core library: `env.py`, `rewards.py`, `metrics.py` + `tests/unit/` | `docs/phase-2-core-library.md` | **Done** — `scripts/verify_phase2.py` passes; see phase doc's Handoff notes for the confirmed `reset()` contract and a flagged Phase 3/4 gap (dataset's `prompt` column needs replacing) |
```

- [ ] **Step 6: Commit the documentation updates**

```bash
git add docs/phase-2-core-library.md CLAUDE.md
git commit -m "Mark Phase 2 done; record handoff notes for Phase 3"
```

---

## Self-Review Notes

- **Spec coverage**: package plumbing (Task 1), `metrics.py` + tests (Task 2), `env.py` + tests
  including the pooled-instance-reuse/reset and capped-at-1.0 cases (Task 3), `rewards.py` + tests
  covering all five required scenarios plus both `get_reward_funcs` conditions (Task 4), and the
  validation-loop gate + Handoff notes/roadmap update (Task 5) — every spec item has a task.
- **Placeholder scan**: no TBD/TODO; every step has literal, complete code.
- **Type consistency**: `SearchEnv.retrieve_fn` signature (`Callable[[str, int], DocList]`) matches
  every fake retriever defined in `test_env.py` (`retrieve(query, topk) -> list[dict]`);
  `turn_reward`'s `environments: list[Any]` matches `FakeEnvironment`'s duck-typed
  `.retrieval_fraction` attribute in `test_rewards.py`; `get_reward_funcs` return order
  (`format_reward, outcome_reward[, turn_reward]`) matches the unpacking order used in every
  `test_rewards.py` test.
