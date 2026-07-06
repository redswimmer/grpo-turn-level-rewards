# Phase 5 Full Training Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two remaining code changes Phase 5 needs (periodic eval/save + per-metric
trackio logging), a static exit-criteria script, and a restructured README — everything needed
before the real GPU training runs can be launched and verified.

**Architecture:** Two small, additive changes to already-existing modules (`train.py`'s
`build_config`, `rewards.py`'s three reward functions), one new static verification script
mirroring the existing `scripts/verify_phase{2,3,4}.py` pattern, and a README restructure. No new
modules, no new external dependencies.

**Tech Stack:** Python 3.13, `trl.GRPOConfig`, `pytest`, `ruff`, `ty`.

## Global Constraints

- Test location: `tests/unit/` only — no new test tiers (CLAUDE.md's "Repo layout" section).
- `build_config`'s new fields apply identically to both conditions: `num_iterations=2`,
  `eval_strategy="steps"`, `eval_steps=20`, `save_strategy="steps"`, `save_steps=50`,
  `save_total_limit=3`.
- `log_metric` is an optional keyword parameter on each reward function, defaulting to a no-op —
  every existing test that calls these functions without `log_metric` must keep passing unchanged.
- README structure: `## What this compares` → `## Results` → `## Roadmap` → `## Reproducing this`
  → `## Contributing`. No phase numbers or internal doc paths (`docs/phase-*.md`) anywhere in
  `README.md`.
- Canary command (used after this plan, not in it):
  `python -m turn_level_rewards.train --condition outcome_only --num-generations 21 --max-steps 3`
- Full-run launch flags (both conditions, used after this plan):
  `--max-steps 300 --num-generations 21 --eval-size 64 --train-size 90447`

---

### Task 1: `build_config` — periodic eval and checkpoint saves

**Files:**
- Modify: `src/turn_level_rewards/train.py` (the `build_config` function, `src/turn_level_rewards/train.py:23-52`)
- Modify: `tests/unit/test_train.py` (append after `test_build_config_per_device_train_batch_size_matches_num_generations`, `tests/unit/test_train.py:65-68`)

**Interfaces:**
- Consumes: nothing new — `build_config`'s existing signature
  `(condition: Condition, seed: int, max_steps: int, num_generations: int) -> GRPOConfig` is
  unchanged.
- Produces: the returned `GRPOConfig` now also has `num_iterations=2`, `eval_strategy="steps"`,
  `eval_steps=20`, `save_strategy="steps"`, `save_steps=50`, `save_total_limit=3` set. Task 3's
  verification script and the post-run evidence gate (after this plan) both rely on these exact
  values.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_train.py` (after `test_build_config_per_device_train_batch_size_matches_num_generations`):

```python
def test_build_config_periodic_eval_and_save_fields():
    outcome_config = _build("outcome_only")
    turn_config = _build("turn_level")

    for config in (outcome_config, turn_config):
        assert config.num_iterations == 2
        assert config.eval_strategy == "steps"
        assert config.eval_steps == 20
        assert config.save_strategy == "steps"
        assert config.save_steps == 50
        assert config.save_total_limit == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train.py::test_build_config_periodic_eval_and_save_fields -v`
Expected: FAIL (`AssertionError` — `num_iterations` defaults to `1`, `eval_strategy` defaults to
`"no"`, `save_strategy` defaults to `"steps"` with `save_steps=500`, `save_total_limit` defaults
to `None`).

- [ ] **Step 3: Update `build_config`**

In `src/turn_level_rewards/train.py`, replace the `build_config` function body's `return
GRPOConfig(...)` call:

```python
def build_config(
    condition: Condition,
    seed: int,
    max_steps: int,
    num_generations: int,
) -> GRPOConfig:
    """Build the GRPOConfig for a training run.

    per_device_train_batch_size is set equal to num_generations, not passed independently --
    GRPOConfig requires generation_batch_size (which defaults to per_device_train_batch_size *
    num_processes * steps_per_generation) to be evenly divisible by num_generations; setting
    them equal satisfies this trivially on a single GPU.

    num_iterations, eval_strategy/eval_steps, and save_strategy/save_steps/save_total_limit are
    fixed per the Phase 5 design spec's paper-grounded config; see
    docs/superpowers/specs/2026-07-05-phase-5-full-training-runs-design.md.
    """
    return GRPOConfig(
        output_dir=f"outputs/{condition}",
        seed=seed,
        max_steps=max_steps,
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        max_tool_calling_iterations=4,
        beta=0.0,
        max_completion_length=2048,
        logging_steps=1,
        logging_nan_inf_filter=False,
        log_completions=True,
        gradient_checkpointing=True,
        num_iterations=2,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        report_to="trackio",
        project="turn-level-rewards",
        run_name=condition,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_train.py -v`
Expected: PASS, all tests in the file (including the new one and every pre-existing one — the
`test_build_config_only_condition_derived_fields_differ` test's `{"output_dir", "run_name"}`
assertion is unaffected since the new fields are identical across both conditions).

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/train.py tests/unit/test_train.py
git commit -m "Add periodic eval/save fields to build_config for Phase 5's full runs"
```

---

### Task 2: `log_metric` calls in `rewards.py`

**Files:**
- Modify: `src/turn_level_rewards/rewards.py` (full file — see below)
- Modify: `tests/unit/test_rewards.py` (add a `_FakeLogMetric` helper near `FakeEnvironment`, and
  three new tests after `test_get_reward_funcs_rejects_unknown_condition`)

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `format_reward`, `outcome_reward`, and `turn_reward` each gain an optional
  `log_metric: Callable[[str, float], None] = _noop_log_metric` keyword parameter. Calling any of
  them without `log_metric` (as every pre-existing test in `tests/unit/test_rewards.py` does) must
  behave exactly as before. When a real `log_metric` is passed:
  - `format_reward` calls `log_metric("format_compliance_rate", 1.0)` or `(..., 0.0)` once per
    completion.
  - `outcome_reward` calls `log_metric("exact_match", <0.0 or 1.0>)` and `log_metric("f1", <float>)`
    once each per completion, using the `golden_answers` entry that produced the max reward (not
    independently maxed).
  - `turn_reward` calls `log_metric("retrieval_fraction", <float>)` once per completion, using the
    raw (unscaled by 0.4) value.

- [ ] **Step 1: Write the failing tests**

Add near the top of `tests/unit/test_rewards.py`, right after the `FakeEnvironment` class:

```python
class _FakeLogMetric:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def __call__(self, name: str, value: float) -> None:
        self.calls.append((name, value))
```

Append at the end of `tests/unit/test_rewards.py`:

```python
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

    outcome_reward(
        completions=completions, golden_answers=golden_answers, log_metric=log_metric
    )

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
```

Note: `format_reward`, `outcome_reward`, `turn_reward` need to be imported directly for these
tests (the existing tests only import `get_reward_funcs` and destructure its return value). Add
them to the existing import line:

```python
from turn_level_rewards.rewards import (
    format_reward,
    get_reward_funcs,
    outcome_reward,
    turn_reward,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_rewards.py -v`
Expected: FAIL with `TypeError: format_reward() got an unexpected keyword argument 'log_metric'`
(and equivalent for the other two — `**kwargs` currently swallows it silently instead of the
function acting on it, so the assertions on `log_metric.calls` fail).

- [ ] **Step 3: Update `rewards.py`**

Replace the full contents of `src/turn_level_rewards/rewards.py`:

```python
"""Reward functions for GRPO training (see CLAUDE.md's "Reward design" section).

turn_reward implements turn-level credit assignment via reward density -- GRPO scores one
scalar per completed trajectory, so there is no per-timestep value function here, and this is
not a literal per-step RL change.
"""

import re
from typing import Any, Callable, Literal

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_rewards.py -v`
Expected: PASS, all tests in the file (every pre-existing test calls these functions without
`log_metric`, exercising the `_noop_log_metric` default).

- [ ] **Step 5: Commit**

```bash
git add src/turn_level_rewards/rewards.py tests/unit/test_rewards.py
git commit -m "Log exact_match/f1/format_compliance_rate/retrieval_fraction to trackio per-step"
```

---

### Task 3: Phase 5 exit-criteria script (static portion)

**Files:**
- Create: `scripts/verify_phase5.py`

**Interfaces:**
- Consumes: `turn_level_rewards.train.build_config` (Task 1's new fields).
- Produces: a script exiting `0` with a `PASS` message, or `1` with a `FAIL` message listing every
  failing check — same contract as `scripts/verify_phase4.py`.

- [ ] **Step 1: Write the script**

Create `scripts/verify_phase5.py`:

```python
#!/usr/bin/env python3
"""Phase 5 exit-criteria check (static/code portion only).

Mirrors scripts/verify_phase4.py's pattern: prints exactly which check failed, or PASS and exits
0, only if every check below passes. Run this after the build_config/rewards.py changes and
before spending any GPU time on the full training runs.

This only covers the static/testable subset of Phase 5's exit criteria -- the canary dry-run, the
full 300-step runs, and the post-run evidence gate (checkpoint loadability, trackio alerts/curves)
are NOT scripted here; see
docs/superpowers/specs/2026-07-05-phase-5-full-training-runs-design.md's "Verification plan"
section for that layer.

Usage: uv run python scripts/verify_phase5.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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

    from turn_level_rewards.train import build_config

    outcome_config = build_config(
        condition="outcome_only", seed=42, max_steps=2, num_generations=2
    )
    turn_config = build_config(condition="turn_level", seed=42, max_steps=2, num_generations=2)

    fixed_checks = {
        "num_iterations": 2,
        "eval_strategy": "steps",
        "eval_steps": 20,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 3,
    }
    for config, label in [(outcome_config, "outcome_only"), (turn_config, "turn_level")]:
        for field, expected in fixed_checks.items():
            actual = getattr(config, field)
            if actual != expected:
                failures.append(
                    f"build_config({label!r}).{field} == {actual!r}, expected {expected!r}"
                )

    return failures


if __name__ == "__main__":
    failures = check()
    if failures:
        print("FAIL -- Phase 5's static checks are not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(
        "PASS: unit tests, ruff, ty are clean, and build_config's new num_iterations/eval/save "
        "fields match the design spec."
    )
    print(
        "This does NOT cover the canary dry-run, the full training runs, or the post-run "
        "evidence gate -- run those manually per the design spec before sign-off."
    )
    sys.exit(0)
```

- [ ] **Step 2: Run the script to verify it passes**

Run: `uv run python scripts/verify_phase5.py`
Expected: exits 0, prints the `PASS` message above.

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_phase5.py
git commit -m "Add Phase 5 exit-criteria validation script (static portion)"
```

---

### Task 4: README restructure — Results / Roadmap / Reproducing this

**Files:**
- Modify: `README.md` (full file)

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces: a `README.md` with sections in this order: `# ...` title, `## What this compares`,
  `## Results` (placeholder until real runs finish), `## Roadmap` (conceptual, no phase numbers),
  `## Reproducing this` (the current Prerequisites/Retrieval server/Training content, renamed and
  moved), `## Contributing`.

- [ ] **Step 1: Replace `README.md`**

Replace the full contents of `README.md`:

```markdown
# Outcome vs. Turn-Level Reward for Multi-Turn Search Agents

A small-scale experiment testing whether rewarding an AI agent's intermediate actions — not just
its final answer — helps it learn faster and more reliably.

## What this compares

This repo trains a multi-turn Wikipedia-search agent under two reward regimes:

- **Outcome reward** — the agent is scored only on its final answer's correctness. Sparse: no
  signal until the very end of the episode.
- **Turn-level reward** — the same outcome scoring, plus a bonus for surfacing a real
  supporting-fact passage during search. Denser: the agent gets credit for good intermediate
  behavior, not just a good final answer.

The interesting question isn't just "does the denser signal help" — it's **whether that holds up
across genuinely different reinforcement learning algorithms**, not just one. So this repo tests
the same outcome-vs-turn-level comparison twice:

- **GRPO** — scores a group of the agent's attempts at the same question against each other,
  using the ones that did relatively better within the group as the learning signal.
- **PPO** — learns a running estimate of how good a position is (a value function), and nudges
  the policy toward actions that beat that estimate, turn by turn.

If turn-level reward helps in the same way under both, that's a real finding about reward shaping
in multi-turn agent RL, not an artifact of one algorithm's mechanics.

Concretely, this is a simplified reproduction of two ablations from ["Reinforcing Multi-Turn
Reasoning in LLM Agents via Turn-Level Reward Design"](https://arxiv.org/abs/2505.11821)
(arXiv:2505.11821): its Appendix E GRPO case study (`GRPO-OR`/`GRPO-MR`), and its main-results PPO
comparison (`PPO`/`MT-PPO`).

## Results

_No runs have completed yet — this section fills in as each one finishes, so a reader can learn
what was found without running anything themselves._

## Roadmap

- **GRPO: outcome-only vs. merged-reward** — training infrastructure built; full runs not yet run.
- **PPO: outcome-only vs. merged-reward** — design complete; not yet started.
- **LLM-as-judge reward** (an alternative to exact-match/F1 scoring, explored on top of the PPO
  comparison) — not yet started.

## Reproducing this

### Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- JDK 21 (needed by the retrieval server's Lucene bridge)

```bash
uv sync
sudo apt install openjdk-21-jdk
```

### Retrieval server

Training and evaluation search a local BM25 server backed by the real wiki-18
Wikipedia dump (~21M passages). Set it up once:

```bash
bash scripts/setup_retrieval.sh   # downloads the wiki-18 BM25 index (+corpus if needed) into data/wiki18/
```

The script downloads the index, checks whether it also needs the separate
corpus file, and prints the exact command to launch the server — something
like:

```bash
uv run python scripts/retrieval_server.py \
    --index_path data/wiki18/bm25-repo/bm25 \
    --corpus_path data/wiki18/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
    --port 8000
```

Run that (in the background or a separate terminal — it needs to stay up for
the rest of setup and for training/evaluation later), then confirm it's
working:

```bash
uv run python scripts/verify_retrieval.py
```

```
PASS: retrieval server is up, wired correctly, and returns real documents.
```

### Training

```bash
uv run python -m turn_level_rewards.train --condition outcome_only
uv run python -m turn_level_rewards.train --condition turn_level
```

The bare invocation above (no extra flags) runs at smoke-test scale — 8 rows, 2 steps, a real
`Qwen/Qwen3.5-0.8B` model against the retrieval server started above. Pass `--train-size`,
`--max-steps`, `--num-generations`, etc. explicitly for a full-scale run. Both conditions
log to the same [trackio](https://github.com/gradio-app/trackio) project
(`turn-level-rewards`) — run `trackio show --project turn-level-rewards` to view.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, quality gates, and running tests.
```

- [ ] **Step 2: Sanity-check the rendered structure**

Run: `grep -n "^## " README.md`
Expected:
```
6:## What this compares
32:## Results
37:## Roadmap
44:## Reproducing this
88:## Contributing
```
(exact line numbers may differ slightly; the important thing is the section order and that no
`docs/phase-` or phase-numbered text appears — confirm with `grep -n "phase-" README.md`, expected
no output).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Restructure README: separate Results/Roadmap from Reproducing this"
```

---

## After this plan: not part of any task above

Once all 4 tasks are reviewed and merged, the real GPU work happens manually in the foreground
(not delegated to a subagent), per the design spec's "Sequence" and "Verification plan" sections:

1. **Canary dry-run**: `python -m turn_level_rewards.train --condition outcome_only
   --num-generations 21 --max-steps 3`, watching `nvidia-smi` and wall-clock directly. If OOM,
   attempt installing `flash-linear-attention`/`causal-conv1d` before falling back to anything
   else (per the design spec's decision gate).
2. **Launch `outcome_only`** at full scale (`--max-steps 300 --num-generations 21 --eval-size 64
   --train-size 90447`) in the background; verify healthy for ~20-30 steps before considering it
   launched.
3. **Launch `turn_level`** the same way, once `outcome_only` is confirmed healthy.
4. **Post-run evidence gate** for each condition once its 300 steps finish: checkpoint actually
   loads via `AutoModelForCausalLM.from_pretrained`, `trackio list alerts ... --json` is empty (or
   every entry explained), the run's step metric reached 300, and each of the 4 new per-metric
   curves plus the periodic eval-reward curve has a real, non-trivial number of data points.
5. **Populate README's `## Results` section** with real findings once both conditions pass their
   evidence gate — concise, concept-level, comparing to the paper's own reported direction.
6. **Update** `docs/phase-5-full-training-runs.md`'s Handoff notes and CLAUDE.md's Roadmap table
   Status column to mark Phase 5 done — this plan's 4 tasks alone don't satisfy Phase 5's real
   exit criteria, which include the full runs and the evidence gate.
