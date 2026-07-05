# Phase 4 Training Script + Live Smoke Test — Design

Status: approved by user, 2026-07-05. Implements `docs/phase-4-training-smoke-test.md`.

## Context and evidence gathered

This design was grounded by reading TRL's actual installed source (`trl==1.7.1`,
`transformers==5.13.0`) and the paper's HTML text directly, and by pulling real data through the
live retrieval server — not by trusting CLAUDE.md's prose to still be exactly right:

- **`GRPOTrainer.__init__` signature** (inspected directly): `environment_factory`,
  `reward_funcs`, `tools`, `rollout_func` all exist as documented. `environments` is populated on
  `self` at `grpo_trainer.py:545` (`[environment_factory() for _ in range(generation_batch_size)]`)
  and passed into `reward_kwargs["environments"]` at line 1387-1388 — confirms CLAUDE.md's claims
  about environment pooling and the `environments` reward kwarg still hold on the installed
  version.
- **`_tool_call_loop` (`grpo_trainer.py:1635-1843`), read in full**: `iteration_num` increments
  once per *(execute pending tool calls → generate next turn)* round; the pre-tool-call initial
  generation doesn't count. A fully-compliant 2-search rollout consumes exactly 2 iterations. This
  settles `max_tool_calling_iterations=4` (2 rounds of slack above the soft "at most 2 searches"
  prompt instruction) — full reasoning already recorded in CLAUDE.md's "TRL mechanics" section, not
  repeated here.
- **Paper (arXiv:2505.11821) Appendix E.1, fetched directly**: the paper's own GRPO case study caps
  the agent at exactly one search call, not two. This repo's "at most 2" is a confirmed, deliberate
  deviation (HotpotQA is genuinely 2-hop; see CLAUDE.md for the full comparison) — also already
  recorded in CLAUDE.md, not repeated here.
- **Real `topk=3` retrieval-server responses, tokenized with the actual `Qwen/Qwen3.5-0.8B`
  tokenizer**: individual passages run 124-155 tokens; one full formatted `search()` result (3
  docs) runs 442-478 tokens. A typical compliant 2-search rollout needs roughly 1,100-1,350
  completion tokens end to end. This grounds `max_completion_length=2048` (~1.5-1.9x headroom over
  the typical case) as a measured choice, not a guess.
- **`GRPOConfig` defaults, inspected directly**: `logging_steps=10`, `logging_nan_inf_filter=True`,
  `max_completion_length=256`, `num_generations=8`, `per_device_train_batch_size=8`,
  `learning_rate=1e-6`, `max_tool_calling_iterations=None` (unlimited).
- **`transformers/trainer.py:1745-1751`, read directly**: when `logging_nan_inf_filter=True` (the
  default), a NaN/Inf loss step is *not* recorded as NaN — the accumulator substitutes a decayed
  average of previous losses instead, specifically to keep the logged loss curve from looking
  broken. This means the default setting would hide the exact failure this phase's NaN alert
  exists to catch.
- **`grpo_trainer.py:2399`, read directly**: the logged `reward` metric is computed via
  `torch.nanmean(rewards).item()`, which silently drops NaN entries from the average rather than
  propagating them. This is internal to `GRPOTrainer` (not exposed via any `GRPOConfig` field) —
  patching it would mean touching non-public trainer internals, which CLAUDE.md already rules out
  of scope for the same reason as `MT-GRPO`. Reward-side NaN detection is dropped from this design
  for that reason (see "Alert callback" below for the accepted-gap reasoning).
- **`log_completions` behavior (`grpo_trainer.py:2891-2918`), read directly**: when
  `log_completions=True`, on the main process it (a) prints a `rich`-formatted sample transcript to
  stdout via `print_prompt_completions_sample`, and (b) logs a `{step, prompt, completion, rewards,
  extra, advantage}` table into every backend in `report_to` — including trackio, since it's in
  `report_to` here. `rich` is confirmed installed (`is_rich_available() == True`), so both paths
  are live.
- **`TrackioCallback.setup()` (`transformers/integrations/integration_utils.py:985-992`), read
  directly**: `GRPOConfig.project`/`.run_name` map straight through to `trackio.init(project=...,
  name=...)` — confirms the exact fields to set for CLAUDE.md's "same trackio project, one run per
  condition" requirement, rather than guessing at a mechanism.
- **Trackio's alerts API (`trackio.alert(title, text=None, level=WARN, ...)`) and the
  `TrainerCallback.on_log` integration pattern**, read from the trackio skill's reference docs:
  confirms the exact callback shape to use, and importantly that trackio has **no built-in alert
  deduplication** — repeatedly firing on every `on_log` call while a condition holds is the
  documented example pattern, so this design's own dedup/re-arm logic is necessary, not redundant.

## Goal

Implement `train.py` wiring `GRPOTrainer` + `SearchEnv` + `get_reward_funcs` + `GRPOConfig` +
trackio (per `docs/phase-4-training-smoke-test.md`), then run and manually verify a live smoke
test for both conditions. Keep the module split so `build_config` and the alert callback are fast,
GPU-free unit-test surface, per CLAUDE.md's "Guiding principles" (dependency inversion, thin
seams, high-gear-by-default testing) — the same treatment already given to `env.py`/`rewards.py`.

## Module structure (`src/turn_level_rewards/train.py`)

```python
def build_config(
    condition: Literal["outcome_only", "turn_level"],
    seed: int,
    max_steps: int,
    num_generations: int,
    per_device_train_batch_size: int,
) -> GRPOConfig:
    """Pure function, no model/GPU/network touched. Unit-testable."""

class TrackioAlertCallback(TrainerCallback):
    """Diagnostic alerts for silent-failure modes that a clean exit code wouldn't catch.
    Holds its own rolling state; fires each condition at most once per streak (re-arms after
    the condition clears and re-trips later)."""
    def on_log(self, args, state, control, logs=None, **kwargs): ...

def build_trainer(
    condition: str, train_size: int | None, eval_size: int | None, config: GRPOConfig
) -> GRPOTrainer:
    """Composition root piece: real model, real SearchEnv (hits localhost:8000), real datasets.
    Not unit-tested — this is exactly what the live smoke test validates instead."""

def main() -> None:
    """CLI entrypoint: argparse -> build_config -> build_trainer -> .train()."""
```

## CLI

```
train.py --condition {outcome_only,turn_level}   # required
  --seed 42
  --train-size 8            # Phase 4's own "4-8 rows" smoke-test slice
  --eval-size 8
  --max-steps 2
  --num-generations 2
  --per-device-train-batch-size 1
```

**The bare invocation *is* the smoke test.** `train.py --condition outcome_only` with no other
flags runs Phase 4's smoke test directly — no separate "smoke test mode" branch in the code.
Phase 5's full runs must deliberately pass every scale flag explicitly (e.g. `--train-size 90447
--num-generations 8 --per-device-train-batch-size 4 --max-steps <N>`). This means the smoke test
exercises the exact same code path the full runs will use, not a parallel branch that could
silently diverge from it.

## `build_config`'s fixed hyperparameters (same for both conditions, not CLI flags)

| Field | Value | Why |
|---|---|---|
| `max_tool_calling_iterations` | `4` | See CLAUDE.md's "TRL mechanics" section — already fully justified there |
| `beta` | `0.0` | Locked by CLAUDE.md — no reference model needed |
| `max_completion_length` | `2048` | Measured against real tokenized retrieval results (see Context above) |
| `logging_steps` | `1` | Default `10` would make "~20 steps" alert thresholds actually span 200 real training steps; per-step resolution is needed for both the dashboard and the alert callback to mean what CLAUDE.md says |
| `logging_nan_inf_filter` | `False` | Default `True` hides real NaN losses from `logs["loss"]` (see Context above) — this repo wants NaN to surface, not be smoothed away |
| `log_completions` | `True` | Needed for the manual transcript-reading exit criterion |
| `output_dir` | `f"outputs/{condition}"` | Matches what Phase 5's doc already expects to find |
| `report_to` | `"trackio"` | Locked by CLAUDE.md |
| `project` | `"turn-level-rewards"` | Confirmed via `TrackioCallback.setup()` (`transformers/integrations/integration_utils.py:985-992`): `GRPOConfig.project`/`.run_name` map directly to `trackio.init(project=..., name=...)` — same trackio project for both conditions, per CLAUDE.md |
| `run_name` | `condition` | See above — one run per condition within that shared project |

`learning_rate` is left at TRL's own default (`1e-6`) — not specified anywhere in CLAUDE.md, and
not worth inventing a number for now; flagged as a Phase 5 tuning knob if the full runs' curves
look too slow/fast to converge.

A unit test mechanically diffs the two conditions' built configs and asserts *only*
condition-derived fields (`output_dir`, run name, `reward_funcs`/`get_reward_funcs(condition)`
choice) differ — directly enforcing CLAUDE.md's own stated invariant ("the only thing that should
differ between the two conditions is `--condition` itself").

## `TrackioAlertCallback`

Three checks, each with its own re-armable "already fired this streak" state:

| Condition | Threshold | Level | Notes |
|---|---|---|---|
| Dead/miswired reward | `state.global_step > 20` and `reward` has been exactly `0.0` at *every* logged point so far | ERROR | Reads CLAUDE.md's "staying at exactly 0" as "zero since the start" (wiring bug) — a later dip back to zero after being nonzero is normal variance and does not trip this |
| No learning signal | `frac_reward_zero_std == 1.0` for 20 consecutive logged points | WARN | Every group scoring identically for a sustained stretch = zero policy gradient even if other metrics look fine |
| NaN loss | `logs["loss"]` is NaN (works because `logging_nan_inf_filter=False`, see above) | ERROR | Fires immediately, no threshold; also sets `control.should_training_stop = True` since a NaN loss poisons optimizer state and continuing wastes GPU time on an unrecoverable run |

**Reward-side NaN detection is explicitly dropped**, not forgotten: `torch.nanmean` inside
`GRPOTrainer` masks it structurally (see Context above), and this repo's own `format_reward`/
`outcome_reward`/`turn_reward` are pure string ops and a bounded multiply with no real numerical
path to NaN — the loss-side check covers the realistic failure mode.

Tests (`tests/unit/test_train.py`) feed synthetic `logs` dicts through `on_log()` with an injected
fake `trackio.alert`, using a plain object/`SimpleNamespace` for `state`/`control` (no real
`Trainer` needed):
- Dead reward: 0.0 for 25 straight logged points → fires once past step 20.
- Reward nonzero at some point, later drops to 0.0 → does not fire (variance, not a wiring bug).
- `frac_reward_zero_std==1.0` for 20 consecutive points → fires once; drops below 1.0, then holds
  at 1.0 for another 20-point stretch later → fires again (re-arm).
- `logs["loss"] = float("nan")` → fires immediately, `control.should_training_stop` set `True`.
- A normal healthy log sequence (nonzero reward from the start, `frac_reward_zero_std` varying,
  no NaN) → no alerts fire at all.

## Smoke test execution (manual, foreground — not delegated to a subagent)

Run directly in this session, one condition at a time:
```
uv run python -m turn_level_rewards.train --condition outcome_only
uv run python -m turn_level_rewards.train --condition turn_level
```
Prerequisite: retrieval server up at `localhost:8000` (relaunch command in
`docs/phase-1-retrieval-infra.md`'s Handoff notes if it's died again).

Verification, matching Phase 4's stated exit criteria exactly:
1. Both commands exit 0.
2. Read the `rich`-printed transcripts in stdout directly: does the model actually emit a `search`
   tool call? Does the retrieval server respond with real, non-empty passages? Does the chat
   template render tool calls/results without garbling? Does the final turn contain a well-formed
   `<answer>` tag?
3. `trackio show --project turn-level-rewards` (or `trackio list runs/metrics --json`) shows both
   smoke-test runs with logged metrics.
4. `trackio list alerts --project turn-level-rewards --json` — confirm no unexplained alerts; if
   one fired, inspect the metric snapshot around that step and understand why before proceeding
   (per CLAUDE.md's own alert-triage workflow), rather than ignoring it.

## Out of scope for this phase

- Phase 5's full-scale runs (this phase only proves the mechanism works at smoke-test scale).
- Tuning `learning_rate` or any hyperparameter beyond what's fixed above — left for Phase 5 if the
  full runs' curves warrant it.
- vLLM colocate mode — CLAUDE.md defers this to "later if generation throughput bottlenecks";
  Phase 4's tiny smoke test has no throughput problem to solve.
- Patching `GRPOTrainer` internals to fix `torch.nanmean`'s reward-NaN masking — same "no
  non-public trainer internals" boundary CLAUDE.md already draws around `MT-GRPO`.
