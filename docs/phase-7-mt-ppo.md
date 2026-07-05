# Phase 7: Multi-Turn PPO / MT-PPO

## Goal

Implement a custom multi-turn PPO trainer (`src/turn_level_rewards/train_ppo.py`) supporting two
conditions, `--condition {ppo,mt_ppo}`, reproducing the paper's actual Table 2 PPO/MT-PPO
methodology on this repo's existing HotpotQA/wiki-18 pipeline. Deterministic rewards only — the
LLM judge is Phase 8, built on top of a working Phase 7.

## Read first

`CLAUDE.md`'s Goal section (the `PPO`/`MT-PPO` framing has been updated — this is now in scope,
not excluded) and `docs/superpowers/specs/2026-07-05-phase-7-mt-ppo-design.md` in full — it
contains the paper's exact PPO/MT-PPO hyperparameters (Section 6.2/C.1.3), the confirmed absence
of any TRL multi-turn PPO trainer (checked fresh against both the installed `1.7.1` and the dev
branch, not assumed stale), and two concrete verified blueprints (a working critic architecture,
a fully manual multi-turn tool-calling round-trip) to build from directly.

## Prerequisites (entry state)

- Phases 1-6 done (`env.py`, `rewards.py`, `metrics.py`, `data.py`, `train.py` all exist and are
  reusable unmodified). Phase 6 itself doesn't need to be complete first — Phase 7 doesn't depend
  on `evaluate.py`/`compare_runs.py`.
- Retrieval server running and stable.

## Tasks

- [ ] `src/turn_level_rewards/train_ppo.py`:
  - GAE computation as a pure function (`gamma=1.0`, `lambda=1.0` per the paper).
  - Eq. 9 reward-placement logic as a pure function: `R^O` (`format_reward`+`outcome_reward`) at
    the trajectory's final token always; `R^I` (`turn_reward`'s per-turn contribution) at each
    intermediate turn boundary for `mt_ppo` only, always `0` for `ppo`.
  - `MTPPOTrainer(transformers.Trainer)`: the rollout loop (render with `SearchEnv.search`'s tool
    schema via `apply_chat_template` + `add_response_schema`/`get_training_chat_template`,
    generate, `parse_response`, execute the tool call for real, append the tool message, repeat up
    to `N_max=4` turns per the paper's Section 6.2, then require a final `<answer>`), the critic
    forward pass (`AutoModelForSequenceClassification(num_labels=1)`, `.score()` on the backbone's
    per-token hidden states — verified working directly against `Qwen/Qwen3.5-0.8B` during design),
    and the PPO-clip (`ε=0.2`) + KL-penalty (`β=0.001`) + value-loss (`coef=0.5`, an assumed
    default, not paper-derived) update.
  - `build_ppo_config`/`build_ppo_trainer`/`main()`, mirroring `train.py`'s Phase 4 composition-root
    pattern (CLI flags, fixed hyperparameters baked into `build_ppo_config`, not exposed
    independently).
- [ ] `tests/unit/test_train_ppo.py`: GAE and Eq. 9 placement logic, fast/deterministic, no GPU —
  same DI/seam principle as the rest of the repo. The rollout loop and critic construction are
  **not** unit-tested (require a real model/GPU/chat-template) — that's what the live smoke test
  below is for.
- [ ] `scripts/verify_phase7.py`, mirroring `verify_phase2.py`'s/`verify_phase4.py`'s existing
  pattern: run `tests/unit/`, `ruff check`, `ty check`, plus direct calls into
  `build_ppo_config` asserting the fixed hyperparameters above hold for both conditions.
- [ ] Live smoke test (mirroring Phase 4's own): both conditions, `--max-steps 2` or similar tiny
  scale, real model, real retrieval server, `log_completions`-equivalent transcript reading —
  confirm real tool calls, real retrieved passages, real critic values, and (specifically for
  `mt_ppo`) that `R^I` actually lands at intermediate turn boundaries, not just at the end.

## Exit criteria (all must be true before handing off)

- [ ] Live smoke test completes without errors for **both** `--condition ppo` and
      `--condition mt_ppo`.
- [ ] Transcripts manually confirmed to show real tool calls, real retrieved passages, and correct
      reward placement (spot-check that `mt_ppo`'s intermediate-turn reward is nonzero when a gold
      title is actually surfaced, and `ppo`'s is always 0 there).
- [ ] `tests/unit/test_train_ppo.py` passes; `scripts/verify_phase7.py` passes.
- [ ] No obviously-broken critic values (e.g. NaN, or values that never move across a few smoke-test
      steps) — a real trained critic should show *some* signal even at tiny scale.

## Handoff notes

<!-- Fill in after completing this phase: any TRL/transformers API surprises the design's manual
verification didn't already catch, the actual value_loss_coef/other assumed defaults if they
needed adjusting, real observed per-step wall-clock time (this will likely be slower than GRPO's,
given the extra critic forward/backward pass and multiple PPO epochs per batch), and anything
that would help Phase 8 wire in the LLM judge cleanly. Leave this section for the next fresh agent
to read first. -->

(not yet started)
