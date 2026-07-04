# Phase 4: Training script + live smoke test

## Goal

Wire everything built so far into `train.py` (real `GRPOTrainer` + `SearchEnv` +
`get_reward_funcs` + `GRPOConfig` + trackio), and prove with a tiny real-GPU smoke test that the
whole tool-calling loop actually works end to end before committing to a full run.

## Read first

`CLAUDE.md` — especially "TRL mechanics being relied on", "Hardware", and "Experiment tracking
(trackio)". This is the highest-risk phase for surfacing surprises that no amount of unit testing
catches (chat-template/tool-call rendering, `environment_factory` wiring) — budget time to
actually read the live completions transcripts, not just check for a clean exit code.

## Prerequisites (entry state)

- Phase 1: retrieval server running and reachable at a known URL.
- Phase 2: `env.py`, `rewards.py`, `metrics.py` implemented and unit-tested.
- Phase 3: `data.py` implemented and unit-tested.
- JDK + pyserini installed (Phase 1).

## Tasks

- [ ] `src/turn_level_rewards/train.py` — CLI entrypoint:
      - Args: `--condition {outcome_only,turn_level}`, `--seed`, `--train-size`, `--eval-size`,
        `--max-steps` (all defaulted identically across conditions per CLAUDE.md — the only
        thing that should differ between the two conditions is `--condition` itself).
      - Builds `GRPOTrainer(model="Qwen/Qwen3.5-0.8B", environment_factory=SearchEnv,
        reward_funcs=get_reward_funcs(condition), train_dataset=..., eval_dataset=...,
        args=GRPOConfig(...))` per the recommended hyperparameters in CLAUDE.md
        (`num_generations`, `max_tool_calling_iterations`, `beta=0.0`, etc.).
      - `report_to="trackio"`; project name and per-condition run name per CLAUDE.md's
        "Experiment tracking" section (both conditions in the *same* trackio project).
      - `trackio.alert()` calls for the diagnostic conditions listed in CLAUDE.md (dead reward,
        `frac_reward_zero_std` stuck at 1.0, NaN).
- [ ] Smoke test: run with `--max-steps 2`, `num_generations=2`, `per_device_train_batch_size=1`,
      a tiny real slice of the real HotpotQA data (4-8 rows), `log_completions=True`, against the
      real retrieval server from Phase 1 and the real `Qwen/Qwen3.5-0.8B` model.
- [ ] Manually read the logged completions transcripts (not just exit code): does the model
      actually call `search`? Does the retrieval server respond with real passages? Do
      `environments` populate (`retrieval_fraction` isn't silently stuck at 0 for the wrong
      reason — e.g. `environment_factory` wiring failure vs. genuinely no hits)? Does the chat
      template render tool calls/results correctly for Qwen3.5?

## Exit criteria (all must be true before handing off)

- [ ] Smoke test completes without errors for **both** `--condition outcome_only` and
      `--condition turn_level`.
- [ ] Transcripts manually confirmed to show real tool calls and real retrieved passages (not
      empty/malformed tool turns).
- [ ] trackio dashboard shows both smoke-test runs and their logged metrics.
- [ ] No trackio alerts fired during the smoke test (or any that did fire were investigated and
      are understood, not just ignored).

## Handoff notes

<!-- Fill in after completing this phase: any TRL API mismatches vs. CLAUDE.md's notes, final
hyperparameters actually used if different from CLAUDE.md's recommendations, chat-template
gotchas, and a realistic per-step wall-clock estimate observed on real hardware (needed to size
Phase 5's `--max-steps`). Leave this section for the next fresh agent to read first. -->

(not yet started)
