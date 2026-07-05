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
- **Phase 3 is complete and merged** — `src/turn_level_rewards/data.py` exists with
  `load_train_dataset(n: int | None, seed: int = 42, *, load_dataset_fn=...) -> Dataset` and
  `load_eval_dataset(n: int | None, seed: int = 42, *, load_dataset_fn=...) -> Dataset`
  (`load_dataset_fn` is an injectable seam for tests — call both with no third argument to get the
  real loader). Both return rows with an identical column contract: `prompt`
  (`list[{"role": "system"/"user", "content": str}]`, already conversational-format and ready to
  hand straight to `GRPOTrainer`), `question` (`str`), `golden_answers` (`list[str]`), `metadata`
  (`dict`, consumed by `SearchEnv.reset(self, metadata, **kwargs)`). See
  `docs/phase-3-data-pipeline.md`'s Handoff notes for the full contract and a real bug fix
  (`data_files` pin) that's already baked into these functions — nothing extra needed here.
- JDK + pyserini installed (Phase 1).

## Tasks

- [x] `src/turn_level_rewards/train.py` — CLI entrypoint:
      - Args: `--condition {outcome_only,turn_level}`, `--seed`, `--train-size`, `--eval-size`,
        `--max-steps` (all defaulted identically across conditions per CLAUDE.md — the only
        thing that should differ between the two conditions is `--condition` itself).
      - Builds `GRPOTrainer(model="Qwen/Qwen3.5-0.8B", environment_factory=SearchEnv,
        reward_funcs=get_reward_funcs(condition),
        train_dataset=data.load_train_dataset(n=train_size, seed=seed),
        eval_dataset=data.load_eval_dataset(n=eval_size, seed=seed),
        args=GRPOConfig(...))` per the recommended hyperparameters in CLAUDE.md
        (`num_generations`, `beta=0.0`, etc.).
      - **`GRPOConfig(max_tool_calling_iterations=N)` must be set with `N` strictly above 2** —
        `data.py`'s system prompt (see Phase 3's Handoff notes) states a soft "at most 2 searches"
        limit to the model; this hard cutoff exists as a safety net *above* that soft limit so a
        not-yet-compliant rollout isn't truncated mid-trajectory. CLAUDE.md's "TRL mechanics"
        section recommends `N=4`.
      - `report_to="trackio"`; project name and per-condition run name per CLAUDE.md's
        "Experiment tracking" section (both conditions in the *same* trackio project).
      - `trackio.alert()` calls for the diagnostic conditions listed in CLAUDE.md (dead reward,
        `frac_reward_zero_std` stuck at 1.0, NaN).
      - **As actually built** (see Handoff notes): `--per-device-train-batch-size` is not a
        separate CLI flag — `per_device_train_batch_size` is always derived equal to
        `--num-generations` (a real TRL constraint, not a stylistic choice). `--num-generations`
        also defaults to `2`, so the bare CLI invocation (just `--condition`) *is* this phase's
        smoke test.
- [x] Smoke test: run with `--max-steps 2`, `num_generations=2` (which also fixes
      `per_device_train_batch_size=2`, not `1` as originally sketched here — see Handoff notes), a
      tiny real slice of the real HotpotQA data (8 rows), `log_completions=True`, against the real
      retrieval server from Phase 1 and the real `Qwen/Qwen3.5-0.8B` model.
- [x] Manually read the logged completions transcripts (not just exit code): does the model
      actually call `search`? Does the retrieval server respond with real passages? Do
      `environments` populate (`retrieval_fraction` isn't silently stuck at 0 for the wrong
      reason — e.g. `environment_factory` wiring failure vs. genuinely no hits)? Does the chat
      template render tool calls/results correctly for Qwen3.5?

## Exit criteria (all must be true before handing off)

- [x] Smoke test completes without errors for **both** `--condition outcome_only` and
      `--condition turn_level`.
- [x] Transcripts manually confirmed to show real tool calls and real retrieved passages (not
      empty/malformed tool turns).
- [x] trackio dashboard shows both smoke-test runs and their logged metrics.
- [x] No trackio alerts fired during the smoke test (or any that did fire were investigated and
      are understood, not just ignored).

## Handoff notes

- **Three real bugs surfaced by the live smoke test that no unit test could have caught**
  (each required a real model, real chat template, or real GPU — exactly the integration surface
  `tests/unit/` deliberately fakes around):
  1. **`GRPOConfig` divisibility constraint, found before ever running the model**: TRL requires
     `generation_batch_size` (defaults to `per_device_train_batch_size × num_processes ×
     steps_per_generation`) to be evenly divisible by `num_generations`. On a single GPU this
     reduces to `per_device_train_batch_size` needing to be a multiple of `num_generations`. This
     phase's own doc originally sketched `num_generations=2, per_device_train_batch_size=1` for
     the smoke test — that combination is invalid and raises `ValueError` immediately. Fixed
     during design (not left for implementation to discover) by deriving
     `per_device_train_batch_size = num_generations` always, dropping the separate CLI flag
     entirely. See `docs/superpowers/specs/2026-07-05-phase-4-training-smoke-test-design.md` for
     the full verification trail.
  2. **Two missing runtime dependencies**, only reachable once a real model/tool-schema was
     involved: `torchvision` (Qwen3.5 is natively multimodal — `AutoProcessor.from_pretrained`
     tries to build a video sub-processor even for text-only use) and `jmespath` (TRL's
     tool-response parsing). Both added to `pyproject.toml`/`uv.lock`. Confirmed `Qwen/Qwen3.5-0.8B`
     is genuinely ~752M parameters (no hidden multimodal-tower bloat) — the dependency gap was
     about processor construction, not model size.
  3. **A docstring-format bug in already-merged Phase 2 code**: `SearchEnv.search`'s docstring
     used Sphinx-style `query (str): description`, but transformers'
     `chat_template_utils.parse_google_format_docstring` (which builds the tool's JSON schema for
     the chat template) requires bare Google-style `query: description` — its regex
     (`args_split_re`) requires the argument name immediately followed by `:`, so the inline type
     annotation silently broke parsing and raised `DocstringParsingException` the moment a real
     `apply_chat_template(tools=...)` call happened. Fixed directly in `env.py`.
- **A real CUDA OOM on step 2** (not step 1) of the smoke test, during `loss.backward()`. Root
  cause: confirmed the model is genuinely ~752M params (not a size-mismatch issue), but Qwen3.5's
  hybrid linear-attention/Mamba-style layers (18 of 24 layers) fall back to a memory-inefficient
  reference PyTorch implementation without `flash-linear-attention`/`causal-conv1d` installed
  (both require compiling CUDA extensions against `torch==2.12.1+cu130`, uncertain to build
  cleanly). Fixed with `gradient_checkpointing=True` in `build_config` instead — standard
  compute-for-memory tradeoff, no new dependencies. If Phase 5's full runs hit memory pressure
  again at larger `--train-size`/`--num-generations`, revisit installing the fused kernels
  properly rather than assuming `gradient_checkpointing` alone scales indefinitely.
- **Confirmed working end-to-end for both conditions**: real `search` tool calls, real retrieved
  Wikipedia passages (readable in the `rich`-printed transcripts, e.g. a real "Red Bull Arena
  (Salzburg)" article correctly answering "What event occurred at the Red Bull Arena... in
  2008?" with `<answer>UEFA Euro 2008</answer>`), real reward computation
  (`rewards/outcome_reward`, `rewards/format_reward`, and for `turn_level` specifically
  `rewards/turn_reward` genuinely nonzero — confirming `SearchEnv.retrieval_fraction` is wired
  correctly, not silently stuck at 0), zero tool-call failures, zero trackio alerts (expected at
  only 2 steps — well below the alert callback's 20-step thresholds).
- **Per-step wall-clock observed**: roughly 6-9 seconds/step at this smoke-test scale
  (`num_generations=2`, 8 train rows, `max_completion_length=2048`, `gradient_checkpointing=True`)
  on a single RTX 4090. Use this only as a rough floor for sizing Phase 5's `--max-steps` — a
  larger `--num-generations`/`--train-size` will scale per-step time up, and
  `gradient_checkpointing`'s recompute overhead is part of this number.
- **`TRL_EXPERIMENTAL_SILENCE=1`** environment variable silences a `UserWarning` about
  `environment_factory` being an experimental, may-change-without-notice TRL API — worth knowing
  it's experimental for Phase 5 (a future TRL upgrade could change its behavior), but not itself a
  bug or blocker.
