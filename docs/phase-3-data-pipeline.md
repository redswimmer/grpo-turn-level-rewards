# Phase 3: Data pipeline

## Goal

Implement `data.py`: load/filter the training dataset and the held-out eval dataset into a
consistent shape that `train.py` (Phase 4) and `SearchEnv`/`rewards.py` (Phase 2) can consume
without special-casing which dataset a row came from.

## Read first

`CLAUDE.md` — especially the "Dataset" section (exact columns, the broken `test`-split gotcha,
row-count facts already verified) and the reward-design section (recall `outcome_reward` expects
a `golden_answers` **list**, not a single string).

## Prerequisites (entry state)

- **Phase 2 is complete and merged to `main`** — `src/turn_level_rewards/{metrics,env,rewards}.py`
  and `tests/unit/` exist, `scripts/verify_phase2.py` passes. Check
  `docs/phase-2-core-library.md`'s Handoff notes for the exact field names `env.py`/`rewards.py`
  ended up expecting (`golden_answers` list, nested `metadata.supporting_facts.title`) before
  finalizing this phase's output schema — those names are now confirmed, not provisional.

## Tasks

- [x] **Replace the dataset's `prompt` column (known gap, flagged in Phase 2's Handoff notes —
      do this first, it affects the row-formatting helper below).** `PeterJinGo/nq_hotpotqa_train`'s
      own `prompt` column is Search-R1's original **text-tag** ReAct prompt
      (`<search>...</search>` → `<information>...</information>` → `<answer>...</answer>`), which
      assumes a regex-parsed rollout loop. This repo uses TRL's native `environment_factory`
      tool-calling instead (structured `tool_calls`, not text tags) — so the dataset's own
      `prompt` column must be discarded and replaced with a system/user prompt that teaches native
      tool use (describe the `search` tool, instruct the model to reason and call it as needed,
      and to give its final answer wrapped in `<answer>...</answer>` — that one convention is kept,
      since `rewards.py`'s `_extract_answer` already depends on it). **State an explicit soft
      search-count limit in the prompt text itself** (e.g. "at most 2 searches") — CLAUDE.md's
      "TRL mechanics being relied on" section recommends Phase 4 set
      `GRPOConfig(max_tool_calling_iterations=N)` as a hard cutoff *above* whatever soft limit the
      prompt states, so a mismatch here (or no stated limit at all) would leave that hard cutoff
      with nothing to sit above. Write this as a shared row-formatting step so both
      `load_train_dataset` and `load_eval_dataset` produce rows with this new prompt, not the
      dataset's original one.
- [x] `src/turn_level_rewards/data.py`:
      - `load_train_dataset(n: int | None, seed: int = 42)` — loads
        `PeterJinGo/nq_hotpotqa_train`, `default` config, `train` split; filters to
        `data_source == "hotpotqa"` (confirmed: 90,447 rows); shuffles with the given seed;
        selects the first `n` if given. **Do not load this dataset's `test` split** — confirmed
        broken/mixed schema, throws `DatasetGenerationError` (see CLAUDE.md).
      - `load_eval_dataset(n: int | None, seed: int = 42)` — loads `hotpotqa/hotpot_qa`,
        `distractor` config, `validation` split (7,405 rows) directly; reshapes it to the *same*
        column contract as the training set (wrap singular `answer` into a `golden_answers` list
        of one; keep `supporting_facts`/`context` under a `metadata` dict matching the training
        set's nesting, since `env.py`/`rewards.py` should not need to know which source dataset a
        row came from).
      - A shared row-formatting helper so train/eval don't duplicate the reshaping logic.
- [x] `tests/unit/test_data.py` — high-level and fast per CLAUDE.md's testing principles: assert
      the filtering/reshaping *logic* (e.g. against a small local fixture or a mocked
      `datasets.load_dataset`), not a live multi-GB download inside the test. If it's genuinely
      unclear how to fake this dataset load cleanly, stop and ask the user rather than reaching
      for a new test tier (live-download integration test) without checking first.

## Exit criteria (all must be true before handing off)

- [x] A one-off manual (not-necessarily-in-`tests/unit/`) load of the real data confirms row
      counts / schema match CLAUDE.md's already-confirmed facts (90,447 hotpotqa-sourced train
      rows; 7,405 eval rows; avg exactly 2.00 supporting facts/row).
- [x] `load_train_dataset` and `load_eval_dataset` return rows with an identical column contract
      (verified by a test or a manual check), so `env.py`/`rewards.py` work unmodified on either.
- [x] `pytest tests/unit/` (including the new `test_data.py`) still passes fast.

## Handoff notes

- **Column contract (final, confirmed identical for both loaders)**: `prompt`
  (`list[{"role", "content"}]`, system+user), `question` (`str`), `golden_answers` (`list[str]`),
  `metadata` (`dict` with `type`, `level`, `supporting_facts: {"title", "sent_id"}`,
  `context: {"title", "sentences"}`). `env.py`'s `reset(self, metadata, **kwargs)` and
  `rewards.py`'s `outcome_reward` consume `metadata`/`golden_answers` unmodified from either
  loader.
- **`load_dataset_fn` is an injectable keyword-only seam** (defaulting to the real
  `datasets.load_dataset`) on both `load_train_dataset`/`load_eval_dataset` -- a deviation from
  this doc's originally-sketched signature, added deliberately per CLAUDE.md's dependency-inversion
  principle (see `docs/superpowers/specs/2026-07-04-phase-3-data-pipeline-design.md`). Phase 4's
  `train.py` calls both with no `load_dataset_fn` argument, getting the real loader by default.
- **System prompt** lives in `data.py`'s `_SYSTEM_PROMPT` and states "at most 2 searches" verbatim
  -- Phase 4 must set `GRPOConfig(max_tool_calling_iterations=N)` with `N` above this (CLAUDE.md
  recommends 4).
- **Real row counts confirmed** via the manual check in Task 3: 90,447 train rows, 7,405 eval
  rows, identical column contract -- matches CLAUDE.md's already-documented facts, no surprises
  found there.
- **Bug found and fixed during Task 3's manual real-data check**: `load_train_dataset`'s original
  call (`load_dataset_fn("PeterJinGo/nq_hotpotqa_train", "default", split="train")`) crashed with
  `DatasetGenerationError`, even though only the `train` split was requested. Root cause: 🤗
  `datasets`' `download_and_prepare()` runs over *every* split a script-less parquet-config repo
  exposes (both `train.parquet` and `test.parquet` here) before slicing out the requested split —
  and this repo's `test.parquet` has the broken/mixed schema CLAUDE.md's Dataset section already
  documents, so it throws even when `test` is never used. Fixed by pinning
  `data_files={"train": "train.parquet"}` on the same call, so `test.parquet` is never touched.
  Confirmed the fix doesn't change the filtered row count (still 90,447).
- **"Avg supporting facts/row" needed one clarification, not a fix**: computing
  `len(row["metadata"]["supporting_facts"]["title"])` directly (the brief's literal one-liner)
  gives **2.385**, not 2.00 — because HotpotQA's `supporting_facts` is a list of `(title,
  sent_id)` pairs, and 26,771 of the 90,447 rows repeat the same title for more than one
  supporting sentence. Computing the average of `len(set(...))` (**unique titles per row**)
  gives exactly **2.00**, matching CLAUDE.md's already-documented figure. This is also the metric
  that actually matters operationally: `env.py`'s `SearchEnv` already dedupes via
  `frozenset(metadata["supporting_facts"]["title"])`, so `retrieval_fraction` was never affected
  by the raw/deduped distinction — this was purely a difference in how the verification one-liner
  counted, not a data or filtering bug.
- **`scripts/verify_phase3.py`** is the exit-criteria gate -- re-run it after any future change to
  `data.py`.
