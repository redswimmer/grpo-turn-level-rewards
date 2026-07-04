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

- Phase 2's `rewards.py`/`env.py` should exist so this phase's output shape can be checked
  against what they expect (e.g. field names `golden_answers`, `metadata.supporting_facts`) —
  but this phase is fairly independent and could be built in parallel if needed; check
  `docs/phase-2-core-library.md`'s Handoff notes for the exact field names those modules ended up
  expecting before finalizing this phase's output schema.

## Tasks

- [ ] **Replace the dataset's `prompt` column (known gap, flagged in Phase 2's Handoff notes —
      do this first, it affects the row-formatting helper below).** `PeterJinGo/nq_hotpotqa_train`'s
      own `prompt` column is Search-R1's original **text-tag** ReAct prompt
      (`<search>...</search>` → `<information>...</information>` → `<answer>...</answer>`), which
      assumes a regex-parsed rollout loop. This repo uses TRL's native `environment_factory`
      tool-calling instead (structured `tool_calls`, not text tags) — so the dataset's own
      `prompt` column must be discarded and replaced with a system/user prompt that teaches native
      tool use (describe the `search` tool, instruct the model to reason and call it as needed,
      and to give its final answer wrapped in `<answer>...</answer>` — that one convention is kept,
      since `rewards.py`'s `_extract_answer` already depends on it). Write this as a shared
      row-formatting step so both `load_train_dataset` and `load_eval_dataset` produce rows with
      this new prompt, not the dataset's original one.
- [ ] `src/turn_level_rewards/data.py`:
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
- [ ] `tests/unit/test_data.py` — high-level and fast per CLAUDE.md's testing principles: assert
      the filtering/reshaping *logic* (e.g. against a small local fixture or a mocked
      `datasets.load_dataset`), not a live multi-GB download inside the test. If it's genuinely
      unclear how to fake this dataset load cleanly, stop and ask the user rather than reaching
      for a new test tier (live-download integration test) without checking first.

## Exit criteria (all must be true before handing off)

- [ ] A one-off manual (not-necessarily-in-`tests/unit/`) load of the real data confirms row
      counts / schema match CLAUDE.md's already-confirmed facts (90,447 hotpotqa-sourced train
      rows; 7,405 eval rows; avg exactly 2.00 supporting facts/row).
- [ ] `load_train_dataset` and `load_eval_dataset` return rows with an identical column contract
      (verified by a test or a manual check), so `env.py`/`rewards.py` work unmodified on either.
- [ ] `pytest tests/unit/` (including the new `test_data.py`) still passes fast.

## Handoff notes

<!-- Fill in after completing this phase: exact final column contract chosen, any deviations from
the plan above, and anything Phase 4 needs to know about how to call `data.py`. Leave this
section for the next fresh agent to read first. -->

(not yet started)
