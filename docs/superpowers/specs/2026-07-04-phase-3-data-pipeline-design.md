# Phase 3 Data Pipeline — Design

Status: approved by user, 2026-07-04. Implements `docs/phase-3-data-pipeline.md`.

## Context and evidence gathered

This design was grounded by reading TRL's actual source and streaming one real row from each
source dataset, rather than assuming CLAUDE.md's prose holds exactly:

- **`trl/trainer/grpo_trainer.py`** (`_tokenize_prompts`, read directly): the `prompt` column must
  be **conversational** — a list of `{"role", "content"}` dicts — not a plain string. The
  tool-call loop appends assistant/tool messages onto that same list, and `is_conversational()`
  branches on this shape. This settles that the new prompt must be message-list shaped.
- **Real row streamed from `PeterJinGo/nq_hotpotqa_train`** (`data_source == "hotpotqa"`)
  confirms `metadata = {"type", "level", "supporting_facts": {"title": [...], "sent_id": [...]},
  "context": {"title": [...], "sentences": [...]}}`, `golden_answers` a top-level list, and the
  existing `prompt` column is Search-R1's original text-tag ReAct prompt (confirming Phase 2's
  handoff-note gap).
- **Real row streamed from `hotpotqa/hotpot_qa` (distractor, validation)** confirms `answer` is a
  singular string and `supporting_facts`/`context`/`type`/`level` are **top-level** columns, not
  nested under `metadata` — the eval-side reshape must relocate all four fields, not just wrap
  `answer` into a list.
- **Paper's Appendix E.5 ("System Prompt (GRPO)"), read directly from the arXiv PDF**: the exact
  case study this repo targets uses a **system**-role block for tool description + step
  instructions, with the question as a separate turn — confirming a system+user split rather than
  one combined message. (Its concrete mechanics — JSON-in-XML-tags tool calls, "use the tool
  exactly once" — don't transfer here: TRL's native `environment_factory` gives the model a
  structured tool schema automatically, and this repo's own `CLAUDE.md`/phase-3 doc already
  independently settled on "at most 2 searches" as the soft limit, matching this dataset's
  real ~2.00 avg supporting-facts/row rather than the paper's own dataset choice of one search.)

## Column contract (identical for train and eval)

| Column | Type | Source |
|---|---|---|
| `prompt` | `list[{"role": "system"/"user", "content": str}]` | Built fresh, replaces the dataset's own `prompt` |
| `question` | `str` | Passed through (train) / already named `question` (eval) |
| `golden_answers` | `list[str]` | Passed through (train) / `[answer]` (eval) |
| `metadata` | `dict` — `{"type", "level", "supporting_facts": {"title", "sent_id"}, "context": {"title", "sentences"}}` | Passed through (train) / assembled from eval's four top-level fields |

`env.py`'s `reset(self, metadata, **kwargs)` and `rewards.py`'s `outcome_reward(completions,
golden_answers, **kwargs)` only ever read `metadata["supporting_facts"]["title"]` and
`golden_answers` — `type`/`level` aren't functionally required downstream, but are included in
eval's `metadata` anyway so the two loaders produce a genuinely identical shape (this phase's
stated exit criterion), not just a functionally-equivalent one.

## System prompt content

Adapted from the paper's Appendix E.5 template, with the JSON-in-XML-tag tool-call mechanics
dropped (TRL's native tool-calling replaces that entirely) and the soft search-count limit stated
explicitly per the phase doc's task:

```
You are a research assistant that answers questions by searching Wikipedia when needed.

You have access to a `search` tool that looks up Wikipedia passages for a query. Reason about
what you need to find out, and call `search` with a focused query if you need more information
(at most 2 searches). Read the results and reason further before deciding whether you need
another search. Once you are confident in the answer, give it wrapped in <answer>...</answer>
tags (e.g. <answer>Paris</answer>) and nothing else. If search results aren't helpful, rely on
your own knowledge rather than searching repeatedly.
```

`_build_prompt(question)` returns `[{"role": "system", "content": <above>}, {"role": "user",
"content": question}]`.

## `src/turn_level_rewards/data.py`

```python
_SYSTEM_PROMPT: str
_build_prompt(question: str) -> list[dict]
_row_with_prompt(question, golden_answers, metadata) -> dict   # shared final assembly
_format_train_row(row: dict) -> dict     # golden_answers/metadata already correctly shaped
_format_eval_row(row: dict) -> dict      # wraps answer, relocates 4 fields under metadata

def load_train_dataset(
    n: int | None, seed: int = 42, *, load_dataset_fn: Callable = datasets.load_dataset
) -> Dataset:
    # PeterJinGo/nq_hotpotqa_train, "default", split="train"
    # .filter(data_source == "hotpotqa") -> .shuffle(seed) -> .select(range(n)) if n is not None
    # -> .map(_format_train_row, remove_columns=<all original columns>)

def load_eval_dataset(
    n: int | None, seed: int = 42, *, load_dataset_fn: Callable = datasets.load_dataset
) -> Dataset:
    # hotpotqa/hotpot_qa, "distractor", split="validation"
    # .shuffle(seed) -> .select(range(n)) if n is not None
    # -> .map(_format_eval_row, remove_columns=<all original columns>)
```

- `load_dataset_fn` is an injectable, keyword-only seam (defaulting to the real
  `datasets.load_dataset`) per CLAUDE.md's guiding principle 1 — dataset loading is a network call,
  so it gets the same DI treatment as `SearchEnv`'s `retrieve_fn`. This is an intentional addition
  beyond the phase-3 doc's literal `load_train_dataset(n, seed=42)` signature — raised as an open
  question and approved before writing this spec — not an unnoticed deviation.
- Never references the `test` split for either dataset (hardcoded `split="train"` /
  `split="validation"`), so the known-broken schema in `nq_hotpotqa_train`'s `test` split can't be
  reached.
- `remove_columns=<all original columns>` on `.map()` guarantees the two loaders can't leak
  source-specific extra columns (`id`, `ability`, `reward_model`, `extra_info` from train; `id`
  from eval) into the shared contract.

## Tests (`tests/unit/test_data.py`)

Fake `load_dataset_fn` returns `datasets.Dataset.from_list([...])` built from small literal
fixtures shaped like the real streamed rows (same style as `test_env.py`'s `HOTPOT_ROW_1`/`_2`).
Covers:
- Train-side filtering drops non-`"hotpotqa"` `data_source` rows.
- `n` truncates to exactly `n` rows; `n=None` returns all fixture rows.
- `golden_answers` is a list on both train and eval output.
- Eval's singular `answer` wraps into a one-element `golden_answers` list.
- `metadata` has identical key structure between train and eval output rows.
- `prompt` is `[system, user]`; the user message's content contains the row's `question` text; the
  system message mentions the search-count limit.

No network, no GPU — matches CLAUDE.md's "every test in `tests/unit/` must be fast and
deterministic" requirement.

## Manual verification (not a `tests/unit/` test — this phase's stated exit criterion)

After implementation, a one-off script/REPL check against the real data: `load_train_dataset(None)`
row count is 90,447; `load_eval_dataset(None)` row count is 7,405; both loaders' `.column_names`
are identical; avg `len(metadata["supporting_facts"]["title"])` across a sample is 2.00.

## Out of scope for this phase

- `train.py` / the live smoke test (Phase 4).
- Setting `GRPOConfig(max_tool_calling_iterations=N)` — that's Phase 4's hard cutoff, sitting above
  this phase's soft "at most 2 searches" prompt text.
- Renaming `SearchEnv.search()`/its tool name to match the paper's `wiki_search` — out of scope for
  a data-only phase; would touch already-merged Phase 2 code for no functional gain.
