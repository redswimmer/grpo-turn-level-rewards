"""Dataset loading for train/eval, reshaped to one shared column contract.

env.py/rewards.py never need to know which source dataset a row came from -- both loaders
produce identical prompt/question/golden_answers/metadata columns.
"""

from collections.abc import Callable

import datasets
from datasets import Dataset

_SYSTEM_PROMPT = (
    "You are a research assistant that answers questions by searching Wikipedia when needed.\n\n"
    "You have access to a `search` tool that looks up Wikipedia passages for a query. Reason "
    "about what you need to find out, and call `search` with a focused query if you need more "
    "information (at most 2 searches). Read the results and reason further before deciding "
    "whether you need another search. Once you are confident in the answer, give it wrapped in "
    "<answer>...</answer> tags (e.g. <answer>Paris</answer>) and nothing else. If search results "
    "aren't helpful, rely on your own knowledge rather than searching repeatedly."
)

_SYSTEM_PROMPT_NO_SEARCH_CAP = (
    "You are a research assistant that answers questions by searching Wikipedia when needed.\n\n"
    "You have access to a `search` tool that looks up Wikipedia passages for a query. Reason "
    "about what you need to find out, and call `search` with a focused query if you need more "
    "information. Read the results and reason further before deciding whether you need another "
    "search. Once you are confident in the answer, give it wrapped in <answer>...</answer> tags "
    "(e.g. <answer>Paris</answer>) and nothing else."
)


def _build_prompt(question: str, search_cap_in_prompt: bool = True) -> list[dict[str, str]]:
    """Build the system+user prompt that teaches native tool-calling for a question.

    search_cap_in_prompt=False drops the "(at most 2 searches)" instruction and the "rely on
    your own knowledge rather than searching repeatedly" hint -- used for the
    search_count_penalty experiment, which replaces this prompt-engineered guidance with a
    reward-shaped one (rewards.py's search_count_penalty). See
    docs/phase-6-evaluation-comparison.md's Handoff notes for why.
    """
    system_prompt = _SYSTEM_PROMPT if search_cap_in_prompt else _SYSTEM_PROMPT_NO_SEARCH_CAP
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def _row_with_prompt(
    question: str, golden_answers: list[str], metadata: dict, search_cap_in_prompt: bool = True
) -> dict:
    """Assemble the shared output row shape used by both loaders."""
    return {
        "prompt": _build_prompt(question, search_cap_in_prompt),
        "question": question,
        "golden_answers": golden_answers,
        "metadata": metadata,
    }


def _format_train_row(row: dict, search_cap_in_prompt: bool = True) -> dict:
    """Reshape a PeterJinGo/nq_hotpotqa_train row -- golden_answers/metadata already match."""
    return _row_with_prompt(
        row["question"], row["golden_answers"], row["metadata"], search_cap_in_prompt
    )


def _format_eval_row(row: dict, search_cap_in_prompt: bool = True) -> dict:
    """Reshape a hotpotqa/hotpot_qa row -- wraps answer, nests 4 top-level fields under metadata."""
    metadata = {
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": row["supporting_facts"],
        "context": row["context"],
    }
    return _row_with_prompt(row["question"], [row["answer"]], metadata, search_cap_in_prompt)


def load_train_dataset(
    n: int | None,
    seed: int = 42,
    *,
    load_dataset_fn: Callable[..., Dataset] = datasets.load_dataset,
    search_cap_in_prompt: bool = True,
) -> Dataset:
    """Load PeterJinGo/nq_hotpotqa_train, filtered to hotpotqa rows, reshaped to the shared contract.

    Args:
        n: Number of rows to select after shuffling, or None for all filtered rows.
        seed: Shuffle seed.
        load_dataset_fn: Injectable seam for the real datasets.load_dataset call -- tests pass a
            fake returning an in-memory Dataset.
        search_cap_in_prompt: False drops the "(at most 2 searches)" prompt instruction -- see
            _build_prompt's docstring.

    Note: `data_files` pins this to the repo's `train.parquet` explicitly. Without it,
    `datasets.load_dataset(..., split="train")` still runs `download_and_prepare()` over *every*
    split the repo exposes (both `train.parquet` and `test.parquet`) before slicing out the
    requested one -- and this repo's `test.parquet` has the broken/mixed schema documented in
    CLAUDE.md's "Dataset" section, so it throws `DatasetGenerationError` even though `test` is
    never requested. Confirmed by direct reproduction (Task 3's manual real-data check).
    """
    ds = load_dataset_fn(
        "PeterJinGo/nq_hotpotqa_train",
        "default",
        data_files={"train": "train.parquet"},
        split="train",
    )
    ds = ds.filter(lambda row: row["data_source"] == "hotpotqa")
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(n))
    return ds.map(
        lambda row: _format_train_row(row, search_cap_in_prompt), remove_columns=ds.column_names
    )


def load_eval_dataset(
    n: int | None,
    seed: int = 42,
    *,
    load_dataset_fn: Callable[..., Dataset] = datasets.load_dataset,
    search_cap_in_prompt: bool = True,
) -> Dataset:
    """Load hotpotqa/hotpot_qa (distractor, validation), reshaped to the shared contract.

    Args:
        n: Number of rows to select after shuffling, or None for all rows.
        seed: Shuffle seed.
        load_dataset_fn: Injectable seam for the real datasets.load_dataset call -- tests pass a
            fake returning an in-memory Dataset.
        search_cap_in_prompt: False drops the "(at most 2 searches)" prompt instruction -- see
            _build_prompt's docstring.
    """
    ds = load_dataset_fn("hotpotqa/hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(n))
    return ds.map(
        lambda row: _format_eval_row(row, search_cap_in_prompt), remove_columns=ds.column_names
    )
