"""Dataset loading for train/eval, reshaped to one shared column contract.

env.py/rewards.py never need to know which source dataset a row came from -- both loaders
produce identical prompt/question/golden_answers/metadata columns.
"""

_SYSTEM_PROMPT = (
    "You are a research assistant that answers questions by searching Wikipedia when needed.\n\n"
    "You have access to a `search` tool that looks up Wikipedia passages for a query. Reason "
    "about what you need to find out, and call `search` with a focused query if you need more "
    "information (at most 2 searches). Read the results and reason further before deciding "
    "whether you need another search. Once you are confident in the answer, give it wrapped in "
    "<answer>...</answer> tags (e.g. <answer>Paris</answer>) and nothing else. If search results "
    "aren't helpful, rely on your own knowledge rather than searching repeatedly."
)


def _build_prompt(question: str) -> list[dict[str, str]]:
    """Build the system+user prompt that teaches native tool-calling for a question."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _row_with_prompt(question: str, golden_answers: list[str], metadata: dict) -> dict:
    """Assemble the shared output row shape used by both loaders."""
    return {
        "prompt": _build_prompt(question),
        "question": question,
        "golden_answers": golden_answers,
        "metadata": metadata,
    }


def _format_train_row(row: dict) -> dict:
    """Reshape a PeterJinGo/nq_hotpotqa_train row -- golden_answers/metadata already match."""
    return _row_with_prompt(row["question"], row["golden_answers"], row["metadata"])


def _format_eval_row(row: dict) -> dict:
    """Reshape a hotpotqa/hotpot_qa row -- wraps answer, nests 4 top-level fields under metadata."""
    metadata = {
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": row["supporting_facts"],
        "context": row["context"],
    }
    return _row_with_prompt(row["question"], [row["answer"]], metadata)
