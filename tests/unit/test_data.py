import datasets
from turn_level_rewards.data import (
    _format_eval_row,
    _format_train_row,
    load_eval_dataset,
    load_train_dataset,
)

TRAIN_ROW = {
    "id": "train_0",
    "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
    "golden_answers": ["Arthur's Magazine"],
    "data_source": "hotpotqa",
    "prompt": [{"role": "user", "content": "old text-tag prompt, discarded"}],
    "ability": "fact-reasoning",
    "metadata": {
        "type": "comparison",
        "level": "medium",
        "supporting_facts": {"title": ["Arthur's Magazine", "First for Women"], "sent_id": [0, 0]},
        "context": {
            "title": ["Arthur's Magazine", "First for Women"],
            "sentences": [["Arthur's Magazine sentence."], ["First for Women sentence."]],
        },
    },
}

EVAL_ROW = {
    "id": "5a8b57f2",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
    "context": {
        "title": ["Scott Derrickson", "Ed Wood"],
        "sentences": [["Scott Derrickson sentence."], ["Ed Wood sentence."]],
    },
}


def test_format_train_row_builds_system_then_user_prompt_with_question():
    row = _format_train_row(TRAIN_ROW)

    roles = [m["role"] for m in row["prompt"]]
    assert roles == ["system", "user"]
    assert row["prompt"][1]["content"] == TRAIN_ROW["question"]
    assert "at most 2 searches" in row["prompt"][0]["content"]


def test_format_train_row_passes_through_golden_answers_and_metadata():
    row = _format_train_row(TRAIN_ROW)

    assert row["golden_answers"] == ["Arthur's Magazine"]
    assert row["metadata"] == TRAIN_ROW["metadata"]


def test_format_train_row_drops_original_prompt_and_source_columns():
    row = _format_train_row(TRAIN_ROW)

    assert set(row.keys()) == {"prompt", "question", "golden_answers", "metadata"}


def test_format_eval_row_wraps_answer_into_golden_answers_list():
    row = _format_eval_row(EVAL_ROW)

    assert row["golden_answers"] == ["yes"]


def test_format_eval_row_nests_four_top_level_fields_under_metadata():
    row = _format_eval_row(EVAL_ROW)

    assert row["metadata"] == {
        "type": "comparison",
        "level": "hard",
        "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
        "context": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sentences": [["Scott Derrickson sentence."], ["Ed Wood sentence."]],
        },
    }


def test_format_eval_row_builds_same_prompt_shape_as_train():
    row = _format_eval_row(EVAL_ROW)

    roles = [m["role"] for m in row["prompt"]]
    assert roles == ["system", "user"]
    assert row["prompt"][1]["content"] == EVAL_ROW["question"]


def test_train_and_eval_rows_have_identical_column_contract():
    train_row = _format_train_row(TRAIN_ROW)
    eval_row = _format_eval_row(EVAL_ROW)

    assert set(train_row.keys()) == set(eval_row.keys())
    assert set(train_row["metadata"].keys()) == set(eval_row["metadata"].keys())


TRAIN_ROWS = [
    TRAIN_ROW,
    {
        "id": "train_1",
        "question": "What year was 127 Hours released?",
        "golden_answers": ["2010"],
        "data_source": "hotpotqa",
        "prompt": [{"role": "user", "content": "old text-tag prompt, discarded"}],
        "ability": "fact-reasoning",
        "metadata": {
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["127 Hours"], "sent_id": [0]},
            "context": {"title": ["127 Hours"], "sentences": [["A 2010 survival drama film."]]},
        },
    },
    {
        "id": "train_2",
        "question": "A natural-questions-sourced row that must be filtered out.",
        "golden_answers": ["irrelevant"],
        "data_source": "nq",
        "prompt": [{"role": "user", "content": "old text-tag prompt, discarded"}],
        "ability": "fact-reasoning",
        "metadata": {
            "type": "single",
            "level": "easy",
            "supporting_facts": {"title": [], "sent_id": []},
            "context": {"title": [], "sentences": []},
        },
    },
]

EVAL_ROWS = [
    EVAL_ROW,
    {
        "id": "5a90620b",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {"title": ["Paris"], "sent_id": [0]},
        "context": {"title": ["Paris"], "sentences": [["Paris is the capital of France."]]},
    },
]


def _fake_loader(rows):
    def load_dataset_fn(*args, **kwargs):
        return datasets.Dataset.from_list(rows)

    return load_dataset_fn


def test_load_train_dataset_filters_to_hotpotqa_only():
    ds = load_train_dataset(None, load_dataset_fn=_fake_loader(TRAIN_ROWS))

    assert len(ds) == 2
    assert all(row["question"] != TRAIN_ROWS[2]["question"] for row in ds)


def test_load_train_dataset_n_selects_exactly_n_rows():
    ds = load_train_dataset(1, seed=0, load_dataset_fn=_fake_loader(TRAIN_ROWS))

    assert len(ds) == 1


def test_load_train_dataset_n_none_returns_all_filtered_rows():
    ds = load_train_dataset(None, load_dataset_fn=_fake_loader(TRAIN_ROWS))

    assert len(ds) == 2


def test_load_eval_dataset_wraps_answer_into_golden_answers_list():
    ds = load_eval_dataset(None, load_dataset_fn=_fake_loader(EVAL_ROWS))

    assert sorted(row["golden_answers"][0] for row in ds) == ["Paris", "yes"]


def test_load_eval_dataset_nests_supporting_facts_and_context_under_metadata():
    ds = load_eval_dataset(1, load_dataset_fn=_fake_loader(EVAL_ROWS))
    row = ds[0]
    supporting_titles = row["metadata"]["supporting_facts"]["title"]
    context_titles = row["metadata"]["context"]["title"]

    # Row order isn't guaranteed after shuffling, so check internal consistency
    # (supporting_facts and context must agree) and that it's one of the two EVAL_ROWS
    # fixtures declared above (["Scott Derrickson", "Ed Wood"] from EVAL_ROW, ["Paris"]
    # from the second EVAL_ROWS entry).
    assert supporting_titles == context_titles
    assert supporting_titles in (["Scott Derrickson", "Ed Wood"], ["Paris"])


def test_load_train_and_eval_datasets_have_identical_column_contract():
    train_ds = load_train_dataset(1, load_dataset_fn=_fake_loader(TRAIN_ROWS))
    eval_ds = load_eval_dataset(1, load_dataset_fn=_fake_loader(EVAL_ROWS))

    assert set(train_ds.column_names) == set(eval_ds.column_names)
    assert set(train_ds[0]["metadata"].keys()) == set(eval_ds[0]["metadata"].keys())
