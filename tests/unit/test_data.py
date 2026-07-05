from turn_level_rewards.data import _format_eval_row, _format_train_row

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
