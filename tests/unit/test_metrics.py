import pytest
from turn_level_rewards.metrics import exact_match, f1_score, normalize_answer


def test_normalize_answer_strips_case_punctuation_articles_and_whitespace():
    assert normalize_answer("The Beatles!") == "beatles"
    assert normalize_answer("  a   dog  ") == "dog"


def test_exact_match_identical_strings():
    assert exact_match("Paris", "Paris") is True


def test_exact_match_normalizes_before_comparing():
    assert exact_match("The Beatles", "beatles") is True


def test_exact_match_fully_disjoint_answers():
    assert exact_match("Paris", "London") is False


def test_f1_score_identical_strings_is_one():
    assert f1_score("Paris, France", "Paris, France") == 1.0


def test_f1_score_partial_overlap():
    assert f1_score("New", "New York") == pytest.approx(2 / 3)


def test_f1_score_fully_disjoint_is_zero():
    assert f1_score("Paris", "London") == 0.0


def test_f1_score_empty_prediction_against_nonempty_ground_truth_is_zero():
    assert f1_score("", "Paris") == 0.0
