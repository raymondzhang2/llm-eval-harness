import pytest

from app.services.metrics import (
    ScoredItem,
    biggest_disagreements,
    exact_agreement,
    mean_score,
    quadratic_weighted_kappa,
    score_distribution,
    spearman_correlation,
    summarize_criterion,
    within_one_agreement,
)


def test_mean_and_distribution():
    assert mean_score([1, 2, 3, 4]) == 2.5
    assert mean_score([]) is None
    assert score_distribution([1, 1, 5], 1, 5) == {1: 2, 2: 0, 3: 0, 4: 0, 5: 1}


def test_exact_and_within_one_hand_computed():
    judge, human = [1, 3, 5, 2], [2, 3, 1, 2]  # gaps: 1, 0, 4, 0
    assert exact_agreement(judge, human) == 0.5
    assert within_one_agreement(judge, human) == 0.75


def test_agreement_on_empty_input_is_undefined():
    assert exact_agreement([], []) is None
    assert within_one_agreement([], []) is None
    assert quadratic_weighted_kappa([], [], 1, 5) is None


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        exact_agreement([1, 2], [1])


def test_kappa_perfect_agreement():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1, 5) == pytest.approx(1.0)


def test_kappa_total_disagreement():
    # Observed weighted disagreement 1.0, chance-expected 0.5 -> kappa = 1 - 1/0.5 = -1.
    assert quadratic_weighted_kappa([1, 5], [5, 1], 1, 5) == pytest.approx(-1.0)


def test_kappa_hand_computed_value():
    # Scale 1-2: observed disagreement 1/4; expected 0.5*0.75 + 0.5*0.25 = 0.5 -> kappa 0.5.
    assert quadratic_weighted_kappa([1, 1, 2, 2], [1, 2, 2, 2], 1, 2) == pytest.approx(0.5)


def test_kappa_penalizes_big_misses_more_than_small_ones():
    human = [1, 2, 3, 4, 5, 3]
    near = quadratic_weighted_kappa([2, 2, 3, 4, 4, 3], human, 1, 5)
    far = quadratic_weighted_kappa([5, 2, 3, 4, 1, 3], human, 1, 5)
    assert near > far


def test_kappa_uses_full_scale_even_if_a_score_is_unused():
    # Same data, but declared scale is 1-5 though only 1-2 appear: must not raise.
    assert quadratic_weighted_kappa([1, 1, 2, 2], [1, 2, 2, 2], 1, 5) == pytest.approx(0.5)


def test_kappa_undefined_when_both_raters_constant():
    assert quadratic_weighted_kappa([3, 3, 3], [3, 3, 3], 1, 5) is None


def test_spearman():
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman_correlation([3, 3, 3], [1, 2, 3]) is None
    assert spearman_correlation([1], [1]) is None


def test_biggest_disagreements_sorted_and_skips_unreviewed_and_matches():
    items = [
        ScoredItem(1, 3, 3),  # match: excluded
        ScoredItem(2, 5, 1),  # gap 4
        ScoredItem(3, 2, 3),  # gap 1
        ScoredItem(4, 4, None),  # unreviewed: excluded
        ScoredItem(5, 1, 4),  # gap 3
    ]
    assert [i.item_id for i in biggest_disagreements(items)] == [2, 5, 3]
    assert [i.item_id for i in biggest_disagreements(items, limit=1)] == [2]


def test_summarize_criterion_only_uses_reviewed_items_for_agreement():
    items = [
        ScoredItem(1, 1, 1),
        ScoredItem(2, 1, 2),
        ScoredItem(3, 2, 2),
        ScoredItem(4, 2, 2),
        ScoredItem(5, 5, None),  # unreviewed: counts toward mean/distribution only
    ]
    s = summarize_criterion(items, 1, 5)
    assert s["n_items"] == 5 and s["n_reviewed"] == 4
    assert s["mean_score"] == pytest.approx(2.2)
    assert s["distribution"][5] == 1
    assert s["exact_agreement"] == 0.75
    assert s["within_one_agreement"] == 1.0
    assert [i.item_id for i in s["biggest_disagreements"]] == [2]


def test_summarize_with_no_reviews_has_undefined_agreement():
    s = summarize_criterion([ScoredItem(1, 3)], 1, 5)
    assert s["n_reviewed"] == 0
    assert s["exact_agreement"] is None and s["quadratic_kappa"] is None
