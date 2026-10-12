"""Judge metrics as pure functions (no DB, no I/O) so they are easy to unit test.

Agreement metrics compare the judge's score with the human's score on reviewed items only.
Metrics that are mathematically undefined for the given data return None, never NaN.
"""

import math
import warnings
from collections import Counter
from dataclasses import dataclass

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


@dataclass(frozen=True)
class ScoredItem:
    item_id: int
    judge_score: int
    human_score: int | None = None  # None until the item has been reviewed


def mean_score(scores: list[int]) -> float | None:
    return sum(scores) / len(scores) if scores else None


def score_distribution(scores: list[int], scale_min: int, scale_max: int) -> dict[int, int]:
    """Count of items at every point on the scale, including zeros."""
    counts = Counter(scores)
    return {s: counts.get(s, 0) for s in range(scale_min, scale_max + 1)}


def _check_paired(judge: list[int], human: list[int]) -> None:
    if len(judge) != len(human):
        raise ValueError("judge and human scores must be the same length")


def exact_agreement(judge: list[int], human: list[int]) -> float | None:
    _check_paired(judge, human)
    if not judge:
        return None
    return sum(j == h for j, h in zip(judge, human, strict=True)) / len(judge)


def within_one_agreement(judge: list[int], human: list[int]) -> float | None:
    _check_paired(judge, human)
    if not judge:
        return None
    return sum(abs(j - h) <= 1 for j, h in zip(judge, human, strict=True)) / len(judge)


def quadratic_weighted_kappa(
    judge: list[int], human: list[int], scale_min: int, scale_max: int
) -> float | None:
    """Quadratic weighted Cohen's kappa over the full scale.

    Undefined (None) when there is no data or when both raters used a single identical
    score, since chance agreement is then already perfect and the ratio is 0/0.
    """
    _check_paired(judge, human)
    if not judge:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns on the undefined case handled below
        kappa = cohen_kappa_score(
            judge, human, labels=list(range(scale_min, scale_max + 1)), weights="quadratic"
        )
    return None if math.isnan(kappa) else float(kappa)


def spearman_correlation(judge: list[int], human: list[int]) -> float | None:
    """Rank correlation; None with fewer than 2 items or when either side is constant."""
    _check_paired(judge, human)
    if len(judge) < 2 or len(set(judge)) < 2 or len(set(human)) < 2:
        return None
    rho = spearmanr(judge, human).statistic
    return None if math.isnan(rho) else float(rho)


def biggest_disagreements(items: list[ScoredItem], limit: int = 10) -> list[ScoredItem]:
    """Reviewed items with the largest absolute judge-human gap, largest first."""
    reviewed = [i for i in items if i.human_score is not None]
    reviewed.sort(key=lambda i: (-abs(i.judge_score - i.human_score), i.item_id))
    return [i for i in reviewed[:limit] if i.judge_score != i.human_score]


def summarize_criterion(items: list[ScoredItem], scale_min: int, scale_max: int) -> dict:
    """All metrics for one criterion, in the shape the metrics endpoint will return."""
    reviewed = [i for i in items if i.human_score is not None]
    judge = [i.judge_score for i in reviewed]
    human = [i.human_score for i in reviewed]
    return {
        "n_items": len(items),
        "n_reviewed": len(reviewed),
        "mean_score": mean_score([i.judge_score for i in items]),
        "distribution": score_distribution([i.judge_score for i in items], scale_min, scale_max),
        "exact_agreement": exact_agreement(judge, human),
        "within_one_agreement": within_one_agreement(judge, human),
        "quadratic_kappa": quadratic_weighted_kappa(judge, human, scale_min, scale_max),
        "spearman": spearman_correlation(judge, human),
        "biggest_disagreements": biggest_disagreements(items),
    }
