"""
Statistical significance tests for distribution comparison.
"""

from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy import stats


def paired_t_test(scores_a: NDArray[np.float64], scores_b: NDArray[np.float64]) -> float:
    """
    Paired t-test for comparing two distributions' scores.

    H0: Mean scores are equal
    H1: Mean scores are different

    Args:
        scores_a: Scores for distribution A (e.g., CRPS values)
        scores_b: Scores for distribution B

    Returns:
        p-value (small = significant difference)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")

    _, pvalue = stats.ttest_rel(scores_a, scores_b)
    return float(pvalue)


def determine_confidence(
    best_scores: NDArray[np.float64],
    second_scores: NDArray[np.float64],
    significance_level: float = 0.05,
) -> Literal["high", "medium", "low"]:
    """
    Determine confidence in distribution selection.

    Confidence levels:
    - high: p < significance_level AND gap > 10%
    - medium: p < significance_level
    - low: p >= significance_level (not statistically significant)

    Args:
        best_scores: Scores for best distribution
        second_scores: Scores for second-best distribution
        significance_level: p-value threshold for significance

    Returns:
        Confidence level as string
    """
    pvalue = paired_t_test(best_scores, second_scores)

    # Calculate relative gap
    mean_best = np.mean(best_scores)
    mean_second = np.mean(second_scores)
    gap = (mean_second - mean_best) / abs(mean_best) if mean_best != 0 else 0

    if pvalue < significance_level and gap > 0.10:
        return "high"
    elif pvalue < significance_level:
        return "medium"
    else:
        return "low"


def diebold_mariano_test(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    h: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    More robust than paired t-test for autocorrelated scores
    (e.g., from rolling forecasts).

    Args:
        scores_a: Scores for model A
        scores_b: Scores for model B
        h: Forecast horizon (for HAC standard error adjustment)

    Returns:
        (DM statistic, p-value)
    """
    d = scores_a - scores_b  # Score differences
    n = len(d)
    mean_d = np.mean(d)

    # Newey-West HAC standard error
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[:-k], d[k:])[0, 1]
        gamma_sum += (1 - k / h) * gamma_k

    var_d = (gamma_0 + 2 * gamma_sum) / n
    se_d = np.sqrt(max(var_d, 1e-10))

    dm_stat = mean_d / se_d
    pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(pvalue)
