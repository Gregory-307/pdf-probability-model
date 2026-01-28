"""
Automatic distribution selection via proper scoring rules.
"""

from dataclasses import dataclass
from typing import Sequence, Literal
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .scoring import crps_from_samples, crps_normal, log_score
from .significance import paired_t_test, determine_confidence
from ..utilities import fit_nig, fit_student_t, fit_normal


@dataclass(frozen=True)
class DiscoveryResult:
    """
    Result of automatic distribution selection.

    Attributes:
        best: Name of best distribution
        confidence: Confidence level ('high', 'medium', 'low')
        scores: Mean score for each distribution
        std_scores: Standard deviation of scores
        pairwise_pvalues: p-values for pairwise comparisons
        best_params: Fitted parameters for best distribution
    """
    best: str
    confidence: Literal["high", "medium", "low"]
    scores: dict[str, float]
    std_scores: dict[str, float]
    pairwise_pvalues: dict[tuple[str, str], float]
    best_params: object

    def summary(self) -> str:
        """Return formatted summary table."""
        lines = [
            "Distribution Selection Results",
            "=" * 50,
            f"Best: {self.best} (confidence: {self.confidence})",
            "",
            "Scores (lower is better):",
        ]

        for name in sorted(self.scores.keys()):
            score = self.scores[name]
            std = self.std_scores.get(name, 0.0)
            marker = " *" if name == self.best else ""
            lines.append(f"  {name:15s}: {score:.4f} (+/- {std:.4f}){marker}")

        return "\n".join(lines)


def discover(
    data: NDArray[np.float64],
    candidates: Sequence[str] = ("normal", "student_t", "nig"),
    scoring: Sequence[Literal["crps", "log_score"]] = ("crps",),
    test_fraction: float = 0.2,
    cv_folds: int = 5,
    significance_level: float = 0.05,
    n_samples: int = 2000,
) -> DiscoveryResult:
    """
    Determine best distribution family with statistical rigor.

    Process:
    1. For each candidate distribution, fit to training data
    2. Compute CRPS and/or log_score on test data
    3. Run k-fold cross-validation for robust scoring
    4. Perform pairwise paired t-tests between best and others
    5. Assign confidence level based on p-values and score gaps

    Args:
        data: Array of observations (e.g., returns)
        candidates: Distribution names to compare
        scoring: Scoring metrics to use ('crps' and/or 'log_score')
        test_fraction: Fraction of data to use for testing
        cv_folds: Number of cross-validation folds
        significance_level: p-value threshold for significance
        n_samples: Monte Carlo samples for CRPS calculation

    Returns:
        DiscoveryResult with best distribution and statistics

    Example:
        >>> result = discover(returns, candidates=['normal', 'student_t', 'nig'])
        >>> print(result.summary())
        >>> if result.confidence == 'high':
        ...     model = TemporalModel(distribution=result.best)
    """
    data = np.asarray(data)
    n = len(data)
    fold_size = n // cv_folds

    # Collect scores for each distribution across folds
    fold_scores: dict[str, list[float]] = {c: [] for c in candidates}

    for fold in range(cv_folds):
        # Create train/test split
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        test_data = data[test_start:test_end]
        train_data = np.concatenate([data[:test_start], data[test_end:]])

        # Score each candidate
        for dist_name in candidates:
            scores = _score_distribution(
                dist_name, train_data, test_data, scoring, n_samples
            )
            fold_scores[dist_name].append(np.mean(scores))

    # Compute mean and std scores
    mean_scores = {c: float(np.mean(fold_scores[c])) for c in candidates}
    std_scores = {c: float(np.std(fold_scores[c])) for c in candidates}

    # Find best
    best = min(mean_scores.keys(), key=lambda k: mean_scores[k])

    # Compute pairwise p-values
    pairwise_pvalues: dict[tuple[str, str], float] = {}
    best_scores_array = np.array(fold_scores[best])

    for other in candidates:
        if other != best:
            other_scores_array = np.array(fold_scores[other])
            pvalue = paired_t_test(best_scores_array, other_scores_array)
            pairwise_pvalues[(best, other)] = pvalue

    # Determine confidence
    if len(candidates) >= 2:
        sorted_dists = sorted(mean_scores.keys(), key=lambda k: mean_scores[k])
        second_best = sorted_dists[1]
        confidence = determine_confidence(
            np.array(fold_scores[best]),
            np.array(fold_scores[second_best]),
            significance_level,
        )
    else:
        confidence = "n/a"

    # Fit best distribution on full data
    best_params = _fit_distribution(best, data)

    return DiscoveryResult(
        best=best,
        confidence=confidence,
        scores=mean_scores,
        std_scores=std_scores,
        pairwise_pvalues=pairwise_pvalues,
        best_params=best_params,
    )


def _fit_distribution(dist_name: str, data: NDArray[np.float64]) -> object:
    """Fit distribution to data."""
    if dist_name == "normal":
        return fit_normal(data)
    elif dist_name == "student_t":
        return fit_student_t(data)
    elif dist_name == "nig":
        return fit_nig(data)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


def _score_distribution(
    dist_name: str,
    train_data: NDArray[np.float64],
    test_data: NDArray[np.float64],
    scoring: Sequence[str],
    n_samples: int,  # kept for backward compatibility, not used for numerical methods
) -> list[float]:
    """Compute scores for a distribution on test data using numerical integration."""
    from ..scoring.rules import crps as crps_numerical

    # Fit on training data
    params = _fit_distribution(dist_name, train_data)

    # Get distribution instance
    if dist_name == "normal":
        from ..distributions.normal import NormalDistribution
        dist = NormalDistribution()
    elif dist_name == "student_t":
        from ..distributions.student_t import StudentTDistribution
        dist = StudentTDistribution()
    elif dist_name == "nig":
        from ..distributions.nig import NIGDistribution
        dist = NIGDistribution()
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

    scores = []

    for y in test_data:
        score = 0.0

        if "crps" in scoring:
            if dist_name == "normal":
                # Use closed-form for Normal (fastest)
                score += crps_normal(y, params.mu_0, params.sigma_0)
            else:
                # Use numerical integration for others
                score += crps_numerical(dist, params, y, t=0.0)

        if "log_score" in scoring:
            pdf_val = dist.pdf(np.array([y]), t=0.0, params=params)[0]
            score += log_score(y, pdf_val)

        scores.append(score)

    return scores
