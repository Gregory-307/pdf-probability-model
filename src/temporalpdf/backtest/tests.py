"""
Statistical tests for VaR model validation.

References:
    Kupiec, P.H. (1995). Techniques for Verifying the Accuracy of
    Risk Measurement Models. Journal of Derivatives, 3(2), 73-84.

    Christoffersen, P.F. (1998). Evaluating Interval Forecasts.
    International Economic Review, 39(4), 841-862.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def kupiec_test(
    exceedances: NDArray[np.bool_],
    alpha: float,
    significance: float = 0.05,
) -> tuple[float, float, bool]:
    """
    Kupiec unconditional coverage test.

    Tests whether the observed exceedance rate equals the expected rate (alpha).

    H0: True exceedance rate equals alpha
    H1: True exceedance rate != alpha

    Args:
        exceedances: Boolean array where True = VaR exceeded
        alpha: Expected exceedance rate (e.g., 0.05 for 95% VaR)
        significance: p-value threshold for rejection

    Returns:
        (LR statistic, p-value, reject_null)
    """
    n = len(exceedances)
    n_exc = np.sum(exceedances)

    if n_exc == 0 or n_exc == n:
        return 0.0, 0.0, True  # Edge case

    p_hat = n_exc / n

    # Likelihood ratio
    lr = -2 * (
        n_exc * np.log(alpha / p_hat) +
        (n - n_exc) * np.log((1 - alpha) / (1 - p_hat))
    )

    p_value = 1 - stats.chi2.cdf(lr, 1)
    reject = p_value < significance

    return float(lr), float(p_value), reject


def christoffersen_test(
    exceedances: NDArray[np.bool_],
    significance: float = 0.05,
) -> tuple[float, float, bool]:
    """
    Christoffersen independence test.

    Tests whether exceedances are independent (no clustering).

    H0: Exceedances are independent
    H1: Exceedances cluster (not independent)

    Args:
        exceedances: Boolean array where True = VaR exceeded
        significance: p-value threshold for rejection

    Returns:
        (LR statistic, p-value, reject_null)
    """
    exceedances = np.asarray(exceedances, dtype=int)

    # Count transitions
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, len(exceedances)):
        prev, curr = exceedances[i-1], exceedances[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Edge cases
    if n00 + n01 == 0 or n10 + n11 == 0:
        return 0.0, 1.0, False

    # Transition probabilities
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Edge cases for log
    if pi01 == 0 or pi01 == 1 or pi11 == 0 or pi11 == 1 or pi == 0 or pi == 1:
        return 0.0, 1.0, False

    # Likelihood ratio
    lr = -2 * (
        n00 * np.log(1 - pi) + n01 * np.log(pi) +
        n10 * np.log(1 - pi) + n11 * np.log(pi) -
        n00 * np.log(1 - pi01) - n01 * np.log(pi01) -
        n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )

    p_value = 1 - stats.chi2.cdf(abs(lr), 1)
    reject = p_value < significance

    return float(lr), float(p_value), reject


def conditional_coverage_test(
    exceedances: NDArray[np.bool_],
    alpha: float,
    significance: float = 0.05,
) -> tuple[float, float, bool]:
    """
    Christoffersen conditional coverage test.

    Combined test for correct coverage AND independence.

    H0: Coverage is correct AND exceedances are independent
    H1: Coverage is wrong OR exceedances cluster

    Args:
        exceedances: Boolean array where True = VaR exceeded
        alpha: Expected exceedance rate
        significance: p-value threshold for rejection

    Returns:
        (LR statistic, p-value, reject_null)
    """
    lr_uc, _, _ = kupiec_test(exceedances, alpha, significance)
    lr_ind, _, _ = christoffersen_test(exceedances, significance)

    lr_cc = lr_uc + lr_ind
    p_value = 1 - stats.chi2.cdf(lr_cc, 2)
    reject = p_value < significance

    return float(lr_cc), float(p_value), reject
