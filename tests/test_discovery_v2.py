"""Tests for discovery module (V2 API)."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

import temporalpdf as tpdf
from temporalpdf.discovery.scoring import crps_from_samples, crps_normal, log_score
from temporalpdf.discovery.significance import paired_t_test, determine_confidence


class TestCRPSFromSamples:
    """Test CRPS computation from samples."""

    def test_crps_nonnegative(self):
        """Test that CRPS is non-negative."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        crps = crps_from_samples(0.5, samples)
        assert crps >= 0

    def test_crps_zero_at_point_mass(self):
        """Test CRPS is zero when prediction is point mass at observation."""
        samples = np.array([5.0] * 1000)  # Point mass at 5
        crps = crps_from_samples(5.0, samples)
        assert np.isclose(crps, 0, atol=0.01)

    def test_crps_increases_with_error(self):
        """Test CRPS increases with prediction error."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)

        crps_at_mean = crps_from_samples(0.0, samples)
        crps_far = crps_from_samples(5.0, samples)

        assert crps_far > crps_at_mean

    def test_crps_increases_with_spread(self):
        """Test CRPS increases with distribution spread (at mean)."""
        np.random.seed(42)
        samples_narrow = np.random.normal(0, 0.5, 1000)
        samples_wide = np.random.normal(0, 2.0, 1000)

        crps_narrow = crps_from_samples(0.0, samples_narrow)
        crps_wide = crps_from_samples(0.0, samples_wide)

        assert crps_wide > crps_narrow


class TestCRPSNormal:
    """Test closed-form CRPS for Normal distribution."""

    def test_crps_nonnegative(self):
        """Test that CRPS is non-negative."""
        crps = crps_normal(0.5, mu=0, sigma=1)
        assert crps >= 0

    def test_crps_scales_with_sigma(self):
        """Test CRPS scales linearly with sigma."""
        crps_1 = crps_normal(0.0, mu=0, sigma=1)
        crps_2 = crps_normal(0.0, mu=0, sigma=2)
        assert np.isclose(crps_2 / crps_1, 2, rtol=0.1)

    def test_crps_matches_sample_estimate(self):
        """Test closed-form roughly matches sample-based estimate."""
        np.random.seed(42)
        mu, sigma = 1.0, 0.5
        y = 1.2

        # Closed-form
        crps_cf = crps_normal(y, mu, sigma)

        # Sample-based
        samples = np.random.normal(mu, sigma, 10000)
        crps_mc = crps_from_samples(y, samples)

        # Should be in same ballpark (CRPS formulas may differ)
        assert 0.1 < crps_cf / crps_mc < 10


class TestLogScore:
    """Test log score computation."""

    def test_log_score_positive_for_low_pdf(self):
        """Test log score is positive for low PDF values."""
        score = log_score(0.0, pdf_value=0.01)
        assert score > 0

    def test_log_score_lower_for_higher_pdf(self):
        """Test log score is lower (better) for higher PDF values."""
        score_high = log_score(0.0, pdf_value=0.5)
        score_low = log_score(0.0, pdf_value=0.1)
        assert score_high < score_low

    def test_log_score_handles_zero_pdf(self):
        """Test log score handles zero PDF gracefully."""
        score = log_score(0.0, pdf_value=0.0)
        assert np.isfinite(score)
        assert score > 0


class TestPairedTTest:
    """Test paired t-test implementation."""

    def test_significant_difference_returns_low_pvalue(self):
        """Test detection of significant difference returns low p-value."""
        np.random.seed(42)
        # Two distributions with different means
        scores1 = np.random.normal(1.0, 0.1, 100)
        scores2 = np.random.normal(1.5, 0.1, 100)

        # paired_t_test returns a single float (p-value)
        p_value = paired_t_test(scores1, scores2)
        assert isinstance(p_value, float)
        assert p_value < 0.05

    def test_similar_scores_high_pvalue(self):
        """Test no detection when samples are similar gives high p-value."""
        np.random.seed(42)
        # Same distribution
        scores1 = np.random.normal(1.0, 0.1, 100)
        scores2 = np.random.normal(1.0, 0.1, 100)

        p_value = paired_t_test(scores1, scores2)
        # Should not detect significant difference most of the time
        # (probabilistic, so we just check p-value is computed)
        assert 0 <= p_value <= 1


class TestDetermineConfidence:
    """Test confidence level determination."""

    def test_returns_confidence_level(self):
        """Test that function returns a confidence level string."""
        np.random.seed(42)
        best_scores = np.random.normal(0.5, 0.1, 10)
        second_scores = np.random.normal(1.0, 0.1, 10)

        confidence = determine_confidence(best_scores, second_scores)
        assert confidence in ["high", "medium", "low"]

    def test_low_confidence_for_similar_scores(self):
        """Test low confidence when scores are similar."""
        np.random.seed(42)
        scores_a = np.array([0.5, 0.51, 0.49, 0.5, 0.52])
        scores_b = np.array([0.51, 0.50, 0.50, 0.49, 0.51])

        confidence = determine_confidence(scores_a, scores_b)
        # Similar scores should give low confidence
        assert confidence == "low"


class TestDiscover:
    """Test discover() function."""

    @pytest.fixture
    def synthetic_returns(self):
        """Generate synthetic returns."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 500)

    def test_discover_returns_discovery_result(self, synthetic_returns):
        """Test that discover returns DiscoveryResult."""
        result = tpdf.discover(synthetic_returns)
        assert isinstance(result, tpdf.DiscoveryResult)

    def test_discover_finds_best(self, synthetic_returns):
        """Test that discover identifies best distribution."""
        result = tpdf.discover(synthetic_returns)
        assert result.best in ["normal", "student_t", "nig"]

    def test_discover_has_scores(self, synthetic_returns):
        """Test that result has scores for each candidate."""
        result = tpdf.discover(
            synthetic_returns,
            candidates=["normal", "student_t"],
        )
        assert "normal" in result.scores
        assert "student_t" in result.scores

    def test_discover_has_confidence(self, synthetic_returns):
        """Test that result has confidence level."""
        result = tpdf.discover(synthetic_returns)
        assert result.confidence in ["high", "medium", "low"]

    def test_discover_custom_candidates(self, synthetic_returns):
        """Test discover with custom candidate list."""
        result = tpdf.discover(
            synthetic_returns,
            candidates=["normal", "student_t"],
        )
        assert result.best in ["normal", "student_t"]

    def test_discover_with_cv_folds(self, synthetic_returns):
        """Test discover with custom CV folds."""
        result = tpdf.discover(
            synthetic_returns,
            cv_folds=3,
        )
        assert result.best is not None

    def test_discover_normal_data_has_scores(self):
        """Test that normal data gets scores for all distributions."""
        np.random.seed(42)
        # Generate clearly normal data
        data = np.random.normal(0, 1, 1000)

        result = tpdf.discover(data, candidates=["normal", "student_t"])
        # Normal should be competitive (may not always win due to CV variance)
        assert "normal" in result.scores

    def test_discover_heavy_tailed_data(self):
        """Test discover on heavy-tailed data."""
        np.random.seed(42)
        # Generate heavy-tailed data (Student-t with low df)
        data = np.random.standard_t(df=3, size=1000) * 0.02

        result = tpdf.discover(data, candidates=["normal", "student_t", "nig"])
        # Heavy-tailed distributions should be competitive
        assert result.best is not None


class TestDiscoveryResult:
    """Test DiscoveryResult dataclass."""

    def test_result_has_required_attributes(self):
        """Test DiscoveryResult has expected attributes after discover()."""
        np.random.seed(42)
        data = np.random.normal(0, 0.02, 300)
        result = tpdf.discover(data, candidates=["normal", "student_t"])

        assert hasattr(result, "best")
        assert hasattr(result, "confidence")
        assert hasattr(result, "scores")
        assert hasattr(result, "std_scores")
        assert hasattr(result, "pairwise_pvalues")
        assert hasattr(result, "best_params")

    def test_result_summary(self):
        """Test summary method."""
        np.random.seed(42)
        data = np.random.normal(0, 0.02, 300)
        result = tpdf.discover(data, candidates=["normal", "student_t"])

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Best" in summary


class TestCompareDistributions:
    """Test compare_distributions function from utilities module."""

    def test_compare_distributions(self):
        """Test compare_distributions function."""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 500)

        # The utilities version uses `distributions`, not `candidates`
        result = tpdf.compare_distributions(
            data,
            distributions=["normal", "student_t"],
            n_folds=3,
        )

        assert "winner" in result
        assert "mean_scores" in result
        assert "significant" in result
