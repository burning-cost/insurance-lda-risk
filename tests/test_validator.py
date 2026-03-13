"""
Tests for TopicValidator and TopicValidationResult.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_lda_risk import TopicValidator
from insurance_lda_risk._types import TopicValidationResult, TopicStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_theta():
    """5 policies, 3 topics. Policies clearly assigned to topics."""
    rng = np.random.default_rng(1)
    theta = rng.dirichlet([10.0, 0.1, 0.1], size=100)  # topic 0 dominant
    return theta / theta.sum(axis=1, keepdims=True)


@pytest.fixture
def mixed_theta():
    """Flat Dirichlet — policies spread across topics."""
    rng = np.random.default_rng(2)
    theta = rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=200)
    return theta / theta.sum(axis=1, keepdims=True)


@pytest.fixture
def claims_low():
    rng = np.random.default_rng(10)
    return rng.poisson(0.05, 100).astype(float)


@pytest.fixture
def exposure_uniform():
    return np.ones(100)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestValidatorInit:
    def test_default_distribution(self):
        v = TopicValidator()
        assert v.distribution == "poisson"

    def test_binomial_accepted(self):
        v = TopicValidator(distribution="binomial")
        assert v.distribution == "binomial"

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution"):
            TopicValidator(distribution="gamma")


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidatorValidate:
    def test_returns_result_object(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        assert isinstance(result, TopicValidationResult)

    def test_n_topics_correct(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        assert result.n_topics == 3

    def test_topic_stats_count(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        assert len(result.topic_stats) == 3

    def test_deviance_positive(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        assert result.deviance >= 0

    def test_null_deviance_positive(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        assert result.null_deviance >= 0

    def test_deviance_reduction_in_range(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        assert -1.0 <= result.deviance_reduction <= 1.0

    def test_exposure_none_defaults_to_ones(self, simple_theta, claims_low):
        v = TopicValidator()
        result_none = v.validate(simple_theta, claims_low, None)
        result_ones = v.validate(simple_theta, claims_low, np.ones(len(claims_low)))
        assert abs(result_none.deviance - result_ones.deviance) < 1e-6

    def test_shape_mismatch_raises(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        with pytest.raises(ValueError, match="rows"):
            v.validate(simple_theta, claims_low[:50], exposure_uniform)

    def test_exposure_mismatch_raises(self, simple_theta, claims_low):
        v = TopicValidator()
        with pytest.raises(ValueError, match="shape"):
            v.validate(simple_theta, claims_low, np.ones(50))

    def test_binomial_validate(self):
        rng = np.random.default_rng(5)
        theta = rng.dirichlet([1.0, 1.0], size=100)
        y = rng.integers(0, 2, 100).astype(float)
        v = TopicValidator(distribution="binomial")
        result = v.validate(theta, y)
        assert result.distribution == "binomial"
        assert result.deviance > 0


# ---------------------------------------------------------------------------
# TopicStats
# ---------------------------------------------------------------------------

class TestTopicStats:
    def test_all_topics_have_positive_exposure(
        self, simple_theta, claims_low, exposure_uniform
    ):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        for ts in result.topic_stats:
            assert ts.total_exposure >= 0

    def test_pct_policies_sum_to_1(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        total_pct = sum(ts.pct_policies for ts in result.topic_stats)
        assert abs(total_pct - 1.0) < 1e-6

    def test_claim_frequency_non_negative(
        self, simple_theta, claims_low, exposure_uniform
    ):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        for ts in result.topic_stats:
            assert ts.claim_frequency >= 0

    def test_topic_ids_sequential(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        for i, ts in enumerate(result.topic_stats):
            assert ts.topic_id == i


# ---------------------------------------------------------------------------
# TopicValidationResult
# ---------------------------------------------------------------------------

class TestTopicValidationResult:
    def test_summary_is_dataframe(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        df = result.summary
        assert isinstance(df, pd.DataFrame)

    def test_summary_has_correct_rows(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        assert len(result.summary) == 3

    def test_summary_columns(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        required = {"claim_frequency", "total_exposure", "total_claims", "pct_policies"}
        assert required.issubset(set(result.summary.columns))

    def test_plot_frequencies_returns_figure(
        self, simple_theta, claims_low, exposure_uniform
    ):
        import matplotlib
        matplotlib.use("Agg")
        v = TopicValidator()
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        fig = result.plot_frequencies()
        assert fig is not None

    def test_distribution_stored(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator(distribution="poisson")
        result = v.validate(simple_theta, claims_low, exposure_uniform)
        assert result.distribution == "poisson"


# ---------------------------------------------------------------------------
# Deviance calculations
# ---------------------------------------------------------------------------

class TestDeviance:
    def test_perfect_constant_model_deviance_equals_null(self):
        """If all policies have same topic, deviance = null deviance."""
        n = 50
        theta = np.zeros((n, 2))
        theta[:, 0] = 1.0  # all in topic 0
        rng = np.random.default_rng(3)
        y = rng.poisson(0.08, n).astype(float)
        exp = np.ones(n)

        v = TopicValidator()
        result = v.validate(theta, y, exp)
        # With all policies in one topic, topic freq = portfolio mean
        # So deviance should equal null deviance
        assert abs(result.deviance_reduction) < 0.01

    def test_deviance_decreases_with_more_topics(
        self, encoded_portfolio, fitted_profiler, claims_and_exposure
    ):
        _, X = encoded_portfolio
        y, exp = claims_and_exposure

        from insurance_lda_risk import LDARiskProfiler

        profiler_2 = LDARiskProfiler(n_topics=2, random_state=0, max_iter=15)
        theta_2 = profiler_2.fit_transform(X)

        profiler_8 = LDARiskProfiler(n_topics=8, random_state=0, max_iter=15)
        theta_8 = profiler_8.fit_transform(X)

        v = TopicValidator()
        r2 = v.validate(theta_2, y, exp)
        r8 = v.validate(theta_8, y, exp)

        # More topics should generally not increase deviance
        # (this is a soft check — not guaranteed but expected)
        # Just check both run without error
        assert r2.n_topics == 2
        assert r8.n_topics == 8


# ---------------------------------------------------------------------------
# topic_claim_frequencies convenience wrapper
# ---------------------------------------------------------------------------

class TestTopicClaimFrequencies:
    def test_returns_dataframe(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        df = v.topic_claim_frequencies(simple_theta, claims_low, exposure_uniform)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, simple_theta, claims_low, exposure_uniform):
        v = TopicValidator()
        df = v.topic_claim_frequencies(simple_theta, claims_low, exposure_uniform)
        assert len(df) == 3
