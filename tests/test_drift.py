"""
Tests for PortfolioDrift and DriftResult.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from insurance_lda_risk import LDARiskProfiler, PortfolioDrift
from insurance_lda_risk._types import DriftResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_profiler_5():
    X = sp.random(200, 40, density=0.3, format="csr", random_state=5).astype(np.float64)
    p = LDARiskProfiler(n_topics=5, random_state=5, max_iter=10)
    p.fit(X)
    return p


@pytest.fixture(scope="module")
def theta_t0(fitted_profiler_5):
    rng = np.random.default_rng(10)
    raw = rng.dirichlet([1.0] * 5, size=300)
    return raw / raw.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def theta_t1(fitted_profiler_5):
    # Shifted — topic 0 grows, topic 4 shrinks
    rng = np.random.default_rng(11)
    raw = rng.dirichlet([2.0, 1.0, 1.0, 1.0, 0.5], size=280)
    return raw / raw.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def identical_theta():
    """Two identical theta matrices — zero drift expected."""
    rng = np.random.default_rng(99)
    raw = rng.dirichlet([1.0] * 3, size=100)
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDriftInit:
    def test_default_threshold(self, fitted_profiler_5):
        d = PortfolioDrift(fitted_profiler_5)
        assert d.alert_threshold == 0.05

    def test_custom_threshold(self, fitted_profiler_5):
        d = PortfolioDrift(fitted_profiler_5, alert_threshold=0.1)
        assert d.alert_threshold == 0.1


# ---------------------------------------------------------------------------
# compute_drift()
# ---------------------------------------------------------------------------

class TestComputeDrift:
    def test_returns_drift_result(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        result = d.compute_drift(theta_t0, theta_t1)
        assert isinstance(result, DriftResult)

    def test_jsd_range(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        result = d.compute_drift(theta_t0, theta_t1)
        assert 0.0 <= result.jsd <= 1.0

    def test_identical_distributions_zero_jsd(self, fitted_profiler_5, identical_theta):
        from insurance_lda_risk import LDARiskProfiler
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        X = sp.random(100, 20, density=0.3, format="csr", random_state=0).astype(np.float64)
        p.fit(X)
        d = PortfolioDrift(p)
        result = d.compute_drift(identical_theta, identical_theta)
        assert result.jsd < 1e-8

    def test_per_topic_shift_length(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        result = d.compute_drift(theta_t0, theta_t1)
        assert len(result.per_topic_shift) == 5

    def test_per_topic_shift_sums_to_zero(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        result = d.compute_drift(theta_t0, theta_t1)
        assert abs(result.per_topic_shift.sum()) < 1e-6

    def test_alert_flag_low_threshold(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5, alert_threshold=0.0)
        result = d.compute_drift(theta_t0, theta_t1)
        assert result.alert is True

    def test_alert_flag_high_threshold(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5, alert_threshold=1.0)
        result = d.compute_drift(theta_t0, theta_t1)
        assert result.alert is False

    def test_labels_stored(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        result = d.compute_drift(theta_t0, theta_t1, labels=("2023", "2024"))
        assert result.labels == ("2023", "2024")

    def test_exposure_weighted(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        w0 = np.ones(len(theta_t0))
        w1 = np.ones(len(theta_t1))
        result = d.compute_drift(theta_t0, theta_t1, weights_t0=w0, weights_t1=w1)
        assert isinstance(result, DriftResult)

    def test_wrong_k_raises(self, fitted_profiler_5, theta_t0):
        d = PortfolioDrift(fitted_profiler_5)
        wrong_theta = theta_t0[:, :3]  # 3 topics instead of 5
        with pytest.raises(ValueError, match="topics"):
            d.compute_drift(theta_t0, wrong_theta)

    def test_different_n_policies_ok(self, fitted_profiler_5, theta_t0, theta_t1):
        """D0 and D1 can differ."""
        d = PortfolioDrift(fitted_profiler_5)
        result = d.compute_drift(theta_t0[:100], theta_t1)
        assert isinstance(result, DriftResult)


# ---------------------------------------------------------------------------
# compute_drift_series()
# ---------------------------------------------------------------------------

class TestDriftSeries:
    def test_returns_dataframe(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        df = d.compute_drift_series(
            [theta_t0, theta_t1, theta_t0],
            labels=["2022", "2023", "2024"],
        )
        assert isinstance(df, pd.DataFrame)

    def test_n_rows_is_n_periods_minus_1(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        df = d.compute_drift_series(
            [theta_t0, theta_t1, theta_t0],
            labels=["2022", "2023", "2024"],
        )
        assert len(df) == 2

    def test_columns(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        df = d.compute_drift_series([theta_t0, theta_t1], labels=["A", "B"])
        assert set(df.columns) >= {"period_from", "period_to", "jsd", "alert"}

    def test_too_few_periods_raises(self, fitted_profiler_5, theta_t0):
        d = PortfolioDrift(fitted_profiler_5)
        with pytest.raises(ValueError, match="two periods"):
            d.compute_drift_series([theta_t0], labels=["A"])

    def test_mismatched_labels_raises(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5)
        with pytest.raises(ValueError):
            d.compute_drift_series([theta_t0, theta_t1], labels=["A", "B", "C"])


# ---------------------------------------------------------------------------
# DriftResult methods
# ---------------------------------------------------------------------------

class TestDriftResult:
    def test_plot_shift_returns_figure(self, fitted_profiler_5, theta_t0, theta_t1):
        import matplotlib
        matplotlib.use("Agg")
        d = PortfolioDrift(fitted_profiler_5)
        result = d.compute_drift(theta_t0, theta_t1)
        fig = result.plot_shift()
        assert fig is not None

    def test_alert_threshold_stored(self, fitted_profiler_5, theta_t0, theta_t1):
        d = PortfolioDrift(fitted_profiler_5, alert_threshold=0.07)
        result = d.compute_drift(theta_t0, theta_t1)
        assert result.alert_threshold == 0.07


# ---------------------------------------------------------------------------
# plot_composition()
# ---------------------------------------------------------------------------

class TestDriftCompositionPlot:
    def test_plot_composition_returns_figure(self, fitted_profiler_5, theta_t0, theta_t1):
        import matplotlib
        matplotlib.use("Agg")
        d = PortfolioDrift(fitted_profiler_5)
        fig = d.plot_composition(
            [theta_t0, theta_t1],
            labels=["2023", "2024"],
        )
        assert fig is not None

    def test_plot_composition_multi_period(self, fitted_profiler_5, theta_t0, theta_t1):
        import matplotlib
        matplotlib.use("Agg")
        d = PortfolioDrift(fitted_profiler_5)
        fig = d.plot_composition(
            [theta_t0, theta_t1, theta_t0, theta_t1],
            labels=["2021", "2022", "2023", "2024"],
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# JSD implementation
# ---------------------------------------------------------------------------

class TestJSD:
    def test_jsd_symmetric(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.3, 0.5, 0.2])
        assert abs(PortfolioDrift._jsd(p, q) - PortfolioDrift._jsd(q, p)) < 1e-12

    def test_jsd_self_zero(self):
        p = np.array([0.4, 0.4, 0.2])
        assert PortfolioDrift._jsd(p, p) < 1e-10

    def test_jsd_bounded(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        jsd = PortfolioDrift._jsd(p + 1e-10, q + 1e-10)
        assert 0.0 <= jsd <= 1.0
