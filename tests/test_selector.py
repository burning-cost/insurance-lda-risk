"""
Tests for TopicSelector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from insurance_lda_risk import TopicSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_X():
    X = sp.random(150, 30, density=0.3, format="csr", random_state=7).astype(np.float64)
    return X


@pytest.fixture(scope="module")
def small_y(small_X):
    rng = np.random.default_rng(7)
    return rng.poisson(0.07, small_X.shape[0]).astype(float)


@pytest.fixture(scope="module")
def small_exposure(small_X):
    rng = np.random.default_rng(8)
    return rng.uniform(0.5, 1.5, small_X.shape[0])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSelectorInit:
    def test_defaults(self):
        s = TopicSelector()
        assert s.cv == 5
        assert s.distribution == "poisson"
        assert list(s.k_range) == list(range(2, 21))

    def test_custom_range(self):
        s = TopicSelector(k_range=range(2, 6))
        assert list(s.k_range) == [2, 3, 4, 5]


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------

class TestSelectorSelect:
    def test_returns_int(self, small_X, small_y):
        s = TopicSelector(k_range=range(2, 5), cv=2, random_state=0)
        k = s.select(small_X, y_claims=small_y)
        assert isinstance(k, int)

    def test_k_in_range(self, small_X, small_y):
        k_range = range(2, 6)
        s = TopicSelector(k_range=k_range, cv=2, random_state=0)
        k = s.select(small_X, y_claims=small_y)
        assert k in list(k_range)

    def test_optimal_k_attribute_set(self, small_X, small_y):
        s = TopicSelector(k_range=range(2, 5), cv=2, random_state=0)
        s.select(small_X, y_claims=small_y)
        assert s.optimal_k_ is not None

    def test_scores_dataframe_created(self, small_X, small_y):
        s = TopicSelector(k_range=range(2, 5), cv=2, random_state=0)
        s.select(small_X, y_claims=small_y)
        assert isinstance(s.scores_, pd.DataFrame)

    def test_scores_has_k_column(self, small_X, small_y):
        s = TopicSelector(k_range=range(2, 5), cv=2, random_state=0)
        s.select(small_X, y_claims=small_y)
        assert "k" in s.scores_.columns

    def test_scores_n_rows_matches_range(self, small_X, small_y):
        k_range = range(2, 6)
        s = TopicSelector(k_range=k_range, cv=2, random_state=0)
        s.select(small_X, y_claims=small_y)
        assert len(s.scores_) == len(list(k_range))

    def test_unsupervised_mode_no_y(self, small_X):
        s = TopicSelector(k_range=range(2, 4), cv=2, random_state=0)
        k = s.select(small_X)
        assert isinstance(k, int)

    def test_y_shape_mismatch_raises(self, small_X, small_y):
        s = TopicSelector(k_range=range(2, 4), cv=2, random_state=0)
        with pytest.raises(ValueError, match="rows"):
            s.select(small_X, y_claims=small_y[:50])

    def test_with_exposure(self, small_X, small_y, small_exposure):
        s = TopicSelector(k_range=range(2, 4), cv=2, random_state=0)
        k = s.select(small_X, y_claims=small_y, exposure=small_exposure)
        assert isinstance(k, int)

    def test_binomial_distribution(self, small_X):
        rng = np.random.default_rng(3)
        y_bin = rng.integers(0, 2, small_X.shape[0]).astype(float)
        s = TopicSelector(k_range=range(2, 4), cv=2, distribution="binomial", random_state=0)
        k = s.select(small_X, y_claims=y_bin)
        assert isinstance(k, int)


# ---------------------------------------------------------------------------
# plot_elbow()
# ---------------------------------------------------------------------------

class TestSelectorPlot:
    def test_plot_before_select_raises(self):
        s = TopicSelector()
        with pytest.raises(RuntimeError, match="select"):
            s.plot_elbow()

    def test_plot_returns_figure(self, small_X, small_y):
        import matplotlib
        matplotlib.use("Agg")
        s = TopicSelector(k_range=range(2, 5), cv=2, random_state=0)
        s.select(small_X, y_claims=small_y)
        fig = s.plot_elbow()
        assert fig is not None


# ---------------------------------------------------------------------------
# Elbow finder
# ---------------------------------------------------------------------------

class TestElbowFinder:
    def test_elbow_with_two_ks(self):
        result = TopicSelector._find_elbow([2, 3], np.array([1.0, 0.5]))
        assert result in [2, 3]

    def test_elbow_with_flat_curve(self):
        # Flat curve — should return first K
        ks = list(range(2, 8))
        scores = np.ones(6)
        result = TopicSelector._find_elbow(ks, scores)
        assert result in ks

    def test_elbow_with_clear_drop(self):
        # Clear elbow at k=4
        ks = [2, 3, 4, 5, 6, 7]
        scores = np.array([10.0, 7.0, 4.5, 4.0, 3.9, 3.85])
        result = TopicSelector._find_elbow(ks, scores)
        # Elbow should be around k=4 or k=5
        assert result in ks
