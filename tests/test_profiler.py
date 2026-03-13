"""
Tests for LDARiskProfiler.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from insurance_lda_risk import LDARiskProfiler


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestProfilerInit:
    def test_default_params(self):
        p = LDARiskProfiler()
        assert p.n_topics == 8
        assert p.exposure_weighted is False
        assert p.max_iter == 50

    def test_custom_params(self):
        p = LDARiskProfiler(n_topics=5, alpha=0.1, eta=0.05, random_state=99)
        assert p.n_topics == 5
        assert p.alpha == 0.1
        assert p.eta == 0.05
        assert p.random_state == 99

    def test_exposure_weighted_raises_not_implemented(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, exposure_weighted=True)
        with pytest.raises(NotImplementedError, match="v0.2"):
            p.fit(small_sparse_matrix)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestProfilerFit:
    def test_fit_returns_self(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        result = p.fit(small_sparse_matrix)
        assert result is p

    def test_attributes_set_after_fit(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        assert p.components_ is not None
        assert p.topic_modality_dist_ is not None
        assert p.perplexity_ is not None
        assert p.lda_ is not None

    def test_components_shape(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        assert p.components_.shape == (3, small_sparse_matrix.shape[1])

    def test_topic_modality_rows_sum_to_1(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        row_sums = p.topic_modality_dist_.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_perplexity_is_positive(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        assert p.perplexity_ > 0

    def test_fit_before_transform_raises(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3)
        with pytest.raises(RuntimeError, match="fitted"):
            p.transform(small_sparse_matrix)

    def test_empty_matrix_raises(self):
        p = LDARiskProfiler(n_topics=2)
        with pytest.raises(ValueError, match="row"):
            p.fit(sp.csr_matrix((0, 10)))

    def test_zero_columns_raises(self):
        p = LDARiskProfiler(n_topics=2)
        with pytest.raises(ValueError, match="column"):
            p.fit(sp.csr_matrix((10, 0)))

    def test_accepts_non_csr_input(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix.tocsc())
        assert p._fitted is True

    def test_accepts_dense_input(self):
        p = LDARiskProfiler(n_topics=2, random_state=0, max_iter=3)
        X_dense = np.random.default_rng(0).integers(0, 2, size=(30, 15)).astype(float)
        p.fit(X_dense)
        assert p._fitted is True


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

class TestProfilerTransform:
    def test_theta_shape(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        theta = p.transform(small_sparse_matrix)
        assert theta.shape == (small_sparse_matrix.shape[0], 3)

    def test_theta_rows_sum_to_1(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        theta = p.transform(small_sparse_matrix)
        assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-5)

    def test_theta_non_negative(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        theta = p.transform(small_sparse_matrix)
        assert np.all(theta >= 0)


# ---------------------------------------------------------------------------
# fit_transform
# ---------------------------------------------------------------------------

class TestProfilerFitTransform:
    def test_fit_transform_matches_fit_then_transform(self, small_sparse_matrix):
        p1 = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        theta1 = p1.fit_transform(small_sparse_matrix)

        p2 = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p2.fit(small_sparse_matrix)
        theta2 = p2.transform(small_sparse_matrix)

        # Both use same random_state so should be identical
        assert np.allclose(theta1, theta2, atol=1e-6)

    def test_fit_transform_returns_ndarray(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        theta = p.fit_transform(small_sparse_matrix)
        assert isinstance(theta, np.ndarray)


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

class TestProfilerHelpers:
    def test_get_dominant_topic_shape(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        theta = p.fit_transform(small_sparse_matrix)
        dominant = p.get_dominant_topic(theta)
        assert dominant.shape == (small_sparse_matrix.shape[0],)

    def test_get_dominant_topic_range(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        theta = p.fit_transform(small_sparse_matrix)
        dominant = p.get_dominant_topic(theta)
        assert np.all(dominant >= 0)
        assert np.all(dominant < 3)

    def test_top_modalities_per_topic(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        feature_names = [f"f_{i}" for i in range(small_sparse_matrix.shape[1])]
        top = p.top_modalities_per_topic(feature_names, top_n=5)
        assert len(top) == 3
        for k in range(3):
            assert k in top
            assert len(top[k]) == 5

    def test_top_modalities_probs_sorted_desc(self, small_sparse_matrix):
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(small_sparse_matrix)
        feature_names = [f"f_{i}" for i in range(small_sparse_matrix.shape[1])]
        top = p.top_modalities_per_topic(feature_names, top_n=5)
        for k in range(3):
            probs = [x[1] for x in top[k]]
            assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestProfilerIntegration:
    def test_fit_on_motor_portfolio(self, encoded_portfolio):
        _, X = encoded_portfolio
        p = LDARiskProfiler(n_topics=5, random_state=0, max_iter=15)
        p.fit(X)
        assert p.components_.shape[0] == 5
        assert p.components_.shape[1] == X.shape[1]

    def test_transform_preserves_n_topics(self, fitted_profiler, encoded_portfolio):
        _, X = encoded_portfolio
        theta = fitted_profiler.transform(X)
        assert theta.shape[1] == fitted_profiler.n_topics_

    def test_n_topics_attribute(self, fitted_profiler):
        assert fitted_profiler.n_topics_ == 5
