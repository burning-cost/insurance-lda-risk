"""
End-to-end integration tests.

These tests run the full pipeline: DataFrame -> encode -> profile -> validate ->
drift.  They confirm the classes compose correctly and the outputs are
numerically sensible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from insurance_lda_risk import (
    InsuranceLDAEncoder,
    LDARiskProfiler,
    PortfolioDrift,
    TopicValidator,
)


class TestFullPipeline:
    def test_encode_profile_validate(self, motor_portfolio, cat_cols, cont_cols, claims_and_exposure):
        y, exp = claims_and_exposure

        enc = InsuranceLDAEncoder()
        X = enc.fit_transform(motor_portfolio, cat_cols=cat_cols, cont_cols=cont_cols, n_bins=5)

        profiler = LDARiskProfiler(n_topics=4, random_state=0, max_iter=15)
        theta = profiler.fit_transform(X)

        assert theta.shape == (len(motor_portfolio), 4)
        assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-5)

        validator = TopicValidator(distribution="poisson")
        result = validator.validate(theta, y, exp)

        assert result.n_topics == 4
        assert result.deviance > 0
        assert result.null_deviance > 0
        assert len(result.topic_stats) == 4

    def test_drift_detection_on_two_years(self, motor_portfolio, cat_cols, cont_cols):
        rng = np.random.default_rng(20)
        n = len(motor_portfolio)

        enc = InsuranceLDAEncoder()
        X = enc.fit_transform(motor_portfolio, cat_cols=cat_cols, cont_cols=cont_cols)

        profiler = LDARiskProfiler(n_topics=4, random_state=0, max_iter=10)
        profiler.fit(X)
        theta = profiler.transform(X)

        # Simulate a shifted portfolio (year 2)
        idx_shift = rng.choice(n, size=int(n * 0.7), replace=False)
        motor_2 = motor_portfolio.iloc[idx_shift].reset_index(drop=True)
        X_2 = enc.transform(motor_2)
        theta_2 = profiler.transform(X_2)

        drift = PortfolioDrift(profiler)
        result = drift.compute_drift(theta, theta_2, labels=("2023", "2024"))

        assert 0 <= result.jsd <= 1
        assert isinstance(result.alert, bool)

    def test_topic_descriptions_with_encoder(self, motor_portfolio, cat_cols, cont_cols):
        enc = InsuranceLDAEncoder()
        X = enc.fit_transform(motor_portfolio, cat_cols=cat_cols, cont_cols=cont_cols)

        profiler = LDARiskProfiler(n_topics=3, random_state=0, max_iter=10)
        profiler.fit(X)

        top = profiler.top_modalities_per_topic(enc.feature_names_, top_n=5)
        assert len(top) == 3
        for k, mods in top.items():
            for name, prob in mods:
                assert name in enc.vocabulary_
                assert 0 <= prob <= 1


class TestNumericalStability:
    def test_all_zero_claims(self, encoded_portfolio, fitted_profiler):
        _, X = encoded_portfolio
        theta = fitted_profiler.transform(X)
        y_zeros = np.zeros(len(theta))

        v = TopicValidator()
        result = v.validate(theta, y_zeros, np.ones(len(y_zeros)))
        # Zero claims -> null deviance is zero, model deviance near zero
        assert result.deviance >= 0

    def test_single_topic(self):
        # K=1 edge case
        X = sp.random(50, 20, density=0.4, format="csr", random_state=1).astype(np.float64)
        p = LDARiskProfiler(n_topics=1, random_state=0, max_iter=5)
        theta = p.fit_transform(X)
        assert theta.shape == (50, 1)
        assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-5)

    def test_single_policy(self):
        X = sp.random(1, 15, density=0.5, format="csr", random_state=2).astype(np.float64)
        # Ensure at least one nonzero
        if X.nnz == 0:
            X[0, 0] = 1.0
        # Need to fit on bigger matrix, then transform single
        X_train = sp.random(100, 15, density=0.4, format="csr", random_state=3).astype(np.float64)
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        p.fit(X_train)
        theta = p.transform(X)
        assert theta.shape == (1, 3)

    def test_high_exposure_variance(self):
        rng = np.random.default_rng(30)
        X = sp.random(100, 25, density=0.3, format="csr", random_state=30).astype(np.float64)
        p = LDARiskProfiler(n_topics=3, random_state=0, max_iter=5)
        theta = p.fit_transform(X)

        y = rng.poisson(0.05, 100).astype(float)
        # Highly variable exposures
        exp = rng.uniform(0.001, 100.0, 100)
        v = TopicValidator()
        result = v.validate(theta, y, exp)
        assert np.isfinite(result.deviance)


class TestImports:
    def test_all_public_classes_importable(self):
        from insurance_lda_risk import (
            InsuranceLDAEncoder,
            LDARiskProfiler,
            PortfolioDrift,
            TopicSelector,
            TopicValidator,
        )
        assert InsuranceLDAEncoder is not None
        assert LDARiskProfiler is not None
        assert PortfolioDrift is not None
        assert TopicSelector is not None
        assert TopicValidator is not None

    def test_type_imports(self):
        from insurance_lda_risk import DriftResult, TopicStats, TopicValidationResult
        assert DriftResult is not None
        assert TopicStats is not None
        assert TopicValidationResult is not None

    def test_version_string(self):
        import insurance_lda_risk
        assert hasattr(insurance_lda_risk, "__version__")
        assert insurance_lda_risk.__version__ == "0.1.0"
