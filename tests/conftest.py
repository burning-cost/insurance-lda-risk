"""
Shared fixtures for insurance-lda-risk tests.

All synthetic data is generated deterministically via np.random.default_rng.
Portfolio is a simplified UK motor portfolio with:
- vehicle_group: A, B, C, D, E
- area: urban, suburban, rural
- age_band: 17-25, 26-35, 36-50, 51-65, 66+
- vehicle_age: continuous 0-20 years
- annual_mileage: continuous 1000-30000
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def motor_portfolio(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic UK motor portfolio, 500 policies."""
    n = 500
    vehicle_groups = rng.choice(["A", "B", "C", "D", "E"], size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    areas = rng.choice(["urban", "suburban", "rural"], size=n, p=[0.5, 0.3, 0.2])
    age_bands = rng.choice(
        ["17-25", "26-35", "36-50", "51-65", "66+"],
        size=n,
        p=[0.1, 0.2, 0.35, 0.25, 0.1],
    )
    vehicle_age = rng.uniform(0, 20, size=n)
    annual_mileage = rng.lognormal(mean=9.5, sigma=0.5, size=n)

    # Introduce ~3% missing values in vehicle_group
    missing_mask = rng.random(n) < 0.03
    vehicle_groups = np.where(missing_mask, None, vehicle_groups).tolist()

    return pd.DataFrame(
        {
            "vehicle_group": vehicle_groups,
            "area": areas,
            "age_band": age_bands,
            "vehicle_age": vehicle_age,
            "annual_mileage": annual_mileage,
        }
    )


@pytest.fixture(scope="session")
def cat_cols() -> list[str]:
    return ["vehicle_group", "area", "age_band"]


@pytest.fixture(scope="session")
def cont_cols() -> list[str]:
    return ["vehicle_age", "annual_mileage"]


@pytest.fixture(scope="session")
def claims_and_exposure(rng: np.random.Generator, motor_portfolio: pd.DataFrame):
    """Synthetic Poisson claims and exposure for the motor portfolio."""
    n = len(motor_portfolio)
    exposure = rng.uniform(0.5, 1.5, size=n)
    freq = 0.06 + 0.04 * (motor_portfolio["age_band"] == "17-25").astype(float).values
    claims = rng.poisson(freq * exposure).astype(float)
    return claims, exposure


@pytest.fixture(scope="session")
def encoded_portfolio(motor_portfolio, cat_cols, cont_cols):
    """Fitted encoder + sparse matrix for the motor portfolio."""
    from insurance_lda_risk import InsuranceLDAEncoder

    enc = InsuranceLDAEncoder()
    X = enc.fit_transform(motor_portfolio, cat_cols=cat_cols, cont_cols=cont_cols, n_bins=5)
    return enc, X


@pytest.fixture(scope="session")
def fitted_profiler(encoded_portfolio):
    """LDARiskProfiler fitted on the motor portfolio."""
    from insurance_lda_risk import LDARiskProfiler

    _, X = encoded_portfolio
    profiler = LDARiskProfiler(n_topics=5, random_state=0, max_iter=10)
    profiler.fit(X)
    return profiler


@pytest.fixture(scope="session")
def theta(encoded_portfolio, fitted_profiler):
    """Policy-topic distribution for the motor portfolio."""
    _, X = encoded_portfolio
    return fitted_profiler.transform(X)


@pytest.fixture
def small_sparse_matrix(rng: np.random.Generator) -> sp.csr_matrix:
    """Small random sparse matrix for unit tests."""
    return sp.random(50, 20, density=0.4, format="csr", random_state=7).astype(np.float64)
