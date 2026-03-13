"""
TopicValidator
==============

Validates the quality of the discovered risk profiles by measuring how well
they separate policies by claim experience.

The core question for a pricing actuary is: "Do these topics mean anything
from a loss perspective?"  If topic 1 policies have a 3x higher claim frequency
than topic 5, the topics are doing useful work.  If every topic has the same
frequency, the segmentation is actuarially worthless regardless of perplexity.

**Deviance metric** (Jamotton & Hainaut 2024, Section 3.3):

For Poisson (frequency models)::

    D = 2 Σ_d [ n_d * log(n_d / ê_d) - (n_d - ê_d) ]  (for policies with n_d > 0)
    + 2 Σ_d ê_d  (for policies with n_d = 0)

where ê_d = exposure_d * λ̂_d and λ̂_d is the estimated claim frequency for
policy d (a weighted average of per-topic frequencies using θ_d as weights).

The null model uses the portfolio mean frequency for every policy.

Examples
--------
>>> import numpy as np
>>> from insurance_lda_risk import TopicValidator

>>> rng = np.random.default_rng(0)
>>> n = 300
>>> theta = rng.dirichlet([0.5] * 5, size=n)
>>> y = rng.poisson(0.1, size=n).astype(float)
>>> exposure = np.ones(n)

>>> validator = TopicValidator(distribution='poisson')
>>> result = validator.validate(theta, y, exposure)
>>> result.n_topics
5
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from insurance_lda_risk._types import TopicStats, TopicValidationResult


class TopicValidator:
    """Validate LDA topics against observed claim outcomes.

    Parameters
    ----------
    distribution : str, default ``'poisson'``
        Loss model to use.  ``'poisson'`` for frequency modelling (standard
        for UK motor / home portfolio analysis).  ``'binomial'`` for binary
        default / conversion targets.

    Raises
    ------
    ValueError
        If ``distribution`` is not ``'poisson'`` or ``'binomial'``.
    """

    _SUPPORTED = ("poisson", "binomial")

    def __init__(self, distribution: str = "poisson") -> None:
        if distribution not in self._SUPPORTED:
            raise ValueError(
                f"distribution must be one of {self._SUPPORTED}, got {distribution!r}"
            )
        self.distribution = distribution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        theta: np.ndarray,
        y_claims: np.ndarray,
        exposure: Optional[np.ndarray] = None,
    ) -> TopicValidationResult:
        """Run topic validation against claim data.

        Parameters
        ----------
        theta : np.ndarray, shape (D, K)
            Policy-topic distribution from LDARiskProfiler.transform().
        y_claims : np.ndarray, shape (D,)
            Observed claim counts (Poisson) or indicators (Binomial).
        exposure : np.ndarray | None, shape (D,)
            Policy exposure.  If None, defaults to ones (each policy = 1 unit).

        Returns
        -------
        TopicValidationResult
            Per-topic statistics plus overall deviance metrics.

        Raises
        ------
        ValueError
            If shape mismatch between theta, y_claims, exposure.
        """
        theta, y, exp = self._validate_inputs(theta, y_claims, exposure)
        n_policies, n_topics = theta.shape

        # Per-topic claim frequency: weighted average of observed data
        topic_stats = self._compute_topic_stats(theta, y, exp, n_topics)

        # Per-policy estimated frequency: θ_d · λ_topics
        topic_freqs = np.array([ts.claim_frequency for ts in topic_stats])
        lambda_hat = theta @ topic_freqs  # shape (D,)

        deviance = self._compute_deviance(y, exp, lambda_hat)
        null_deviance = self._compute_null_deviance(y, exp)
        dev_reduction = (
            (null_deviance - deviance) / null_deviance
            if null_deviance > 0
            else 0.0
        )

        return TopicValidationResult(
            topic_stats=topic_stats,
            deviance=float(deviance),
            null_deviance=float(null_deviance),
            deviance_reduction=float(dev_reduction),
            n_topics=n_topics,
            distribution=self.distribution,
        )

    def topic_claim_frequencies(
        self,
        theta: np.ndarray,
        y_claims: np.ndarray,
        exposure: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Convenience wrapper returning a DataFrame of per-topic frequencies.

        Parameters
        ----------
        theta : np.ndarray, shape (D, K)
        y_claims : np.ndarray, shape (D,)
        exposure : np.ndarray | None

        Returns
        -------
        pd.DataFrame
            Columns: topic, claim_frequency, total_exposure, total_claims,
            pct_policies.
        """
        result = self.validate(theta, y_claims, exposure)
        return result.summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_topic_stats(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        exp: np.ndarray,
        n_topics: int,
    ) -> list[TopicStats]:
        """Compute per-topic exposure, claims, and frequency.

        The soft-membership approach: policy d contributes θ_{d,k} fraction
        of its exposure and claims to topic k.
        """
        dominant = np.argmax(theta, axis=1)
        pct_policies = np.bincount(dominant, minlength=n_topics) / len(dominant)

        stats = []
        for k in range(n_topics):
            weights = theta[:, k]
            topic_exposure = float((weights * exp).sum())
            topic_claims = float((weights * y).sum())
            if self.distribution == "poisson":
                freq = topic_claims / max(topic_exposure, 1e-9)
            else:  # binomial
                total_w = float(weights.sum())
                freq = topic_claims / max(total_w, 1e-9)
            stats.append(
                TopicStats(
                    topic_id=k,
                    claim_frequency=float(freq),
                    total_exposure=float(topic_exposure),
                    total_claims=float(topic_claims),
                    pct_policies=float(pct_policies[k]),
                )
            )
        return stats

    def _compute_deviance(
        self,
        y: np.ndarray,
        exp: np.ndarray,
        lambda_hat: np.ndarray,
    ) -> float:
        """Poisson or Binomial deviance for the topic model predictions."""
        if self.distribution == "poisson":
            return self._poisson_deviance(y, exp, lambda_hat)
        return self._binomial_deviance(y, lambda_hat)

    def _compute_null_deviance(self, y: np.ndarray, exp: np.ndarray) -> float:
        """Deviance of the null (portfolio-mean) model."""
        if self.distribution == "poisson":
            mu = y.sum() / max(exp.sum(), 1e-9)
            lambda_null = np.full(len(y), mu)
            return self._poisson_deviance(y, exp, lambda_null)
        mu = y.mean()
        lambda_null = np.full(len(y), max(mu, 1e-9))
        return self._binomial_deviance(y, lambda_null)

    @staticmethod
    def _poisson_deviance(
        y: np.ndarray,
        exp: np.ndarray,
        lambda_hat: np.ndarray,
    ) -> float:
        """Poisson deviance: 2 Σ [ y log(y/mu) - (y - mu) ] where mu = exp * λ.

        Policies with y=0 contribute 2 * mu_d.
        """
        mu = exp * lambda_hat
        mu = np.maximum(mu, 1e-300)
        # term for y > 0: 2[y log(y/mu) - (y - mu)]
        mask = y > 0
        d = np.zeros_like(y, dtype=float)
        d[mask] = 2.0 * (y[mask] * np.log(y[mask] / mu[mask]) - (y[mask] - mu[mask]))
        d[~mask] = 2.0 * mu[~mask]
        return float(d.sum())

    @staticmethod
    def _binomial_deviance(y: np.ndarray, p_hat: np.ndarray) -> float:
        """Binomial deviance: -2 Σ [ y log(p) + (1-y) log(1-p) ]."""
        p = np.clip(p_hat, 1e-9, 1 - 1e-9)
        d = -2.0 * (y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(d.sum())

    def _validate_inputs(
        self,
        theta: np.ndarray,
        y_claims: np.ndarray,
        exposure: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        y = np.asarray(y_claims, dtype=float)
        if theta.ndim != 2:
            raise ValueError("theta must be 2-D, shape (D, K).")
        if y.ndim != 1:
            raise ValueError("y_claims must be 1-D, shape (D,).")
        if theta.shape[0] != y.shape[0]:
            raise ValueError(
                f"theta and y_claims must have the same number of rows. "
                f"Got theta {theta.shape[0]} vs y {y.shape[0]}."
            )
        if exposure is not None:
            exp = np.asarray(exposure, dtype=float)
            if exp.shape != y.shape:
                raise ValueError(
                    f"exposure shape {exp.shape} does not match y_claims {y.shape}."
                )
        else:
            exp = np.ones(len(y), dtype=float)
        return theta, y, exp
