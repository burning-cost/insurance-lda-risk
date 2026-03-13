"""
PortfolioDrift
==============

Detect and visualise year-on-year shifts in the portfolio's risk composition.

A UK personal lines portfolio is rarely static.  Aggregators, underwriting
appetite changes, and macro trends all shift which risk archetypes make up the
book.  If your renewal pricing is calibrated against last year's mix, and this
year's mix has shifted materially towards higher-risk topics, you are under-
pricing the current book.

This class quantifies portfolio drift as the Jensen-Shannon divergence (JSD)
between the portfolio-level topic distributions across two periods.  JSD is
symmetric, bounded [0, 1], and well-defined even when some topics have zero
weight in one period.

The portfolio-level topic distribution for period t is::

    π_t = (1/D_t) Σ_d θ_{d,t}     (simple average over policies)

or, with exposure weights::

    π_t = Σ_d (e_d * θ_{d,t}) / Σ_d e_d

JSD between π_0 and π_1::

    JSD(π_0 || π_1) = 0.5 * KL(π_0 || m) + 0.5 * KL(π_1 || m)
    where m = 0.5 * (π_0 + π_1)

    KL(p || q) = Σ_k p_k * log(p_k / q_k)

JSD ranges from 0 (no drift) to 1 (completely disjoint composition, rare in
practice).  A practical alert threshold is JSD > 0.05 for annual monitoring.

Examples
--------
>>> import numpy as np
>>> from insurance_lda_risk import LDARiskProfiler, PortfolioDrift
>>> import scipy.sparse as sp

>>> rng = np.random.default_rng(0)
>>> X_t0 = sp.random(500, 60, density=0.3, format='csr', random_state=0)
>>> profiler = LDARiskProfiler(n_topics=5, random_state=0)
>>> profiler.fit(X_t0)  # doctest: +SKIP
>>> theta_t0 = profiler.transform(X_t0)  # doctest: +SKIP

>>> X_t1 = sp.random(480, 60, density=0.3, format='csr', random_state=1)
>>> theta_t1 = profiler.transform(X_t1)  # doctest: +SKIP

>>> drift_detector = PortfolioDrift(profiler)
>>> result = drift_detector.compute_drift(theta_t0, theta_t1,
...                                       labels=("2023", "2024"))  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from insurance_lda_risk._types import DriftResult
from insurance_lda_risk.profiler import LDARiskProfiler


class PortfolioDrift:
    """Measure composition drift in a portfolio between two periods.

    Parameters
    ----------
    profiler : LDARiskProfiler
        A fitted profiler.  Provides topic count and labelling context.
    alert_threshold : float, default 0.05
        JSD threshold above which ``DriftResult.alert`` is set to True.
        0.05 corresponds roughly to a noticeable composition shift in practice.

    Attributes
    ----------
    profiler : LDARiskProfiler
    alert_threshold : float
    """

    def __init__(
        self,
        profiler: LDARiskProfiler,
        alert_threshold: float = 0.05,
    ) -> None:
        self.profiler = profiler
        self.alert_threshold = alert_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_drift(
        self,
        theta_t0: np.ndarray,
        theta_t1: np.ndarray,
        weights_t0: Optional[np.ndarray] = None,
        weights_t1: Optional[np.ndarray] = None,
        labels: tuple[str, str] = ("t0", "t1"),
    ) -> DriftResult:
        """Compute Jensen-Shannon divergence between two period compositions.

        Parameters
        ----------
        theta_t0 : np.ndarray, shape (D0, K)
            Policy-topic distributions for period 0.
        theta_t1 : np.ndarray, shape (D1, K)
            Policy-topic distributions for period 1.  D1 may differ from D0.
        weights_t0 : np.ndarray | None, shape (D0,)
            Optional exposure or premium weights for period 0 policies.
            If None, each policy contributes equally.
        weights_t1 : np.ndarray | None, shape (D1,)
            Optional weights for period 1 policies.
        labels : tuple[str, str], default ``('t0', 't1')``
            Human-readable period names for plots and reports.

        Returns
        -------
        DriftResult
            JSD, per-topic shift, and alert flag.

        Raises
        ------
        ValueError
            If theta shapes are incompatible with the fitted profiler's K.
        """
        theta_t0 = np.asarray(theta_t0, dtype=float)
        theta_t1 = np.asarray(theta_t1, dtype=float)
        k = self.profiler.n_topics_
        self._validate_theta(theta_t0, k, "theta_t0")
        self._validate_theta(theta_t1, k, "theta_t1")

        pi0 = self._portfolio_distribution(theta_t0, weights_t0)
        pi1 = self._portfolio_distribution(theta_t1, weights_t1)

        jsd = float(self._jsd(pi0, pi1))
        shift = pd.Series(pi1 - pi0, name="shift")

        return DriftResult(
            jsd=jsd,
            per_topic_shift=shift,
            alert=jsd > self.alert_threshold,
            alert_threshold=self.alert_threshold,
            labels=labels,
        )

    def compute_drift_series(
        self,
        thetas_by_period: list[np.ndarray],
        labels: list[str],
        weights_by_period: Optional[list[Optional[np.ndarray]]] = None,
    ) -> pd.DataFrame:
        """Compute pairwise consecutive drift across multiple periods.

        Parameters
        ----------
        thetas_by_period : list[np.ndarray]
            Ordered list of (D_t, K) theta matrices, one per period.
        labels : list[str]
            Period labels.  Must have same length as ``thetas_by_period``.
        weights_by_period : list | None
            Optional list of weight arrays.

        Returns
        -------
        pd.DataFrame
            Columns: period_from, period_to, jsd, alert.
        """
        if len(thetas_by_period) != len(labels):
            raise ValueError(
                f"thetas_by_period has {len(thetas_by_period)} entries but "
                f"labels has {len(labels)}."
            )
        if len(thetas_by_period) < 2:
            raise ValueError("Need at least two periods to compute drift.")

        weights = weights_by_period or [None] * len(thetas_by_period)
        rows = []
        for i in range(len(thetas_by_period) - 1):
            result = self.compute_drift(
                thetas_by_period[i],
                thetas_by_period[i + 1],
                weights_t0=weights[i],
                weights_t1=weights[i + 1],
                labels=(labels[i], labels[i + 1]),
            )
            rows.append(
                {
                    "period_from": labels[i],
                    "period_to": labels[i + 1],
                    "jsd": result.jsd,
                    "alert": result.alert,
                }
            )
        return pd.DataFrame(rows)

    def plot_composition(
        self,
        thetas_by_period: list[np.ndarray],
        labels: list[str],
        weights_by_period: Optional[list[Optional[np.ndarray]]] = None,
    ) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]  # noqa: F821
        """Stacked area chart of portfolio topic composition over time.

        Parameters
        ----------
        thetas_by_period : list[np.ndarray]
            (D_t, K) theta matrices per period.
        labels : list[str]
            Period labels for the x-axis.
        weights_by_period : list | None
            Optional exposure weights per period.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        weights = weights_by_period or [None] * len(thetas_by_period)
        k = self.profiler.n_topics_
        compositions = np.array(
            [
                self._portfolio_distribution(t, w)
                for t, w in zip(thetas_by_period, weights)
            ]
        )  # shape (n_periods, K)

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8 + 2), 4))
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        cmap = plt.cm.tab20  # type: ignore[attr-defined]
        for k_idx in range(k):
            values = compositions[:, k_idx]
            ax.fill_between(
                x,
                bottom,
                bottom + values,
                alpha=0.85,
                color=cmap(k_idx / max(k, 1)),
                label=f"Topic {k_idx}",
            )
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Portfolio weight")
        ax.set_ylim(0, 1)
        ax.set_title("Portfolio risk composition over time")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _portfolio_distribution(
        theta: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute the weighted average topic distribution across policies."""
        if weights is None:
            pi = theta.mean(axis=0)
        else:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
            pi = (theta * w[:, np.newaxis]).sum(axis=0)
        # Ensure sums to 1 (floating point safety)
        pi = pi / pi.sum()
        return pi

    @staticmethod
    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence between distributions p and q."""
        p = np.clip(p, 1e-300, None)
        q = np.clip(q, 1e-300, None)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        # Clamp numerical noise to [0, 1]
        return float(np.clip(jsd, 0.0, 1.0))

    @staticmethod
    def _validate_theta(theta: np.ndarray, k: int, name: str) -> None:
        if theta.ndim != 2:
            raise ValueError(f"{name} must be 2-D, shape (D, K).")
        if theta.shape[1] != k:
            raise ValueError(
                f"{name} has {theta.shape[1]} topics but profiler has {k}."
            )
