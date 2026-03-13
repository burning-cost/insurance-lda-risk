"""
TopicSelector
=============

Automated selection of the number of risk profiles K.

The standard NLP approach — minimise perplexity — does not translate well to
insurance.  Perplexity measures in-sample reconstruction quality; what pricing
teams care about is whether the topics discriminate claim experience.

The approach here (following Jamotton & Hainaut 2024, Section 3.3) is:

1.  Fit LDA with K = k_min, k_min+1, ..., k_max.
2.  For each K, compute the Poisson deviance of the resulting topic model on a
    held-out fold.
3.  Plot deviance vs K and identify the elbow — the K after which additional
    topics produce diminishing deviance reduction.
4.  ``optimal_k_`` is the elbow point, found via the second-difference method.

If ``y_claims`` is not provided, fall back to perplexity (unsupervised use).

This is intentionally conservative: it does not validate topics against held-out
*claim data* but against held-out *encoded matrices* (because some portfolios
have no claim labels attached at segmentation time).

Examples
--------
>>> import numpy as np
>>> import scipy.sparse as sp
>>> from insurance_lda_risk import TopicSelector

>>> rng = np.random.default_rng(42)
>>> X = sp.random(400, 60, density=0.3, format='csr', random_state=42)
>>> y = rng.poisson(0.08, 400)

>>> sel = TopicSelector(k_range=range(2, 8), cv=2)
>>> optimal_k = sel.select(X, y_claims=y)
>>> isinstance(optimal_k, int)
True
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold

from insurance_lda_risk.profiler import LDARiskProfiler
from insurance_lda_risk.validator import TopicValidator


class TopicSelector:
    """Select the optimal number of LDA risk topics via deviance elbow.

    Parameters
    ----------
    k_range : range, default range(2, 21)
        K values to evaluate.
    cv : int, default 5
        Number of cross-validation folds.
    distribution : str, default ``'poisson'``
        Passed to TopicValidator.  Use ``'binomial'`` for binary outcomes.
    random_state : int | None, default None
        Seed for LDA fitting reproducibility.

    Attributes
    ----------
    optimal_k_ : int
        Selected number of topics.
    scores_ : pd.DataFrame
        Columns: k, mean_deviance, std_deviance (if y_claims provided) or
        mean_perplexity, std_perplexity (if no y_claims).
    """

    def __init__(
        self,
        k_range: range = range(2, 21),
        cv: int = 5,
        distribution: str = "poisson",
        random_state: Optional[int] = None,
    ) -> None:
        self.k_range = k_range
        self.cv = cv
        self.distribution = distribution
        self.random_state = random_state

        self.optimal_k_: Optional[int] = None
        self.scores_: Optional[pd.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        X: sp.spmatrix,
        y_claims: Optional[np.ndarray] = None,
        exposure: Optional[np.ndarray] = None,
    ) -> int:
        """Run K selection and return optimal K.

        Parameters
        ----------
        X : scipy.sparse matrix, shape (D, V)
            Encoded portfolio from InsuranceLDAEncoder.
        y_claims : np.ndarray | None, shape (D,)
            Observed claim counts.  If provided, uses Poisson/Binomial deviance
            on held-out folds.  If None, falls back to LDA perplexity.
        exposure : np.ndarray | None, shape (D,)
            Policy exposures.  Only used when ``y_claims`` is provided.

        Returns
        -------
        int
            Optimal K.
        """
        X = sp.csr_matrix(X).astype(np.float64)
        n = X.shape[0]

        if y_claims is not None:
            y_claims = np.asarray(y_claims, dtype=float)
            if y_claims.shape[0] != n:
                raise ValueError(
                    f"X has {n} rows but y_claims has {len(y_claims)}."
                )
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)

        use_deviance = y_claims is not None
        scores: list[dict] = []

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for k in self.k_range:
            fold_scores: list[float] = []
            for train_idx, test_idx in kf.split(X):
                X_train = X[train_idx]
                X_test = X[test_idx]
                profiler = LDARiskProfiler(
                    n_topics=k,
                    random_state=self.random_state,
                )
                profiler.fit(X_train)

                if use_deviance:
                    theta_test = profiler.transform(X_test)
                    y_test = y_claims[test_idx]  # type: ignore[index]
                    exp_test = (
                        exposure[test_idx]  # type: ignore[index]
                        if exposure is not None
                        else None
                    )
                    validator = TopicValidator(distribution=self.distribution)
                    result = validator.validate(theta_test, y_test, exp_test)
                    fold_scores.append(result.deviance)
                else:
                    fold_scores.append(profiler.lda_.perplexity(X_test))  # type: ignore[union-attr]

            scores.append(
                {
                    "k": k,
                    "mean_score": float(np.mean(fold_scores)),
                    "std_score": float(np.std(fold_scores)),
                }
            )

        metric_col = "mean_deviance" if use_deviance else "mean_perplexity"
        std_col = "std_deviance" if use_deviance else "std_perplexity"

        self.scores_ = pd.DataFrame(scores).rename(
            columns={"mean_score": metric_col, "std_score": std_col}
        )
        mean_vals = np.array([s["mean_score"] for s in scores])
        self.optimal_k_ = int(self._find_elbow(list(self.k_range), mean_vals))
        self._fitted = True
        return self.optimal_k_

    def plot_elbow(self) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]  # noqa: F821
        """Plot the deviance / perplexity elbow curve.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If called before ``select()``.
        """
        import matplotlib.pyplot as plt

        if not self._fitted or self.scores_ is None:
            raise RuntimeError("Call select() before plot_elbow().")

        df = self.scores_
        metric_col = [c for c in df.columns if c.startswith("mean_")][0]
        std_col = [c for c in df.columns if c.startswith("std_")][0]
        label = metric_col.replace("mean_", "").replace("_", " ").title()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["k"], df[metric_col], marker="o", linewidth=1.8, label=label)
        ax.fill_between(
            df["k"],
            df[metric_col] - df[std_col],
            df[metric_col] + df[std_col],
            alpha=0.2,
        )
        if self.optimal_k_ is not None:
            best_score = df.loc[df["k"] == self.optimal_k_, metric_col].values[0]
            ax.axvline(
                self.optimal_k_,
                color="red",
                linestyle="--",
                linewidth=1,
                label=f"Optimal K = {self.optimal_k_}",
            )
            ax.scatter([self.optimal_k_], [best_score], color="red", zorder=5)
        ax.set_xlabel("Number of topics (K)")
        ax.set_ylabel(label)
        ax.set_title(f"Topic count selection — {label} elbow")
        ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_elbow(k_values: list[int], scores: np.ndarray) -> int:
        """Find the elbow via maximum second difference.

        The elbow is where the curve stops decreasing sharply.  We use the
        index of maximum absolute second difference of the score vector.

        Falls back to the K with the minimum score if only two K values
        are provided.

        Parameters
        ----------
        k_values : list[int]
        scores : np.ndarray, shape (len(k_values),)

        Returns
        -------
        int
            Selected K.
        """
        if len(k_values) <= 2:
            return k_values[int(np.argmin(scores))]

        # Normalise to [0,1] for scale-invariant second difference
        s_range = scores.max() - scores.min()
        if s_range < 1e-12:
            return k_values[0]
        s_norm = (scores - scores.min()) / s_range

        second_diff = np.abs(np.diff(np.diff(s_norm)))
        # +1 because second diff has 2 fewer elements and elbow is at index+1
        elbow_idx = int(np.argmax(second_diff)) + 1
        return k_values[elbow_idx]
