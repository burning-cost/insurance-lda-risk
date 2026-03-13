"""
LDARiskProfiler
===============

Wraps sklearn's LatentDirichletAllocation to produce the (D x K) policy-topic
distribution matrix θ — the soft cluster membership output for every policy.

Design decisions
----------------
-   **sklearn backend** — sklearn's online variational Bayes implementation
    (Hoffman, Bach & Blei 2010) is battle-hardened and handles large sparse
    matrices efficiently.  We do not reimplement VB from scratch.
-   **exposure_weighted=True** raises NotImplementedError in v0.1.  The
    mathematical modification (multiplying the γ update by e_d) is documented
    in the code and planned for v0.2.
-   **alpha and eta** default to 1/K following Jamotton & Hainaut (2024), which
    encourages sparse topic membership rather than the sklearn default of 1/K as
    well (but with a different doc_topic_prior path).
-   **components_** mirrors sklearn's attribute: unnormalised λ matrix of shape
    (K, V).  The normalised per-row version is ``topic_modality_dist_``.

Examples
--------
>>> import numpy as np
>>> import scipy.sparse as sp
>>> from insurance_lda_risk import LDARiskProfiler

>>> rng = np.random.default_rng(0)
>>> X = sp.random(200, 50, density=0.4, format="csr", dtype=np.float64,
...               random_state=0)
>>> profiler = LDARiskProfiler(n_topics=4, random_state=0)
>>> theta = profiler.fit_transform(X)
>>> theta.shape
(200, 4)
>>> np.allclose(theta.sum(axis=1), 1.0, atol=1e-6)
True
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import LatentDirichletAllocation


class LDARiskProfiler:
    """Fit LDA on the encoded portfolio and produce soft topic memberships.

    Parameters
    ----------
    n_topics : int, default 8
        Number of latent risk profiles (K).
    alpha : float | None, default None
        Dirichlet prior on the policy-topic distribution (doc_topic_prior).
        None defaults to ``1 / n_topics`` — sparse, each policy skews towards
        one dominant topic.  Higher values produce flatter, more mixed profiles.
    eta : float | None, default None
        Dirichlet prior on the topic-modality distribution (topic_word_prior).
        None defaults to ``1 / n_topics`` — sparse topics.  Higher values
        produce more diffuse topics.
    exposure_weighted : bool, default False
        If True, weight each policy's contribution to the M-step by its
        exposure.  Currently raises NotImplementedError — planned for v0.2.
    learning_method : str, default ``'online'``
        Passed to sklearn LDA.  ``'online'`` is efficient for large portfolios;
        ``'batch'`` can be more stable on small portfolios.
    max_iter : int, default 50
        Maximum EM iterations.
    random_state : int | None, default None
        Seed for reproducibility.

    Attributes
    ----------
    components_ : np.ndarray, shape (K, V)
        Unnormalised topic-modality counts λ (direct from sklearn).
    topic_modality_dist_ : np.ndarray, shape (K, V)
        Row-normalised β matrix.  ``topic_modality_dist_[k, v]`` is the
        probability of modality v given risk profile k.
    n_topics_ : int
        Number of topics (same as ``n_topics`` parameter).
    perplexity_ : float
        Per-word perplexity on the training data.  Lower is better for LDA;
        not the primary selection criterion for insurance — use deviance instead
        (see TopicSelector).
    lda_ : sklearn.decomposition.LatentDirichletAllocation
        Underlying sklearn LDA instance.
    """

    def __init__(
        self,
        n_topics: int = 8,
        alpha: Optional[float] = None,
        eta: Optional[float] = None,
        exposure_weighted: bool = False,
        learning_method: str = "online",
        max_iter: int = 50,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_topics = n_topics
        self.alpha = alpha
        self.eta = eta
        self.exposure_weighted = exposure_weighted
        self.learning_method = learning_method
        self.max_iter = max_iter
        self.random_state = random_state

        # Set after fit
        self.components_: Optional[np.ndarray] = None
        self.topic_modality_dist_: Optional[np.ndarray] = None
        self.n_topics_: int = n_topics
        self.perplexity_: Optional[float] = None
        self.lda_: Optional[LatentDirichletAllocation] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: sp.spmatrix,
        exposure: Optional[np.ndarray] = None,
    ) -> "LDARiskProfiler":
        """Fit LDA on the encoded portfolio matrix.

        Parameters
        ----------
        X : scipy.sparse matrix, shape (D, V)
            Encoded portfolio from InsuranceLDAEncoder.transform().
        exposure : np.ndarray | None, shape (D,)
            Policy exposure (e.g. years at risk).  Currently only supported
            when ``exposure_weighted=False``, in which case it is ignored.
            Providing exposure with ``exposure_weighted=True`` raises
            NotImplementedError.

        Returns
        -------
        LDARiskProfiler
            Fitted profiler (self).

        Raises
        ------
        NotImplementedError
            If ``exposure_weighted=True``.
        ValueError
            If ``X`` has zero rows or zero columns.
        """
        if self.exposure_weighted:
            raise NotImplementedError(
                "exposure_weighted=True is planned for v0.2.  "
                "The mathematical extension (multiplying the γ update rule by "
                "e_d) is documented in the source.  For now, pass "
                "exposure_weighted=False (the default)."
            )

        X = self._validate_X(X)
        doc_topic_prior = self.alpha if self.alpha is not None else (1.0 / self.n_topics)
        topic_word_prior = self.eta if self.eta is not None else (1.0 / self.n_topics)

        self.lda_ = LatentDirichletAllocation(
            n_components=self.n_topics,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            learning_method=self.learning_method,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.lda_.fit(X)
        self._store_attributes(X)
        self._fitted = True
        return self

    def transform(self, X: sp.spmatrix) -> np.ndarray:
        """Compute policy-topic distributions for new data.

        Parameters
        ----------
        X : scipy.sparse matrix, shape (D, V)
            Encoded portfolio.

        Returns
        -------
        np.ndarray, shape (D, K)
            θ matrix.  Each row is a probability distribution over K topics.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        self._check_fitted()
        X = self._validate_X(X)
        theta = self.lda_.transform(X)  # type: ignore[union-attr]
        return theta

    def fit_transform(
        self,
        X: sp.spmatrix,
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit then transform in one call.

        Parameters
        ----------
        X : scipy.sparse matrix, shape (D, V)
        exposure : np.ndarray | None, shape (D,)

        Returns
        -------
        np.ndarray, shape (D, K)
            θ matrix.
        """
        return self.fit(X, exposure=exposure).transform(X)

    def top_modalities_per_topic(
        self,
        feature_names: list[str],
        top_n: int = 10,
    ) -> dict[int, list[tuple[str, float]]]:
        """Return the top modalities for each topic.

        Parameters
        ----------
        feature_names : list[str]
            Vocabulary term names from ``InsuranceLDAEncoder.feature_names_``.
        top_n : int, default 10
            Number of top modalities per topic.

        Returns
        -------
        dict[int, list[tuple[str, float]]]
            Mapping topic_id → list of (modality_name, probability).
        """
        self._check_fitted()
        beta = self.topic_modality_dist_  # type: ignore[union-attr]
        result: dict[int, list[tuple[str, float]]] = {}
        for k in range(self.n_topics_):
            row = beta[k]
            top_idx = np.argsort(row)[::-1][:top_n]
            result[k] = [(feature_names[i], float(row[i])) for i in top_idx]
        return result

    def get_dominant_topic(self, theta: np.ndarray) -> np.ndarray:
        """Return the plurality topic index for each policy.

        Parameters
        ----------
        theta : np.ndarray, shape (D, K)

        Returns
        -------
        np.ndarray, shape (D,), dtype int
            Argmax topic index per policy.
        """
        return np.argmax(theta, axis=1).astype(int)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_attributes(self, X: sp.spmatrix) -> None:
        lda = self.lda_  # type: ignore[union-attr]
        self.components_ = lda.components_
        # Normalise rows to get β = P(modality | topic)
        row_sums = self.components_.sum(axis=1, keepdims=True)
        self.topic_modality_dist_ = self.components_ / np.where(
            row_sums > 0, row_sums, 1.0
        )
        self.n_topics_ = self.n_topics
        self.perplexity_ = float(lda.perplexity(X))

    @staticmethod
    def _validate_X(X: sp.spmatrix) -> sp.csr_matrix:
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()
        if X.shape[0] == 0:
            raise ValueError("X must have at least one row (policy).")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one column (modality).")
        # sklearn LDA expects float64
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        return X

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Profiler has not been fitted. Call fit() before transform()."
            )
