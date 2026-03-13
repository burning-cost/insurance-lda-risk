"""
InsuranceLDAEncoder
===================

Converts a portfolio DataFrame into the (D x V) sparse count matrix that
sklearn's LatentDirichletAllocation expects.

The key differences from a generic one-hot encoder are:

1.  **Global vocabulary** — modalities from all variables are indexed into a
    single vocabulary V.  Vocabulary position is ``variable__modality``.
2.  **Continuous discretisation** — continuous variables are binned via
    ``pd.cut`` and the resulting interval string becomes a modality.
3.  **Missing as modality** — NaN is a valid observation, not an error.  It
    gets the modality ``__MISSING__``.
4.  **Sparse int8 output** — entries are 0 or 1 (one active modality per
    variable per policy).  Using int8 keeps memory low for large portfolios.

Why not sklearn's OneHotEncoder?

    OneHotEncoder produces one sparse column per category but does not build a
    *shared* vocabulary across variables.  LDA needs a single V-dimensional
    vocabulary so that β_k is a distribution over all modalities jointly.

Examples
--------
>>> import pandas as pd
>>> from insurance_lda_risk import InsuranceLDAEncoder

>>> df = pd.DataFrame({
...     "vehicle_group": ["A", "A", "B", "C"],
...     "area": ["urban", "rural", "urban", "urban"],
...     "age": [25.0, 45.0, 35.0, 22.0],
... })
>>> enc = InsuranceLDAEncoder()
>>> X = enc.fit_transform(df, cat_cols=["vehicle_group", "area"],
...                        cont_cols=["age"], n_bins=3)
>>> X.shape
(4, 9)
>>> list(enc.vocabulary_.keys())[:3]  # doctest: +SKIP
['vehicle_group__A', 'vehicle_group__B', 'vehicle_group__C']
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


class InsuranceLDAEncoder:
    """Encode a portfolio DataFrame as a (D x V) sparse count matrix for LDA.

    Parameters
    ----------
    missing_as_modality : bool, default True
        When True, NaN values are encoded as the ``__MISSING__`` modality for
        that variable rather than being dropped or raising an error.

    Attributes
    ----------
    vocabulary_ : dict[str, int]
        Mapping ``"variable__modality"`` → column index in V.
    feature_names_ : list[str]
        Ordered list of vocabulary terms (``vocabulary_`` keys).
    variable_ranges_ : dict[str, list[str]]
        Per-variable list of modality labels (including ``__MISSING__`` if
        ``missing_as_modality=True`` and any NaN was seen at fit time).
    n_variables_ : int
        Number of input variables (cat + discretised cont).
    n_modalities_ : int
        Total vocabulary size V.
    cat_cols_ : list[str]
        Categorical column names set at fit time.
    cont_cols_ : list[str]
        Continuous column names set at fit time.
    bin_edges_ : dict[str, np.ndarray]
        Bin edges used for each continuous variable.
    n_bins_ : int
        Number of bins used for continuous variables.
    """

    def __init__(self, missing_as_modality: bool = True) -> None:
        self.missing_as_modality = missing_as_modality

        # Set after fit
        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.variable_ranges_: dict[str, list[str]] = {}
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.cat_cols_: list[str] = []
        self.cont_cols_: list[str] = []
        self.n_bins_: int = 10
        self.n_variables_: int = 0
        self.n_modalities_: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        cont_cols: Optional[list[str]] = None,
        n_bins: int = 10,
    ) -> "InsuranceLDAEncoder":
        """Learn vocabulary from a portfolio DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio with one row per policy.
        cat_cols : list[str]
            Column names to treat as categorical.  Values are cast to str.
        cont_cols : list[str] | None, default None
            Column names to discretise into ``n_bins`` equal-frequency bins.
        n_bins : int, default 10
            Number of bins for continuous variables.

        Returns
        -------
        InsuranceLDAEncoder
            Fitted encoder (self).

        Raises
        ------
        ValueError
            If any column in ``cat_cols`` or ``cont_cols`` is not in ``df``.
        """
        cont_cols = cont_cols or []
        self._validate_columns(df, cat_cols, cont_cols)
        self.cat_cols_ = list(cat_cols)
        self.cont_cols_ = list(cont_cols)
        self.n_bins_ = n_bins

        # Build bin edges for continuous variables
        for col in cont_cols:
            vals = df[col].dropna().values
            if len(vals) == 0:
                # All missing — single bin
                self.bin_edges_[col] = np.array([-np.inf, np.inf])
            else:
                quantiles = np.linspace(0, 100, n_bins + 1)
                edges = np.unique(np.percentile(vals, quantiles))
                if len(edges) < 2:
                    edges = np.array([vals.min() - 1e-9, vals.max() + 1e-9])
                self.bin_edges_[col] = edges

        # Build vocabulary
        vocab_index = 0
        for col in cat_cols:
            modalities = self._categorical_modalities(df[col])
            self.variable_ranges_[col] = modalities
            for mod in modalities:
                key = f"{col}__{mod}"
                self.vocabulary_[key] = vocab_index
                self.feature_names_.append(key)
                vocab_index += 1

        for col in cont_cols:
            binned = self._bin_continuous(df[col], col)
            modalities = self._categorical_modalities(binned)
            self.variable_ranges_[col] = modalities
            for mod in modalities:
                key = f"{col}__{mod}"
                self.vocabulary_[key] = vocab_index
                self.feature_names_.append(key)
                vocab_index += 1

        self.n_variables_ = len(cat_cols) + len(cont_cols)
        self.n_modalities_ = vocab_index
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Encode a DataFrame using the fitted vocabulary.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio to encode.  May contain policies not seen at fit time;
            unseen modality values are mapped to ``__MISSING__`` if that modality
            exists, otherwise the variable contributes a zero row for that
            policy (rare but handled gracefully).

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape (D, V), dtype int8.  Each row sums to the number of active
            variables for that policy (typically ``n_variables_``).

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        self._check_fitted()
        n_policies = len(df)

        rows: list[int] = []
        cols: list[int] = []

        for col in self.cat_cols_:
            if col not in df.columns:
                continue
            series = df[col].copy()
            for policy_idx, value in enumerate(series):
                mod = self._resolve_modality(col, value)
                if mod is None:
                    continue
                key = f"{col}__{mod}"
                if key in self.vocabulary_:
                    rows.append(policy_idx)
                    cols.append(self.vocabulary_[key])

        for col in self.cont_cols_:
            if col not in df.columns:
                continue
            binned = self._bin_continuous(df[col], col)
            for policy_idx, value in enumerate(binned):
                mod = self._resolve_modality(col, value)
                if mod is None:
                    continue
                key = f"{col}__{mod}"
                if key in self.vocabulary_:
                    rows.append(policy_idx)
                    cols.append(self.vocabulary_[key])

        data = np.ones(len(rows), dtype=np.int8)
        X = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_policies, self.n_modalities_),
            dtype=np.int8,
        )
        return X

    def fit_transform(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        cont_cols: Optional[list[str]] = None,
        n_bins: int = 10,
    ) -> sp.csr_matrix:
        """Fit then transform in one call.

        Parameters
        ----------
        df : pd.DataFrame
        cat_cols : list[str]
        cont_cols : list[str] | None, default None
        n_bins : int, default 10

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape (D, V), dtype int8.
        """
        return self.fit(df, cat_cols, cont_cols, n_bins).transform(df)

    def decode_topic(
        self,
        topic_weights: np.ndarray,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Translate a topic-word vector into its most prominent modalities.

        Parameters
        ----------
        topic_weights : np.ndarray
            Shape (V,) — one row of the K×V components_ matrix, normalised or
            not (will be normalised here).
        top_n : int, default 10
            Number of top modalities to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``variable``, ``modality``, ``weight``.
        """
        self._check_fitted()
        w = np.asarray(topic_weights, dtype=float)
        w = w / w.sum()
        top_idx = np.argsort(w)[::-1][:top_n]

        records = []
        for idx in top_idx:
            feat = self.feature_names_[idx]
            variable, modality = feat.split("__", 1)
            records.append(
                {
                    "feature": feat,
                    "variable": variable,
                    "modality": modality,
                    "weight": float(w[idx]),
                }
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_columns(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        cont_cols: list[str],
    ) -> None:
        missing = set(cat_cols + cont_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Columns not found in DataFrame: {sorted(missing)}"
            )

    def _categorical_modalities(self, series: pd.Series) -> list[str]:
        """Sorted unique modalities (+ __MISSING__ if applicable)."""
        values = series.dropna().astype(str).unique()
        modalities = sorted(values.tolist())
        if self.missing_as_modality and series.isna().any():
            modalities.append("__MISSING__")
        return modalities

    def _bin_continuous(self, series: pd.Series, col: str) -> pd.Series:
        """Bin a continuous series using stored edges for ``col``."""
        edges = self.bin_edges_.get(col)
        if edges is None:
            raise RuntimeError(
                f"No bin edges for column '{col}'. "
                "Did you call fit() with this column in cont_cols?"
            )
        # Clamp to edges so out-of-range values fall in the outermost bin
        finite_min = edges[0] if np.isfinite(edges[0]) else -1e18
        finite_max = edges[-1] if np.isfinite(edges[-1]) else 1e18
        clipped = series.clip(lower=finite_min, upper=finite_max)
        # pd.cut with the stored edges
        binned = pd.cut(
            clipped,
            bins=edges,
            include_lowest=True,
            right=True,
        )
        return binned.astype(str).where(series.notna(), other=np.nan)

    def _resolve_modality(self, col: str, value) -> Optional[str]:
        """Map a raw value to its modality label."""
        if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
            if self.missing_as_modality:
                return "__MISSING__"
            return None
        str_val = str(value)
        known = self.variable_ranges_.get(col, [])
        if str_val in known:
            return str_val
        if self.missing_as_modality:
            return "__MISSING__"
        return None

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Encoder has not been fitted. Call fit() before transform()."
            )
