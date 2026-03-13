"""
Tests for InsuranceLDAEncoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from insurance_lda_risk import InsuranceLDAEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_group": ["A", "A", "B", "C", "B"],
            "area": ["urban", "rural", "urban", "suburban", "rural"],
        }
    )


@pytest.fixture
def df_with_cont() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_group": ["A", "A", "B", "C", "B"],
            "area": ["urban", "rural", "urban", "suburban", "rural"],
            "age": [25.0, 45.0, 35.0, 60.0, 22.0],
        }
    )


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_group": ["A", None, "B", "C", None],
            "area": ["urban", "rural", "urban", None, "rural"],
        }
    )


# ---------------------------------------------------------------------------
# Construction & fit
# ---------------------------------------------------------------------------

class TestEncoderFit:
    def test_fit_returns_self(self, simple_df):
        enc = InsuranceLDAEncoder()
        result = enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        assert result is enc

    def test_vocabulary_built(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        # vehicle_group: A, B, C = 3 modalities
        # area: rural, suburban, urban = 3 modalities
        assert len(enc.vocabulary_) == 6

    def test_feature_names_match_vocabulary(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        assert len(enc.feature_names_) == len(enc.vocabulary_)
        for name in enc.feature_names_:
            assert name in enc.vocabulary_

    def test_variable_ranges_populated(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        assert "vehicle_group" in enc.variable_ranges_
        assert "area" in enc.variable_ranges_
        assert sorted(enc.variable_ranges_["vehicle_group"]) == ["A", "B", "C"]

    def test_n_modalities_correct(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        assert enc.n_modalities_ == 6

    def test_missing_column_raises(self, simple_df):
        enc = InsuranceLDAEncoder()
        with pytest.raises(ValueError, match="not found"):
            enc.fit(simple_df, cat_cols=["vehicle_group", "does_not_exist"])

    def test_fit_with_cont_cols(self, df_with_cont):
        enc = InsuranceLDAEncoder()
        enc.fit(df_with_cont, cat_cols=["vehicle_group", "area"], cont_cols=["age"], n_bins=3)
        # 3 cat modalities for vg + 3 for area + 3 bins for age
        assert enc.n_modalities_ >= 6
        assert "age" in enc.bin_edges_

    def test_bin_edges_stored(self, df_with_cont):
        enc = InsuranceLDAEncoder()
        enc.fit(df_with_cont, cat_cols=["vehicle_group"], cont_cols=["age"], n_bins=3)
        assert "age" in enc.bin_edges_
        assert len(enc.bin_edges_["age"]) >= 2


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class TestEncoderTransform:
    def test_transform_shape(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        X = enc.transform(simple_df)
        assert X.shape == (5, 6)

    def test_transform_is_sparse(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        X = enc.transform(simple_df)
        assert sp.issparse(X)

    def test_each_row_sums_to_n_variables(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        X = enc.transform(simple_df)
        row_sums = np.array(X.sum(axis=1)).ravel()
        assert np.all(row_sums == 2), f"Expected 2 per row, got {row_sums}"

    def test_entries_are_0_or_1(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        X = enc.transform(simple_df)
        unique_vals = set(X.data.tolist())
        assert unique_vals <= {1}, f"Unexpected values: {unique_vals}"

    def test_transform_before_fit_raises(self, simple_df):
        enc = InsuranceLDAEncoder()
        with pytest.raises(RuntimeError, match="fitted"):
            enc.transform(simple_df)

    def test_transform_with_cont_cols(self, df_with_cont):
        enc = InsuranceLDAEncoder()
        enc.fit(df_with_cont, cat_cols=["vehicle_group", "area"], cont_cols=["age"], n_bins=3)
        X = enc.transform(df_with_cont)
        assert X.shape[0] == 5
        assert X.shape[1] == enc.n_modalities_
        # Each row should have one active entry per variable (3 variables)
        row_sums = np.array(X.sum(axis=1)).ravel()
        assert np.all(row_sums == 3)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

class TestEncoderMissing:
    def test_missing_creates_modality(self, df_with_missing):
        enc = InsuranceLDAEncoder(missing_as_modality=True)
        enc.fit(df_with_missing, cat_cols=["vehicle_group", "area"])
        vg_mods = enc.variable_ranges_["vehicle_group"]
        assert "__MISSING__" in vg_mods

    def test_missing_rows_still_have_entries(self, df_with_missing):
        enc = InsuranceLDAEncoder(missing_as_modality=True)
        enc.fit(df_with_missing, cat_cols=["vehicle_group", "area"])
        X = enc.transform(df_with_missing)
        row_sums = np.array(X.sum(axis=1)).ravel()
        # Every policy should have exactly 2 active entries (one per variable)
        assert np.all(row_sums == 2)

    def test_missing_excluded_when_flag_false(self):
        df = pd.DataFrame({"x": ["A", None, "B"]})
        enc = InsuranceLDAEncoder(missing_as_modality=False)
        enc.fit(df, cat_cols=["x"])
        mods = enc.variable_ranges_["x"]
        assert "__MISSING__" not in mods


# ---------------------------------------------------------------------------
# fit_transform
# ---------------------------------------------------------------------------

class TestEncoderFitTransform:
    def test_fit_transform_equivalent(self, df_with_cont):
        enc1 = InsuranceLDAEncoder()
        X1 = enc1.fit_transform(df_with_cont, cat_cols=["vehicle_group", "area"], cont_cols=["age"])

        enc2 = InsuranceLDAEncoder()
        enc2.fit(df_with_cont, cat_cols=["vehicle_group", "area"], cont_cols=["age"])
        X2 = enc2.transform(df_with_cont)

        assert (X1 - X2).nnz == 0

    def test_fit_transform_returns_sparse(self, simple_df):
        enc = InsuranceLDAEncoder()
        X = enc.fit_transform(simple_df, cat_cols=["vehicle_group", "area"])
        assert sp.issparse(X)


# ---------------------------------------------------------------------------
# Vocabulary and decode
# ---------------------------------------------------------------------------

class TestEncoderVocabulary:
    def test_vocabulary_keys_have_double_underscore(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        for key in enc.vocabulary_:
            assert "__" in key

    def test_decode_topic_returns_dataframe(self, df_with_cont):
        enc = InsuranceLDAEncoder()
        enc.fit(df_with_cont, cat_cols=["vehicle_group", "area"], cont_cols=["age"])
        fake_weights = np.ones(enc.n_modalities_)
        df = enc.decode_topic(fake_weights, top_n=3)
        assert len(df) == 3
        assert "feature" in df.columns
        assert "weight" in df.columns

    def test_decode_topic_weights_sum_to_1(self, simple_df):
        enc = InsuranceLDAEncoder()
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        w = np.random.default_rng(0).dirichlet(np.ones(enc.n_modalities_))
        df = enc.decode_topic(w, top_n=enc.n_modalities_)
        assert abs(df["weight"].sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Integration with larger portfolio
# ---------------------------------------------------------------------------

class TestEncoderIntegration:
    def test_large_portfolio(self, motor_portfolio, cat_cols, cont_cols):
        enc = InsuranceLDAEncoder()
        X = enc.fit_transform(motor_portfolio, cat_cols=cat_cols, cont_cols=cont_cols, n_bins=5)
        assert X.shape[0] == len(motor_portfolio)
        assert X.shape[1] == enc.n_modalities_

    def test_transform_on_subset(self, motor_portfolio, cat_cols):
        enc = InsuranceLDAEncoder()
        enc.fit(motor_portfolio, cat_cols=cat_cols)
        subset = motor_portfolio.iloc[:50]
        X = enc.transform(subset)
        assert X.shape[0] == 50

    def test_out_of_sample_unseen_values_use_missing(self, simple_df):
        enc = InsuranceLDAEncoder(missing_as_modality=True)
        enc.fit(simple_df, cat_cols=["vehicle_group", "area"])
        new_df = pd.DataFrame(
            {"vehicle_group": ["X"], "area": ["urban"]}
        )
        X = enc.transform(new_df)
        # Row should still sum to 2 (unseen vg -> __MISSING__, known area)
        assert X.sum() == 2
