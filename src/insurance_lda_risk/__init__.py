"""
insurance-lda-risk
==================

LDA-based probabilistic risk profiling for insurance portfolios.

Adapts Latent Dirichlet Allocation from NLP to tabular insurance data,
following Jamotton & Hainaut (2024), LIDAM Discussion Paper ISBA 2024/008.

Each policy is treated as a "document" whose "words" are the observed values
of its categorical rating factors. LDA then discovers K latent risk archetypes
(topics) and assigns every policy a soft membership vector across them.

Typical usage
-------------
>>> import pandas as pd
>>> from insurance_lda_risk import InsuranceLDAEncoder, LDARiskProfiler
>>> from insurance_lda_risk import TopicValidator, TopicSelector, PortfolioDrift

>>> encoder = InsuranceLDAEncoder()
>>> X = encoder.fit_transform(df, cat_cols=["vehicle_group", "area", "age_band"])

>>> profiler = LDARiskProfiler(n_topics=8, random_state=42)
>>> theta = profiler.fit_transform(X)  # shape (n_policies, 8)

References
----------
Jamotton, C. & Hainaut, D. (2024). Topic Modelling for Insurance Losses.
LIDAM Discussion Paper ISBA 2024/008, UCLouvain.
https://dial.uclouvain.be/pr/boreal/object/boreal:285770
"""

from insurance_lda_risk.encoder import InsuranceLDAEncoder
from insurance_lda_risk.profiler import LDARiskProfiler
from insurance_lda_risk.validator import TopicValidator
from insurance_lda_risk.selector import TopicSelector
from insurance_lda_risk.drift import PortfolioDrift
from insurance_lda_risk._types import (
    TopicValidationResult,
    DriftResult,
    TopicStats,
)

__version__ = "0.1.0"
__all__ = [
    "InsuranceLDAEncoder",
    "LDARiskProfiler",
    "TopicValidator",
    "TopicSelector",
    "PortfolioDrift",
    "TopicValidationResult",
    "DriftResult",
    "TopicStats",
]
