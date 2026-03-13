# insurance-lda-risk

LDA-based probabilistic risk profiling for insurance portfolios.

## What problem does this solve for a UK pricing actuary?

Your GLM has 400 rating cells.  Your portfolio has 60,000 policies.  You have a reasonable fit for the policies you understand, but you have no clean answer to these questions:

- **What kind of risk is this book, really?**  Not "vehicle group B, area suburban, age 36-50" — but in broad terms, how many distinct risk archetypes exist, and what mix does your portfolio have?
- **Is the composition shifting?**  Renewals, aggregator volumes, and underwriting appetite changes all move the mix.  If your 2024 book has proportionally more high-risk policies than 2023, your flat renewal rate change is under-priced.
- **What makes a good book-transfer candidate?**  When you're pricing a TPA transfer or a binder, you want to know whether the incoming book looks like your existing portfolio or a different risk universe.

Standard tools — GLM cells, k-means, decision trees — all give hard cluster assignments.  A policy is either in segment 3 or it is not.  Real portfolios don't work that way.  A young driver with a group A vehicle and low mileage is *mostly* low-risk but *partly* high-risk.  Soft membership matters.

This library applies Latent Dirichlet Allocation to tabular insurance data, following Jamotton & Hainaut (2024).  Each policy gets a probability vector across K latent risk profiles (topics).  You can use that vector as features for downstream models, as a drift signal, or as a segmentation for actuarial review.

## How it works

LDA was designed for text.  The analogy to insurance is direct:

| NLP concept | Insurance equivalent |
|---|---|
| Corpus | Portfolio (D × V matrix) |
| Document | Policy |
| Word | A specific modality value (e.g. `area=urban`, `age_band=17-25`) |
| Topic | Latent risk profile |
| Document-topic distribution θ_d | Policy's soft membership across K risk profiles |
| Topic-word distribution β_k | Risk profile's characteristic modality mix |

Each policy gets a vector θ_d summing to 1.  θ_d = [0.8, 0.15, 0.05] means "80% like profile 0, 15% like profile 1, 5% like profile 2".

Inference uses sklearn's online variational Bayes (Hoffman, Bach & Blei 2010).

**Reference:** Jamotton, C. & Hainaut, D. (2024). *Topic Modelling for Insurance Losses.* LIDAM Discussion Paper ISBA 2024/008, UCLouvain. https://dial.uclouvain.be/pr/boreal/object/boreal:285770

## Installation

```bash
pip install insurance-lda-risk
```

Dependencies: scikit-learn, scipy, numpy, pandas, matplotlib.

## Quick start

```python
import pandas as pd
from insurance_lda_risk import InsuranceLDAEncoder, LDARiskProfiler, TopicValidator

# 1. Encode portfolio to (D x V) sparse matrix
enc = InsuranceLDAEncoder()
X = enc.fit_transform(
    df,
    cat_cols=["vehicle_group", "area", "age_band", "ncb_band"],
    cont_cols=["vehicle_age", "annual_mileage"],
    n_bins=10,
)

# 2. Fit LDA and get soft memberships
profiler = LDARiskProfiler(n_topics=8, random_state=42)
theta = profiler.fit_transform(X)  # shape (n_policies, 8)

# 3. Validate topics against claims
validator = TopicValidator(distribution="poisson")
result = validator.validate(theta, y_claims=df["n_claims"], exposure=df["exposure"])
print(result.summary)
#        claim_frequency  total_exposure  total_claims  pct_policies
# topic
# 0             0.041234       12450.23        512.34         23.14
# 1             0.072891        8234.11        600.12         15.43
# ...

result.plot_frequencies()
```

## API reference

### InsuranceLDAEncoder

Converts a portfolio DataFrame to a (D × V) sparse count matrix.

```python
enc = InsuranceLDAEncoder(missing_as_modality=True)
enc.fit(df, cat_cols, cont_cols=None, n_bins=10)
X = enc.transform(df)               # scipy.sparse.csr_matrix (D, V)
X = enc.fit_transform(df, ...)      # fit + transform in one call

enc.vocabulary_                     # dict: "variable__modality" -> index
enc.feature_names_                  # list of vocabulary terms
enc.variable_ranges_                # dict: variable -> list of modalities
enc.decode_topic(weights, top_n=10) # explain a topic in terms of modalities
```

Continuous variables are discretised into equal-frequency bins.  Missing values become the `__MISSING__` modality by default, so you do not lose policies with sparse covariate data.

### LDARiskProfiler

```python
profiler = LDARiskProfiler(
    n_topics=8,
    alpha=None,       # Dirichlet prior on policy-topic dist (default: 1/K)
    eta=None,         # Dirichlet prior on topic-modality dist (default: 1/K)
    max_iter=50,
    random_state=42,
)
profiler.fit(X)
theta = profiler.transform(X)       # np.ndarray (D, K)
theta = profiler.fit_transform(X)

profiler.components_               # (K, V) unnormalised β from sklearn
profiler.topic_modality_dist_      # (K, V) normalised β
profiler.perplexity_               # float (lower is better, but use deviance for K selection)
profiler.get_dominant_topic(theta) # (D,) argmax topic per policy
profiler.top_modalities_per_topic(enc.feature_names_, top_n=10)
```

`exposure_weighted=True` is planned for v0.2.  It will weight each policy's contribution to the M-step by its exposure, following the modified γ update: γ_{d,k} = α + e_d · Σ_v n_{d,v} ϕ_{d,v,k}.

### TopicValidator

```python
validator = TopicValidator(distribution="poisson")  # or 'binomial'
result = validator.validate(theta, y_claims, exposure=None)

result.deviance            # float
result.null_deviance       # float
result.deviance_reduction  # float, 1 - deviance/null_deviance
result.summary             # pd.DataFrame with per-topic stats
result.plot_frequencies()  # bar chart of claim frequency by topic
```

The deviance metric is Poisson deviance (or Binomial cross-entropy for binary outcomes).  The null model is the portfolio mean frequency applied uniformly.  A positive deviance reduction means the topics are discriminating claim experience.

### TopicSelector

```python
selector = TopicSelector(k_range=range(2, 21), cv=5, distribution="poisson")
optimal_k = selector.select(X, y_claims=y, exposure=exp)

selector.optimal_k_        # int
selector.scores_           # pd.DataFrame: k, mean_deviance, std_deviance
selector.plot_elbow()      # elbow curve
```

Uses held-out Poisson deviance rather than perplexity.  Perplexity is the NLP metric; for insurance, you want to know whether more topics improve claim frequency discrimination.

If you do not have claim labels, omit `y_claims` to fall back to perplexity.

### PortfolioDrift

```python
drift = PortfolioDrift(profiler, alert_threshold=0.05)
result = drift.compute_drift(theta_t0, theta_t1,
                              labels=("2023", "2024"))

result.jsd               # float [0, 1]: Jensen-Shannon divergence
result.per_topic_shift   # pd.Series: t1 - t0 per topic
result.alert             # bool: True if jsd > alert_threshold
result.plot_shift()      # horizontal bar chart of composition change

# Multi-period drift
df = drift.compute_drift_series(
    [theta_2021, theta_2022, theta_2023, theta_2024],
    labels=["2021", "2022", "2023", "2024"],
)

# Stacked area chart
fig = drift.plot_composition(
    [theta_2021, theta_2022, theta_2023, theta_2024],
    labels=["2021", "2022", "2023", "2024"],
)
```

JSD is symmetric and bounded [0, 1].  A JSD of 0.05 between consecutive years is a practical alert threshold for UK personal lines.  Values above 0.15 indicate a material composition shift that warrants pricing review.

## Selecting K

There is no ground truth K for a portfolio.  The right approach:

1.  Run `TopicSelector` over a plausible range (e.g. K = 2 to 20).
2.  Look at the deviance elbow — the K after which additional topics produce diminishing claim frequency discrimination.
3.  Check that topics are interpretable: use `top_modalities_per_topic` to read the dominant modalities for each topic.  If topic 3 is "young, urban, high vehicle group" and topic 7 is "rural, mature, low vehicle group", the topics are meaningful.
4.  Prefer smaller K if the elbow is ambiguous.  Ten interpretable topics beat twenty noisy ones.

Jamotton & Hainaut (2024) found K = 10 optimal for a 62,000-policy Swedish motorcycle portfolio.

## Worked example

See `notebooks/insurance_lda_risk_demo.ipynb` for a complete walkthrough on a synthetic UK motor portfolio, including topic interpretation, K selection, and multi-year drift analysis.

## Limitations

-   **Mutual exclusivity**: Standard LDA ignores the fact that modalities within a single variable are mutually exclusive (a policy cannot be simultaneously in age band 17-25 and 36-50).  The paper acknowledges this and uses standard LDA anyway because it is computationally tractable and empirically effective.
-   **Exposure weighting**: v0.1 does not weight policies by exposure.  A 3-year fleet policy and a 1-month private car policy contribute equally to the inference.  Exposure weighting is planned for v0.2.
-   **Topic instability**: LDA results depend on the random seed and can vary across runs.  Fix `random_state` for reproducibility, and consider running `PortfolioDrift` across multiple seeds to assess stability.

## Licence

MIT
