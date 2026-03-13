# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-lda-risk: Probabilistic Risk Profiling Demo
# MAGIC
# MAGIC This notebook demonstrates the full `insurance-lda-risk` workflow on a
# MAGIC synthetic UK motor portfolio.
# MAGIC
# MAGIC **Steps covered:**
# MAGIC 1. Generate a synthetic portfolio with realistic structure
# MAGIC 2. Encode to (D x V) sparse count matrix
# MAGIC 3. Select K via deviance elbow
# MAGIC 4. Fit LDA and inspect risk profiles
# MAGIC 5. Validate topics against claim outcomes
# MAGIC 6. Detect year-on-year portfolio composition drift
# MAGIC
# MAGIC **Reference:** Jamotton & Hainaut (2024), LIDAM Discussion Paper ISBA 2024/008

# COMMAND ----------

# MAGIC %pip install insurance-lda-risk matplotlib scikit-learn scipy pandas numpy

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_lda_risk import (
    InsuranceLDAEncoder,
    LDARiskProfiler,
    TopicValidator,
    TopicSelector,
    PortfolioDrift,
)

print(f"insurance-lda-risk loaded OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK motor portfolio

# COMMAND ----------

def generate_motor_portfolio(n: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate a synthetic UK motor portfolio with realistic risk structure.

    Risk archetypes (latent):
    - Low risk:  Group A-B vehicle, 36-65 age, rural/suburban, low mileage
    - Standard:  Group B-C vehicle, 26-50 age, suburban
    - High risk: Group D-E vehicle, 17-25 age, urban, high mileage
    """
    rng = np.random.default_rng(seed)
    n = int(n)

    # Draw latent archetype
    archetype = rng.choice(["low_risk", "standard", "high_risk"],
                            size=n, p=[0.55, 0.30, 0.15])

    vehicle_group_map = {
        "low_risk":  ["A", "A", "B"],
        "standard":  ["B", "C", "C"],
        "high_risk": ["D", "D", "E"],
    }
    area_map = {
        "low_risk":  ["rural", "suburban", "rural"],
        "standard":  ["suburban", "suburban", "urban"],
        "high_risk": ["urban", "urban", "suburban"],
    }
    age_band_map = {
        "low_risk":  ["36-50", "51-65", "36-50"],
        "standard":  ["26-35", "36-50", "51-65"],
        "high_risk": ["17-25", "17-25", "26-35"],
    }
    ncb_map = {
        "low_risk":  [4, 5, 5],
        "standard":  [2, 3, 4],
        "high_risk": [0, 0, 1],
    }

    vehicle_group = [rng.choice(vehicle_group_map[a]) for a in archetype]
    area = [rng.choice(area_map[a]) for a in archetype]
    age_band = [rng.choice(age_band_map[a]) for a in archetype]
    ncb_band = [str(rng.choice(ncb_map[a])) + " years" for a in archetype]

    # Continuous variables
    vehicle_age_mu = {"low_risk": 4.0, "standard": 5.5, "high_risk": 7.0}
    mileage_mu = {"low_risk": 8000, "standard": 12000, "high_risk": 18000}

    vehicle_age = np.clip(
        rng.normal(loc=[vehicle_age_mu[a] for a in archetype], scale=2.0), 0.5, 20.0
    )
    annual_mileage = np.clip(
        rng.lognormal(
            mean=np.log([mileage_mu[a] for a in archetype]),
            sigma=0.4,
        ),
        1000, 40000,
    )

    # Introduce 2% missing in vehicle_group
    missing_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    vehicle_group = np.array(vehicle_group, dtype=object)
    vehicle_group[missing_idx] = None

    df = pd.DataFrame({
        "vehicle_group": vehicle_group,
        "area": area,
        "age_band": age_band,
        "ncb_band": ncb_band,
        "vehicle_age": vehicle_age,
        "annual_mileage": annual_mileage,
    })

    # Exposure (years at risk)
    exposure = rng.uniform(0.5, 1.5, size=n)

    # Claim frequency by archetype
    freq_map = {"low_risk": 0.04, "standard": 0.08, "high_risk": 0.18}
    freq = np.array([freq_map[a] for a in archetype])
    n_claims = rng.poisson(freq * exposure).astype(float)

    return df, n_claims, exposure


# Year 1: 5,000 policies
df_2023, y_2023, exp_2023 = generate_motor_portfolio(5000, seed=42)

# Year 2: 5,500 policies — shift towards more high-risk (aggregator growth)
# We achieve this by oversampling the high_risk archetype
df_2024_main, y_2024_main, exp_2024_main = generate_motor_portfolio(3500, seed=43)
df_2024_agg, y_2024_agg, exp_2024_agg = generate_motor_portfolio(
    2000, seed=44
)
# Bias the aggregator cohort towards high-risk by re-generating with shifted params
rng_bias = np.random.default_rng(44)
df_2024_agg["age_band"] = rng_bias.choice(
    ["17-25", "26-35"], size=len(df_2024_agg), p=[0.6, 0.4]
)
df_2024_agg["area"] = rng_bias.choice(
    ["urban", "urban", "suburban"], size=len(df_2024_agg)
)

df_2024 = pd.concat([df_2024_main, df_2024_agg], ignore_index=True)
y_2024 = np.concatenate([y_2024_main, y_2024_agg])
exp_2024 = np.concatenate([exp_2024_main, exp_2024_agg])

print(f"2023 portfolio: {len(df_2023):,} policies, "
      f"{y_2023.sum():.0f} claims, "
      f"freq = {y_2023.sum() / exp_2023.sum():.4f}")
print(f"2024 portfolio: {len(df_2024):,} policies, "
      f"{y_2024.sum():.0f} claims, "
      f"freq = {y_2024.sum() / exp_2024.sum():.4f}")
print(df_2023.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Encode portfolio

# COMMAND ----------

cat_cols = ["vehicle_group", "area", "age_band", "ncb_band"]
cont_cols = ["vehicle_age", "annual_mileage"]

enc = InsuranceLDAEncoder(missing_as_modality=True)
X_2023 = enc.fit_transform(df_2023, cat_cols=cat_cols, cont_cols=cont_cols, n_bins=8)
X_2024 = enc.transform(df_2024)

print(f"Encoded matrix shape: {X_2023.shape}")
print(f"Vocabulary size V = {enc.n_modalities_}")
print(f"\nSample vocabulary terms:")
for term in enc.feature_names_[:12]:
    print(f"  {enc.vocabulary_[term]:>3}: {term}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Select optimal K

# COMMAND ----------

# Note: This runs 5-fold CV for K=2..12, fitting 55 LDA models.
# On Databricks serverless it takes ~2-3 minutes for 5,000 policies.

selector = TopicSelector(
    k_range=range(2, 13),
    cv=5,
    distribution="poisson",
    random_state=42,
)
optimal_k = selector.select(X_2023, y_claims=y_2023, exposure=exp_2023)

print(f"\nOptimal K = {optimal_k}")
print(f"\nDeviance by K:")
print(selector.scores_.to_string(index=False))

fig = selector.plot_elbow()
fig.savefig("/tmp/k_selection_elbow.png", dpi=120, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fit LDA and inspect risk profiles

# COMMAND ----------

profiler = LDARiskProfiler(
    n_topics=optimal_k,
    random_state=42,
    max_iter=50,
)
theta_2023 = profiler.fit_transform(X_2023)

print(f"Theta shape: {theta_2023.shape}")
print(f"LDA perplexity (training): {profiler.perplexity_:.2f}")
print(f"\nSample policy topic distributions (first 5):")
df_theta = pd.DataFrame(
    theta_2023[:5],
    columns=[f"Topic {k}" for k in range(optimal_k)]
)
print(df_theta.round(3).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dominant modalities per topic

# COMMAND ----------

top = profiler.top_modalities_per_topic(enc.feature_names_, top_n=6)

for k in range(optimal_k):
    print(f"\n--- Topic {k} ---")
    for modality, prob in top[k]:
        print(f"  {prob:.4f}  {modality}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validate topics against claim outcomes

# COMMAND ----------

validator = TopicValidator(distribution="poisson")
result = validator.validate(theta_2023, y_2023, exposure=exp_2023)

print(f"Model deviance:          {result.deviance:.2f}")
print(f"Null deviance:           {result.null_deviance:.2f}")
print(f"Deviance reduction:      {result.deviance_reduction:.1%}")
print(f"\nPer-topic statistics:")
print(result.summary.to_string())

fig = result.plot_frequencies()
fig.savefig("/tmp/topic_frequencies.png", dpi=120, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Detect portfolio drift (2023 -> 2024)

# COMMAND ----------

theta_2024 = profiler.transform(X_2024)

drift = PortfolioDrift(profiler, alert_threshold=0.05)
drift_result = drift.compute_drift(
    theta_2023, theta_2024,
    weights_t0=exp_2023, weights_t1=exp_2024,
    labels=("2023", "2024"),
)

print(f"JSD (2023 -> 2024): {drift_result.jsd:.4f}")
print(f"Alert triggered:    {drift_result.alert}")
print(f"\nPer-topic composition shift (2023 -> 2024):")
print(drift_result.per_topic_shift.round(4).to_string())

fig = drift_result.plot_shift()
fig.savefig("/tmp/drift_shift.png", dpi=120, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stacked composition chart

# COMMAND ----------

fig = drift.plot_composition(
    [theta_2023, theta_2024],
    labels=["2023", "2024"],
    weights_by_period=[exp_2023, exp_2024],
)
fig.savefig("/tmp/composition_chart.png", dpi=120, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary
# MAGIC
# MAGIC What we found on this synthetic portfolio:
# MAGIC
# MAGIC - **K selection** identified the elbow in held-out Poisson deviance — more
# MAGIC   informative than perplexity because it directly measures claim frequency
# MAGIC   discrimination.
# MAGIC - **Risk profiles** correspond to interpretable archetypes: low-risk
# MAGIC   mature rural drivers, standard suburban commuters, high-risk young urban
# MAGIC   drivers.
# MAGIC - **Deviance reduction** confirms the topics are doing actuarial work —
# MAGIC   each topic has a materially different claim frequency.
# MAGIC - **Drift detection** (JSD = 0.05+) flags the composition shift as the
# MAGIC   2024 book took on more aggregator-sourced young urban risk.  A flat
# MAGIC   renewal rate change would have under-priced this book.
# MAGIC
# MAGIC ### Production usage
# MAGIC
# MAGIC ```python
# MAGIC # Fit once on historical portfolio
# MAGIC enc = InsuranceLDAEncoder()
# MAGIC X_hist = enc.fit_transform(df_hist, cat_cols, cont_cols)
# MAGIC profiler = LDARiskProfiler(n_topics=8, random_state=42)
# MAGIC profiler.fit(X_hist)
# MAGIC
# MAGIC # Score new policies at quote time
# MAGIC X_new = enc.transform(df_new)
# MAGIC theta_new = profiler.transform(X_new)
# MAGIC
# MAGIC # Use theta as features in downstream GLM or GBM
# MAGIC features = np.hstack([X_numeric, theta_new])
# MAGIC
# MAGIC # Monthly drift monitoring
# MAGIC drift = PortfolioDrift(profiler, alert_threshold=0.05)
# MAGIC result = drift.compute_drift(theta_hist, theta_new, labels=("historical", "current"))
# MAGIC if result.alert:
# MAGIC     send_alert(f"Portfolio drift detected: JSD = {result.jsd:.3f}")
# MAGIC ```

# COMMAND ----------

print("Demo complete.")
print(f"Outputs saved to /tmp/")
