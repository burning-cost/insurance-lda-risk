"""
Shared dataclasses and result types.

These are the structured outputs from TopicValidator and PortfolioDrift.
They carry both the numeric results and rendering methods so callers do not
have to import matplotlib themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure


@dataclass
class TopicStats:
    """Per-topic summary statistics from a TopicValidator run.

    Attributes
    ----------
    topic_id : int
        Zero-based topic index.
    claim_frequency : float
        Exposure-adjusted claim frequency for policies weighted by this topic.
        For Poisson: E[claims / exposure]. For Binomial: E[claims / policy].
    total_exposure : float
        Sum of exposure attributed to this topic (sum of membership * exposure).
    total_claims : float
        Expected claims attributed to this topic.
    pct_policies : float
        Fraction of portfolio policies with this topic as the plurality topic.
    dominant_modalities : list[str]
        Top modality names by topic-word weight (requires encoder).
    """

    topic_id: int
    claim_frequency: float
    total_exposure: float
    total_claims: float
    pct_policies: float
    dominant_modalities: list[str] = field(default_factory=list)


@dataclass
class TopicValidationResult:
    """Full output from TopicValidator.validate().

    Attributes
    ----------
    topic_stats : list[TopicStats]
        Per-topic statistics, ordered by topic_id.
    deviance : float
        Overall model deviance (Poisson or Binomial).
    null_deviance : float
        Deviance of the null model (constant frequency = portfolio mean).
    deviance_reduction : float
        Proportional deviance reduction vs null: 1 - deviance / null_deviance.
    n_topics : int
        Number of topics.
    distribution : str
        ``'poisson'`` or ``'binomial'``.
    summary : pd.DataFrame
        Table form of topic_stats for display.
    """

    topic_stats: list[TopicStats]
    deviance: float
    null_deviance: float
    deviance_reduction: float
    n_topics: int
    distribution: str

    @property
    def summary(self) -> pd.DataFrame:
        """Return per-topic statistics as a DataFrame."""
        rows = [
            {
                "topic": s.topic_id,
                "claim_frequency": round(s.claim_frequency, 6),
                "total_exposure": round(s.total_exposure, 2),
                "total_claims": round(s.total_claims, 4),
                "pct_policies": round(s.pct_policies * 100, 2),
            }
            for s in self.topic_stats
        ]
        return pd.DataFrame(rows).set_index("topic")

    def plot_frequencies(self) -> "matplotlib.figure.Figure":
        """Bar chart of claim frequency by topic, sorted low to high.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        df = self.summary.reset_index().sort_values("claim_frequency")
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            range(len(df)),
            df["claim_frequency"],
            color=plt.cm.RdYlGn_r(  # type: ignore[attr-defined]
                np.linspace(0.2, 0.8, len(df))
            ),
        )
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f"Topic {t}" for t in df["topic"]], rotation=45, ha="right")
        ax.set_ylabel("Claim frequency")
        ax.set_title(
            f"Claim frequency by risk profile  "
            f"(deviance reduction {self.deviance_reduction:.1%})"
        )
        ax.axhline(
            sum(s.total_claims for s in self.topic_stats)
            / max(sum(s.total_exposure for s in self.topic_stats), 1e-9),
            color="k",
            linestyle="--",
            linewidth=0.8,
            label="Portfolio mean",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig


@dataclass
class DriftResult:
    """Output from PortfolioDrift.compute_drift().

    Attributes
    ----------
    jsd : float
        Jensen-Shannon divergence between the two portfolio-level topic
        distributions. Range [0, 1]; 0 = identical, 1 = completely different.
    per_topic_shift : pd.Series
        Signed shift in topic weight (t1 - t0) per topic. Positive = growing.
    alert : bool
        True if JSD exceeds ``alert_threshold``.
    alert_threshold : float
        The threshold used when computing ``alert``.
    labels : tuple[str, str]
        Human-readable period labels.
    """

    jsd: float
    per_topic_shift: pd.Series
    alert: bool
    alert_threshold: float
    labels: tuple[str, str] = ("t0", "t1")

    def plot_shift(self) -> "matplotlib.figure.Figure":
        """Horizontal bar chart of per-topic composition shift.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        series = self.per_topic_shift.sort_values()
        colours = ["#d73027" if v > 0 else "#4575b4" for v in series]
        fig, ax = plt.subplots(figsize=(7, max(3, len(series) * 0.4)))
        ax.barh(
            [f"Topic {i}" for i in series.index],
            series.values,
            color=colours,
        )
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel(f"Composition shift  ({self.labels[0]} → {self.labels[1]})")
        ax.set_title(f"Portfolio drift  JSD = {self.jsd:.4f}" + ("  ⚠ ALERT" if self.alert else ""))
        fig.tight_layout()
        return fig
