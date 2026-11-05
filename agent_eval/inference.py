"""Coefficient-aware trajectory influence functions and pointwise t intervals."""

import numpy as np
from scipy.stats import t as student_t


def coefficient_influence(j_by_trajectory, b_by_trajectory, beta):
    """First-order coefficient contributions from independent trajectories."""
    j_by_trajectory = np.asarray(j_by_trajectory, float)
    b_by_trajectory = np.asarray(b_by_trajectory, float)
    beta = np.asarray(beta, float)
    scores = b_by_trajectory - np.einsum("tij,j->ti", j_by_trajectory, beta)
    return np.linalg.solve(j_by_trajectory.sum(0), scores.T).T


def ratio_effect(feature_totals, counts, fit):
    denominator = counts.sum()
    if denominator == 0 or not fit["supported"]:
        return np.nan, None
    total_features = feature_totals.sum(0)
    point = float(total_features @ fit["beta"] / denominator)
    influence = (
        feature_totals @ fit["beta"] - point * counts + fit["derivative"] @ total_features
    ) / denominator
    return point, influence


def policy_value(outcomes, episode_counts, correction, fit):
    if not fit["supported"]:
        return np.nan, None
    total = correction.sum(0)
    n = episode_counts.sum()
    point = float((outcomes.sum() + total @ fit["beta"]) / n)
    influence = (
        outcomes + correction @ fit["beta"] - point * episode_counts + fit["derivative"] @ total
    ) / n
    return point, influence


def interval(point, influence):
    if influence is None or len(influence) < 2 or not np.isfinite(influence).all():
        return dict(se=None, low=None, high=None)
    g = len(influence)
    se = float(np.sqrt(g / (g - 1) * np.sum(influence**2)))
    half = float(student_t.ppf(0.975, g - 1)) * se
    return dict(se=se, low=float(point - half), high=float(point + half))
