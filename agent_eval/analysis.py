"""Fit the three paper estimands with their experiment-specific specifications."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import checkpoints, inference, moments, phases

ROOT = Path(__file__).resolve().parents[1]
STUDIES = (
    "retail_nano",
    "retail_haiku",
    "toolsandbox_nano",
    "toolsandbox_haiku",
    "impossiblebench",
)


def load(study):
    folder = ROOT / "data" / study
    metadata = json.loads((folder / "metadata.json").read_text())
    with np.load(folder / "panel.npz", allow_pickle=False) as saved:
        panel = dict(saved)
    panel.update(
        g=metadata["trajectories"],
        names=metadata["effect_features"],
        trajectory_index=panel["task_index"],
        episode_counts=np.ones(metadata["trajectories"]),
        phase_count=4 if study.startswith("retail") else 3,
    )
    if not np.array_equal(panel["reward"], panel["outcome_sums"][panel["task_index"]]):
        raise ValueError("Decision outcomes differ from trajectory outcomes")
    for i in range(panel["g"]):
        rows = panel["task_index"] == i
        if len(np.unique(panel["fold"][rows])) > 1:
            raise ValueError("A trajectory crosses nuisance folds")
        if np.any(np.diff(panel["t"][rows]) <= 0):
            raise ValueError("Decision order is not strictly increasing")
    return panel, metadata


def outcome_predictions(panel, separate_arms):
    """Five-fold gradient boosting on the exact recorded history columns."""
    x, a, y, fold = (panel[k] for k in ("x", "treatment", "reward", "fold"))
    result = {k: np.zeros(len(y)) for k in ("m0", "m1", "q")}
    for k in sorted(np.unique(fold)):
        train, test = fold != k, fold == k
        settings = dict(
            loss="squared_error",
            learning_rate=0.1,
            max_iter=120,
            max_leaf_nodes=12,
            min_samples_leaf=20,
            l2_regularization=2.0,
            random_state=42 + int(k),
        )
        if separate_arms:
            for name, arm in (("q", None), ("m0", 0), ("m1", 1)):
                selected = train if arm is None else train & (a == arm)
                model = HistGradientBoostingRegressor(**settings, early_stopping=False)
                model.fit(x[selected], y[selected])
                result[name][test] = model.predict(x[test])
        else:
            model = HistGradientBoostingRegressor(**settings, early_stopping="auto")
            model.fit(np.column_stack([x[train], a[train]]), y[train])
            for arm in (0, 1):
                result[f"m{arm}"][test] = model.predict(
                    np.column_stack([x[test], np.full(int(test.sum()), arm)])
                )
    if not separate_arms:
        result["q"] = (result["m0"] + result["m1"]) / 2
    return result


def solve(j_by_episode, b_by_episode, blocks, support):
    """Unpenalized joint moments, solved in reverse block order."""
    j, b = j_by_episode.sum(0), b_by_episode.sum(0)
    rank, condition = int(np.linalg.matrix_rank(j)), float(np.linalg.cond(j))
    supported = rank == len(b) and condition < 1e8 and all(s["supported"] for s in support)
    beta = np.zeros_like(b)
    try:
        for block in reversed(blocks):
            if len(block):
                beta[block] = np.linalg.solve(j[np.ix_(block, block)], b[block] - j[block] @ beta)
    except np.linalg.LinAlgError:
        supported = False
        beta[:] = np.nan
    if supported:
        error = float(
            np.max(np.abs(j @ beta - b), initial=0) / max(1.0, np.max(np.abs(b), initial=0))
        )
        if not np.isfinite(error) or error > 1e-7:
            raise ValueError("Moment reconstruction failed")
    derivative = (
        inference.coefficient_influence(j_by_episode, b_by_episode, beta) if supported else None
    )
    return dict(
        beta=beta,
        derivative=derivative,
        supported=supported,
        rank=rank,
        dimension=len(b),
        condition=condition,
    )


def fit(study, refit_nuisance=False):
    panel, metadata = load(study)
    steering = study == "impossiblebench"
    pooled = study.startswith("toolsandbox")
    nuisance = (
        outcome_predictions(panel, steering)
        if refit_nuisance
        else {k: panel[k] for k in ("m0", "m1", "q")}
    )
    if refit_nuisance:
        for key in nuisance:
            np.testing.assert_allclose(
                nuisance[key],
                panel[key],
                atol=1e-8,
                rtol=1e-8,
                err_msg=f"{study}: {key} nuisance predictions differ",
            )
    a, y, stage = panel["treatment"], panel["reward"], panel["stage"]
    pseudo = (
        nuisance["m1"]
        - nuisance["m0"]
        + 2 * a * (y - nuisance["m1"])
        - 2 * (1 - a) * (y - nuisance["m0"])
    )
    if steering:
        design = checkpoints.basis(panel, panel["f"], phase=stage)
        f, blocks, support = design["features"], design["blocks"], design["support"]
        controls = checkpoints.FutureRegression(panel["x"], panel["fold"])

        def future_residual(target):
            future = moments.future_sum(panel, f, target)
            predicted = controls.predict(future, design, panel["t"] == 5)
            predicted[panel["t"][:, None] >= design["column_phases"][None, :]] = 0.0
            return future - predicted
    elif pooled:
        design = phases.pooled_system(panel, y, nuisance["q"])
        f, blocks, support = design["projected"], design["fit_blocks"], design["fit_support"]

        def future_residual(target):
            return phases.policy_future_residual(panel, design, target)
    else:
        design = phases.stage_skeleton(panel, stage, panel["f"])
        f, blocks, support = design["projected"], design["blocks"], design["support"]

        def future_residual(target):
            future = moments.future_sum(panel, f, target)
            return future - phases.predict_future(panel, stage, blocks, future)

    z = (a - 0.5)[:, None] * f
    rhs = moments.clustered_rhs(panel, z, y - nuisance["q"])
    local_j = moments.clustered_cross(panel, z, z)
    local = solve(local_j, rhs, blocks, support)
    observed_residual = future_residual(a)
    never_j = moments.clustered_cross(panel, z, z + observed_residual)
    never = solve(never_j, rhs, blocks, support)
    zero = np.zeros(len(a))

    def value(target, model):
        correction = moments.correction_sums(panel, f, target)
        return inference.policy_value(
            panel["outcome_sums"], panel["episode_counts"], correction, model
        )

    reference, reference_if = value(zero, never)
    rows, contributions, cache = [], {}, {}
    unreachable = set(metadata["structural_zero_policies"])

    def keep(group, mask, target, method, point, derivative, fit_ok=True, exact=False):
        n, treated = int(mask.sum()), int(a[mask].sum())
        trajectories = len(np.unique(panel["task_index"][mask]))
        group_ok = n >= 20 and min(treated, n - treated) >= 5 and trajectories >= 5
        ci = inference.interval(point, derivative)
        if exact:
            ci = dict(se=0.0, low=0.0, high=0.0)
        supported = bool(group_ok and fit_ok and ci["low"] is not None)
        if steering and not exact and ci["se"] is not None and ci["se"] <= 1e-12:
            supported = False
        if exact:
            supported = True
        row = dict(
            estimand=target,
            group_id=group["group_id"],
            method=method,
            supported=supported,
            matching_decisions=n,
            treated_decisions=treated,
            skip_decisions=n - treated,
            total_episodes=panel["g"],
            estimate=100 * point,
            lower=None if ci["low"] is None else 100 * ci["low"],
            upper=None if ci["high"] is None else 100 * ci["high"],
            standard_error=None if ci["se"] is None else 100 * ci["se"],
        )
        if steering and not supported:
            for key in ("estimate", "lower", "upper", "standard_error"):
                row[key] = np.nan
        rows.append(row)
        if derivative is not None:
            contributions[f"{target}|{method}|{group['group_id']}"] = derivative

    for j, group in enumerate(metadata["groups"]):
        mask = panel["group_masks"][:, j].astype(bool)
        counts = moments.aggregate(panel, mask.astype(float))
        total = counts.sum()
        point = float(pseudo[mask].mean()) if total else np.nan
        derivative = (
            (moments.aggregate(panel, pseudo * mask) - point * counts) / total if total else None
        )
        keep(group, mask, "blip_randomized", "AIPW", point, derivative)
        feature_totals = moments.aggregate(panel, f * mask[:, None])
        for target, model in (("blip_randomized", local), ("blip_no_future", never)):
            point, derivative = inference.ratio_effect(feature_totals, counts, model)
            fit_ok = model["supported"]
            if steering and target == "blip_no_future":
                fit_ok = fit_ok and np.isfinite(reference) and 0 <= reference <= 1
            keep(group, mask, target, "SNMM", point, derivative, fit_ok)
        if group["group_id"] in unreachable:
            keep(group, mask, "selective_policy", "SNMM", 0.0, np.zeros(panel["g"]), exact=True)
            continue
        key = mask.tobytes()
        if key not in cache:
            target_residual = future_residual(mask.astype(float))
            if steering:
                target_j = moments.clustered_cross(
                    panel, z, z + observed_residual - target_residual
                )
            else:
                target_j = never_j - moments.clustered_cross(panel, z, target_residual)
            model = solve(target_j, rhs, blocks, support)
            policy_value, policy_if = value(mask.astype(float), model)
            cache[key] = model, policy_value, policy_if
        model, policy_value, policy_if = cache[key]
        fit_ok = bool(
            never["supported"]
            and model["supported"]
            and 0 <= reference <= 1
            and 0 <= policy_value <= 1
        )
        derivative = None if policy_if is None or reference_if is None else policy_if - reference_if
        keep(group, mask, "selective_policy", "SNMM", policy_value - reference, derivative, fit_ok)
    return pd.DataFrame(rows), contributions


def verify(study, frame):
    keys = ["estimand", "group_id", "method"]
    expected = pd.read_csv(ROOT / "data" / study / "estimates.csv").set_index(keys).sort_index()
    actual = frame.set_index(keys).sort_index()
    if not actual.index.equals(expected.index):
        raise AssertionError(f"{study}: result rows differ")
    for field in (
        "supported",
        "matching_decisions",
        "treated_decisions",
        "skip_decisions",
        "total_episodes",
    ):
        np.testing.assert_array_equal(actual[field], expected[field], err_msg=f"{study}: {field}")
    for field in ("estimate", "lower", "upper", "standard_error"):
        np.testing.assert_allclose(
            actual[field],
            expected[field],
            rtol=1e-7,
            atol=1e-6,
            equal_nan=True,
            err_msg=f"{study}: {field}",
        )
    return len(actual)
