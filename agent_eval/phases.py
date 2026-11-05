"""The paper's Retail and ToolSandbox coefficient structures."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .moments import observed_basis, future_sum, clustered_cross, clustered_rhs


def stage_skeleton(panel, stage, f):
    active = np.any(f != 0, axis=1)
    bases = []
    for s in range(panel["phase_count"]):
        values = f[(stage == s) & active]
        bases.append(observed_basis(values) if len(values) else np.zeros((f.shape[1], 0)))
    edges = np.r_[0, np.cumsum([basis.shape[1] for basis in bases])]
    blocks = [np.arange(edges[s], edges[s + 1]) for s in range(panel["phase_count"])]
    projected = np.zeros((len(f), edges[-1]))
    support = []
    for s, (basis, block) in enumerate(zip(bases, blocks)):
        ii = (stage == s) & active
        projected[np.ix_(ii, block)] = f[ii] @ basis
        treated, n, rank = int(panel["treatment"][ii].sum()), int(ii.sum()), int(len(block))
        trajectories = int(len(np.unique(panel["trajectory_index"][ii])))
        support.append(
            dict(
                stage=s,
                name=str(s),
                n=n,
                treated=treated,
                skip=n - treated,
                trajectories=trajectories,
                rank=rank,
                feature_count=f.shape[1],
                supported=bool(
                    rank == 0
                    or (
                        n >= max(20, 2 * rank)
                        and min(treated, n - treated) >= 5
                        and trajectories >= 5
                    )
                ),
            )
        )
    return dict(v=bases, blocks=blocks, projected=projected, support=support)


def predict_future(panel, stage, blocks, future):
    """Cross-fitted pre-assignment variance control for future feature totals."""
    prediction = np.zeros_like(future)
    x, fold = panel["target_control_x"], panel["fold"]
    for k in sorted(np.unique(fold)):
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        active = np.flatnonzero(np.any(future[train] != 0, axis=0))
        if not len(train) or not len(test) or not len(active):
            continue
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=10.0, solver="cholesky")
        model.fit(scaler.transform(x[train]), future[np.ix_(train, active)])
        prediction[np.ix_(test, active)] = model.predict(scaler.transform(x[test]))
    # Monotonicity makes future features from an earlier progress stage impossible.
    for s, block in enumerate(blocks):
        prediction[np.ix_(stage > s, block)] = 0.0
    return prediction


def pooled_candidates(panel):
    """Three phase baselines plus one shared coefficient per remaining feature."""
    intercept = panel["names"].index("intercept")
    keep = [k for k in range(len(panel["names"])) if k != intercept]
    names = [f"phase:{s}" for s in range(panel["phase_count"])] + [panel["names"][k] for k in keep]
    values = np.column_stack([np.eye(panel["phase_count"])[panel["stage"]], panel["f"][:, keep]])
    return values, names, intercept, keep


def pooled_system(panel, outcome, q):
    """ToolSandbox joint moments with phase baselines and shared slopes."""
    stage, action = panel["stage"], panel["treatment"]
    skeleton = stage_skeleton(panel, stage, panel["f"])
    candidate, names, intercept, keep = pooled_candidates(panel)
    basis = observed_basis(candidate)
    mapping = np.zeros((skeleton["projected"].shape[1], len(names)))
    for s, (phase_basis, block) in enumerate(zip(skeleton["v"], skeleton["blocks"])):
        raw_map = np.zeros((panel["f"].shape[1], len(names)))
        raw_map[intercept, s] = 1.0
        raw_map[np.ix_(keep, np.arange(panel["phase_count"], len(names)))] = np.eye(len(keep))
        mapping[block] = phase_basis.T @ raw_map
    mapping = mapping @ basis
    projected = skeleton["projected"] @ mapping
    if not np.allclose(projected, candidate @ basis, atol=1e-10, rtol=1e-10):
        raise ValueError(
            "Phase-expanded nuisance map does not reproduce the pooled effect features"
        )
    residual = (action - 0.5)[:, None] * projected
    observed = future_sum(panel, skeleton["projected"], action)
    predicted = predict_future(panel, stage, skeleton["blocks"], observed)
    future_residual = (observed - predicted) @ mapping
    n, treated, rank = len(action), int(action.sum()), projected.shape[1]
    trajectories = len(np.unique(panel["trajectory_index"]))
    support = dict(
        n=n,
        treated=treated,
        skip=n - treated,
        trajectories=trajectories,
        rank=rank,
        feature_count=len(names),
        supported=bool(
            rank > 0
            and n >= max(20, 2 * rank)
            and min(treated, n - treated) >= 5
            and trajectories >= 5
        ),
    )
    return dict(
        **{k: v for k, v in skeleton.items() if k != "projected"},
        projected=projected,
        candidate_features=candidate,
        candidate_names=names,
        effect_basis=basis,
        expanded_to_effect=mapping,
        future_projected=skeleton["projected"],
        fit_blocks=[np.arange(rank)],
        fit_support=[support],
        stage_j=clustered_cross(panel, residual, residual + future_residual),
        stage_b=clustered_rhs(panel, residual, outcome - q),
        z=residual,
    )


def policy_future_residual(panel, system, target):
    """Future target-action features, controlled at the current history."""
    expanded = system.get("future_projected", system["projected"])
    future = future_sum(panel, expanded, target)
    predicted = predict_future(panel, panel["stage"], system["blocks"], future)
    residual = future - predicted
    return residual @ system["expanded_to_effect"] if "expanded_to_effect" in system else residual
