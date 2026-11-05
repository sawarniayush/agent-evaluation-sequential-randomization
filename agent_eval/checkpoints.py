"""Separate ImpossibleBench coefficients at each of five checkpoints."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def basis(panel, features, phase=None):
    f = np.asarray(features, float)
    stage = np.zeros(len(f), int) if phase is None else np.asarray(phase)
    levels = sorted(np.unique(stage))
    matrices, blocks, support, column_phases = [], [], [], []
    start = 0
    for s in levels:
        mask = stage == s
        _, singular, vt = np.linalg.svd(f[mask], full_matrices=False)
        tolerance = max(f[mask].shape) * max(singular[0], 1.0) * 1e-10
        rank = int((singular > tolerance).sum())
        transform = vt[:rank].T * (np.sqrt(mask.sum()) / singular[:rank])
        values = np.zeros((len(f), rank))
        values[mask] = f[mask] @ transform
        matrices.append(values)
        blocks.append(np.arange(start, start + rank))
        start += rank
        column_phases.extend([s] * rank)
        trajectories = len(np.unique(panel.get("trajectory_index", panel["task_index"])[mask]))
        help_count = int(panel["treatment"][mask].sum())
        n = int(mask.sum())
        support.append(
            dict(
                phase=int(s),
                decisions=n,
                trajectories=trajectories,
                dimension=rank,
                help=help_count,
                skip=n - help_count,
                candidate_dimension=f.shape[1],
                supported=bool(
                    rank
                    and trajectories >= rank + 5
                    and n >= max(20, 2 * rank)
                    and min(help_count, n - help_count) >= 5
                ),
                transform=transform.tolist(),
            )
        )
    return dict(
        features=np.column_stack(matrices),
        blocks=blocks,
        support=support,
        phase=stage,
        column_phases=np.asarray(column_phases),
        ordered=phase is not None,
    )


class FutureRegression:
    """Training-only scaling and ridge controls; same model for all continuations."""

    def __init__(self, x, folds):
        self.folds = []
        for k in np.unique(folds):
            train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            scaler = StandardScaler().fit(x[train])
            self.folds.append((train, test, scaler.transform(x[train]), scaler.transform(x[test])))

    def predict(self, future, design, final_horizon):
        result = np.zeros_like(future)
        for train, test, xt, xv in self.folds:
            model = Ridge(alpha=10.0, solver="cholesky").fit(xt, future[train])
            # Some sklearn versions squeeze a single response column. Keep the
            # declared future-feature dimension after exact alias removal.
            result[test] = np.asarray(model.predict(xv)).reshape(len(test), -1)
        if design["ordered"]:
            result[design["phase"][:, None] > design["column_phases"][None, :]] = 0.0
        # Only the prespecified cap is known before assignment. Do not zero at
        # the last *observed* block, whose termination can depend on treatment.
        result[final_horizon] = 0.0
        return result
