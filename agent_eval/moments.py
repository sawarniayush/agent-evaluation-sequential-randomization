"""Chronological moments, accumulated within complete trajectories."""

import numpy as np


def observed_basis(f):
    """Orthonormal basis for empirically estimable feature combinations; no outcomes."""
    _, s, v = np.linalg.svd(f, full_matrices=False)
    rank = int(np.sum(s > max(f.shape) * max(s[0], 1.0) * 1e-10))
    return v[:rank].T


def future_sum(panel: dict, projected: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    """Sum later proposal features; the current proposal is excluded."""
    result = np.zeros_like(projected)
    for eid in np.unique(panel["episode_id"]):
        ii = np.flatnonzero(panel["episode_id"] == eid)
        ii = ii[np.argsort(panel["t"][ii])]
        running = np.zeros(projected.shape[1])
        for row in ii[::-1]:
            result[row] = running
            running = running + multiplier[row] * projected[row]
    return result


def clustered_cross(panel: dict, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros((panel["g"], left.shape[1], right.shape[1]))
    for k in range(panel["g"]):
        ii = panel["task_index"] == k
        result[k] = left[ii].T @ right[ii]
    return result


def clustered_rhs(panel: dict, instrument: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = np.zeros((panel["g"], instrument.shape[1]))
    np.add.at(result, panel["task_index"], instrument * residual[:, None])
    return result


def correction_sums(panel: dict, projected: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = np.zeros((panel["g"], projected.shape[1]))
    np.add.at(
        result,
        panel["task_index"],
        (target - panel["treatment"])[:, None] * projected,
    )
    return result


def aggregate(panel, values):
    result = np.zeros((panel["g"], *values.shape[1:]))
    np.add.at(result, panel.get("trajectory_index", panel["task_index"]), values)
    return result
